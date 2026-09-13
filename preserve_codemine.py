#!/usr/bin/env python3
"""Create a verified, read-only snapshot of Codemine's persistent NeuroGraph.

This module intentionally uses only the Python standard library and must remain
safe to run before any Codemine application or NeuroGraph module is imported.
"""

# ---- Changelog ----
# [2026-09-12] Codex — Add preservation-only Codemine NeuroGraph archive.
# What: Verify the persistent mount and graph, copy the entire workspace, hash
#       every file, write shape-appropriate manifests, and publish atomically.
# Why: Codemine's third CC hemisphere must be preserved before its sleeping
#      Hugging Face Space is upgraded or its retired synchronization is replaced.
# How: Standard-library-only startup path; no app, substrate, or Tonic imports.
# -------------------

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REQUIRED_CHECKPOINTS = (
    Path("checkpoints/main.msgpack"),
    Path("checkpoints/vector_db.npz"),
)


class PreservationError(RuntimeError):
    """Raised when preservation cannot be proved safe and complete."""


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    sha256: str


def _unescape_mount_field(value: str) -> str:
    """Decode the octal escapes used in /proc/self/mountinfo path fields."""
    for escaped, literal in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(escaped, literal)
    return value


def mount_points(mountinfo: str) -> set[Path]:
    """Return normalized mount points parsed from Linux mountinfo text."""
    points: set[Path] = set()
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 6 or "-" not in fields:
            continue
        points.add(Path(_unescape_mount_field(fields[4])).resolve())
    return points


def assert_persistent_mount(
    source: Path,
    persistent_root: Path,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> None:
    """Require the configured persistent root to be the source's mount."""
    source = source.resolve()
    persistent_root = persistent_root.resolve()
    try:
        source.relative_to(persistent_root)
    except ValueError as exc:
        raise PreservationError(
            f"source {source} is outside persistent root {persistent_root}"
        ) from exc

    try:
        points = mount_points(mountinfo_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PreservationError(f"cannot read mount table: {exc}") from exc

    if persistent_root not in points:
        raise PreservationError(
            f"persistent root {persistent_root} is not a distinct mounted filesystem"
        )

    nested = [
        point
        for point in points
        if point != persistent_root and point.is_relative_to(persistent_root)
    ]
    owners = [point for point in nested if source == point or source.is_relative_to(point)]
    if owners:
        owner = max(owners, key=lambda path: len(path.parts))
        raise PreservationError(
            f"source {source} is backed by unexpected nested mount {owner}"
        )


def validate_source(source: Path) -> Path:
    """Validate the existing graph without creating or modifying anything."""
    source = source.resolve()
    if not source.exists():
        raise PreservationError(f"graph source does not exist: {source}")
    if not source.is_dir():
        raise PreservationError(f"graph source is not a directory: {source}")
    for relative in REQUIRED_CHECKPOINTS:
        path = source / relative
        if not path.is_file():
            raise PreservationError(f"required graph checkpoint is missing: {path}")
        if path.stat().st_size == 0:
            raise PreservationError(f"required graph checkpoint is empty: {path}")
    return source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[FileRecord]:
    """Hash every regular file below root, rejecting special files and links."""
    records: list[FileRecord] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PreservationError(f"refusing graph symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PreservationError(f"refusing special graph entry: {relative}")
        records.append(FileRecord(relative, path.stat().st_size, sha256_file(path)))
    return records


def code_revision() -> str | None:
    """Return the deployed revision when HF or git makes it available."""
    for variable in ("SPACE_COMMIT_SHA", "CODEMINE_CODE_REVISION"):
        value = os.environ.get(variable, "").strip()
        if value:
            return value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_manifest(
    archive: Path,
    source: Path,
    records: Iterable[FileRecord],
    created_at: str,
) -> None:
    records = list(records)
    with (archive / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "size", "sha256"))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)

    summary = {
        "format_version": 1,
        "created_at": created_at,
        "source": str(source),
        "graph_copy": "neurograph_worker",
        "file_count": len(records),
        "total_bytes": sum(record.size for record in records),
        "code_revision": code_revision(),
        "required_checkpoints": [str(path) for path in REQUIRED_CHECKPOINTS],
    }
    (archive / "MANIFEST.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_manifest(archive: Path) -> tuple[dict, list[FileRecord]]:
    """Load and validate the two self-describing archive manifests."""
    try:
        summary = json.loads((archive / "MANIFEST.json").read_text(encoding="utf-8"))
        with (archive / "MANIFEST.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        records = [
            FileRecord(path=row["path"], size=int(row["size"]), sha256=row["sha256"])
            for row in rows
        ]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise PreservationError(f"cannot read archive manifest: {exc}") from exc
    if summary.get("format_version") != 1:
        raise PreservationError("unsupported or missing archive format_version")
    return summary, records


def verify_archive(archive: Path) -> None:
    """Prove a copied archive still exactly matches its recorded inventory."""
    archive = archive.resolve()
    summary, expected = read_manifest(archive)
    if summary.get("graph_copy") != "neurograph_worker":
        raise PreservationError("archive manifest has an invalid graph_copy path")
    graph_copy = archive / "neurograph_worker"
    if not graph_copy.is_dir():
        raise PreservationError(f"archived graph is missing: {graph_copy}")
    actual = inventory(graph_copy)
    if actual != expected:
        raise PreservationError("archived graph does not match its manifest")
    if summary.get("file_count") != len(actual):
        raise PreservationError("archive file count does not match its manifest")
    if summary.get("total_bytes") != sum(record.size for record in actual):
        raise PreservationError("archive byte count does not match its manifest")
    for relative in REQUIRED_CHECKPOINTS:
        if not (graph_copy / relative).is_file():
            raise PreservationError(f"required archived checkpoint is missing: {relative}")
    print(
        f"VERIFIED: {len(actual)} files "
        f"({sum(record.size for record in actual)} bytes) in {archive}"
    )


def preserve(
    source: Path,
    archive_root: Path,
    persistent_root: Path,
    *,
    dry_run: bool = False,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
    timestamp: str | None = None,
) -> Path | None:
    """Validate, copy, re-hash, and atomically publish one graph archive."""
    source = validate_source(source)
    assert_persistent_mount(source, persistent_root, mountinfo_path)
    assert_persistent_mount(archive_root, persistent_root, mountinfo_path)
    archive_root = archive_root.resolve()
    if archive_root == source or archive_root.is_relative_to(source):
        raise PreservationError(
            f"archive root {archive_root} must not be inside graph source {source}"
        )
    source_records = inventory(source)
    if not source_records:
        raise PreservationError(f"graph source contains no files: {source}")

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_path = archive_root.resolve() / f"codemine-ng-{stamp}"
    if final_path.exists():
        raise PreservationError(f"archive already exists: {final_path}")

    if dry_run:
        print(
            f"DRY RUN: validated {len(source_records)} files "
            f"({sum(record.size for record in source_records)} bytes) at {source}"
        )
        print(f"DRY RUN: would create {final_path}")
        return None

    archive_root.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{final_path.name}.", dir=archive_root))
    try:
        graph_copy = temp_path / "neurograph_worker"
        shutil.copytree(source, graph_copy, copy_function=shutil.copy2)
        copied_records = inventory(graph_copy)
        if copied_records != source_records:
            raise PreservationError("copied graph does not match source inventory")
        write_manifest(temp_path, source, source_records, created_at)
        temp_path.rename(final_path)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise

    print(
        f"PRESERVED: {len(source_records)} files "
        f"({sum(record.size for record in source_records)} bytes) to {final_path}"
    )
    return final_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="ARCHIVE",
        help="verify a previously copied archive; no runtime environment is required",
    )
    args = parser.parse_args(argv)
    if args.verify is not None:
        return args
    settings = {
        "source": "NEUROGRAPH_WORKSPACE_DIR",
        "archive_root": "CODEMINE_PRESERVATION_DIR",
        "persistent_root": "CODEMINE_PERSISTENT_ROOT",
    }
    for attribute, variable in settings.items():
        value = os.environ.get(variable, "").strip()
        if not value:
            parser.error(f"required environment variable is unset: {variable}")
        setattr(args, attribute, Path(value))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify is not None:
            verify_archive(args.verify)
        else:
            if os.environ.get("CODEMINE_PRESERVATION_DRY_RUN", "").lower() in {
                "1",
                "true",
                "yes",
            }:
                args.dry_run = True
            preserve(
                args.source,
                args.archive_root,
                args.persistent_root,
                dry_run=args.dry_run,
            )
    except PreservationError as exc:
        print(f"PRESERVATION REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
