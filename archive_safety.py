"""Pure validation helpers for untrusted ZIP uploads."""

# ---- Changelog ----
# [2026-09-13] Codex — Bound archive destination as well as archive members.
# What: Derive a safe per-upload root outside the source repository.
# Why: A basename such as ..zip could widen extraction to the checkout root.
# How: Validate the environment root, extension, stem, and resolved destination.
# [2026-09-13] Codex — Refuse ZIP traversal and symlink members before extraction.
# What: Validate every member against the resolved extraction root.
# Why: Size checks alone do not prevent an uploaded archive from writing elsewhere.
# How: Reject absolute, parent, backslash, NUL, symlink, and resolved escape paths.
# -------------------

from __future__ import annotations

import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo


def safe_upload_extract_root(
    upload_root: Path,
    archive_name: str,
    *,
    repository_root: Path,
) -> Path:
    """Return a stable extraction directory outside the source repository."""
    root = Path(upload_root).expanduser().resolve()
    repository = Path(repository_root).expanduser().resolve()
    if root == repository or root.is_relative_to(repository):
        raise ValueError("upload root must be outside the source repository")

    basename = Path(archive_name).name
    if not basename or "\x00" in basename or "\\" in basename:
        raise ValueError("unsafe ZIP archive name")
    if not basename.lower().endswith(".zip"):
        raise ValueError("unsafe ZIP archive name")
    stem = basename[:-4]
    if stem in {"", ".", ".."}:
        raise ValueError("unsafe ZIP archive name")

    destination = (root / stem).resolve()
    if destination == root or not destination.is_relative_to(root):
        raise ValueError("ZIP extraction destination escapes upload root")
    return destination


def validated_zip_members(archive: ZipFile, extract_root: Path) -> list[ZipInfo]:
    """Return safe members or raise ``ValueError`` before any filesystem write."""
    root = Path(extract_root).resolve()
    members = archive.infolist()
    for info in members:
        name = info.filename
        member_path = Path(name)
        file_type = (info.external_attr >> 16) & 0o170000
        if (
            not name
            or "\x00" in name
            or "\\" in name
            or member_path.is_absolute()
            or ".." in member_path.parts
            or stat.S_ISLNK(file_type)
        ):
            raise ValueError(f"unsafe ZIP member path: {name!r}")
        destination = (root / member_path).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"ZIP member escapes extraction root: {name!r}")
    return members
