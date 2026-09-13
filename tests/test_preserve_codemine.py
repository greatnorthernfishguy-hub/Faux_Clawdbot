import json
from pathlib import Path

import pytest

import preserve_codemine as preservation

# ---- Changelog ----
# [2026-09-12] Codex — Verify preservation-only startup safety.
# What: Cover mount validation, non-creation, dry-run, complete copy, manifests,
#       unsafe archive placement, and fail-closed hash mismatch behavior.
# Why: The sleeping HF graph must not be omitted, replaced, or partially copied.
# How: Use isolated fake graph trees and mountinfo fixtures; never touch live data.
# -------------------


def make_graph(root: Path) -> Path:
    graph = root / "neurograph_worker"
    checkpoint = graph / "checkpoints"
    checkpoint.mkdir(parents=True)
    (checkpoint / "main.msgpack").write_bytes(b"topology")
    (checkpoint / "vector_db.npz").write_bytes(b"vectors")
    (checkpoint / "main.msgpack.activations.json").write_text('{"warm": true}')
    (graph / "tracts").mkdir()
    (graph / "tracts" / "experience.jsonl").write_text("one\ntwo\n")
    return graph


def mountinfo(path: Path, mount: Path) -> Path:
    table = path / "mountinfo"
    table.write_text(f"36 25 0:32 / {mount} rw,relatime - ext4 /dev/test rw\n")
    return table


def test_mount_parser_decodes_paths():
    points = preservation.mount_points(
        r"36 25 0:32 / /data\040disk rw - ext4 /dev/test rw" + "\n"
    )
    assert Path("/data disk") in points


def test_missing_graph_is_not_created(tmp_path):
    missing = tmp_path / "data" / "neurograph_worker"
    with pytest.raises(preservation.PreservationError, match="does not exist"):
        preservation.validate_source(missing)
    assert not missing.exists()


def test_rejects_nonmounted_persistent_root(tmp_path):
    data = tmp_path / "data"
    graph = make_graph(data)
    empty_mountinfo = tmp_path / "mountinfo"
    empty_mountinfo.write_text("36 25 0:32 / / rw - ext4 /dev/root rw\n")
    with pytest.raises(preservation.PreservationError, match="not a distinct"):
        preservation.assert_persistent_mount(graph, data, empty_mountinfo)


def test_dry_run_writes_nothing(tmp_path):
    data = tmp_path / "data"
    graph = make_graph(data)
    archive_root = data / "archives"
    result = preservation.preserve(
        graph,
        archive_root,
        data,
        dry_run=True,
        mountinfo_path=mountinfo(tmp_path, data),
        timestamp="20260913T010203Z",
    )
    assert result is None
    assert not archive_root.exists()


def test_rejects_archive_inside_graph(tmp_path):
    data = tmp_path / "data"
    graph = make_graph(data)
    with pytest.raises(preservation.PreservationError, match="must not be inside"):
        preservation.preserve(
            graph,
            graph / "archives",
            data,
            mountinfo_path=mountinfo(tmp_path, data),
        )


def test_preserves_whole_graph_and_self_describing_manifest(tmp_path, monkeypatch):
    data = tmp_path / "data"
    graph = make_graph(data)
    archive_root = data / "archives"
    monkeypatch.setenv("CODEMINE_CODE_REVISION", "abc123")

    result = preservation.preserve(
        graph,
        archive_root,
        data,
        mountinfo_path=mountinfo(tmp_path, data),
        timestamp="20260913T010203Z",
    )

    assert result == archive_root / "codemine-ng-20260913T010203Z"
    assert (result / "neurograph_worker/checkpoints/main.msgpack").read_bytes() == b"topology"
    assert (result / "neurograph_worker/checkpoints/vector_db.npz").read_bytes() == b"vectors"
    assert (result / "neurograph_worker/checkpoints/main.msgpack.activations.json").is_file()
    summary = json.loads((result / "MANIFEST.json").read_text())
    assert summary["file_count"] == 4
    assert summary["code_revision"] == "abc123"
    assert (result / "MANIFEST.csv").read_text().count("\n") == 5
    preservation.verify_archive(result)


def test_verification_detects_damage(tmp_path):
    data = tmp_path / "data"
    graph = make_graph(data)
    result = preservation.preserve(
        graph,
        data / "archives",
        data,
        mountinfo_path=mountinfo(tmp_path, data),
        timestamp="20260913T010203Z",
    )
    (result / "neurograph_worker/checkpoints/main.msgpack").write_bytes(b"damaged")
    with pytest.raises(preservation.PreservationError, match="does not match"):
        preservation.verify_archive(result)


def test_copy_mismatch_fails_closed_and_removes_temporary_archive(tmp_path, monkeypatch):
    data = tmp_path / "data"
    graph = make_graph(data)
    archive_root = data / "archives"
    real_inventory = preservation.inventory
    calls = 0

    def mismatched_inventory(root):
        nonlocal calls
        calls += 1
        records = real_inventory(root)
        if calls == 2:
            records[0] = preservation.FileRecord(records[0].path, records[0].size, "0" * 64)
        return records

    monkeypatch.setattr(preservation, "inventory", mismatched_inventory)
    with pytest.raises(preservation.PreservationError, match="does not match"):
        preservation.preserve(
            graph,
            archive_root,
            data,
            mountinfo_path=mountinfo(tmp_path, data),
            timestamp="20260913T010203Z",
        )

    assert not (archive_root / "codemine-ng-20260913T010203Z").exists()
    assert not list(archive_root.glob(".codemine-ng-20260913T010203Z.*"))


def test_cli_requires_environment_paths(monkeypatch):
    for variable in (
        "NEUROGRAPH_WORKSPACE_DIR",
        "CODEMINE_PRESERVATION_DIR",
        "CODEMINE_PERSISTENT_ROOT",
    ):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(SystemExit):
        preservation.parse_args([])


def test_verify_cli_does_not_require_runtime_environment(monkeypatch, tmp_path):
    for variable in (
        "NEUROGRAPH_WORKSPACE_DIR",
        "CODEMINE_PRESERVATION_DIR",
        "CODEMINE_PERSISTENT_ROOT",
    ):
        monkeypatch.delenv(variable, raising=False)
    args = preservation.parse_args(["--verify", str(tmp_path)])
    assert args.verify == tmp_path
