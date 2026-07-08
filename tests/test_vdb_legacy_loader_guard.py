# ---- Changelog ----
# [2026-07-08] Tabitha (TQB/QB build) — Test suite for the vdb legacy-loader guard
# What: New file. Covers PRD §7's four cases against the real SimpleVectorDB
#       (no mocks): (1) legacy np.savez round-trip — synthetic archive built
#       with the empirically confirmed shapes (object-array ids/content,
#       metadata as JSON strings like the real 2026-04-22 file, float32
#       (N,768) embeddings) migrates fully: ids/content/parsed-metadata/
#       normalized embeddings, backup-first bak, loud WARNING; (2) current-
#       format regression — JSON (under the realistic vector_db.npz name)
#       and .msgpack round-trips unchanged, no bak, no migration log;
#       (3) corrupt PK-magic and non-PK garbage files raise with the
#       original byte-identical and prior state untouched, plus loud-caller
#       behavior at both swallow sites (ng_topology_sync._export returns 0
#       with exactly one ERROR containing the path; NeuroGraphMemory boot
#       proceeds fail-soft with an empty vdb and exactly one ERROR
#       containing the path); (4) idempotency — post-migration save()
#       writes current format that reloads via the normal branch with no
#       re-migration, and the bak is never re-copied or overwritten even
#       when a legacy-magic file reappears at the path. Plus metadata edge
#       cases: unparseable string -> {}, JSON non-dict -> {}, real dict kept.
# Why: PRD "Codemine VDB Legacy-Loader Guard #364" §7 test plan. The guard
#      is a deploy gate: 1932 entries of accumulated worker memory hinge on
#      load() migrating the legacy archive instead of crashing into a
#      swallowed except.
# How: Same fixture patterns as sibling tests: real SimpleVectorDB
#      throughout; real NeuroGraphMemory against pytest's tmp_path with
#      Tonic disabled (tests/test_gsg_stamping.py pattern) for the boot-side
#      loud-caller case; ng_topology_sync._CLONE_DIR monkeypatched to a tmp
#      dir (tests/test_ng_topology_sync_export.py pattern). Migration
#      loudness observed via caplog on the "neurograph.vector_db",
#      "ng_topology_sync", and "neurograph" loggers.
# -------------------

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

import ng_topology_sync
from openclaw_hook import NeuroGraphMemory
from universal_ingestor import SimpleVectorDB


NG_CONFIG = {"tonic": {"enabled": False}}
MIGRATION_MARK = "LEGACY NPZ MIGRATION"
DIM = 768  # matches the real legacy archive's embedding dimension


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_legacy_npz(path, n=4, seed=42, content_prefix="chunk"):
    """A synthetic legacy archive with the empirically confirmed shapes:
    ids/content/metadata as object arrays, metadata as JSON *strings*
    (matching the real 2026-04-22 file), embeddings float32 (N, 768)
    deliberately NOT unit-norm so migration must normalize them."""
    rng = np.random.RandomState(seed)
    ids = np.array([f"id-{seed}-{i}" for i in range(n)], dtype=object)
    embeddings = (rng.randn(n, DIM) * 3.0).astype(np.float32)
    content = np.array([f"{content_prefix} {i}" for i in range(n)], dtype=object)
    metadata = np.array(
        [json.dumps({"strategy": "semantic", "position": i}) for i in range(n)],
        dtype=object,
    )
    np.savez(path, ids=ids, embeddings=embeddings, content=content,
             metadata=metadata)
    assert Path(path).read_bytes()[:4] == b"PK\x03\x04"
    return ids, embeddings, content, metadata


def _migration_records(caplog):
    return [r for r in caplog.records
            if r.name == "neurograph.vector_db" and MIGRATION_MARK in r.getMessage()]


# ---------------------------------------------------------------------------
# Case 1 — legacy round-trip (PRD §7.1)
# ---------------------------------------------------------------------------

def test_legacy_npz_migrates_full_roundtrip(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = tmp_path / "vector_db.npz"
    ids, embeddings, content, metadata = _write_legacy_npz(path, n=4)
    original_hash = _sha256(path)

    db = SimpleVectorDB()
    count = db.load(str(path))

    assert count == 4
    assert db.count() == 4
    assert set(db.all_ids()) == set(str(i) for i in ids)
    for i, entry_id in enumerate(str(x) for x in ids):
        assert db.content[entry_id] == str(content[i])
        # metadata JSON strings are parsed back into dicts
        assert db.metadata[entry_id] == json.loads(metadata[i])
        vec = db.embeddings[entry_id]
        assert vec.dtype == np.float32
        assert vec.shape == (DIM,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
        # direction preserved (normalization, not replacement)
        src = embeddings[i] / np.linalg.norm(embeddings[i])
        assert np.allclose(vec, src, atol=1e-5)

    # backup-first bak, byte-identical to the original; original untouched
    bak = Path(str(path) + ".legacy-npz-bak")
    assert bak.exists()
    assert _sha256(bak) == original_hash
    assert _sha256(path) == original_hash

    # loud migration log: WARNING with path, entry count, and bak path
    marks = _migration_records(caplog)
    assert len(marks) == 1
    assert marks[0].levelno == logging.WARNING
    msg = marks[0].getMessage()
    assert str(path) in msg and "1932" not in msg and "4" in msg
    assert str(bak) in msg


# ---------------------------------------------------------------------------
# Case 2 — current-format regression (PRD §7.2)
# ---------------------------------------------------------------------------

def _populated_db():
    db = SimpleVectorDB()
    rng = np.random.RandomState(7)
    for i in range(3):
        db.insert(f"cur-{i}", rng.randn(DIM).astype(np.float32),
                  f"current content {i}", {"n": i})
    return db


def test_current_json_under_npz_name_unchanged(tmp_path, caplog):
    """The realistic current-era case: JSON content under the vector_db.npz
    filename (SimpleVectorDB.save() picks JSON for any non-.msgpack path)."""
    caplog.set_level(logging.INFO)
    path = tmp_path / "vector_db.npz"
    src = _populated_db()
    src.save(str(path))
    assert Path(path).read_bytes()[:4] != b"PK\x03\x04"

    db = SimpleVectorDB()
    assert db.load(str(path)) == 3
    assert db.content["cur-1"] == "current content 1"
    assert db.metadata["cur-1"] == {"n": 1}
    assert not Path(str(path) + ".legacy-npz-bak").exists()
    assert _migration_records(caplog) == []


def test_current_msgpack_unchanged(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = tmp_path / "vector_db.msgpack"
    src = _populated_db()
    src.save(str(path))

    db = SimpleVectorDB()
    assert db.load(str(path)) == 3
    assert db.content["cur-2"] == "current content 2"
    assert db.metadata["cur-2"] == {"n": 2}
    assert abs(float(np.linalg.norm(db.embeddings["cur-0"])) - 1.0) < 1e-5
    assert not Path(str(path) + ".legacy-npz-bak").exists()
    assert _migration_records(caplog) == []


# ---------------------------------------------------------------------------
# Case 3 — corrupt/unknown files + loud callers (PRD §7.3)
# ---------------------------------------------------------------------------

def test_corrupt_pk_magic_raises_state_and_file_untouched(tmp_path):
    path = tmp_path / "vector_db.npz"
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    original_hash = _sha256(path)

    db = _populated_db()
    with pytest.raises(Exception):
        db.load(str(path))

    # parse-fully-first discipline: prior state untouched
    assert db.count() == 3 and "cur-0" in db.embeddings
    # original never deleted or overwritten
    assert _sha256(path) == original_hash
    # backup-first design: bak was made before the parse and stays
    bak = Path(str(path) + ".legacy-npz-bak")
    assert bak.exists() and _sha256(bak) == original_hash


def test_non_pk_garbage_raises_state_and_file_untouched(tmp_path):
    path = tmp_path / "vector_db.npz"
    path.write_text("this is not json {{{")
    original_hash = _sha256(path)

    db = _populated_db()
    with pytest.raises(Exception):
        db.load(str(path))

    assert db.count() == 3 and "cur-0" in db.embeddings
    assert _sha256(path) == original_hash
    # non-legacy path performs no backup
    assert not Path(str(path) + ".legacy-npz-bak").exists()


def test_export_returns_zero_and_logs_error_with_path(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(ng_topology_sync, "_CLONE_DIR", tmp_path / "clone")
    workspace = tmp_path / "workspace"
    (workspace / "checkpoints").mkdir(parents=True)
    vdb_path = workspace / "checkpoints" / "vector_db.npz"
    vdb_path.write_text("this is not json {{{")

    assert ng_topology_sync._export(str(workspace)) == 0

    errors = [r for r in caplog.records
              if r.name == "ng_topology_sync" and r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert str(vdb_path) in errors[0].getMessage()


def test_boot_proceeds_empty_and_logs_error_with_path(tmp_path, caplog):
    """openclaw_hook loud-caller: booting over a garbage vdb sidecar must
    fail soft (empty vdb, boot completes) and emit exactly one ERROR naming
    the path. Real NeuroGraphMemory over tmp_path, Tonic disabled — same
    proportionate-boot pattern as tests/test_gsg_stamping.py."""
    caplog.set_level(logging.INFO)
    (tmp_path / "checkpoints").mkdir(parents=True)
    vdb_path = tmp_path / "checkpoints" / "vector_db.npz"
    vdb_path.write_text("this is not json {{{")

    ng = NeuroGraphMemory(workspace_dir=str(tmp_path), config=NG_CONFIG)

    # fail-soft: boot completed, vdb empty, file untouched
    assert ng.vector_db.count() == 0
    assert vdb_path.read_text() == "this is not json {{{"

    errors = [r for r in caplog.records
              if r.name == "neurograph" and r.levelno == logging.ERROR
              and "Failed to restore vector DB" in r.getMessage()]
    assert len(errors) == 1
    assert str(vdb_path) in errors[0].getMessage()


# ---------------------------------------------------------------------------
# Case 4 — idempotency (PRD §7.4)
# ---------------------------------------------------------------------------

def test_post_migration_save_reloads_via_normal_branch(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = tmp_path / "vector_db.npz"
    _write_legacy_npz(path, n=4)

    db = SimpleVectorDB()
    db.load(str(path))
    bak = Path(str(path) + ".legacy-npz-bak")
    bak_state = (_sha256(bak), os.stat(bak).st_mtime_ns)

    # save() over the same path writes current format (JSON for .npz name)
    db.save(str(path))
    assert Path(path).read_bytes()[:4] != b"PK\x03\x04"

    db2 = SimpleVectorDB()
    assert db2.load(str(path)) == 4
    assert db2.content == db.content
    assert db2.metadata == db.metadata

    # exactly one migration ever logged (the first load), bak untouched
    assert len(_migration_records(caplog)) == 1
    assert (_sha256(bak), os.stat(bak).st_mtime_ns) == bak_state


def test_bak_never_recopied_or_overwritten(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = tmp_path / "vector_db.npz"
    _write_legacy_npz(path, n=3, seed=1, content_prefix="first")

    db = SimpleVectorDB()
    assert db.load(str(path)) == 3
    bak = Path(str(path) + ".legacy-npz-bak")
    bak_state = (_sha256(bak), os.stat(bak).st_mtime_ns)

    # a DIFFERENT legacy-magic file reappears at the path
    _write_legacy_npz(path, n=5, seed=2, content_prefix="second")
    assert db.load(str(path)) == 5
    assert db.content[db.all_ids()[0]].startswith("second")

    # migration ran again, but the original bak was NOT re-copied/overwritten
    assert len(_migration_records(caplog)) == 2
    assert (_sha256(bak), os.stat(bak).st_mtime_ns) == bak_state


# ---------------------------------------------------------------------------
# Metadata edge cases (sub-step 1 deviation, QB-approved)
# ---------------------------------------------------------------------------

def test_metadata_edge_cases(tmp_path):
    path = tmp_path / "vector_db.npz"
    rng = np.random.RandomState(9)
    np.savez(
        path,
        ids=np.array(["m-0", "m-1", "m-2"], dtype=object),
        embeddings=rng.randn(3, DIM).astype(np.float32),
        content=np.array(["a", "b", "c"], dtype=object),
        metadata=np.array(
            ["not json {{{",          # unparseable string -> {}
             "[1, 2]",                # JSON, but not a dict -> {}
             {"k": "v"}],             # already a dict -> kept
            dtype=object,
        ),
    )

    db = SimpleVectorDB()
    assert db.load(str(path)) == 3
    assert db.metadata["m-0"] == {}
    assert db.metadata["m-1"] == {}
    assert db.metadata["m-2"] == {"k": "v"}
