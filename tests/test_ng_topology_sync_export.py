# ---- Changelog ----
# [2026-07-08] Ratchet (TQB/QB build) — Test suite for the fixed ng_topology_sync._export()
# What: New file. Covers _export() against a synthetic current-era vector_db.npz
#       sidecar (JSON, written by the real SimpleVectorDB.save()), the
#       no-checkpoint-file regression case, and the fail-silent path on a
#       legacy genuine-npz-era file.
# Why: PRD 2026-07-06-codemine-ng-topology-sync-checkpoint-fix §9 acceptance
#      criteria: (a) a save()-produced sidecar with >=1 non-empty content entry
#      yields codemine_export.jsonl with {"hash": <sha256-of-content>,
#      "content": <content>} lines; (b) a workspace with NO checkpoint file
#      returns 0 without raising.
# How: Real SimpleVectorDB (no mocks) writes the sidecar exactly as the runtime
#      writer does (openclaw_hook.py save path). _CLONE_DIR is monkeypatched to
#      a per-test tmp dir that is not a git repo, so _export()'s unchanged git
#      block fails at `git add` (exit 128), is swallowed by its existing except
#      (fail-silent by design), and no real clone/commit/push ever fires.
# -------------------

import hashlib
import json

import numpy as np
import pytest

import ng_topology_sync
from universal_ingestor import SimpleVectorDB


@pytest.fixture
def clone_dir(tmp_path, monkeypatch):
    """Isolated _CLONE_DIR — a tmp path outside any git repo.

    _export() mkdir-s <clone_dir>/ng_topology itself and writes the export
    there; the git commit/push block then runs `git -C <clone_dir> ...` with
    check=True, exits 128 (not a repository), and the function's existing
    except swallows it. Nothing real is cloned, committed, or pushed.
    """
    d = tmp_path / "clone"
    monkeypatch.setattr(ng_topology_sync, "_CLONE_DIR", d)
    return d


def _workspace_with_sidecar(tmp_path, entries):
    """Build a workspace whose checkpoints/vector_db.npz was written by the
    real SimpleVectorDB.save() — the exact current-era writer path."""
    workspace = tmp_path / "workspace"
    (workspace / "checkpoints").mkdir(parents=True)
    db = SimpleVectorDB()
    for node_id, content in entries:
        db.insert(id=node_id, embedding=np.array([0.1, 0.9, 0.2]), content=content, metadata={"source": "test"})
    db.save(str(workspace / "checkpoints" / "vector_db.npz"))
    return workspace


# ---------------------------------------------------------------------------
# Acceptance (a): current-era sidecar -> codemine_export.jsonl
# ---------------------------------------------------------------------------

def test_current_era_sidecar_is_json_not_npz(tmp_path):
    # Guard for the premise of the whole fix (QB Task-1 finding): despite the
    # .npz name, SimpleVectorDB.save() writes JSON to it.
    workspace = _workspace_with_sidecar(tmp_path, [("node-1", "some content")])
    raw = (workspace / "checkpoints" / "vector_db.npz").read_text()
    data = json.loads(raw)
    assert "entries" in data
    assert data["entries"]["node-1"]["content"] == "some content"


def test_export_writes_hash_content_jsonl(tmp_path, clone_dir):
    contents = [
        "a first raw experience shared across the three CC NG instances",
        "a second, distinct experience with unicode — naïve café ☕",
    ]
    workspace = _workspace_with_sidecar(
        tmp_path, [("node-{}".format(i), c) for i, c in enumerate(contents)]
    )

    written = ng_topology_sync._export(str(workspace))

    assert written == 2
    out = clone_dir / "ng_topology" / "codemine_export.jsonl"
    assert out.exists()
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    for line in lines:
        assert set(line.keys()) == {"hash", "content"}
        assert line["hash"] == hashlib.sha256(line["content"].encode()).hexdigest()
    assert sorted(l["content"] for l in lines) == sorted(contents)


def test_export_skips_empty_content_entries(tmp_path, clone_dir):
    workspace = _workspace_with_sidecar(
        tmp_path,
        [("node-real", "the only entry with actual content"), ("node-empty", "")],
    )

    written = ng_topology_sync._export(str(workspace))

    assert written == 1
    out = clone_dir / "ng_topology" / "codemine_export.jsonl"
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["content"] == "the only entry with actual content"


def test_export_never_creates_a_git_repo_in_clone_dir(tmp_path, clone_dir):
    # The git block must fail silently against the non-repo tmp _CLONE_DIR —
    # export still succeeds, and no repository appears as a side effect.
    workspace = _workspace_with_sidecar(tmp_path, [("node-1", "content survives git failure")])

    written = ng_topology_sync._export(str(workspace))

    assert written == 1
    assert not (clone_dir / ".git").exists()


# ---------------------------------------------------------------------------
# Acceptance (b): no checkpoint file -> 0, no raise
# ---------------------------------------------------------------------------

def test_export_returns_zero_when_workspace_has_no_checkpoints_dir(tmp_path, clone_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert ng_topology_sync._export(str(workspace)) == 0
    assert not (clone_dir / "ng_topology" / "codemine_export.jsonl").exists()


def test_export_returns_zero_when_checkpoints_dir_has_no_vdb(tmp_path, clone_dir):
    workspace = tmp_path / "workspace"
    (workspace / "checkpoints").mkdir(parents=True)

    assert ng_topology_sync._export(str(workspace)) == 0
    assert not (clone_dir / "ng_topology" / "codemine_export.jsonl").exists()


# ---------------------------------------------------------------------------
# Fail-silent on the legacy genuine-npz era file (pre-ce16873)
# ---------------------------------------------------------------------------

def test_export_fails_silent_on_legacy_genuine_npz(tmp_path, clone_dir):
    # A real legacy-era sidecar is a genuine numpy zip (PK magic), which the
    # JSON branch of SimpleVectorDB.load() cannot parse — _export()'s existing
    # except must swallow it and return 0, matching the module's by-design
    # fail-silent behavior.
    workspace = tmp_path / "workspace"
    (workspace / "checkpoints").mkdir(parents=True)
    np.savez(
        str(workspace / "checkpoints" / "vector_db.npz"),
        ids=np.array(["n1"]),
        content=np.array(["legacy content"]),
    )

    assert ng_topology_sync._export(str(workspace)) == 0
    assert not (clone_dir / "ng_topology" / "codemine_export.jsonl").exists()
