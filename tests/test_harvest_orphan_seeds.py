# ---- Changelog ----
# [2026-07-08] Meridian (TQB/QB build) — Orphan-seed harvest regression tests
# What: New file. Regression coverage for the _harvest_associations() seed-loop
#       membership guard (`entry_id in self.graph.nodes`): an orphaned top vdb
#       hit no longer empties the harvest; live seeds still surface; an
#       all-orphan seed set returns [] without raising.
# Why: PRD 2026-07-08-codemine-surfacing-parity §2.1/§7.1 — pre-guard, a vdb
#      search hit whose graph node was orphan-pruned entered
#      prime_and_propagate's working set, raised KeyError, and the harvest's
#      own fail-soft returned [] at debug level: Codemine's production recall
#      path silently died on one dead seed.
# How: Same fixture pattern as tests/test_worker_ng_recall.py — real
#      NeuroGraphMemory against pytest's tmp_path (constructor called directly,
#      no singleton bleed, hash-fallback embedder so no model downloads). Tonic
#      disabled via the config path so no background thread perturbs voltages
#      mid-assert. Orphans are real vdb entries with NO graph node — exactly
#      the post-pruning state the guard exists for.
# -------------------

import numpy as np
import pytest

from openclaw_hook import NeuroGraphMemory


QUERY = "spreading activation harvest regression query"


@pytest.fixture
def ng(tmp_path):
    """Real NeuroGraphMemory, isolated workspace per test, no singleton bleed.

    Tonic is disabled through the production config path (snn_config merge →
    tonic_conf["enabled"]) so no background inference thread mutates voltages
    while these tests assert on deterministic firing.
    """
    return NeuroGraphMemory(
        workspace_dir=str(tmp_path),
        config={"tonic": {"enabled": False}},
    )


def _query_vec(ng):
    """The exact (deterministic, hash-fallback, L2-normalized) query embedding."""
    return ng.ingestor.embedder.embed_text(QUERY)


def _insert_orphan(ng, node_id, vec):
    """A vdb entry with NO graph node — the post-pruning orphan state."""
    assert node_id not in ng.graph.nodes
    ng.vector_db.insert(id=node_id, embedding=vec, content=f"orphan content {node_id}")


def _insert_live(ng, node_id, vec):
    """A vdb entry WITH a live graph node, threshold lowered so priming fires it."""
    node = ng.graph.create_node(node_id=node_id)
    node.threshold = 0.1
    ng.vector_db.insert(id=node_id, embedding=vec, content=f"live content {node_id}")
    return node


def _near_vec(vec, seed=42, scale=0.05):
    """Deterministic slight perturbation — high similarity but strictly below
    the orphan's exact-match 1.0, so the orphan is the TOP hit."""
    rng = np.random.RandomState(seed)
    return vec + scale * rng.randn(vec.shape[0]).astype(vec.dtype)


def test_orphaned_top_hit_no_longer_empties_harvest(ng):
    """The PRD's live silent failure: orphan is the TOP vdb hit (exact query
    vector, sim 1.0); pre-guard its dead node ID raised KeyError inside
    prime_and_propagate and the whole harvest returned []. Post-guard the
    live seed still surfaces."""
    qv = _query_vec(ng)
    _insert_orphan(ng, "orphan-top-hit", qv.copy())
    _insert_live(ng, "live-node", _near_vec(qv))

    results = ng.recall(QUERY, k=5, threshold=0.4)

    surfaced_ids = [r["node_id"] for r in results]
    assert "live-node" in surfaced_ids, (
        "live seed must still surface despite the orphaned top hit"
    )
    assert "orphan-top-hit" not in surfaced_ids


def test_live_seeds_still_surface_without_orphans(ng):
    """The guard must not break the normal path: a plain live seed surfaces."""
    qv = _query_vec(ng)
    _insert_live(ng, "live-only", qv.copy())

    results = ng.recall(QUERY, k=5, threshold=0.4)

    assert [r["node_id"] for r in results] == ["live-only"]
    assert results[0]["content"] == "live content live-only"


def test_all_orphan_seed_set_returns_empty_without_raising(ng):
    """Every vdb hit is orphaned → prime_ids is empty → harvest returns []
    cleanly (no exception reaches the caller, no fail-soft swallow needed)."""
    qv = _query_vec(ng)
    _insert_orphan(ng, "orphan-a", qv.copy())
    _insert_orphan(ng, "orphan-b", _near_vec(qv, seed=7))

    results = ng.recall(QUERY, k=5, threshold=0.4)

    assert results == []
