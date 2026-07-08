# ---- Changelog ----
# [2026-07-08] Meridian (TQB/QB build) — GSG poincare_dir stamping + backfill tests
# What: New file. Covers (1) init-time backfill: after constructing a
#       NeuroGraphMemory over a workspace with pre-existing unstamped nodes,
#       every node with a stored vdb embedding has poincare_dir metadata;
#       (2) ingest-time stamping: nodes created by on_message() are stamped at
#       creation; (3) idempotency: a second backfill run stamps 0; (4) nodes
#       without vdb embeddings are left unstamped without raising; (5) fail-soft:
#       a stamping failure inside on_message() does not affect its return dict.
# Why: PRD 2026-07-08-codemine-surfacing-parity §2.3/§7.3/§9 — the fork had
#      zero poincare_dir write sites (#358 audit), leaving shipped GSG Phases
#      3/4 permanent no-ops. These tests flip the audit's inertness finding.
# How: Same fixture pattern as tests/test_harvest_orphan_seeds.py — real
#      NeuroGraphMemory against pytest's tmp_path (direct constructor, no
#      singleton bleed, hash-fallback embedder, Tonic disabled via the
#      production config path). The init-backfill test round-trips through the
#      real save()/restore path: instance one builds unstamped state and
#      save()s; instance two restores and its __init__ backfills.
# -------------------

import numpy as np
import pytest

import openclaw_hook
from openclaw_hook import NeuroGraphMemory, _embed_to_poincare_dir


NG_CONFIG = {"tonic": {"enabled": False}}


@pytest.fixture
def ng(tmp_path):
    """Real NeuroGraphMemory, isolated workspace, Tonic disabled (see
    tests/test_harvest_orphan_seeds.py for rationale)."""
    return NeuroGraphMemory(workspace_dir=str(tmp_path), config=NG_CONFIG)


def _make_unstamped_node(ng, node_id, seed):
    """A graph node + vdb entry with NO poincare_dir — pre-fix state."""
    ng.graph.create_node(node_id=node_id)
    rng = np.random.RandomState(seed)
    ng.vector_db.insert(
        id=node_id,
        embedding=rng.randn(16).astype(np.float32),
        content=f"content {node_id}",
    )
    assert "poincare_dir" not in ng.graph.nodes[node_id].metadata


# ---------------------------------------------------------------------------
# _embed_to_poincare_dir()
# ---------------------------------------------------------------------------

def test_helper_returns_unit_direction():
    vec = np.array([3.0, 4.0], dtype=np.float32)
    out = _embed_to_poincare_dir(vec)
    assert out is not None
    assert np.allclose(out, [0.6, 0.8])
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


def test_helper_returns_none_for_zero_norm():
    assert _embed_to_poincare_dir(np.zeros(8, dtype=np.float32)) is None


# ---------------------------------------------------------------------------
# Backfill (init path + direct)
# ---------------------------------------------------------------------------

def test_init_backfills_existing_unstamped_nodes(tmp_path):
    """The PRD §9 acceptance shape: boot a worker over a workspace whose
    checkpoint holds unstamped nodes → after __init__, every node with a
    stored vdb embedding has poincare_dir."""
    first = NeuroGraphMemory(workspace_dir=str(tmp_path), config=NG_CONFIG)
    for i in range(3):
        _make_unstamped_node(first, f"pre-existing-{i}", seed=100 + i)
    # A node with NO vdb embedding — must stay unstamped, must not raise.
    first.graph.create_node(node_id="no-embedding")
    # Strip stamps the ingest path may add elsewhere: these were created raw.
    first.save()

    second = NeuroGraphMemory(workspace_dir=str(tmp_path), config=NG_CONFIG)

    for i in range(3):
        node = second.graph.nodes[f"pre-existing-{i}"]
        pd = node.metadata.get("poincare_dir")
        assert pd is not None, f"pre-existing-{i} not backfilled"
        assert isinstance(pd, list)
        # Stored embeddings are L2-normalized on insert → stamped dirs are unit.
        assert abs(float(np.linalg.norm(np.asarray(pd))) - 1.0) < 1e-5
        # The stamp IS the stored embedding (verbatim canonical backfill shape).
        assert np.allclose(pd, second.vector_db.embeddings[f"pre-existing-{i}"], atol=1e-6)
    assert "poincare_dir" not in second.graph.nodes["no-embedding"].metadata


def test_backfill_is_idempotent(ng):
    for i in range(2):
        _make_unstamped_node(ng, f"node-{i}", seed=200 + i)

    assert ng._gsg_backfill_poincare_dirs() == 2
    assert ng._gsg_backfill_poincare_dirs() == 0  # second run stamps nothing


def test_backfill_skips_nodes_without_embeddings(ng):
    ng.graph.create_node(node_id="graph-only")
    assert ng._gsg_backfill_poincare_dirs() == 0
    assert "poincare_dir" not in ng.graph.nodes["graph-only"].metadata


# ---------------------------------------------------------------------------
# Ingest-time stamping (on_message)
# ---------------------------------------------------------------------------

def test_newly_ingested_nodes_get_stamped(ng):
    result = ng.on_message(
        "GSG stamping test message: newly ingested nodes gain a poincare_dir "
        "unit direction at creation time from their stored vdb embedding."
    )

    assert result["status"] == "ingested"
    assert result["nodes_created"] > 0
    stamped_new = [
        nid for nid, node in ng.graph.nodes.items()
        if node.metadata.get("poincare_dir") is not None
    ]
    assert len(stamped_new) > 0
    # Every created node with a stored embedding is stamped, and the stamp is
    # the (unit-norm) stored embedding.
    for nid in ng.vector_db.embeddings:
        if nid in ng.graph.nodes:
            pd = ng.graph.nodes[nid].metadata.get("poincare_dir")
            assert pd is not None, f"ingested node {nid} not stamped"
            assert abs(float(np.linalg.norm(np.asarray(pd))) - 1.0) < 1e-5


def test_ingest_stamping_failure_is_fail_soft(ng, monkeypatch):
    """A blown-up stamping path must not touch on_message()'s contract."""
    def _boom(_emb):
        raise RuntimeError("simulated stamping failure")

    monkeypatch.setattr(openclaw_hook, "_embed_to_poincare_dir", _boom)

    result = ng.on_message(
        "fail-soft check: stamping raises but the turn completes normally"
    )

    assert result["status"] == "ingested"
    assert result["nodes_created"] > 0
    # Return dict shape untouched by the stamping failure.
    for key in ("synapses_created", "chunks", "fired", "surfaced", "message_count"):
        assert key in result
