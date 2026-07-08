# ---- Changelog ----
# [2026-07-08] Meridian (TQB/QB build) — #256 anticipate + GSG re-score tests
# What: New file. Covers (1) _anticipate(): primed set non-empty after a step
#       with fired nodes (given outgoing synapses), fired/orphaned targets
#       excluded, empty-fired reset, top-K cap, on_message() wiring;
#       (2) anticipate bonus: TTL expiry honored, primed surfaced node's
#       strength exceeds its unprimed twin by exactly _ANTICIPATE_BONUS and
#       re-sorts above it; (3) GSG re-score: stamped node geometrically close
#       to the query gains <= +_GSG_SCORE_BONUS and re-sorts above an
#       unstamped peer, unstamped untouched, spherical branch exercised;
#       (4) fail-soft on every enrichment; (5) all copied constants pinned to
#       exact canonical values; (6) GSG Phase 3 PROPAGATION modulation
#       executes for stamped nodes during graph.step().
# Why: PRD 2026-07-08-codemine-surfacing-parity §2.4/§7.4, §2.5/§7.5, §9.
# How: Same fixture pattern as tests/test_harvest_orphan_seeds.py (real
#      NeuroGraphMemory, tmp_path, hash-fallback embedder, Tonic disabled).
#      QB addition on §7.3 acceptance: GSG Phase 3's per-step cache
#      (_gsg_cache in neuro_foundation.step()) is a function-LOCAL dict — not
#      inspectable post-hoc without modifying the read-only file. The
#      sanctioned observable is graph._delay_buffer (instance attr): the
#      modulated delivery current is enqueued there, and modulation executes
#      ONLY when both endpoints' cache entries are non-None. The Phase 3 test
#      asserts a stamped pair's enqueued current is strictly below its
#      unstamped control twin's — the audit's inertness finding, flipped
#      directly at the propagation layer. "Fail-soft on missing embedder"
#      (§9) maps to: zero-norm query vector -> re-score returns False,
#      un-rescored; plus a raising re-score never empties the harvest.
# -------------------

import math
import time

import numpy as np
import pytest

import openclaw_hook
from openclaw_hook import (
    NeuroGraphMemory,
    _ANTICIPATE_BONUS,
    _ANTICIPATE_TOP_K,
    _ANTICIPATE_TTL_S,
    _GSG_LAYER_NORMS,
    _GSG_SCORE_BONUS,
    _embed_to_poincare_dir,
)


QUERY = "anticipate and gsg rescore regression query"
NG_CONFIG = {"tonic": {"enabled": False}}


@pytest.fixture
def ng(tmp_path):
    """Real NeuroGraphMemory, isolated workspace, Tonic disabled (see
    tests/test_harvest_orphan_seeds.py for rationale)."""
    return NeuroGraphMemory(workspace_dir=str(tmp_path), config=NG_CONFIG)


def _query_vec(ng):
    return ng.ingestor.embedder.embed_text(QUERY)


def _insert_live(ng, node_id, vec, threshold=0.1):
    """Graph node + vdb entry; threshold lowered so priming fires it."""
    node = ng.graph.create_node(node_id=node_id)
    node.threshold = threshold
    ng.vector_db.insert(id=node_id, embedding=vec, content=f"content {node_id}")
    return node


def _by_id(results, node_id):
    return next(r for r in results if r["node_id"] == node_id)


# ---------------------------------------------------------------------------
# Constants pin — PRD §9: all copied constants asserted at their exact
# canonical values. Do NOT update these without a canonical re-copy.
# ---------------------------------------------------------------------------

def test_constants_pinned_verbatim():
    assert _ANTICIPATE_TOP_K == 15
    assert _ANTICIPATE_TTL_S == 120.0
    assert _ANTICIPATE_BONUS == 0.25
    assert _GSG_LAYER_NORMS == (0.70, 0.50, 0.30)
    assert _GSG_SCORE_BONUS == 0.30


# ---------------------------------------------------------------------------
# _anticipate() — priming mechanics
# ---------------------------------------------------------------------------

def test_anticipate_primes_outgoing_neighbors(ng):
    ng.graph.create_node(node_id="A")
    ng.graph.create_node(node_id="B")
    ng.graph.create_node(node_id="C")
    ng.graph.create_synapse("A", "B", weight=2.0)
    ng.graph.create_synapse("A", "C", weight=1.0)

    before = time.time()
    ng._anticipate(["A"])
    after = time.time()

    assert set(ng._primed_nodes) == {"B", "C"}
    score_b, expiry_b = ng._primed_nodes["B"]
    score_c, _ = ng._primed_nodes["C"]
    assert score_b == pytest.approx(2.0)
    assert score_c == pytest.approx(1.0)
    # TTL pinned: expiry = now + _ANTICIPATE_TTL_S
    assert before + _ANTICIPATE_TTL_S <= expiry_b <= after + _ANTICIPATE_TTL_S


def test_anticipate_excludes_fired_and_orphaned_targets(ng):
    ng.graph.create_node(node_id="A")
    ng.graph.create_node(node_id="B")
    ng.graph.create_node(node_id="C")
    ng.graph.create_synapse("A", "B", weight=1.0)
    ng.graph.create_synapse("A", "C", weight=1.0)
    # Orphan C the same way pruning would leave a stale synapse ref behind.
    ng.graph.nodes.pop("C")

    ng._anticipate(["A", "B"])  # B fired too -> excluded from candidates

    assert ng._primed_nodes == {}


def test_anticipate_empty_fired_resets_primed(ng):
    ng._primed_nodes = {"stale": (1.0, time.time() + 60.0)}
    ng._anticipate([])
    assert ng._primed_nodes == {}


def test_anticipate_top_k_cap_keeps_highest_weights(ng):
    ng.graph.create_node(node_id="hub")
    for i in range(_ANTICIPATE_TOP_K + 5):
        nid = f"n-{i:02d}"
        ng.graph.create_node(node_id=nid)
        ng.graph.create_synapse("hub", nid, weight=0.1 + 0.1 * i)

    ng._anticipate(["hub"])

    assert len(ng._primed_nodes) == _ANTICIPATE_TOP_K
    # The 5 lowest-weight neighbors (n-00..n-04) fell off the top-K.
    for i in range(5):
        assert f"n-{i:02d}" not in ng._primed_nodes
    assert f"n-{_ANTICIPATE_TOP_K + 4:02d}" in ng._primed_nodes


def test_on_message_wiring_populates_primed(ng):
    """PRD §9: after a step with fired nodes (given outgoing synapses), the
    primed set is non-empty — driven through the real on_message() path."""
    ng.graph.create_node(node_id="A")
    ng.graph.create_node(node_id="B")
    ng.graph.create_synapse("A", "B", weight=2.0)
    ng.graph.nodes["A"].voltage = 5.0  # fires on this turn's step()

    result = ng.on_message("anticipate wiring probe message for the step")

    assert result["status"] == "ingested"
    assert result["fired"] > 0
    assert "B" in ng._primed_nodes
    _, expiry = ng._primed_nodes["B"]
    assert expiry > time.time()


# ---------------------------------------------------------------------------
# Anticipate bonus in the harvest — TTL + twin comparison
# ---------------------------------------------------------------------------

def test_primed_surfaced_strength_exceeds_unprimed_twin_by_bonus(ng):
    qv = _query_vec(ng)
    # Twins: identical embeddings and thresholds -> identical base strength.
    _insert_live(ng, "t-unprimed", qv.copy())
    _insert_live(ng, "t-primed", qv.copy())
    ng._primed_nodes = {"t-primed": (1.0, time.time() + 60.0)}

    results = ng.recall(QUERY, k=5, threshold=0.4)

    primed = _by_id(results, "t-primed")
    unprimed = _by_id(results, "t-unprimed")
    assert primed["strength"] - unprimed["strength"] == pytest.approx(
        _ANTICIPATE_BONUS, abs=1e-6
    )
    # Bonus applied -> single re-sort by strength desc puts the primed twin first.
    assert results[0]["node_id"] == "t-primed"


def test_expired_priming_gets_no_bonus(ng):
    """PRD §9: entries expire per TTL — an expired primed node's strength
    equals its never-primed twin's exactly."""
    qv = _query_vec(ng)
    _insert_live(ng, "t-expired", qv.copy())
    _insert_live(ng, "t-never", qv.copy())
    ng._primed_nodes = {"t-expired": (1.0, time.time() - 0.5)}  # already expired

    results = ng.recall(QUERY, k=5, threshold=0.4)

    expired = _by_id(results, "t-expired")
    never = _by_id(results, "t-never")
    assert expired["strength"] == pytest.approx(never["strength"], abs=1e-9)


def test_anticipate_bonus_failure_does_not_empty_harvest(ng):
    """Corrupt primed state -> bonus block fails -> surfaced list intact."""
    qv = _query_vec(ng)
    _insert_live(ng, "t-live", qv.copy())
    ng._primed_nodes = 42  # truthy, no .get -> AttributeError inside the bonus block

    results = ng.recall(QUERY, k=5, threshold=0.4)

    assert [r["node_id"] for r in results] == ["t-live"]


# ---------------------------------------------------------------------------
# GSG surfacing re-score
# ---------------------------------------------------------------------------

def test_gsg_stamped_close_node_gains_bonus_and_resorts_above_unstamped(ng):
    qv = _query_vec(ng)
    query_dir = _embed_to_poincare_dir(qv)
    # Twins: identical base strength; G stamped AT the query direction
    # (geodesic distance 0 -> bonus is exactly _GSG_SCORE_BONUS), U unstamped.
    _insert_live(ng, "u-unstamped", qv.copy())
    g = _insert_live(ng, "g-stamped", qv.copy())
    g.metadata["poincare_dir"] = query_dir.tolist()

    results = ng.recall(QUERY, k=5, threshold=0.4)

    stamped = _by_id(results, "g-stamped")
    unstamped = _by_id(results, "u-unstamped")
    gain = stamped["strength"] - unstamped["strength"]
    assert gain == pytest.approx(_GSG_SCORE_BONUS, abs=1e-9)
    assert gain <= _GSG_SCORE_BONUS  # PRD §9: gains <= +0.30
    # Unstamped peer untouched by the re-score -> stamped re-sorts above it.
    assert results[0]["node_id"] == "g-stamped"


def test_gsg_spherical_branch_exercised(ng):
    qv = _query_vec(ng)
    query_dir = _embed_to_poincare_dir(qv)
    _insert_live(ng, "u-plain", qv.copy())
    s = _insert_live(ng, "s-spherical", qv.copy())
    s.metadata["poincare_dir"] = query_dir.tolist()
    s.manifold_type = "spherical"

    results = ng.recall(QUERY, k=5, threshold=0.4)

    spherical = _by_id(results, "s-spherical")
    plain = _by_id(results, "u-plain")
    # Great-circle branch: cos clipped to 1-1e-7 even for identical dirs,
    # so the bonus is strictly below _GSG_SCORE_BONUS — recompute exactly.
    cos = min(1.0 - 1e-7, max(-1.0 + 1e-7, float(np.dot(query_dir, query_dir))))
    expected = _GSG_SCORE_BONUS / (1.0 + math.acos(cos))
    assert expected < _GSG_SCORE_BONUS
    assert spherical["strength"] - plain["strength"] == pytest.approx(expected, rel=1e-9)


def test_gsg_rescore_returns_false_for_zero_query_vector(ng):
    """§9 'fail-soft on missing embedder': no usable query direction ->
    False, strengths untouched."""
    items = [{"node_id": "whatever", "strength": 1.0}]
    assert ng._gsg_rescore_surfaced(items, np.zeros(8, dtype=np.float32)) is False
    assert items[0]["strength"] == 1.0


def test_gsg_rescore_failure_does_not_empty_harvest(ng, monkeypatch):
    qv = _query_vec(ng)
    _insert_live(ng, "t-survives", qv.copy())

    def _boom(surfaced, query_vec):
        raise RuntimeError("simulated GSG re-score failure")

    monkeypatch.setattr(ng, "_gsg_rescore_surfaced", _boom)

    results = ng.recall(QUERY, k=5, threshold=0.4)

    assert [r["node_id"] for r in results] == ["t-survives"]


def test_no_bonus_leaves_latency_ordering_untouched(ng):
    """With nothing primed and nothing stamped, the (latency, -strength)
    ordering must stand — no strength-desc re-sort."""
    qv = _query_vec(ng)
    # fast fires at step 0; slow (no vdb similarity to the query would be
    # ideal, but simplest deterministic contrast is threshold): 'slow' needs
    # a delivered spike, arriving a step later via the synapse from 'fast'.
    fast = _insert_live(ng, "fast", qv.copy())
    slow = ng.graph.create_node(node_id="slow")
    slow.threshold = 0.1
    # vdb entry dissimilar to the query so 'slow' is NOT a seed, but content
    # exists so it can surface when it fires via propagation.
    rng = np.random.RandomState(7)
    ng.vector_db.insert(id="slow", embedding=rng.randn(qv.shape[0]).astype(np.float32),
                        content="content slow")
    ng.graph.create_synapse("fast", "slow", weight=2.0)

    results = ng.recall(QUERY, k=5, threshold=0.4)

    ids = [r["node_id"] for r in results]
    assert ids == ["fast", "slow"]  # latency order, higher-latency second
    assert _by_id(results, "fast")["latency"] < _by_id(results, "slow")["latency"]


# ---------------------------------------------------------------------------
# GSG Phase 3 propagation modulation (QB addition on §7.3 acceptance)
# ---------------------------------------------------------------------------

def test_gsg_phase3_propagation_modulation_executes_for_stamped_nodes(ng):
    """The #358 audit's inertness finding, flipped at the propagation layer.

    _gsg_cache is local to neuro_foundation.step() (not inspectable), but its
    effect is: modulation multiplies the enqueued delivery current by
    exp(-decay*...) ONLY when both endpoints' cache entries are non-None
    (i.e. both stamped). A stamped pair at distinct positions must therefore
    enqueue current < weight into graph._delay_buffer, while an identical
    unstamped control pair enqueues current == weight.
    """
    d1 = np.zeros(8, dtype=np.float32)
    d1[0] = 1.0
    d2 = np.zeros(8, dtype=np.float32)
    d2[1] = 1.0

    a = ng.graph.create_node(node_id="gsg-pre")
    b = ng.graph.create_node(node_id="gsg-post")
    a.metadata["poincare_dir"] = d1.tolist()
    b.metadata["poincare_dir"] = d2.tolist()
    ng.graph.create_synapse("gsg-pre", "gsg-post", weight=1.0)

    ng.graph.create_node(node_id="ctl-pre")
    ng.graph.create_node(node_id="ctl-post")
    ng.graph.create_synapse("ctl-pre", "ctl-post", weight=1.0)

    ng.graph.nodes["gsg-pre"].voltage = 5.0
    ng.graph.nodes["ctl-pre"].voltage = 5.0
    step_result = ng.graph.step()

    assert "gsg-pre" in step_result.fired_node_ids
    assert "ctl-pre" in step_result.fired_node_ids

    deliveries = [
        (nid, cur)
        for arrivals in ng.graph._delay_buffer.values()
        for (nid, cur) in arrivals
    ]
    gsg_currents = [c for nid, c in deliveries if nid == "gsg-post"]
    ctl_currents = [c for nid, c in deliveries if nid == "ctl-post"]
    assert len(gsg_currents) == 1
    assert len(ctl_currents) == 1

    assert ctl_currents[0] == pytest.approx(1.0)      # unstamped: no modulation
    assert 0.0 < gsg_currents[0] < ctl_currents[0]    # stamped: Phase 3 executed
