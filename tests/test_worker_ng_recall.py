# ---- Changelog ----
# [2026-07-06] Weft (TQB) — Initial test suite for filter_recall_results(),
#              surfacing_context(), and gate_repeat_recall()
# What: New file. Covers the three pattern-completion recall helpers added to
#       worker_ng.py in this sub-step.
# Why: PRD §2-4 (2026-07-06-codemine-pattern-completion-recall) requires real
#      NeuroGraphMemory/graph fixtures (constructor called directly, not the
#      get_instance() singleton) — no Mock()/MagicMock() standing in for the
#      substrate — before these helpers get wired into any .recall() call site.
# How: filter_recall_results()/surfacing_context() tests build a fresh
#      NeuroGraphMemory per test against pytest's tmp_path (no singleton bleed),
#      then use the graph's real create_node()/vector_db.insert() APIs plus the
#      real StepResult dataclass to drive the real SurfacingMonitor. Only the
#      TTL logic in gate_repeat_recall() is exercised as a pure function.
# -------------------

import time

import numpy as np
import pytest

from neuro_foundation import StepResult
from openclaw_hook import NeuroGraphMemory
from surfacing import SurfacingMonitor
from worker_ng import filter_recall_results, gate_repeat_recall, surfacing_context


@pytest.fixture
def ng(tmp_path):
    """Real NeuroGraphMemory, isolated workspace per test, no singleton bleed.

    Constructed directly (not via get_instance()) per the PRD's acceptance
    criteria. CES defaults to enabled, so _surfacing_monitor should be a real
    SurfacingMonitor — verified below rather than assumed.
    """
    return NeuroGraphMemory(workspace_dir=str(tmp_path))


def _insert_vector(ng, node_id, content, metadata=None):
    """Real SimpleVectorDB.insert(). The embedding is a throwaway zero vector —
    none of these tests exercise vector_db.search(), only .get()."""
    ng.vector_db.insert(id=node_id, embedding=np.zeros(8), content=content, metadata=metadata or {})


def test_fixture_has_real_surfacing_monitor(ng):
    """Verify the assumption the PRD asked us to verify, not assume."""
    assert ng._surfacing_monitor is not None
    assert isinstance(ng._surfacing_monitor, SurfacingMonitor)


# ---------------------------------------------------------------------------
# filter_recall_results()
# ---------------------------------------------------------------------------

def test_prefers_forest_content_from_live_node(ng):
    ng.graph.create_node(node_id="n-forest", metadata={
        "_forest_content": "her actual lived turn, long enough to pass the floor",
    })
    raw_results = [{
        "node_id": "n-forest",
        "content": "WANT",  # the short tree-concept shard
        "metadata": {},
        "latency": 1,
        "strength": 0.9,
        "was_predicted": False,
    }]

    filtered = filter_recall_results(ng, raw_results)

    assert len(filtered) == 1
    assert filtered[0]["content"] == "her actual lived turn, long enough to pass the floor"
    # All other keys preserved unchanged.
    assert filtered[0]["node_id"] == "n-forest"
    assert filtered[0]["latency"] == 1
    assert filtered[0]["strength"] == 0.9
    assert filtered[0]["was_predicted"] is False


def test_falls_back_to_result_content_when_node_missing_from_graph(ng):
    # node_id in the result dict doesn't correspond to any live graph node —
    # resolve_surface_content() falls back to the result dict itself as the
    # vdb-entry shard.
    raw_results = [{
        "node_id": "does-not-exist",
        "content": "a raw recall content string, long enough to pass the floor",
    }]

    filtered = filter_recall_results(ng, raw_results)

    assert len(filtered) == 1
    assert filtered[0]["content"] == "a raw recall content string, long enough to pass the floor"


def test_drops_degenerate_fragment(ng):
    ng.graph.create_node(node_id="n-short", metadata={"_forest_content": "hi"})
    raw_results = [{"node_id": "n-short", "content": "no"}]

    filtered = filter_recall_results(ng, raw_results)

    assert filtered == []


def test_allows_ingested_nodes_through_unlike_experiential_surfacers(ng):
    # allow_ingested=True is hardcoded in filter_recall_results — recall may
    # legitimately want a document, unlike CES's experiential surfacers.
    ng.graph.create_node(node_id="n-ingested", metadata={
        "creation_mode": "ingested",
        "_forest_content": "ingested source content, long enough to pass the floor",
    })
    raw_results = [{"node_id": "n-ingested", "content": "shard"}]

    filtered = filter_recall_results(ng, raw_results)

    assert len(filtered) == 1
    assert filtered[0]["content"] == "ingested source content, long enough to pass the floor"


def test_fails_soft_on_bad_entry_and_keeps_good_ones(ng):
    ng.graph.create_node(node_id="n-good", metadata={
        "_forest_content": "a good result that resolves fine, long enough to pass the floor",
    })
    raw_results = [
        # An unhashable node_id makes graph.nodes.get() raise TypeError —
        # this entry must be dropped without taking the good one down with it.
        {"node_id": ["unhashable"], "content": "this entry blows up graph.nodes.get()"},
        {"node_id": "n-good", "content": "shard"},
    ]

    filtered = filter_recall_results(ng, raw_results)

    assert len(filtered) == 1
    assert filtered[0]["node_id"] == "n-good"


def test_empty_input_returns_empty_list(ng):
    assert filter_recall_results(ng, []) == []


# ---------------------------------------------------------------------------
# surfacing_context()
# ---------------------------------------------------------------------------

def _seed_surfacing_queue(ng, node_id, content):
    """Drive the real SurfacingMonitor through its real after_step() path:
    a real graph node, a real vector_db entry, and the real StepResult
    dataclass reporting that node as fired this step."""
    ng.graph.create_node(node_id=node_id)
    _insert_vector(ng, node_id, content)
    ng._surfacing_monitor.after_step(StepResult(fired_node_ids=[node_id]))


def test_both_blocks_present_joined_by_blank_line(ng):
    _seed_surfacing_queue(ng, "n-recency", "a recency signal from SurfacingMonitor")
    filtered = [{"content": "pattern completion result one"}]

    result = surfacing_context(ng, filtered)

    assert "a recency signal from SurfacingMonitor" in result
    assert "pattern completion result one" in result
    assert "\n\n" in result
    assert result.index("a recency signal") < result.index("pattern completion result one")


def test_recency_only_when_filtered_results_empty(ng):
    _seed_surfacing_queue(ng, "n-recency2", "another recency signal")

    result = surfacing_context(ng, [])

    assert "another recency signal" in result
    assert "\n\n" not in result


def test_pattern_only_when_surfacing_monitor_is_none(ng):
    ng._surfacing_monitor = None
    filtered = [{"content": "pattern completion only, no recency"}]

    result = surfacing_context(ng, filtered)

    assert result == "pattern completion only, no recency"


def test_empty_string_when_both_sides_empty(ng):
    # Fresh instance — nothing has fired, nothing filtered through.
    result = surfacing_context(ng, [])

    assert result == ""


def test_multiple_filtered_results_joined_with_triple_dash(ng):
    filtered = [{"content": "first"}, {"content": "second"}]

    result = surfacing_context(ng, filtered)

    assert result == "first\n---\nsecond"


def test_degrades_to_pattern_only_on_surfacing_monitor_error(ng, monkeypatch):
    def _boom():
        raise RuntimeError("simulated SurfacingMonitor failure")

    monkeypatch.setattr(ng._surfacing_monitor, "format_context", _boom)
    filtered = [{"content": "pattern completion survives the recency failure"}]

    result = surfacing_context(ng, filtered)

    assert result == "pattern completion survives the recency failure"


# ---------------------------------------------------------------------------
# gate_repeat_recall() — pure function, no NeuroGraphMemory needed
# ---------------------------------------------------------------------------

def test_first_call_for_key_returns_true_and_records():
    cache = {}
    now = time.time()

    result = gate_repeat_recall(cache, "tool:foo", now)

    assert result is True
    assert cache["tool:foo"] == now


def test_within_ttl_returns_false_and_leaves_cache_untouched():
    cache = {}
    t0 = 1000.0
    gate_repeat_recall(cache, "tool:foo", t0, ttl=1800.0)

    t1 = t0 + 100.0  # well within the 1800s TTL
    result = gate_repeat_recall(cache, "tool:foo", t1, ttl=1800.0)

    assert result is False
    assert cache["tool:foo"] == t0  # untouched — no update on a gated call


def test_past_ttl_returns_true_and_updates_cache():
    cache = {}
    t0 = 1000.0
    gate_repeat_recall(cache, "tool:foo", t0, ttl=1800.0)

    t1 = t0 + 1800.01  # just past the TTL boundary
    result = gate_repeat_recall(cache, "tool:foo", t1, ttl=1800.0)

    assert result is True
    assert cache["tool:foo"] == t1


def test_exactly_at_ttl_boundary_is_still_gated():
    # (now - last) > ttl is strict — exactly at the boundary is NOT "past" it.
    cache = {}
    t0 = 1000.0
    gate_repeat_recall(cache, "tool:foo", t0, ttl=1800.0)

    t1 = t0 + 1800.0  # exactly at the boundary
    result = gate_repeat_recall(cache, "tool:foo", t1, ttl=1800.0)

    assert result is False
    assert cache["tool:foo"] == t0


def test_different_keys_are_independent():
    cache = {}
    now = time.time()

    assert gate_repeat_recall(cache, "tool:a", now) is True
    assert gate_repeat_recall(cache, "tool:b", now) is True
    assert cache == {"tool:a": now, "tool:b": now}


def test_default_ttl_is_1800_seconds():
    t0 = 5000.0
    just_under = t0 + 1799.0
    just_over = t0 + 1801.0

    cache_under = {"k": t0}
    cache_over = {"k": t0}

    assert gate_repeat_recall(cache_under, "k", just_under) is False
    assert gate_repeat_recall(cache_over, "k", just_over) is True
