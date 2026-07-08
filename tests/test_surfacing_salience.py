# ---- Changelog ----
# [2026-07-08] Candor (TQB/QB build) — pin the honest salience label
# What: New file. Pins SurfacingMonitor.format_context()'s render format:
#       "(salience: {score:.2f})" — and the absence of the retired
#       "(confidence: {score:.0%})" rendering.
# Why: PRD 2026-07-08-codemine-surfacing-parity §2.2/§7.2/§9 — _score_node()'s
#      salience (designed range ~[0.8, 1.8], floored >=0.8 for any fired node)
#      is not a probability; the old percentage rendering produced ">100%"
#      strings that read as fictitious and eroded worker trust. Canonical
#      parity (e6eb2f2); this pin prevents drift back.
# How: Real NeuroGraphMemory per test against pytest's tmp_path (no
#      singleton bleed, no Mocks — same convention as
#      tests/test_worker_ng_recall.py). Exact-format tests drive the real
#      format_context() with explicit surfaced_items; the full-path test
#      drives the real after_step() with the real StepResult dataclass.
# -------------------

import numpy as np
import pytest

from neuro_foundation import StepResult
from openclaw_hook import NeuroGraphMemory


@pytest.fixture
def ng(tmp_path):
    """Real NeuroGraphMemory, isolated workspace per test, no singleton bleed."""
    return NeuroGraphMemory(workspace_dir=str(tmp_path))


def _format(ng, surfaced_items):
    """Exercise the REAL SurfacingMonitor.format_context() — no stand-ins."""
    return ng._surfacing_monitor.format_context(surfaced_items)


def test_renders_score_as_salience_two_decimals(ng):
    # A score of 1.23 used to render as "(confidence: 123%)" — fictitious.
    context = _format(ng, [{"node_id": "n1", "content": "a surfaced concept", "score": 1.23}])

    assert "(salience: 1.23)" in context
    assert "- a surfaced concept (salience: 1.23)" in context


def test_contains_salience_and_neither_confidence_nor_percent(ng):
    # PRD §7.2 test pin, verbatim: "salience:" in, "confidence:" out, "%" out.
    context = _format(ng, [{"node_id": "n1", "content": "a surfaced concept", "score": 1.23}])

    assert "salience:" in context
    assert "confidence:" not in context
    assert "%" not in context


def test_above_one_salience_never_renders_as_percentage(ng):
    # The exact failure mode: _score_node() floors >=0.8 and ranges to ~1.8,
    # so scores above 1.0 are NORMAL — they must never read as ">100%".
    context = _format(ng, [{"node_id": "n-hi", "content": "highly salient concept", "score": 1.80}])

    assert "(salience: 1.80)" in context
    assert "180" not in context  # no percentage-scaled artifact of the old :.0% render
    assert "%" not in context


def test_full_after_step_path_renders_salience(ng):
    # Drive the real pipeline: real graph node, real vector_db entry, real
    # StepResult through after_step(), then format_context() with no args
    # (internally calls get_surfaced()).
    ng.graph.create_node(node_id="n-salience")
    ng.vector_db.insert(
        id="n-salience",
        embedding=np.zeros(8),
        content="a fired concept surfaced through the real path",
        metadata={},
    )
    ng._surfacing_monitor.after_step(StepResult(fired_node_ids=["n-salience"]))

    context = ng._surfacing_monitor.format_context()

    assert "[NeuroGraph Surfaced Knowledge]" in context
    assert "a fired concept surfaced through the real path" in context
    assert "salience:" in context
    assert "confidence:" not in context
    assert "%" not in context


def test_empty_surfaced_items_still_returns_empty_string(ng):
    # Display-only change — the empty-queue contract is untouched.
    assert _format(ng, []) == ""
