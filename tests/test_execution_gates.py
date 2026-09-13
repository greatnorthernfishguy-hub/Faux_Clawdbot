# ---- Changelog ----
# [2026-09-12] Codex — Pin gate stop behavior at every nesting level
# What: Exercises human, QB, and automatic gates in top-level and nested steps.
# Why: Pending gates previously allowed later mutating actions to execute.
# How: A recording tool proves no post-gate dispatch occurs.
# -------------------

from pathlib import Path

import pytest

from spec_executor import SpecExecutor


def _spec(gate_type, nested=None):
    gate = {
        "id": "approval",
        "type": "gate",
        "description": "Approve before mutation",
        "gate_type": gate_type,
    }
    action = {
        "id": "after_gate",
        "type": "action",
        "tool": "read_file",
        "params": {"path": "anything"},
        "validation": {"checks": [{"operator": "result_is_string"}]},
        "on_failure": "abort",
    }
    if nested == "group":
        steps = [{"id": "group", "type": "group", "steps": [gate, action]}]
    elif nested == "condition":
        steps = [{
            "id": "condition",
            "type": "condition",
            "check": {"operator": "result_is_string"},
            "if_true": [gate, action],
            "if_false": [],
        }]
    elif nested == "loop":
        steps = [{
            "id": "loop",
            "type": "loop",
            "over": {"items": ["one", "two"]},
            "body": [gate, action],
        }]
    else:
        steps = [gate, action]

    return {
        "spec_version": "1.0.0",
        "block": {
            "id": "BLK-GATE",
            "name": "Gate behavior",
            "scope": "Verify execution stops",
            "acceptance_criteria": ["Nothing after a pending gate executes"],
        },
        "constraints": {"never": ["Never cross an unresolved gate"], "anti_drift": []},
        "steps": steps,
    }


def _executor(tmp_path, calls):
    return SpecExecutor(
        tool_registry={"read_file": lambda params: calls.append(params) or "read"},
        policy_check_fn=lambda *args: (True, "allowed"),
        worker_ng=None,
        workspace=Path(tmp_path),
    )


@pytest.mark.parametrize("gate_type", ["human_review", "qb_checkpoint"])
@pytest.mark.parametrize("nested", [None, "group", "condition", "loop"])
def test_pending_gate_stops_every_following_action(tmp_path, gate_type, nested):
    calls = []

    report = _executor(tmp_path, calls).execute_block(_spec(gate_type, nested))

    assert report["status"] == "paused"
    assert report["pause_reason"] == f"{gate_type}: Approve before mutation"
    assert report["step_results"]["approval"]["status"] == f"pending_{gate_type}"
    assert "after_gate" not in report["step_results"]
    assert calls == []
    if nested == "loop":
        assert report["step_results"]["loop"]["iterations"] == 1


def test_auto_approve_gate_allows_following_action(tmp_path):
    calls = []

    report = _executor(tmp_path, calls).execute_block(_spec("auto_approve"))

    assert report["status"] == "complete"
    assert report["pause_reason"] is None
    assert report["step_results"]["approval"]["status"] == "auto_approved"
    assert report["step_results"]["after_gate"]["status"] == "pass"
    assert len(calls) == 1
