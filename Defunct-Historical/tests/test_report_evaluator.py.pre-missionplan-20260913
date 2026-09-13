# ---- Changelog ----
# [2026-09-12] Codex — Pin explicit Reviewer approval and gate escalation
# What: Covers valid approval, malformed output, provider failure, and pending gates.
# Why: Structural success previously became DONE when Reviewer was unavailable.
# How: Substitute only the external persona response and exercise real evaluator logic.
# -------------------

import persona_client
from report_evaluator import evaluate_report


def _complete_report():
    return {
        "status": "complete",
        "block_id": "BLK-REPORT",
        "block_name": "Reviewer behavior",
        "summary": {
            "total_steps": 1,
            "passed": 1,
            "failed": 0,
            "pending_review": 0,
            "tool_calls": 1,
            "elapsed_seconds": 0.1,
        },
        "step_results": {"test": {"status": "pass", "result_preview": "ok"}},
        "acceptance_criteria": ["Reviewer explicitly approves"],
    }


def test_explicit_reviewer_done_can_complete(monkeypatch):
    monkeypatch.setattr(persona_client, "call_persona", lambda **kwargs: {
        "response": "The work meets the stated bar.\nDECISION: DONE",
        "model": "review/model",
        "elapsed_seconds": 0.1,
    })

    result = evaluate_report(_complete_report())

    assert result["decision"] == "done"


def test_missing_reviewer_marker_escalates(monkeypatch):
    monkeypatch.setattr(persona_client, "call_persona", lambda **kwargs: {
        "response": "Looks fine to me.",
        "model": "review/model",
        "elapsed_seconds": 0.1,
    })

    result = evaluate_report(_complete_report())

    assert result["decision"] == "escalate"
    assert result["escalate_to"] == "qb"
    assert "no recognized decision marker" in result["escalate_reason"]


def test_reviewer_exception_escalates(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(persona_client, "call_persona", fail)

    result = evaluate_report(_complete_report())

    assert result["decision"] == "escalate"
    assert result["escalate_to"] == "qb"
    assert "provider unavailable" in result["escalate_reason"]


def test_human_gate_escalates_to_josh_without_calling_reviewer(monkeypatch):
    monkeypatch.setattr(
        persona_client,
        "call_persona",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Reviewer must not run")),
    )
    report = _complete_report()
    report.update({"status": "paused", "pause_reason": "human_review: approve deploy"})
    report["summary"]["pending_review"] = 1
    report["step_results"] = {
        "gate": {"status": "pending_human_review", "gate_type": "human_review"}
    }

    result = evaluate_report(report)

    assert result["decision"] == "escalate"
    assert result["escalate_to"] == "josh"


def test_qb_gate_escalates_to_qb_without_calling_reviewer(monkeypatch):
    monkeypatch.setattr(
        persona_client,
        "call_persona",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Reviewer must not run")),
    )
    report = _complete_report()
    report.update({"status": "paused", "pause_reason": "qb_checkpoint: inspect diff"})
    report["summary"]["pending_review"] = 1
    report["step_results"] = {
        "gate": {"status": "pending_qb_checkpoint", "gate_type": "qb_checkpoint"}
    }

    result = evaluate_report(report)

    assert result["decision"] == "escalate"
    assert result["escalate_to"] == "qb"
