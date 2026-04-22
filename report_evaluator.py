# ---- Changelog ----
# [2026-04-22] Claude Code (Sonnet 4.6) — qb_checkpoint pending no longer blocks structural pass
#   What: _structural_check() skips pending_qb_checkpoint when counting blocking pending gates.
#   Why:  run_spec.py now calls evaluate_report() directly. pending_qb_checkpoint is the
#         designed terminal state — treating it as structural failure caused qualitative=None,
#         defeating the QB Reviewer loop wired into run_spec.py today.
#   How:  Check 3 in _structural_check skips status == pending_qb_checkpoint.
# [2026-04-07] Josh + Claude — Report evaluator with Reviewer persona (Phase 3)
# What: Structural acceptance check + Reviewer persona qualitative evaluation
# Why: The iteration loop needs a decision — done, iterate, or escalate
# How: Two layers: mechanical criteria check (pass/fail) + persona evaluation (judgment)
# -------------------

"""Report Evaluator — decides done / iterate / escalate.

Two layers of evaluation:
1. Structural: Do the acceptance criteria mechanically pass? (no LLM needed)
2. Qualitative: Reviewer persona evaluates quality, completeness, concerns (RP model)

The structural check is the gate. If criteria don't pass, it's iterate regardless
of what the persona thinks. If criteria pass, the persona can still flag concerns
and recommend iteration or escalation.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger("report_evaluator")


def _structural_check(report: dict) -> dict:
    """Mechanical check — does the report indicate success?

    No LLM. Pure logic on the report structure.
    Returns: {passed: bool, reasons: [str], details: {criterion: pass/fail}}
    """
    status = report.get("status", "unknown")
    summary = report.get("summary", {})
    step_results = report.get("step_results", {})

    reasons = []
    criteria_results = {}

    # Check 1: Block status
    if status == "aborted":
        reasons.append(f"Block aborted: {report.get('abort_reason', 'unknown')}")
    elif status == "partial_failure":
        failed_steps = [sid for sid, r in step_results.items() if r.get("status") == "fail"]
        reasons.append(f"Steps failed: {', '.join(failed_steps)}")

    # Check 2: Any failed steps?
    failed_count = summary.get("failed", 0)
    if failed_count > 0:
        for sid, r in step_results.items():
            if r.get("status") == "fail":
                reason = r.get("reason", "unknown")
                reasons.append(f"Step {sid} failed: {reason}")

    # Check 3: Any pending gates?
    # pending_qb_checkpoint is the designed terminal state when running via run_spec.py —
    # QB resolves it via the Reviewer loop. Only human_review blocks structural pass.
    pending_count = summary.get("pending_review", 0)
    if pending_count > 0:
        for sid, r in step_results.items():
            status_val = r.get("status", "")
            if "pending" in status_val and status_val != "pending_qb_checkpoint":
                reasons.append(f"Step {sid} pending: {r.get('description', r.get('gate_type', 'review'))}")

    # Check 4: Were acceptance criteria addressed?
    # We can't mechanically verify most acceptance criteria — that's the persona's job.
    # But we can flag if the block never ran far enough to address them.
    acceptance = report.get("acceptance_criteria", [])
    total_steps = summary.get("total_steps", 0)
    if total_steps == 0:
        reasons.append("No steps executed — acceptance criteria cannot be evaluated")

    passed = len(reasons) == 0
    return {
        "passed": passed,
        "reasons": reasons,
        "summary": {
            "status": status,
            "steps_passed": summary.get("passed", 0),
            "steps_failed": failed_count,
            "steps_pending": pending_count,
            "tool_calls": summary.get("tool_calls", 0),
            "elapsed_seconds": summary.get("elapsed_seconds", 0),
        },
    }


def _format_report_for_review(report: dict) -> str:
    """Format an execution report as readable text for the Reviewer persona."""
    lines = []
    lines.append(f"# Execution Report: {report.get('block_name', 'unknown')}")
    lines.append(f"**Block ID:** {report.get('block_id', '?')}")
    lines.append(f"**Status:** {report.get('status', '?')}")

    if report.get("abort_reason"):
        lines.append(f"**Abort Reason:** {report['abort_reason']}")

    summary = report.get("summary", {})
    lines.append(f"\n**Summary:** {summary.get('passed', 0)} passed, "
                 f"{summary.get('failed', 0)} failed, "
                 f"{summary.get('pending_review', 0)} pending, "
                 f"{summary.get('tool_calls', 0)} tool calls, "
                 f"{summary.get('elapsed_seconds', 0)}s elapsed")

    lines.append("\n## Step Results\n")
    for sid, result in report.get("step_results", {}).items():
        status = result.get("status", "?")
        tool = result.get("tool", "")
        preview = result.get("result_preview", "")[:200]
        reason = result.get("reason", "")

        if status == "fail":
            lines.append(f"- **{sid}** [{tool}] FAIL: {reason}")
        elif status == "pass":
            lines.append(f"- **{sid}** [{tool}] PASS: {preview[:100]}...")
        elif "pending" in status:
            lines.append(f"- **{sid}** PENDING: {result.get('description', status)}")
        else:
            lines.append(f"- **{sid}** {status}")

    lines.append("\n## Acceptance Criteria\n")
    for criterion in report.get("acceptance_criteria", []):
        lines.append(f"- [ ] {criterion}")

    bindings = report.get("bindings", {})
    if bindings:
        lines.append("\n## Key Bindings (truncated)\n")
        for var, val in bindings.items():
            lines.append(f"- `{var}`: {val[:100]}...")

    return "\n".join(lines)


def evaluate_report(
    report: dict,
    spec: Optional[dict] = None,
    graph_context: Optional[list] = None,
    use_persona: bool = True,
) -> dict:
    """Evaluate an execution report. Returns a decision.

    Args:
        report: The execution report from SpecExecutor
        spec: The original spec (for context — constraints, acceptance criteria)
        graph_context: Graph recall results for Reviewer context
        use_persona: If True, calls Reviewer persona for qualitative evaluation.
                     If False, structural check only (faster, no API call).

    Returns:
        {
            decision: "done" | "iterate" | "escalate",
            structural: {passed, reasons, summary},
            qualitative: {response, concerns, recommendation} | None,
            iteration_hints: [str] — specific things to fix in a follow-up spec,
            escalate_to: str | None — role to escalate to if decision is escalate,
            escalate_reason: str | None,
        }
    """
    # Layer 1: Structural check (no LLM)
    structural = _structural_check(report)

    # If structural check fails, we know we need to iterate (or escalate if aborted)
    if not structural["passed"]:
        status = report.get("status", "")

        # Aborted with escalation = escalate up the chain
        if status == "aborted" and "ESCALATED" in (report.get("abort_reason") or ""):
            return {
                "decision": "escalate",
                "structural": structural,
                "qualitative": None,
                "iteration_hints": structural["reasons"],
                "escalate_to": "strategist",
                "escalate_reason": report.get("abort_reason", "Block escalated"),
            }

        # Aborted for other reasons or partial failure = iterate
        # But build iteration hints from the failure reasons
        return {
            "decision": "iterate",
            "structural": structural,
            "qualitative": None,
            "iteration_hints": structural["reasons"],
            "escalate_to": None,
            "escalate_reason": None,
        }

    # Structural passed — everything green mechanically
    if not use_persona:
        return {
            "decision": "done",
            "structural": structural,
            "qualitative": None,
            "iteration_hints": [],
            "escalate_to": None,
            "escalate_reason": None,
        }

    # Layer 2: Reviewer persona qualitative evaluation
    try:
        from persona_client import call_persona

        report_text = _format_report_for_review(report)

        # Build the review task
        constraints_text = ""
        if spec:
            never = spec.get("constraints", {}).get("never", [])
            anti_drift = spec.get("constraints", {}).get("anti_drift", [])
            if never:
                constraints_text += "\n**Constitutional constraints (never):**\n"
                constraints_text += "\n".join(f"- {n}" for n in never)
            if anti_drift:
                constraints_text += "\n**Anti-drift rules:**\n"
                constraints_text += "\n".join(f"- {a}" for a in anti_drift)

        task = (
            "You are reviewing an execution report. All steps passed structurally. "
            "Your job is to decide if this work is truly DONE, needs ITERATION, or "
            "should be ESCALATED to a higher authority.\n\n"
            "Evaluate:\n"
            "1. Did the work actually accomplish the acceptance criteria, or just not error?\n"
            "2. Are there quality concerns even though steps passed?\n"
            "3. Were any constraints violated (check the 'never' and 'anti-drift' rules)?\n"
            "4. Is there anything a Tracker should investigate further?\n"
            "5. Is there anything Razor should review for security?\n\n"
            "Respond with your assessment, then on the LAST LINE write exactly one of:\n"
            "DECISION: DONE\n"
            "DECISION: ITERATE — [specific reason]\n"
            "DECISION: ESCALATE — [to whom] — [reason]\n\n"
            f"{report_text}"
            f"{constraints_text}"
        )

        result = call_persona(
            role="reviewer",
            task=task,
            graph_context=graph_context,
            temperature=0.4,  # Lower temp for evaluation — we want focused, not creative
        )

        response_text = result.get("response", "")

        # Parse the decision from the last line
        decision = "done"
        escalate_to = None
        escalate_reason = None
        iteration_hints = []

        lines = response_text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("DECISION:"):
                decision_text = line[len("DECISION:"):].strip()
                if decision_text.startswith("DONE"):
                    decision = "done"
                elif decision_text.startswith("ITERATE"):
                    decision = "iterate"
                    reason = decision_text.split("—", 1)[-1].strip() if "—" in decision_text else decision_text
                    iteration_hints = [reason] if reason else []
                elif decision_text.startswith("ESCALATE"):
                    decision = "escalate"
                    parts = decision_text.split("—")
                    if len(parts) >= 3:
                        escalate_to = parts[1].strip().lower()
                        escalate_reason = parts[2].strip()
                    elif len(parts) >= 2:
                        escalate_to = parts[1].strip().lower()
                        escalate_reason = "Reviewer escalation"
                break

        qualitative = {
            "response": response_text,
            "decision_raw": decision_text if 'decision_text' in dir() else "",
            "model": result.get("model", ""),
            "elapsed_seconds": result.get("elapsed_seconds", 0),
        }

        return {
            "decision": decision,
            "structural": structural,
            "qualitative": qualitative,
            "iteration_hints": iteration_hints,
            "escalate_to": escalate_to,
            "escalate_reason": escalate_reason,
        }

    except Exception as e:
        logger.error("Reviewer persona evaluation failed: %s", e, exc_info=True)
        # Fall back to structural-only decision
        return {
            "decision": "done",  # Structural passed, persona unavailable — cautious pass
            "structural": structural,
            "qualitative": {"error": str(e)},
            "iteration_hints": [],
            "escalate_to": None,
            "escalate_reason": None,
        }
