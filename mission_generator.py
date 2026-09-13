"""Bethany's evidence-first MissionPlan generator."""

# ---- Changelog ----
# [2026-09-12] Codex — Deposit Bethany's raw response before plan validation.
# What: Send the untouched planning response to the shared substrate before
#       parsing its JSON or applying the MissionPlan schema.
# Why: A structured control artifact cannot replace raw experience under Law 7.
# How: Use the existing Graph on_message pathway; parsing happens afterward.
# [2026-09-12] Codex — Give Bethany the multi-block planning contract.
# What: Generate, parse, and mechanically validate MissionPlans after read-only
#       repository inspection through the shared specialist tool loop.
# Why: Strategist's canonical contract emits one WorkBlockSpec; Bethany owns
#      whole-mission judgment and must divide work across all six specialists.
# How: Bind the leader persona to read tools, expose only trusted validation IDs,
#      and retry malformed output with exact validator errors.
# -------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from mission_plan import MISSION_PLAN_SCHEMA, validate_mission_plan
from persona_client import call_persona
from specialist_tools import SpecialistToolRuntime


def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "codemine_mission_plan",
            "strict": True,
            "schema": MISSION_PLAN_SCHEMA,
        },
    }


def _catalog_text(validation_catalog: dict[str, tuple[str, ...]]) -> str:
    if not validation_catalog:
        return "- No executable validations are available. Produce read-only blocks only."
    return "\n".join(
        f"- {validation_id}: {json.dumps(list(argv))}"
        for validation_id, argv in sorted(validation_catalog.items())
    )


def _task_prompt(intent: str, constraints: Optional[str], validation_catalog: dict) -> str:
    constraints_text = constraints.strip() if constraints else "No additional constraints supplied."
    return f"""Create one MissionPlan for the approved mission below.

MISSION INTENT
{intent}

PROJECT CONSTRAINTS
{constraints_text}

TRUSTED VALIDATION CATALOG
{_catalog_text(validation_catalog)}

Inspect the actual repository with your read-only tools before planning. Use
only relative repository paths. Assign every block to exactly one of:
strategist, tracker, wrench, forge, razor, reviewer. Any specialist may be a
bounded implementation grunt. Blocks that can run together must have disjoint
write paths. Express every required ordering edge in depends_on. A writable
block must select at least one validation ID from the catalog. Do not invent
commands or validation IDs.

Return only a JSON object matching MissionPlan version 1.0.0. Do not return a
WorkBlockSpec, prose, or markdown fences."""


def generate_mission_plan(
    *,
    intent: str,
    workspace: Path,
    worker_ng,
    validation_catalog: Optional[dict[str, tuple[str, ...]]] = None,
    constraints: Optional[str] = None,
    graph_context: Optional[list] = None,
    read_only_paths=None,
    max_retries: int = 2,
    persona_call=call_persona,
) -> dict:
    """Ask Bethany for a validated plan without granting mutation authority."""
    workspace = Path(workspace).resolve()
    catalog = {
        key: tuple(argv) for key, argv in (validation_catalog or {}).items()
    }
    runtime = SpecialistToolRuntime(
        role="leader",
        workspace=workspace,
        worker_ng=worker_ng,
        read_only_paths=read_only_paths,
        allow_write_tools=False,
        validation_commands=None,
    )
    task = _task_prompt(intent, constraints, catalog)
    attempts = []

    for attempt_number in range(1, max_retries + 2):
        result = persona_call(
            role="leader",
            task=task,
            graph_context=graph_context,
            response_format=_response_format(),
            temperature=0.3,
            max_tokens=8192,
            tool_runtime=runtime,
        )
        raw_response = result.get("response", "")
        if worker_ng is not None and raw_response:
            try:
                worker_ng.on_message(raw_response)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        candidate = result.get("structured") or _extract_json(raw_response)
        valid, errors = validate_mission_plan(candidate or {}, set(catalog))
        attempts.append({
            "attempt": attempt_number,
            "provider": result.get("provider"),
            "model": result.get("model"),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cost": result.get("cost"),
            "tool_events": result.get("tool_events", []),
            "errors": errors,
        })
        if valid:
            return {
                "success": True,
                "plan": candidate,
                "errors": [],
                "attempts": attempts,
            }
        task = _task_prompt(intent, constraints, catalog) + (
            "\n\nYOUR PREVIOUS OUTPUT FAILED MECHANICAL VALIDATION\n- "
            + "\n- ".join(errors)
            + "\nCorrect every error and return the complete JSON object only."
        )

    return {
        "success": False,
        "plan": candidate,
        "errors": errors,
        "attempts": attempts,
    }
