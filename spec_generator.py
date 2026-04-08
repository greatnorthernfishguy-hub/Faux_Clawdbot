# ---- Changelog ----
# [2026-04-07] Josh + Claude — Spec generator via Strategist persona (Phase 5)
# What: Turns natural language intent + constraints into valid WorkBlockSpec JSON
# Why: The system needs to generate its own specs, not have humans write JSON by hand
# How: Strategist persona + Graph context + schema reference + validation retry loop
# -------------------

"""Spec Generator — Strategist creates structured work block specs.

The Strategist persona takes natural language intent and constraints,
queries the Graph for relevant past experience, and produces a valid
WorkBlockSpec JSON. Schema validation on output — malformed specs
trigger a retry with the validation errors fed back in.
"""

import json
import logging
from typing import Optional

from persona_client import call_persona
from work_block_schema import validate_spec, WORK_BLOCK_SCHEMA, _TOOL_NAMES

logger = logging.getLogger("spec_generator")

# Compact schema reference for the Strategist prompt — just enough to generate valid specs
_SCHEMA_REFERENCE = f"""## WorkBlockSpec Schema Reference

Required top-level keys: spec_version, block, steps, constraints

### spec_version
Always "1.0.0"

### block
Required: id (string), name (string, max 120 chars), scope (string), acceptance_criteria (array of strings, min 1)
Optional: agent (string), workspace (string — absolute path for cross-repo work), depends_on (array of strings)

### snap_interface (optional)
inputs: array of {{name, type (file|function|config|state), source_block, path}}
outputs: array of {{name, type, path, contract}}

### constraints
Required: never (array of strings, min 1), anti_drift (array of strings)
Optional: tool_allowlist (array from: {', '.join(_TOOL_NAMES)}), shell_allowlist (array of strings), max_iterations (int, 1-100, default 15), timeout_seconds (int, 30-3600, default 300)

### steps (array, min 1)
Five step types:

**action** (required: id, type:"action", tool, params, validation, on_failure)
- tool: one of {', '.join(_TOOL_NAMES)}
- params: object matching tool's input schema
- bind_result: optional "$variable_name" to store result
- validation.checks: array of {{operator, target, value, description}}
  operators: contains, not_contains, equals, not_equals, matches_regex, result_is_string, result_is_not_error, file_exists, file_contains, output_length_gt, output_length_lt
- on_failure: {{action: abort_block|retry|skip|goto|escalate_to_qb, message}}

**gate** (required: id, type:"gate", description, gate_type)
- gate_type: human_review | qb_checkpoint | auto_approve
- staged_actions: array of step IDs

**condition** (required: id, type:"condition", check, if_true, if_false)
- check: {{operator, target, value, description}}
- if_true / if_false: arrays of steps

**loop** (required: id, type:"loop", over, body)
- over: {{items: [strings]}} or {{from_result: "$var", split_on: "\\n"}}
- bind_item: "$item" (default)
- body: array of steps

**group** (required: id, type:"group", steps)
- steps: array of steps
- on_failure: optional

### CRITICAL RULES
1. EVERY action step MUST have a validation block with at least one check
2. EVERY action step MUST have an on_failure handler
3. NEVER guess file paths — start with shell_execute "find" or "ls" to discover repo structure
4. Start specs by reading the source of truth (vault docs, canonical files) before acting
5. Verify before asserting — add read/grep steps before write steps
6. Use shell_execute for reads outside the default workspace (grep, head, tail, find, ls, etc.)
7. All file paths in params must be ABSOLUTE (starting with /) when workspace is set
8. ONLY use operators from the list above — do NOT invent new operators
"""


def generate_spec(
    intent: str,
    constraints: Optional[str] = None,
    graph_context: Optional[list] = None,
    workspace: Optional[str] = None,
    max_retries: int = 2,
) -> dict:
    """Generate a WorkBlockSpec from natural language intent.

    Args:
        intent: What to build/fix/audit (natural language)
        constraints: Project-specific constraints (natural language or structured)
        graph_context: Graph recall results for relevant past experience
        workspace: Workspace path if cross-repo work needed
        max_retries: How many times to retry on schema validation failure

    Returns:
        {
            success: bool,
            spec: dict | None (the validated spec),
            errors: [str] | None (validation errors if failed),
            generation_attempts: int,
            strategist_response: str (raw response for debugging),
        }
    """
    task = _build_generation_prompt(intent, constraints, workspace)

    for attempt in range(max_retries + 1):
        logger.info("Spec generation attempt %d/%d", attempt + 1, max_retries + 1)

        result = call_persona(
            role="strategist",
            task=task,
            graph_context=graph_context,
            temperature=0.4,  # Low temp for structured output
            max_tokens=8192,  # Specs can be long
        )

        raw = result.get("response", "")

        # Extract JSON from the response
        spec = _extract_json(raw)
        if spec is None:
            logger.warning("Attempt %d: No valid JSON found in response", attempt + 1)
            task = _build_retry_prompt(intent, constraints, workspace,
                                       ["No valid JSON found in response. Output ONLY the JSON spec, no prose."])
            continue

        # Schema validation
        valid, errors = validate_spec(spec)
        if valid:
            logger.info("Spec generated and validated on attempt %d", attempt + 1)
            return {
                "success": True,
                "spec": spec,
                "errors": None,
                "generation_attempts": attempt + 1,
                "strategist_response": raw,
            }

        # Validation failed — retry with errors
        logger.warning("Attempt %d: Schema validation failed with %d errors", attempt + 1, len(errors))
        task = _build_retry_prompt(intent, constraints, workspace, errors)

    return {
        "success": False,
        "spec": spec,  # Return the last attempt even if invalid
        "errors": errors if 'errors' in dir() else ["Max retries exceeded"],
        "generation_attempts": max_retries + 1,
        "strategist_response": raw if 'raw' in dir() else "",
    }


def generate_followup_spec(
    original_spec: dict,
    failed_report: dict,
    evaluation: dict,
    graph_context: Optional[list] = None,
    max_retries: int = 2,
) -> dict:
    """Generate a follow-up spec targeting gaps from a failed execution.

    Args:
        original_spec: The spec that was executed
        failed_report: The execution report showing failures
        evaluation: The evaluate_report() output with iteration hints
        graph_context: Graph recall for context
        max_retries: Retry count for validation

    Returns:
        Same format as generate_spec()
    """
    task = _build_followup_prompt(original_spec, failed_report, evaluation)

    for attempt in range(max_retries + 1):
        logger.info("Follow-up spec generation attempt %d/%d", attempt + 1, max_retries + 1)

        result = call_persona(
            role="strategist",
            task=task,
            graph_context=graph_context,
            temperature=0.4,
            max_tokens=8192,
        )

        raw = result.get("response", "")
        spec = _extract_json(raw)

        if spec is None:
            task = _build_retry_prompt(
                f"Follow-up for {original_spec.get('block', {}).get('name', 'unknown')}",
                None, None,
                ["No valid JSON found. Output ONLY the JSON spec."]
            )
            continue

        valid, errors = validate_spec(spec)
        if valid:
            logger.info("Follow-up spec validated on attempt %d", attempt + 1)
            return {
                "success": True,
                "spec": spec,
                "errors": None,
                "generation_attempts": attempt + 1,
                "strategist_response": raw,
            }

        task = _build_retry_prompt(
            f"Follow-up for {original_spec.get('block', {}).get('name', 'unknown')}",
            None, None, errors
        )

    return {
        "success": False,
        "spec": spec if 'spec' in dir() else None,
        "errors": errors if 'errors' in dir() else ["Max retries exceeded"],
        "generation_attempts": max_retries + 1,
        "strategist_response": raw if 'raw' in dir() else "",
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_generation_prompt(intent: str, constraints: Optional[str], workspace: Optional[str]) -> str:
    """Build the initial spec generation prompt for Strategist."""
    parts = [
        "Generate a WorkBlockSpec JSON for the following task.\n",
        "Output ONLY valid JSON — no prose before or after. No markdown code fences.\n\n",
        f"## Task\n{intent}\n",
    ]

    if constraints:
        parts.append(f"\n## Project Constraints\n{constraints}\n")

    if workspace:
        parts.append(f"\n## Workspace\nSet block.workspace to: {workspace}\n")

    parts.append(f"\n{_SCHEMA_REFERENCE}\n")

    parts.append(
        "\n## Remember\n"
        "- Start by reading the source of truth before acting\n"
        "- Verify paths exist on disk before using them\n"
        "- Every action MUST have validation checks and on_failure\n"
        "- Use shell_execute with head/tail/grep for cross-repo reads\n"
        "- Output ONLY the JSON. No explanation. No wrapping.\n"
    )

    return "\n".join(parts)


def _build_retry_prompt(intent: str, constraints: Optional[str], workspace: Optional[str], errors: list) -> str:
    """Build a retry prompt with validation errors."""
    error_text = "\n".join(f"- {e}" for e in errors[:10])
    return (
        f"Your previous spec had validation errors:\n{error_text}\n\n"
        f"Fix these errors and output a corrected WorkBlockSpec JSON.\n"
        f"Output ONLY valid JSON — no prose, no code fences.\n\n"
        f"{_SCHEMA_REFERENCE}"
    )


def _build_followup_prompt(original_spec: dict, report: dict, evaluation: dict) -> str:
    """Build a follow-up spec generation prompt from a failed execution."""
    hints = evaluation.get("iteration_hints", [])
    hints_text = "\n".join(f"- {h}" for h in hints) if hints else "No specific hints."

    # Summarize what failed
    failed_steps = []
    for sid, r in report.get("step_results", {}).items():
        if r.get("status") == "fail":
            failed_steps.append(f"- {sid}: {r.get('reason', 'unknown')}")
    failed_text = "\n".join(failed_steps) if failed_steps else "No step failures (evaluation flagged other issues)."

    return (
        f"The previous spec execution needs a follow-up.\n\n"
        f"## Original Block\n"
        f"Name: {original_spec.get('block', {}).get('name', '?')}\n"
        f"ID: {original_spec.get('block', {}).get('id', '?')}\n\n"
        f"## What Failed\n{failed_text}\n\n"
        f"## Iteration Hints\n{hints_text}\n\n"
        f"## Acceptance Criteria Still Unmet\n"
        f"{json.dumps(original_spec.get('block', {}).get('acceptance_criteria', []), indent=2)}\n\n"
        f"## Constraints From Original Spec\n"
        f"{json.dumps(original_spec.get('constraints', {}), indent=2)}\n\n"
        f"Generate a NEW WorkBlockSpec JSON that addresses the failures and unmet criteria.\n"
        f"Output ONLY valid JSON — no prose, no code fences.\n\n"
        f"{_SCHEMA_REFERENCE}"
    )


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from text, handling various formats.

    Tries: raw JSON, markdown code blocks, finding { } boundaries.
    """
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass

    if "```" in text:
        start = text.index("```") + 3
        # Skip language identifier if present
        newline = text.index("\n", start)
        content_start = newline + 1
        end = text.index("```", content_start)
        try:
            return json.loads(text[content_start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding outermost { } pair
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None
