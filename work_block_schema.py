# ---- Changelog ----
# [2026-04-06] Josh + Claude — Add edit_file tool + shell_allowlist constraint
# What: (1) edit_file in _TOOL_NAMES (2) shell_allowlist in constraints schema
# Why: Gap 2 — specs need to extend shell allowlist; Gap 3 — edit_file is a new tool
# How: New enum entry in _TOOL_NAMES, new array property in constraints
# [2026-04-05] Josh + Claude — Structured work block spec schema
# What: JSON Schema for QB → Codemine worker handoff
# Why: Structured specs produce +37% better agent execution vs prose (zero improvisation, 97% validation)
# How: jsonschema validation, mandatory validation blocks on every action, on_failure handlers
# -------------------

"""Work Block Spec schema and validation.

Every action step MUST have a validation block and an on_failure handler.
The format is the guardrail — not the model.
"""

import json
import logging
from typing import Tuple

from jsonschema import Draft202012Validator, ValidationError

logger = logging.getLogger("work_block_schema")

# Tools available in Codemine's TOOL_REGISTRY
_TOOL_NAMES = [
    "read_file", "write_file", "edit_file", "list_files",
    "search_code", "search_conversations", "search_testament",
    "ingest_workspace", "shell_execute",
    "push_to_github", "pull_from_github", "create_shadow_branch",
    "notebook_read", "notebook_add", "notebook_delete",
    "map_repository_structure", "get_stats",
]

_CONDITION_OPERATORS = [
    "contains", "not_contains",
    "equals", "not_equals",
    "matches_regex",
    "result_is_string",
    "result_is_not_error",
    "file_exists",
    "file_contains",
    "output_length_gt", "output_length_lt",
]

_FAILURE_ACTIONS = ["abort_block", "retry", "skip", "goto", "escalate_to_qb"]

_GATE_TYPES = ["human_review", "qb_checkpoint", "auto_approve"]

# ---------------------------------------------------------------------------
# Sub-schemas (referenced by $defs in the main schema)
# ---------------------------------------------------------------------------

_CONDITION_CHECK = {
    "type": "object",
    "required": ["operator"],
    "additionalProperties": False,
    "properties": {
        "operator": {"enum": _CONDITION_OPERATORS},
        "target": {"type": "string"},
        "value": {},  # any type — depends on operator
        "description": {"type": "string"},
    },
}

# String shorthands accepted in on_failure for easier spec authoring.
# Executor normalizes these to the full object form before acting on them.
_FAILURE_SHORTHANDS = ["abort", "abort_block", "continue", "skip", "retry", "goto", "escalate_to_qb"]

_FAILURE_HANDLER = {
    "oneOf": [
        {
            "type": "string",
            "enum": _FAILURE_SHORTHANDS,
        },
        {
            "type": "object",
            "required": ["action"],
            "additionalProperties": False,
            "properties": {
                "action": {"enum": _FAILURE_ACTIONS},
                "max_retries": {"type": "integer", "minimum": 0, "default": 0},
                "goto_step": {"type": "string"},
                "message": {"type": "string"},
            },
        },
    ]
}

_VALIDATION_BLOCK = {
    "type": "object",
    "required": ["checks"],
    "additionalProperties": False,
    "properties": {
        "checks": {
            "type": "array",
            "items": _CONDITION_CHECK,
            "minItems": 1,
        },
    },
}

# Step definitions — each is a separate schema, unified via oneOf in the main schema
_ACTION_STEP = {
    "type": "object",
    "required": ["id", "type", "tool", "params", "validation", "on_failure"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "type": {"const": "action"},
        "description": {"type": "string"},
        "tool": {"enum": _TOOL_NAMES},
        "params": {"type": "object"},
        "bind_result": {"type": "string", "pattern": r"^\$[a-z_][a-z0-9_]*$"},
        "validation": _VALIDATION_BLOCK,
        "on_failure": _FAILURE_HANDLER,
    },
}

_GATE_STEP = {
    "type": "object",
    "required": ["id", "type", "description", "gate_type"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "type": {"const": "gate"},
        "description": {"type": "string"},
        "gate_type": {"enum": _GATE_TYPES},
        "staged_actions": {"type": "array", "items": {"type": "string"}},
        "timeout_seconds": {"type": "integer", "default": 300},
        "on_timeout": {"enum": ["abort", "skip", "auto_approve"], "default": "abort"},
    },
}

# Forward-reference placeholder — condition, loop, group contain nested steps.
# jsonschema handles recursive $ref via the $defs mechanism, but since we're
# building the schema as a Python dict, we use a sentinel and patch it below.
_STEP_REF = {"$ref": "#/$defs/step"}

_CONDITION_STEP = {
    "type": "object",
    "required": ["id", "type", "check", "if_true", "if_false"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "type": {"const": "condition"},
        "description": {"type": "string"},
        "check": _CONDITION_CHECK,
        "if_true": {"type": "array", "items": _STEP_REF},
        "if_false": {"type": "array", "items": _STEP_REF},
    },
}

_LOOP_STEP = {
    "type": "object",
    "required": ["id", "type", "over", "body"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "type": {"const": "loop"},
        "description": {"type": "string"},
        "over": {
            "oneOf": [
                {
                    "type": "object",
                    "required": ["items"],
                    "additionalProperties": False,
                    "properties": {
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
                {
                    "type": "object",
                    "required": ["from_result"],
                    "additionalProperties": False,
                    "properties": {
                        "from_result": {"type": "string"},
                        "split_on": {"type": "string", "default": "\n"},
                    },
                },
            ],
        },
        "bind_item": {"type": "string", "pattern": r"^\$[a-z_][a-z0-9_]*$", "default": "$item"},
        "max_iterations": {"type": "integer", "default": 20},
        "body": {"type": "array", "items": _STEP_REF, "minItems": 1},
    },
}

_GROUP_STEP = {
    "type": "object",
    "required": ["id", "type", "steps"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "type": {"const": "group"},
        "description": {"type": "string"},
        "steps": {"type": "array", "items": _STEP_REF, "minItems": 1},
        "on_failure": _FAILURE_HANDLER,
    },
}

# ---------------------------------------------------------------------------
# Main schema
# ---------------------------------------------------------------------------

WORK_BLOCK_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WorkBlockSpec",
    "description": "Structured execution spec for QB -> Codemine worker handoff.",
    "type": "object",
    "required": ["spec_version", "block", "steps", "constraints"],
    "additionalProperties": False,
    "properties": {
        "spec_version": {"const": "1.0.0"},
        "block": {
            "type": "object",
            "required": ["id", "name", "scope", "acceptance_criteria"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string", "maxLength": 120},
                "agent": {"type": "string"},
                "scope": {"type": "string"},
                "workspace": {
                    "type": "string",
                    "description": "Workspace root for PolicyEngine path checks. Defaults to Codemine repo if omitted.",
                },
                "acceptance_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
        },
        "snap_interface": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type", "source_block"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"enum": ["file", "function", "config", "state"]},
                            "source_block": {"type": "string"},
                            "path": {"type": "string"},
                        },
                    },
                    "default": [],
                },
                "outputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type", "path"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"enum": ["file", "function", "config", "state"]},
                            "path": {"type": "string"},
                            "contract": {"type": "string"},
                        },
                    },
                    "default": [],
                },
            },
        },
        "constraints": {
            "type": "object",
            "required": ["never", "anti_drift"],
            "additionalProperties": False,
            "properties": {
                "never": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "anti_drift": {"type": "array", "items": {"type": "string"}},
                "tool_allowlist": {"type": "array", "items": {"enum": _TOOL_NAMES}},
                "shell_allowlist": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional shell command prefixes allowed for this spec, extends PolicyEngine's base allowlist.",
                },
                "max_iterations": {"type": "integer", "minimum": 1, "maximum": 100, "default": 15},
                "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 3600, "default": 300},
            },
        },
        "steps": {
            "type": "array",
            "items": _STEP_REF,
            "minItems": 1,
        },
    },
    "$defs": {
        "step": {
            "oneOf": [
                _ACTION_STEP,
                _GATE_STEP,
                _CONDITION_STEP,
                _LOOP_STEP,
                _GROUP_STEP,
            ],
        },
    },
}

# Pre-compile the validator for reuse
_validator = Draft202012Validator(WORK_BLOCK_SCHEMA)


def validate_spec(spec: dict) -> Tuple[bool, list]:
    """Validate a work block spec against the schema.

    Returns (is_valid, errors) where errors is a list of human-readable strings.
    """
    errors = []
    for error in sorted(_validator.iter_errors(spec), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{path}: {error.message}")

    if errors:
        logger.warning("Spec validation failed with %d errors", len(errors))
    return len(errors) == 0, errors


def validate_spec_file(path: str) -> Tuple[bool, list]:
    """Load and validate a JSON spec file."""
    try:
        with open(path, "r") as f:
            spec = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, [f"Failed to load spec: {e}"]
    return validate_spec(spec)
