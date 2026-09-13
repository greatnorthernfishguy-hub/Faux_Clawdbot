"""Structural MissionPlan contract for Bethany's Lego decomposition."""

# ---- Changelog ----
# [2026-09-12] Codex — Keep executable validations outside model-authored plans.
# What: Mission blocks reference coordinator-owned validation IDs instead of
#       embedding arbitrary argv selected by the planning model.
# Why: shell=False prevents shell syntax attacks, but `python -c` or another
#      executable would still give a model unrestricted process authority.
# How: Accept string IDs in the schema and optionally verify them against the
#      trusted catalog supplied by the coordinator.
# [2026-09-12] Codex — Define the multi-block TQB mission contract.
# What: Validate roles, paths, dependencies, acyclic ordering, and write-set
#       isolation; derive dependency levels for safe parallel dispatch.
# Why: Parallel work is safe only when non-overlap and ordering are mechanical,
#      rather than trusted to a model's prose decomposition.
# How: JSON Schema handles shape; explicit graph/path checks handle semantics.
# -------------------

from __future__ import annotations

from pathlib import PurePosixPath

from jsonschema import Draft202012Validator

from persona_client import TQB_ROLES


MISSION_PLAN_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["plan_version", "mission", "blocks"],
    "additionalProperties": False,
    "properties": {
        "plan_version": {"const": "1.0.0"},
        "mission": {
            "type": "object",
            "required": ["id", "name", "intent", "acceptance_criteria"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
                "intent": {"type": "string", "minLength": 1},
                "acceptance_criteria": {
                    "type": "array", "items": {"type": "string"}, "minItems": 1,
                },
            },
        },
        "blocks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "id", "name", "role", "task", "read_paths", "write_paths",
                    "depends_on", "acceptance_criteria", "validations",
                ],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "name": {"type": "string", "minLength": 1},
                    "role": {"enum": sorted(TQB_ROLES)},
                    "task": {"type": "string", "minLength": 1},
                    "read_paths": {"type": "array", "items": {"type": "string"}},
                    "write_paths": {"type": "array", "items": {"type": "string"}},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "acceptance_criteria": {
                        "type": "array", "items": {"type": "string"}, "minItems": 1,
                    },
                    "validations": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    },
}

_validator = Draft202012Validator(MISSION_PLAN_SCHEMA)


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _paths_overlap(left: str, right: str) -> bool:
    left_path = PurePosixPath(left)
    right_path = PurePosixPath(right)
    return left_path == right_path or left_path in right_path.parents or right_path in left_path.parents


def validate_mission_plan(
    plan: dict, validation_ids: set[str] | frozenset[str] | None = None,
) -> tuple[bool, list[str]]:
    errors = [
        f"{'.'.join(map(str, error.absolute_path)) or '(root)'}: {error.message}"
        for error in sorted(_validator.iter_errors(plan), key=lambda item: list(item.path))
    ]
    if errors:
        return False, errors

    blocks = plan["blocks"]
    ids = [block["id"] for block in blocks]
    if len(ids) != len(set(ids)):
        errors.append("blocks: block IDs must be unique")
    known = set(ids)
    by_id = {block["id"]: block for block in blocks}

    for block in blocks:
        unknown = set(block["depends_on"]) - known
        if unknown:
            errors.append(f"{block['id']}: unknown dependencies: {', '.join(sorted(unknown))}")
        if block["id"] in block["depends_on"]:
            errors.append(f"{block['id']}: block cannot depend on itself")
        for field in ("read_paths", "write_paths"):
            for path in block[field]:
                if not _safe_relative_path(path):
                    errors.append(f"{block['id']}.{field}: unsafe relative path: {path}")
        if block["write_paths"] and not block["validations"]:
            errors.append(f"{block['id']}: writable block requires validation IDs")
        selected_validations = block["validations"]
        if len(selected_validations) != len(set(selected_validations)):
            errors.append(f"{block['id']}: validation IDs must be unique")
        if validation_ids is not None:
            unknown_validations = set(selected_validations) - set(validation_ids)
            if unknown_validations:
                errors.append(
                    f"{block['id']}: unknown validation IDs: "
                    f"{', '.join(sorted(unknown_validations))}"
                )

    if errors:
        return False, errors

    ancestors: dict[str, set[str]] = {}

    def visit(block_id: str, trail: tuple[str, ...] = ()) -> set[str]:
        if block_id in trail:
            cycle = " -> ".join((*trail, block_id))
            raise ValueError(f"dependency cycle: {cycle}")
        if block_id in ancestors:
            return ancestors[block_id]
        result = set(by_id[block_id]["depends_on"])
        for dependency in by_id[block_id]["depends_on"]:
            result.update(visit(dependency, (*trail, block_id)))
        ancestors[block_id] = result
        return result

    try:
        for block_id in ids:
            visit(block_id)
    except ValueError as exc:
        return False, [str(exc)]

    for index, left in enumerate(blocks):
        for right in blocks[index + 1:]:
            serialized = (
                left["id"] in ancestors[right["id"]]
                or right["id"] in ancestors[left["id"]]
            )
            if serialized:
                continue
            for left_path in left["write_paths"]:
                for right_path in right["write_paths"]:
                    if _paths_overlap(left_path, right_path):
                        errors.append(
                            f"parallel blocks {left['id']} and {right['id']} have "
                            f"overlapping writes: {left_path} / {right_path}"
                        )
    return not errors, errors


def dependency_levels(plan: dict) -> list[list[str]]:
    """Return blocks grouped into safe parallel levels."""
    valid, errors = validate_mission_plan(plan)
    if not valid:
        raise ValueError("; ".join(errors))
    pending = {block["id"]: set(block["depends_on"]) for block in plan["blocks"]}
    levels = []
    completed: set[str] = set()
    while pending:
        ready = sorted(
            block_id for block_id, dependencies in pending.items()
            if dependencies <= completed
        )
        if not ready:
            raise ValueError("dependency graph cannot advance")
        levels.append(ready)
        completed.update(ready)
        for block_id in ready:
            del pending[block_id]
    return levels
