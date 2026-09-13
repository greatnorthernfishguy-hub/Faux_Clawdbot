"""Tests for Bethany's MissionPlan generation boundary."""

import json

from mission_generator import generate_mission_plan


def valid_plan():
    return {
        "plan_version": "1.0.0",
        "mission": {
            "id": "m-1",
            "name": "Repair behavior",
            "intent": "Repair current behavior with evidence.",
            "acceptance_criteria": ["Tests pass"],
        },
        "blocks": [{
            "id": "trace",
            "name": "Trace failure",
            "role": "tracker",
            "task": "Inspect current behavior and report the cause.",
            "read_paths": ["src"],
            "write_paths": [],
            "depends_on": [],
            "acceptance_criteria": ["Cause has file evidence"],
            "validations": [],
        }],
    }


def test_generates_valid_plan_with_read_only_leader_runtime(tmp_path):
    captured = {}

    def fake_persona_call(**kwargs):
        captured.update(kwargs)
        return {
            "structured": valid_plan(), "response": "", "provider": "openrouter",
            "model": "moonshotai/kimi-k3", "input_tokens": 10,
            "output_tokens": 5, "cost": 0.01, "tool_events": [],
        }

    result = generate_mission_plan(
        intent="Repair it",
        workspace=tmp_path,
        worker_ng=None,
        validation_catalog={"tests": ("python3", "-m", "pytest", "-q")},
        persona_call=fake_persona_call,
    )

    assert result["success"] is True
    assert captured["role"] == "leader"
    tool_names = {item["name"] for item in captured["tool_runtime"].tool_definitions}
    assert "read_file" in tool_names
    assert "write_file" not in tool_names
    assert "run_validation" not in tool_names
    assert "tests" in captured["task"]


def test_retries_with_exact_validation_errors(tmp_path):
    calls = []
    invalid = valid_plan()
    invalid["blocks"][0]["write_paths"] = ["src/fix.py"]
    invalid["blocks"][0]["validations"] = ["invented"]

    def fake_persona_call(**kwargs):
        calls.append(kwargs)
        chosen = invalid if len(calls) == 1 else valid_plan()
        return {
            "structured": None,
            "response": json.dumps(chosen),
            "provider": "openrouter",
            "model": "test-model",
            "tool_events": [],
        }

    result = generate_mission_plan(
        intent="Repair it", workspace=tmp_path, worker_ng=None,
        validation_catalog={"tests": ("python3", "-m", "pytest", "-q")},
        persona_call=fake_persona_call,
    )

    assert result["success"] is True
    assert len(calls) == 2
    assert "unknown validation IDs: invented" in calls[1]["task"]


def test_fails_closed_after_retry_budget(tmp_path):
    def fake_persona_call(**_kwargs):
        return {"structured": None, "response": "not json", "tool_events": []}

    result = generate_mission_plan(
        intent="Repair it", workspace=tmp_path, worker_ng=None,
        max_retries=1, persona_call=fake_persona_call,
    )

    assert result["success"] is False
    assert len(result["attempts"]) == 2
    assert result["errors"]
