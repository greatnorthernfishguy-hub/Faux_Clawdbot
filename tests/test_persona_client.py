from pathlib import Path
from types import SimpleNamespace

import pytest

import persona_client


EXPECTED_ROLES = {
    "strategist",
    "tracker",
    "wrench",
    "forge",
    "razor",
    "reviewer",
}


def test_complete_six_person_unit_is_packaged():
    assert persona_client.TQB_ROLES == EXPECTED_ROLES
    for role in EXPECTED_ROLES:
        assert (persona_client.PERSONALITY_DIR / f"{role}.md").is_file()
    assert persona_client.UNIT_DISCIPLINES_FILE.is_file()


@pytest.mark.parametrize(
    ("role", "identity_text"),
    [
        ("strategist", "Spec generation, decomposition, architecture planning"),
        ("tracker", "Bug hunting, root cause analysis, regression detection"),
        ("wrench", "Unconventional solutions, edge cases, creative problem-solving"),
        ("forge", "Integration, assembly, cross-module wiring"),
        ("razor", "Security review, attack surface analysis, policy enforcement"),
        ("reviewer", "Code quality, standards enforcement"),
    ],
)
def test_each_role_loads_its_own_canonical_identity(role, identity_text):
    personality, disciplines = persona_client.load_personality(role)

    assert identity_text in personality
    assert "Verify Before Asserting" in disciplines


def test_system_prompt_contains_persona_and_full_unit_disciplines():
    personality, disciplines = persona_client.load_personality("tracker")

    prompt = persona_client._build_system_prompt("tracker", personality, disciplines)

    assert "You are Tracker" in prompt
    assert "## 1. Verify Before Asserting" in prompt
    assert "## 2. The Graph Is Memory, Not Truth" in prompt
    assert "## 3. No Improvisation" in prompt
    assert "## 4. Report What You See, Not What You Expected" in prompt
    assert prompt.count("UNIT DISCIPLINES") == 0


def test_unknown_role_is_rejected_before_path_resolution():
    with pytest.raises(ValueError, match="Unknown TQB role"):
        persona_client.load_personality("../reviewer")


def test_docker_image_copies_the_persona_assets():
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()

    assert "ENV TQB_PERSONALITY_DIR=/app/tqb-personalities" in dockerfile
    assert "COPY tqb-personalities/ ./tqb-personalities/" in dockerfile


def test_persona_reports_requested_and_actual_route(monkeypatch):
    requested = SimpleNamespace(
        provider="huggingface", model_id="source/model", role="tracker"
    )
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Evidence found.")],
        provider="openrouter",
        model="fallback/model",
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7, cost=0.004),
    )
    monkeypatch.setattr(persona_client, "resolve_route", lambda role: requested)
    monkeypatch.setattr(persona_client, "get_client", lambda provider: object())
    monkeypatch.setattr(persona_client, "call_model", lambda *args, **kwargs: response)

    result = persona_client.call_persona("tracker", "Trace the bug.", max_retries=0)

    assert result["provider"] == "openrouter"
    assert result["model"] == "fallback/model"
    assert result["requested_provider"] == "huggingface"
    assert result["requested_model"] == "source/model"
    assert result["input_tokens"] == 12
    assert result["output_tokens"] == 7
    assert result["cost"] == 0.004


def test_persona_uses_shared_runner_when_tool_runtime_is_bound(monkeypatch):
    requested = SimpleNamespace(
        provider="openrouter", model_id="worker/model", role="tracker"
    )
    tool_runtime = SimpleNamespace(
        tool_definitions=[{"name": "read_file"}], execute=lambda name, args: None,
    )
    captured = {}
    monkeypatch.setattr(persona_client, "resolve_route", lambda role: requested)
    monkeypatch.setattr(persona_client, "get_client", lambda provider: object())

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {
            "text": "Root cause found.",
            "model_calls": [{
                "provider": "openrouter", "model": "worker/model",
                "input_tokens": 20, "output_tokens": 5, "cost": 0.002,
            }],
            "staged_actions": [],
            "tool_events": [{"tool": "read_file", "status": "executed"}],
        }

    monkeypatch.setattr(persona_client, "run_tool_loop", fake_runner)
    result = persona_client.call_persona(
        "tracker", "Find the cause.", tool_runtime=tool_runtime,
    )

    assert captured["tools"] == tool_runtime.tool_definitions
    assert captured["execute_tool_fn"] == tool_runtime.execute
    assert result["response"] == "Root cause found."
    assert result["tool_events"][0]["tool"] == "read_file"
