"""Tests for independent per-role provider/model routing."""

# ---- Changelog ----
# [2026-09-12] Codex — Pin concurrent-safe Codemine model routing.
# What: Verify environment precedence, missing-config refusal, explicit adapter
#       dispatch, and Qwen-only thinking control.
# Why: One import-time provider value cannot safely serve a multi-model TQB unit.
# How: Isolate environment settings and inject fake provider clients.
# -------------------

import pytest

import model_client


def clear_routes(monkeypatch):
    for key in list(model_client.os.environ):
        if key.startswith("CODEMINE_") or key in {"HF_MODEL_ID", "CLAUDE_MODEL_ID"}:
            monkeypatch.delenv(key, raising=False)


def test_role_specific_route_wins(monkeypatch):
    clear_routes(monkeypatch)
    monkeypatch.setenv("CODEMINE_PROVIDER", "anthropic")
    monkeypatch.setenv("CLAUDE_MODEL_ID", "claude/default")
    monkeypatch.setenv("CODEMINE_FORGE_PROVIDER", "openrouter")
    monkeypatch.setenv("CODEMINE_FORGE_MODEL_ID", "z-ai/glm-5.3-flash")

    route = model_client.resolve_route("forge")

    assert route == model_client.ModelRoute(
        provider="openrouter",
        model_id="z-ai/glm-5.3-flash",
        role="forge",
    )


def test_default_chat_route_does_not_inherit_worker_model(monkeypatch):
    clear_routes(monkeypatch)
    monkeypatch.setenv("CODEMINE_PROVIDER", "openrouter")
    monkeypatch.setenv("CODEMINE_MODEL_ID", "moonshotai/kimi-k3")
    monkeypatch.setenv("CODEMINE_WORKER_MODEL_ID", "z-ai/glm-5.3-flash")

    assert model_client.resolve_route().model_id == "moonshotai/kimi-k3"


def test_missing_provider_or_model_refuses_to_guess(monkeypatch):
    clear_routes(monkeypatch)
    with pytest.raises(ValueError, match="provider"):
        model_client.resolve_route("tracker")

    monkeypatch.setenv("CODEMINE_TRACKER_PROVIDER", "openrouter")
    with pytest.raises(ValueError, match="Model ID"):
        model_client.resolve_route("tracker")


def test_call_model_dispatches_from_explicit_route(monkeypatch):
    captured = {}

    def fake_openrouter(*args, **kwargs):
        captured.update(kwargs)
        return "response"

    monkeypatch.setattr(model_client, "_call_openrouter", fake_openrouter)
    route = model_client.ModelRoute("openrouter", "provider/model", "forge")

    result = model_client.call_model(
        object(), "system", [], [], route=route, temperature=0.2,
        response_format={"type": "json_object"},
    )

    assert result == "response"
    assert captured["model_id"] == "provider/model"
    assert captured["temperature"] == 0.2
    assert captured["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("model_id", "expected_prefix"),
    [
        ("qwen/qwen3-coder", "/think\n"),
        ("moonshotai/kimi-k3", ""),
        ("z-ai/glm-5.3-flash", ""),
    ],
)
def test_thinking_control_is_only_sent_to_qwen(monkeypatch, model_id, expected_prefix):
    monkeypatch.setenv("QB_THINKING_ENABLED", "true")

    prompt = model_client._prepare_system_prompt("system", model_id)

    assert prompt == expected_prefix + "system"
