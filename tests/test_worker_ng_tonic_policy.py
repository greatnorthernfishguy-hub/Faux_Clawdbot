"""Codemine must never silently fall back to heuristic Tonic inference."""

import pytest

import openclaw_hook
import worker_ng


def test_worker_config_disables_tonic():
    assert worker_ng.WORKER_SNN_CONFIG["tonic"] == {"enabled": False}
    assert openclaw_hook.OPENCLAW_SNN_CONFIG["tonic"] == {"enabled": False}


def test_disabled_tonic_mode_is_accepted(monkeypatch):
    monkeypatch.setenv("CODEMINE_TONIC_MODE", "disabled")
    worker_ng._require_safe_tonic_mode()


@pytest.mark.parametrize("mode", ["heuristic", "transformer", "enabled", "true"])
def test_every_unimplemented_tonic_mode_fails_closed(monkeypatch, mode):
    monkeypatch.setenv("CODEMINE_TONIC_MODE", mode)
    with pytest.raises(RuntimeError, match="heuristic execution is prohibited"):
        worker_ng._require_safe_tonic_mode()


def test_direct_graph_accessor_refuses_preservation_mode(monkeypatch):
    monkeypatch.setenv("CODEMINE_BOOT_MODE", "preserve")
    monkeypatch.setattr(
        worker_ng.NeuroGraphMemory,
        "get_instance",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Graph construction must not be reached")
        ),
    )
    with pytest.raises(RuntimeError, match="live startup refused"):
        worker_ng.get_worker_ng()
