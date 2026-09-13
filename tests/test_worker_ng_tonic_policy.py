"""Codemine must never silently fall back to heuristic Tonic inference."""

import pytest

import worker_ng


def test_worker_config_disables_tonic():
    assert worker_ng.WORKER_SNN_CONFIG["tonic"] == {"enabled": False}


def test_disabled_tonic_mode_is_accepted(monkeypatch):
    monkeypatch.setenv("CODEMINE_TONIC_MODE", "disabled")
    worker_ng._require_safe_tonic_mode()


@pytest.mark.parametrize("mode", ["heuristic", "transformer", "enabled", "true"])
def test_every_unimplemented_tonic_mode_fails_closed(monkeypatch, mode):
    monkeypatch.setenv("CODEMINE_TONIC_MODE", mode)
    with pytest.raises(RuntimeError, match="heuristic execution is prohibited"):
        worker_ng._require_safe_tonic_mode()
