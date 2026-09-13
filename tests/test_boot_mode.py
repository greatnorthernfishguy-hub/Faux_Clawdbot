"""Tests for the fail-closed Python boot authorization boundary."""

# ---- Changelog ----
# [2026-09-13] Codex — Pin preservation-runtime and constructor refusal.
# What: Require a reviewed code gate and verify refusal precedes filesystem writes.
# Why: Environment values alone must not wake this incomplete hemisphere.
# How: Test default refusal, CLI refusal, and an explicit test-only readiness patch.
# -------------------

import pytest

import boot_mode


def test_direct_graph_constructor_refuses_before_filesystem_write(tmp_path, monkeypatch):
    from openclaw_hook import NeuroGraphMemory

    monkeypatch.delenv("CODEMINE_BOOT_MODE", raising=False)
    with pytest.raises(boot_mode.BootModeError, match="live startup refused"):
        NeuroGraphMemory(workspace_dir=str(tmp_path / "graph"))
    assert not (tmp_path / "graph").exists()


def test_missing_mode_defaults_to_preservation_and_refuses(monkeypatch):
    monkeypatch.delenv("CODEMINE_BOOT_MODE", raising=False)
    monkeypatch.delenv("CODEMINE_NORMAL_BOOT_ACK", raising=False)
    assert boot_mode.boot_mode() == "preserve"
    with pytest.raises(boot_mode.BootModeError, match="live startup refused"):
        boot_mode.require_normal_boot()


@pytest.mark.parametrize("mode", ["", "normal ", "NORMAL", "start", "preserve"])
def test_noncanonical_modes_refuse(monkeypatch, mode):
    monkeypatch.setenv("CODEMINE_BOOT_MODE", mode)
    monkeypatch.setenv("CODEMINE_NORMAL_BOOT_ACK", boot_mode.NORMAL_BOOT_ACK)
    with pytest.raises(boot_mode.BootModeError):
        boot_mode.require_normal_boot()


def test_normal_mode_requires_exact_second_acknowledgement(monkeypatch):
    monkeypatch.setenv("CODEMINE_BOOT_MODE", "normal")
    monkeypatch.setenv("CODEMINE_TONIC_MODE", "transformer")
    monkeypatch.setenv("CODEMINE_NORMAL_BOOT_ACK", "almost")
    with pytest.raises(boot_mode.BootModeError, match="acknowledgement"):
        boot_mode.require_normal_boot()

    monkeypatch.setenv("CODEMINE_NORMAL_BOOT_ACK", boot_mode.NORMAL_BOOT_ACK)
    with pytest.raises(boot_mode.BootModeError, match="not implemented"):
        boot_mode.require_normal_boot()


def test_normal_mode_refuses_without_transformer_tonic(monkeypatch):
    monkeypatch.setenv("CODEMINE_BOOT_MODE", "normal")
    monkeypatch.setenv("CODEMINE_NORMAL_BOOT_ACK", boot_mode.NORMAL_BOOT_ACK)
    monkeypatch.setenv("CODEMINE_TONIC_MODE", "disabled")
    with pytest.raises(boot_mode.BootModeError, match="transformer-backed Tonic"):
        boot_mode.require_normal_boot()


def test_cli_contract(monkeypatch, capsys):
    monkeypatch.setenv("CODEMINE_BOOT_MODE", "normal")
    monkeypatch.setenv("CODEMINE_NORMAL_BOOT_ACK", boot_mode.NORMAL_BOOT_ACK)
    monkeypatch.setenv("CODEMINE_TONIC_MODE", "transformer")
    assert boot_mode.main(["--check-normal"]) == 1
    assert "not implemented" in capsys.readouterr().err
    assert boot_mode.main([]) == 2


def test_reviewed_runtime_code_change_is_required(monkeypatch):
    monkeypatch.setenv("CODEMINE_BOOT_MODE", "normal")
    monkeypatch.setenv("CODEMINE_NORMAL_BOOT_ACK", boot_mode.NORMAL_BOOT_ACK)
    monkeypatch.setenv("CODEMINE_TONIC_MODE", "transformer")
    monkeypatch.setattr(boot_mode, "NORMAL_RUNTIME_IMPLEMENTED", True)
    boot_mode.require_normal_boot()
