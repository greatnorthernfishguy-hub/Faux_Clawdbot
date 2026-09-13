"""Shared test gates for isolated NeuroGraph construction."""

# ---- Changelog ----
# [2026-09-13] Codex — Authorize only explicit isolated Graph test modules.
# What: Provide a fixture for the production constructor's fail-closed boot gate.
# Why: Real temporary Graph tests must remain possible without weakening runtime.
# How: Set exact test-process controls and patch the code-owned readiness flag.
# -------------------

import pytest

import boot_mode


@pytest.fixture
def authorized_graph_construction(monkeypatch):
    """Authorize temporary in-test Graphs without weakening production boot."""
    monkeypatch.setenv("CODEMINE_BOOT_MODE", "normal")
    monkeypatch.setenv("CODEMINE_NORMAL_BOOT_ACK", boot_mode.NORMAL_BOOT_ACK)
    monkeypatch.setenv("CODEMINE_TONIC_MODE", "transformer")
    monkeypatch.setattr(boot_mode, "NORMAL_RUNTIME_IMPLEMENTED", True)
