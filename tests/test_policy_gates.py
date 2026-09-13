"""Regression tests for interactive mutation gates."""

# ---- Changelog ----
# [2026-09-12] Codex — Prohibit hook-based gate bypass in the HF runtime.
# What: Assert QB_DISPATCHED cannot disable interactive mutation staging.
# Why: Claude Code hooks do not exist inside the Hugging Face Space.
# How: Reload the policy module with the obsolete environment flag enabled.
# -------------------

import importlib

import policy_engine


def test_qb_dispatched_environment_cannot_bypass_chat_gates(monkeypatch):
    monkeypatch.setenv("QB_DISPATCHED", "true")
    module = importlib.reload(policy_engine)

    assert module.should_gate_for_review("write_file", {}) is True
    assert module.should_gate_for_review("edit_file", {}) is True
    assert module.should_gate_for_review("read_file", {}) is False
