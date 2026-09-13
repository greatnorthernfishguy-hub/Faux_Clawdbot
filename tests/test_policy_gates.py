"""Regression tests for interactive mutation gates."""

# ---- Changelog ----
# [2026-09-13] Codex — Verify vendored writes fail at the tool boundary.
# What: Exercise current, nested, and retired substrate filenames.
# Why: The runtime boundary must hold independently of MissionPlan validation.
# How: Call write-mode path authorization directly in a temporary workspace.
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


def test_vendored_and_retired_files_are_denied_at_tool_boundary(tmp_path):
    for path in ("ng_lite.py", "nested/ng_updater.py", "ng_peer_bridge.py"):
        allowed, reason = policy_engine.can_access_path(path, "write", tmp_path)
        assert allowed is False
        assert "vendored or retired" in reason
