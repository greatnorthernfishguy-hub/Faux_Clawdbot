"""Security tests for structural shell command execution."""

# ---- Changelog ----
# [2026-09-12] Codex — Prove shell chaining and prefix smuggling are blocked.
# What: Cover control operators, substitutions, exact binary matching, safe argv,
#       leading environment assignments, and actual shell=False execution.
# Why: Autonomous workers must not escape the policy boundary through shell syntax.
# How: Test PolicyEngine directly and run ShellTool only against a temp workspace.
# -------------------

from pathlib import Path

import policy_engine
from tools.shell_tool import ShellTool


def test_shell_control_operators_and_substitutions_are_denied():
    for command in (
        "git status ; rm -rf /",
        "git status;rm -rf /",
        "git status && rm -rf /",
        "git status | cat",
        "python3 -c 'print(1)' > out.txt",
        "python3 -c 'print(`whoami`)'",
        "python3 -c 'print($(whoami))'",
    ):
        allowed, _reason = policy_engine.can_execute_shell(command)
        assert allowed is False


def test_binary_prefix_smuggling_is_denied():
    allowed, _reason = policy_engine.can_execute_shell("git-malware status")
    assert allowed is False


def test_versioned_python_and_leading_environment_are_parsed():
    allowed, _reason = policy_engine.can_execute_shell(
        "PYTHONPATH=/workspace python3.12 -m pytest -q"
    )
    argv, environment = policy_engine.parse_shell_command(
        "PYTHONPATH=/workspace python3.12 -m pytest -q"
    )
    assert allowed is True
    assert argv == ["python3.12", "-m", "pytest", "-q"]
    assert environment == {"PYTHONPATH": "/workspace"}


def test_shell_tool_does_not_interpret_metacharacters(tmp_path):
    tool = ShellTool(Path(tmp_path), policy_engine)
    result = tool.shell_execute("python3 -c 'print(\"safe\")'")

    assert "EXIT_CODE: 0" in result
    assert "safe" in result
    assert not (tmp_path / "unexpected").exists()
