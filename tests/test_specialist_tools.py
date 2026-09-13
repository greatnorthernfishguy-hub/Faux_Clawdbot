"""Tests for role and workspace boundaries on specialist tools."""

# ---- Changelog ----
# [2026-09-12] Codex — Verify specialist tool and worktree isolation.
# What: Cover read-only roles, approved Forge writes, path escape refusal, git
#       evidence, exact named validations, and subprocess secret removal.
# Why: Distinct personas need real evidence without unrestricted host access.
# How: Execute against disposable git workspaces with no NeuroGraph instance.
# -------------------

import subprocess

from specialist_tools import SpecialistToolRuntime, tool_definitions_for


def runtime(tmp_path, role="forge", approved=False, validations=None, allow_writes=True):
    return SpecialistToolRuntime(
        role=role,
        workspace=tmp_path,
        worker_ng=None,
        allow_write_tools=allow_writes,
        write_paths=["new.py", "src"],
        mutations_approved=approved,
        validation_commands=validations,
    )


def test_planning_and_review_roles_have_no_write_tools():
    for role in ("strategist", "tracker", "wrench", "razor", "reviewer"):
        names = {tool["name"] for tool in tool_definitions_for(role)}
        assert "write_file" not in names
        assert "edit_file" not in names
        assert "git_diff" in names

    assert "write_file" in {
        tool["name"] for tool in tool_definitions_for("tracker", allow_write_tools=True)
    }


def test_unapproved_forge_write_is_staged(tmp_path):
    result = runtime(tmp_path).execute("write_file", {"path": "new.py", "content": "x = 1\n"})
    assert result["status"] == "staged"
    assert not (tmp_path / "new.py").exists()


def test_approved_forge_write_stays_inside_workspace(tmp_path):
    worker = runtime(tmp_path, approved=True)
    result = worker.execute("write_file", {"path": "new.py", "content": "x = 1\n"})
    escaped = worker.execute("write_file", {"path": "../escape.py", "content": "bad\n"})

    assert result["status"] == "executed"
    assert (tmp_path / "new.py").read_text() == "x = 1\n"
    assert escaped["status"] == "error"
    assert not (tmp_path.parent / "escape.py").exists()


def test_declared_external_root_is_read_only(tmp_path):
    workspace = tmp_path / "workspace"
    reference = tmp_path / "reference"
    workspace.mkdir()
    reference.mkdir()
    source = reference / "law.md"
    source.write_text("canonical\n")
    worker = SpecialistToolRuntime(
        role="forge",
        workspace=workspace,
        worker_ng=None,
        read_only_paths=[reference],
        write_paths=["new.py"],
        allow_write_tools=True,
        mutations_approved=True,
    )

    read = worker.execute("read_file", {"path": str(source)})
    write = worker.execute("write_file", {"path": str(source), "content": "changed\n"})

    assert read["status"] == "executed"
    assert "canonical" in read["result"]
    assert write["status"] == "error"
    assert source.read_text() == "canonical\n"


def test_write_tools_cannot_escape_declared_block_scope(tmp_path):
    worker = runtime(tmp_path, approved=True)

    inside = worker.execute("write_file", {"path": "src/owned.py", "content": "ok\n"})
    outside = worker.execute("write_file", {"path": "other.py", "content": "bad\n"})

    assert inside["status"] == "executed"
    assert outside["status"] == "error"
    assert not (tmp_path / "other.py").exists()


def test_tool_execution_becomes_error_when_raw_deposit_fails(tmp_path):
    class BrokenGraph:
        def recall(self, *_args, **_kwargs):
            return []

        def on_message(self, _message):
            raise OSError("graph unavailable")

    worker = SpecialistToolRuntime(
        role="forge", workspace=tmp_path, worker_ng=BrokenGraph(),
        write_paths=["new.py"], allow_write_tools=True, mutations_approved=True,
    )

    result = worker.execute("write_file", {"path": "new.py", "content": "made\n"})

    assert result["status"] == "error"
    assert "Substrate deposit failed" in result["result"]
    assert worker.substrate_failure


def test_named_validation_runs_exact_argv_with_secrets_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-reach-child")
    worker = runtime(
        tmp_path,
        validations={
            "env_check": (
                "python3", "-c",
                "import os; print(os.getenv('OPENROUTER_API_KEY', 'clean'))",
            )
        },
    )
    result = worker.execute("run_validation", {"validation_id": "env_check"})
    unknown = worker.execute("run_validation", {"validation_id": "invented"})

    assert result["status"] == "executed"
    assert "EXIT_CODE: 0" in result["result"]
    assert "clean" in result["result"]
    assert "must-not-reach-child" not in result["result"]
    assert unknown["status"] == "error"


def test_git_evidence_reads_actual_worktree(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path)
    (tmp_path / "tracked.txt").write_text("before\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "tracked.txt").write_text("after\n")

    worker = runtime(tmp_path, role="reviewer")
    result = worker.execute("git_diff", {})

    assert result["status"] == "executed"
    assert "-before" in result["result"]
    assert "+after" in result["result"]
