"""End-to-end tests for isolated TQB MissionPlan dispatch."""

# ---- Changelog ----
# [2026-09-13] Codex — Exercise assembled validation and review gates.
# What: Cover integration-only failures and rejection of the combined diff.
# Why: Per-block success does not prove that the assembled mission is acceptable.
# How: Use disposable worktrees and deterministic fake reviewers.
# [2026-09-12] Codex — Verify isolated, reviewed, recoverable mission execution.
# What: Cover source preservation, path scope, gates, retries, and patch artifacts.
# Why: Autonomous workers must not blur Lego blocks or strand unfinished work.
# How: Run the coordinator against disposable Git repositories and fake models.
# -------------------

import subprocess
from pathlib import Path

from mission_dispatcher import MissionDispatcher


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    (repo / "src").mkdir()
    (repo / "src" / "feature.py").write_text("VALUE = 1\n")
    (repo / "notes.txt").write_text("tracked\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    return repo


def plan(write_path="src/feature.py", task="Update the feature"):
    return {
        "plan_version": "1.0.0",
        "mission": {
            "id": "test-mission", "name": "Test mission", "intent": task,
            "acceptance_criteria": ["Validation passes"],
        },
        "blocks": [{
            "id": "forge-change", "name": "Change feature", "role": "forge",
            "task": task, "read_paths": ["src/feature.py"],
            "write_paths": [write_path], "depends_on": [],
            "acceptance_criteria": ["VALUE is 2"], "validations": ["check"],
        }],
    }


class RecordingGraph:
    def __init__(self):
        self.messages = []

    def recall(self, *_args, **_kwargs):
        return [
            {"content": "prior", "similarity": 0.9},
            {"content": "prior", "similarity": 0.9},
            {"content": "prior", "similarity": 0.9},
        ]

    def on_message(self, message):
        self.messages.append(message)


def approving_persona(**kwargs):
    role = kwargs["role"]
    runtime = kwargs["tool_runtime"]
    if role == "forge":
        written = runtime.execute(
            "write_file", {"path": "src/feature.py", "content": "VALUE = 2\n"},
        )
        assert written["status"] == "executed"
        structured = {
            "status": "complete", "summary": "Changed VALUE.",
            "evidence": ["src/feature.py now says VALUE = 2"], "blockers": [],
        }
    else:
        diff = runtime.execute("git_diff", {})
        assert diff["status"] == "executed"
        assert "+VALUE = 2" in diff["result"]
        structured = {"decision": "approve", "findings": []}
    return {
        "structured": structured,
        "response": str(structured),
        "provider": "test-provider", "model": f"test-{role}",
        "requested_provider": "test-provider", "requested_model": f"test-{role}",
        "input_tokens": 10, "output_tokens": 5, "cost": 0.001,
        "tool_events": [],
    }


def test_dispatch_preserves_dirty_source_and_stops_at_external_review(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "notes.txt").write_text("Josh's uncommitted work\n")
    graph = RecordingGraph()
    dispatcher = MissionDispatcher(
        repo=repo, worker_ng=graph,
        validation_catalog={
            "check": ("python3", "-c", "from pathlib import Path; assert Path('src/feature.py').read_text() == 'VALUE = 2\\n'"),
        },
        mission_root=tmp_path / "missions", persona_call=approving_persona,
        worktree_root=tmp_path / "worktrees",
    )

    result = dispatcher.dispatch(plan(), mutations_approved=True)

    assert result["status"] == "ready_for_external_review", result
    assert (repo / "src" / "feature.py").read_text() == "VALUE = 1\n"
    assert (repo / "notes.txt").read_text() == "Josh's uncommitted work\n"
    assert result["source"]["modified"] is False
    integration = Path(result["integration"]["path"])
    assert (integration / "src" / "feature.py").read_text() == "VALUE = 2\n"
    assert result["blocks"]["forge-change"]["integrated"] is True
    assert result["integration_validations"][0]["exit_code"] == 0
    assert result["integration_reviews"][0]["decision"] == "approve"
    receipt_path = next((tmp_path / "missions").glob("*/RECEIPT.json"))
    assert receipt_path.is_file()
    patch = Path(result["artifacts"]["integration_patch"]["path"])
    assert patch.is_file()
    assert "+VALUE = 2" in patch.read_text()
    assert graph.messages


def test_write_attempt_outside_block_scope_is_blocked(tmp_path):
    repo = make_repo(tmp_path)

    def escaping_persona(**kwargs):
        if kwargs["role"] == "forge":
            denied = kwargs["tool_runtime"].execute(
                "write_file", {"path": "notes.txt", "content": "escaped\n"},
            )
            assert denied["status"] == "error"
            structured = {
                "status": "blocked", "summary": "Scope denied.",
                "evidence": [], "blockers": ["notes.txt is outside write scope"],
            }
        else:
            raise AssertionError("Review must not run for blocked work")
        return {"structured": structured, "response": str(structured), "tool_events": []}

    result = MissionDispatcher(
        repo=repo, worker_ng=None,
        validation_catalog={"check": ("python3", "-c", "pass")},
        mission_root=tmp_path / "missions", persona_call=escaping_persona,
        worktree_root=tmp_path / "worktrees",
    ).dispatch(plan(), mutations_approved=True)

    assert result["status"] == "blocked"
    assert result["blocks"]["forge-change"]["status"] == "blocked"
    assert (repo / "notes.txt").read_text() == "tracked\n"


def test_unknown_validation_rejected_before_worktree_creation(tmp_path):
    repo = make_repo(tmp_path)
    candidate = plan()
    candidate["blocks"][0]["validations"] = ["invented"]
    missions = tmp_path / "missions"

    result = MissionDispatcher(
        repo=repo, worker_ng=None,
        validation_catalog={"check": ("python3", "-c", "pass")},
        mission_root=missions, persona_call=approving_persona,
        worktree_root=tmp_path / "worktrees",
    ).dispatch(candidate, mutations_approved=True)

    assert result["status"] == "rejected"
    assert not missions.exists()


def test_command_risk_stops_before_worktree_without_human_gate(tmp_path):
    repo = make_repo(tmp_path)
    candidate = plan(write_path="ARCHITECTURE.md", task="Change the architecture")
    missions = tmp_path / "missions"

    result = MissionDispatcher(
        repo=repo, worker_ng=RecordingGraph(),
        validation_catalog={"check": ("python3", "-c", "pass")},
        mission_root=missions, persona_call=approving_persona,
        worktree_root=tmp_path / "worktrees",
    ).dispatch(candidate, mutations_approved=True)

    assert result["status"] == "awaiting_human_approval"
    assert not missions.exists()


def test_reviewer_iteration_returns_to_worker_with_findings(tmp_path):
    repo = make_repo(tmp_path)
    worker_calls = 0
    review_calls = 0

    def iterating_persona(**kwargs):
        nonlocal worker_calls, review_calls
        runtime = kwargs["tool_runtime"]
        if kwargs["role"] == "forge":
            worker_calls += 1
            assert ("Reviewer-specific correction" in kwargs["task"]) is (worker_calls == 2)
            value = 3 if worker_calls == 1 else 2
            runtime.execute(
                "write_file", {"path": "src/feature.py", "content": f"VALUE = {value}\n"},
            )
            structured = {
                "status": "complete", "summary": "Changed value.",
                "evidence": [f"VALUE is {value}"], "blockers": [],
            }
        else:
            review_calls += 1
            diff = runtime.execute("git_diff", {})
            assert diff["status"] == "executed"
            structured = (
                {"decision": "iterate", "findings": ["Reviewer-specific correction"]}
                if review_calls == 1
                else {"decision": "approve", "findings": []}
            )
        return {"structured": structured, "response": str(structured), "tool_events": []}

    result = MissionDispatcher(
        repo=repo, worker_ng=RecordingGraph(),
        validation_catalog={"check": ("python3", "-c", "pass")},
        mission_root=tmp_path / "missions", persona_call=iterating_persona,
        worktree_root=tmp_path / "worktrees",
    ).dispatch(plan(), mutations_approved=True)

    assert result["status"] == "ready_for_external_review"
    assert worker_calls == 2
    assert review_calls == 3


def test_combined_validation_runs_after_integration(tmp_path):
    repo = make_repo(tmp_path)
    result = MissionDispatcher(
        repo=repo,
        worker_ng=RecordingGraph(),
        validation_catalog={
            "check": (
                "python3", "-c",
                "import os,sys; sys.exit(1 if os.path.basename(os.getcwd()) == 'integration' else 0)",
            ),
        },
        mission_root=tmp_path / "missions",
        persona_call=approving_persona,
        worktree_root=tmp_path / "worktrees",
    ).dispatch(plan(), mutations_approved=True)

    assert result["status"] == "blocked"
    assert result["integration_failure"] == "Combined validation failed"
    assert result["integration_validations"][0]["exit_code"] == 1
    assert "integration_reviews" not in result


def test_combined_review_can_block_after_all_validations_pass(tmp_path):
    repo = make_repo(tmp_path)
    review_calls = 0

    def rejecting_integration_persona(**kwargs):
        nonlocal review_calls
        if kwargs["role"] == "forge":
            return approving_persona(**kwargs)
        review_calls += 1
        if review_calls == 1:
            return approving_persona(**kwargs)
        diff = kwargs["tool_runtime"].execute("git_diff", {})
        assert diff["status"] == "executed"
        assert "+VALUE = 2" in diff["result"]
        structured = {"decision": "iterate", "findings": ["Combined behavior conflicts"]}
        return {"structured": structured, "response": str(structured), "tool_events": []}

    result = MissionDispatcher(
        repo=repo, worker_ng=RecordingGraph(),
        validation_catalog={"check": ("python3", "-c", "pass")},
        mission_root=tmp_path / "missions", persona_call=rejecting_integration_persona,
        worktree_root=tmp_path / "worktrees",
    ).dispatch(plan(), mutations_approved=True)

    assert result["status"] == "blocked"
    assert result["integration_failure"] == "Combined diff review did not approve"
    assert result["integration_reviews"][0]["decision"] == "iterate"


def test_blocked_work_is_preserved_as_durable_wip_patch(tmp_path):
    repo = make_repo(tmp_path)

    def escalating_persona(**kwargs):
        if kwargs["role"] == "forge":
            kwargs["tool_runtime"].execute(
                "write_file", {"path": "src/feature.py", "content": "VALUE = 9\n"},
            )
            structured = {
                "status": "complete", "summary": "Risky change.",
                "evidence": ["VALUE is 9"], "blockers": [],
            }
        else:
            diff = kwargs["tool_runtime"].execute("git_diff", {})
            assert diff["status"] == "executed"
            structured = {"decision": "escalate", "findings": ["Needs external judgment"]}
        return {"structured": structured, "response": str(structured), "tool_events": []}

    result = MissionDispatcher(
        repo=repo, worker_ng=RecordingGraph(),
        validation_catalog={"check": ("python3", "-c", "pass")},
        mission_root=tmp_path / "missions", persona_call=escalating_persona,
        worktree_root=tmp_path / "worktrees",
    ).dispatch(plan(), mutations_approved=True)

    block = result["blocks"]["forge-change"]
    assert result["status"] == "blocked"
    assert block["wip_patch"] is not None
    assert "+VALUE = 9" in Path(block["wip_patch"]["path"]).read_text()


def test_reviewer_cannot_approve_without_reading_diff(tmp_path):
    repo = make_repo(tmp_path)

    def rubber_stamp_persona(**kwargs):
        if kwargs["role"] == "forge":
            return approving_persona(**kwargs)
        return {
            "structured": {"decision": "approve", "findings": []},
            "response": "approved without inspection", "tool_events": [],
        }

    result = MissionDispatcher(
        repo=repo, worker_ng=RecordingGraph(),
        validation_catalog={"check": ("python3", "-c", "pass")},
        mission_root=tmp_path / "missions", persona_call=rubber_stamp_persona,
        worktree_root=tmp_path / "worktrees",
    ).dispatch(plan(), mutations_approved=True)

    assert result["status"] == "blocked"
    review = result["blocks"]["forge-change"]["reviews"][0]
    assert review["decision"] == "escalate"
    assert "without inspecting git_diff" in review["findings"][0]
