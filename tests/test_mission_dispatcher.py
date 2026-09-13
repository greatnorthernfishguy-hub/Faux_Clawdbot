"""End-to-end tests for isolated TQB MissionPlan dispatch."""

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
    )

    result = dispatcher.dispatch(plan(), mutations_approved=True)

    assert result["status"] == "ready_for_external_review", result
    assert (repo / "src" / "feature.py").read_text() == "VALUE = 1\n"
    assert (repo / "notes.txt").read_text() == "Josh's uncommitted work\n"
    assert result["source"]["modified"] is False
    integration = Path(result["integration"]["path"])
    assert (integration / "src" / "feature.py").read_text() == "VALUE = 2\n"
    assert result["blocks"]["forge-change"]["integrated"] is True
    assert (Path(result["integration"]["path"]).parent / "RECEIPT.json").is_file()
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
    ).dispatch(plan(), mutations_approved=True)

    assert result["status"] == "ready_for_external_review"
    assert worker_calls == 2
    assert review_calls == 2
