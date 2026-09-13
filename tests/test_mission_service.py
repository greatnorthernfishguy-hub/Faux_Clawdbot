"""Tests for the durable Mission Control service boundary."""

# ---- Changelog ----
# [2026-09-13] Codex — Verify Graph save failures are durable blockers.
# What: Cover planning and post-execution checkpoint failures.
# Why: Mission Control must never hide loss of durable substrate continuity.
# How: Use a graph whose on_message succeeds and whose save raises OSError.
# [2026-09-12] Codex — Verify durable plan-then-execute coordination.
# What: Cover environment configuration, raw deposits, workspace roots, and locks.
# Why: One click must not bypass a concrete plan or execute a proposal twice.
# How: Compose the service with fake generators and dispatchers in temporary roots.
# -------------------

import json
from pathlib import Path

from mission_service import MissionService, load_validation_catalog


def catalog_json():
    return json.dumps({
        "catalog_version": "1.0.0",
        "validations": {
            "check": {"description": "test", "argv": ["git", "diff", "--check"]},
        },
    })


def test_loads_only_structurally_valid_catalog():
    document, commands = load_validation_catalog(catalog_json())

    assert document["catalog_version"] == "1.0.0"
    assert commands == {"check": ("git", "diff", "--check")}


def test_plan_is_persisted_before_execution(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured = {}

    def generator(**kwargs):
        captured.update(kwargs)
        return {"success": True, "plan": {"plan_version": "1.0.0"}, "attempts": []}

    service = MissionService(
        worker_ng=None,
        validation_catalog_json=catalog_json(),
        mission_root=tmp_path / "missions",
        worktree_root=tmp_path / "worktrees",
        workspace_roots=[tmp_path],
        generator_fn=generator,
    )
    result = service.plan(intent="Inspect it", constraints="Stay scoped", workspace=str(workspace))

    assert result["status"] == "planned"
    assert captured["workspace"] == workspace
    assert captured["validation_catalog"] == {"check": ("git", "diff", "--check")}
    persisted = json.loads((tmp_path / "missions" / "proposals" / (result["proposal_path"].split("/")[-1])).read_text())
    assert persisted["generation"]["success"] is True
    assert persisted["proposal_path"] == result["proposal_path"]


def test_plan_deposits_raw_intent_and_constraints_before_generation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events = []

    class Graph:
        def on_message(self, text):
            events.append(("deposit", text))

        def save(self):
            pass

    def generator(**_kwargs):
        events.append(("generate", None))
        return {"success": False, "plan": None, "attempts": [], "errors": ["stop"]}

    service = MissionService(
        worker_ng=Graph(), validation_catalog_json=catalog_json(),
        mission_root=tmp_path / "missions", worktree_root=tmp_path / "worktrees",
        workspace_roots=[tmp_path], generator_fn=generator,
    )

    service.plan(intent="exact intent", constraints="exact constraint", workspace=str(workspace))

    assert events[:3] == [
        ("deposit", "exact intent"),
        ("deposit", "exact constraint"),
        ("generate", None),
    ]


def test_planning_stops_when_raw_intent_cannot_reach_graph(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class BrokenGraph:
        def on_message(self, _text):
            raise OSError("offline")

    service = MissionService(
        worker_ng=BrokenGraph(), validation_catalog_json=catalog_json(),
        mission_root=tmp_path / "missions", worktree_root=tmp_path / "worktrees",
        workspace_roots=[tmp_path],
        generator_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("generator must not run")
        ),
    )

    try:
        service.plan(intent="exact intent", constraints=None, workspace=str(workspace))
    except RuntimeError as exc:
        assert "Raw mission input deposit failed" in str(exc)
    else:
        raise AssertionError("planning continued without raw substrate deposit")


def test_plan_records_graph_save_failure_as_blocked(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class UnsavableGraph:
        def on_message(self, _text):
            pass

        def save(self):
            raise OSError("disk unavailable")

    service = MissionService(
        worker_ng=UnsavableGraph(), validation_catalog_json=catalog_json(),
        mission_root=tmp_path / "missions", worktree_root=tmp_path / "worktrees",
        workspace_roots=[tmp_path],
        generator_fn=lambda **_kwargs: {
            "success": True, "plan": {"plan_version": "1.0.0"}, "attempts": [],
        },
    )

    result = service.plan(intent="Preserve this", constraints=None, workspace=str(workspace))

    assert result["status"] == "blocked"
    assert "Shared NeuroGraph save failed" in result["errors"][0]
    assert json.loads(Path(result["proposal_path"]).read_text())["status"] == "blocked"


def test_workspace_outside_environment_roots_is_rejected(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    service = MissionService(
        worker_ng=None,
        validation_catalog_json=catalog_json(),
        mission_root=tmp_path / "missions",
        worktree_root=tmp_path / "worktrees",
        workspace_roots=[allowed],
    )

    result = service.plan(intent="Inspect it", constraints=None, workspace=str(outside))

    assert result["status"] == "rejected"
    assert "outside CODEMINE_WORKSPACE_ROOTS" in result["errors"][0]


def test_execute_passes_only_reviewed_proposal_to_dispatcher(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured = {}

    class FakeDispatcher:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def dispatch(self, plan, **kwargs):
            captured["plan"] = plan
            captured["dispatch"] = kwargs
            return {"status": "ready_for_external_review"}

    service = MissionService(
        worker_ng=None,
        validation_catalog_json=catalog_json(),
        mission_root=tmp_path / "missions",
        worktree_root=tmp_path / "worktrees",
        workspace_roots=[tmp_path],
        dispatcher_cls=FakeDispatcher,
    )
    proposal = {
        "status": "planned", "workspace": str(workspace),
        "generation": {"plan": {"plan_version": "1.0.0"}},
    }
    proposal_path = tmp_path / "missions" / "proposals" / "proposal.json"
    proposal_path.parent.mkdir(parents=True)
    proposal["proposal_path"] = str(proposal_path)
    proposal_path.write_text(json.dumps(proposal))

    result = service.execute(proposal, command_approved=True)

    assert result["status"] == "ready_for_external_review"
    assert captured["dispatch"] == {
        "mutations_approved": True, "command_approved": True,
    }
    assert captured["init"]["worktree_root"] == tmp_path / "worktrees"
    assert json.loads(proposal_path.read_text())["status"] == "executed"


def test_same_proposal_cannot_execute_twice(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = 0

    class FakeDispatcher:
        def __init__(self, **_kwargs):
            pass

        def dispatch(self, _plan, **_kwargs):
            nonlocal calls
            calls += 1
            return {"status": "ready_for_external_review"}

    service = MissionService(
        worker_ng=None,
        validation_catalog_json=catalog_json(),
        mission_root=tmp_path / "missions",
        worktree_root=tmp_path / "worktrees",
        workspace_roots=[tmp_path],
        generator_fn=lambda **_kwargs: {
            "success": True, "plan": {"plan_version": "1.0.0"}, "attempts": [],
        },
        dispatcher_cls=FakeDispatcher,
    )
    proposal = service.plan(intent="Do it", constraints=None, workspace=str(workspace))

    first = service.execute(proposal)
    second = service.execute(proposal)

    assert first["status"] == "ready_for_external_review"
    assert second["status"] == "rejected"
    assert calls == 1


def test_execute_preserves_result_but_blocks_when_graph_save_fails(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class UnsavableGraph:
        def save(self):
            raise OSError("disk unavailable")

    class FakeDispatcher:
        def __init__(self, **_kwargs):
            pass

        def dispatch(self, _plan, **_kwargs):
            return {"status": "ready_for_external_review", "artifact": "kept"}

    service = MissionService(
        worker_ng=UnsavableGraph(), validation_catalog_json=catalog_json(),
        mission_root=tmp_path / "missions", worktree_root=tmp_path / "worktrees",
        workspace_roots=[tmp_path], dispatcher_cls=FakeDispatcher,
    )
    proposal_path = tmp_path / "missions" / "proposals" / "proposal.json"
    proposal_path.parent.mkdir(parents=True)
    proposal = {
        "status": "planned", "workspace": str(workspace),
        "generation": {"plan": {"plan_version": "1.0.0"}},
        "proposal_path": str(proposal_path),
    }
    proposal_path.write_text(json.dumps(proposal))

    result = service.execute(proposal)

    assert result["status"] == "blocked"
    assert result["artifact"] == "kept"
    assert "Shared NeuroGraph save failed" in result["substrate_save_error"]
    persisted = json.loads(proposal_path.read_text())
    assert persisted["execution"]["status"] == "blocked"
    assert persisted["execution"]["artifact"] == "kept"
