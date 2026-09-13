"""Structural tests for Bethany's MissionPlan Lego decomposition."""

# ---- Changelog ----
# [2026-09-13] Codex — Pin root-scope and vendored-write refusal.
# What: Cover repository-wide write scopes and protected substrate basenames.
# Why: Autonomous plans must fail before worktree creation when scope is unsafe.
# How: Validate representative current, nested, and retired targets.
# [2026-09-12] Codex — Verify safe block dependency and write-set scheduling.
# What: Cover parallel levels, cycles, unknown dependencies, unsafe paths, and
#       write overlap that lacks an explicit serial dependency.
# Why: No two autonomous workers may write the same files concurrently.
# How: Validate compact in-memory plans without starting models or NeuroGraph.
# -------------------

from copy import deepcopy

from mission_plan import dependency_levels, validate_mission_plan


def plan():
    return {
        "plan_version": "1.0.0",
        "mission": {
            "id": "mission-1",
            "name": "Build feature",
            "intent": "Implement and review the feature.",
            "acceptance_criteria": ["Tests pass"],
        },
        "blocks": [
            {
                "id": "tracker",
                "name": "Trace current behavior",
                "role": "tracker",
                "task": "Find the root cause with file evidence.",
                "read_paths": ["src"],
                "write_paths": [],
                "depends_on": [],
                "acceptance_criteria": ["Evidence chain reported"],
                "validations": [],
            },
            {
                "id": "forge",
                "name": "Implement repair",
                "role": "forge",
                "task": "Implement the bounded repair.",
                "read_paths": ["src"],
                "write_paths": ["src/feature.py"],
                "depends_on": ["tracker"],
                "acceptance_criteria": ["Repair is present"],
                "validations": ["tests"],
            },
            {
                "id": "docs",
                "name": "Update usage",
                "role": "wrench",
                "task": "Implement the bounded documentation update.",
                "read_paths": ["README.md"],
                "write_paths": ["README.md"],
                "depends_on": ["tracker"],
                "acceptance_criteria": ["Usage is current"],
                "validations": ["diff-check"],
            },
            {
                "id": "review",
                "name": "Review assembled result",
                "role": "reviewer",
                "task": "Read the actual integrated diff.",
                "read_paths": ["src/feature.py", "README.md"],
                "write_paths": [],
                "depends_on": ["forge", "docs"],
                "acceptance_criteria": ["Explicit verdict reported"],
                "validations": [],
            },
        ],
    }


def test_valid_plan_produces_parallel_levels():
    candidate = plan()
    assert validate_mission_plan(candidate) == (True, [])
    assert dependency_levels(candidate) == [
        ["tracker"], ["docs", "forge"], ["review"],
    ]


def test_unknown_dependency_and_cycle_are_rejected():
    unknown = plan()
    unknown["blocks"][0]["depends_on"] = ["missing"]
    assert validate_mission_plan(unknown)[0] is False

    cycle = plan()
    cycle["blocks"][0]["depends_on"] = ["review"]
    valid, errors = validate_mission_plan(cycle)
    assert valid is False
    assert "cycle" in errors[0]


def test_unsafe_paths_are_rejected():
    candidate = plan()
    candidate["blocks"][1]["write_paths"] = ["../outside.py"]
    valid, errors = validate_mission_plan(candidate)
    assert valid is False
    assert "unsafe relative path" in errors[0]

    candidate = plan()
    candidate["blocks"][1]["write_paths"] = [".git/config"]
    assert validate_mission_plan(candidate)[0] is False

    candidate = plan()
    candidate["blocks"][1]["write_paths"] = ["."]
    valid, errors = validate_mission_plan(candidate)
    assert valid is False
    assert "repository-root write scope is not bounded" in errors[0]


def test_vendored_and_retired_files_are_rejected_as_write_targets():
    for path in ("ng_lite.py", "module/ng_salience_gate.py", "ng_peer_bridge.py"):
        candidate = plan()
        candidate["blocks"][1]["write_paths"] = [path]
        valid, errors = validate_mission_plan(candidate)
        assert valid is False
        assert "vendored or retired file is prohibited" in errors[0]


def test_parallel_write_overlap_requires_dependency():
    candidate = plan()
    duplicate = deepcopy(candidate["blocks"][1])
    duplicate["id"] = "forge-two"
    duplicate["depends_on"] = ["tracker"]
    duplicate["write_paths"] = ["src"]
    candidate["blocks"].append(duplicate)

    valid, errors = validate_mission_plan(candidate)
    assert valid is False
    assert "overlapping writes" in errors[0]

    duplicate["depends_on"] = ["forge"]
    assert validate_mission_plan(candidate) == (True, [])


def test_model_can_only_select_coordinator_owned_validations():
    candidate = plan()
    candidate["blocks"][1]["validations"] = ["tests", "invented-command"]

    valid, errors = validate_mission_plan(candidate, {"tests", "diff-check"})

    assert valid is False
    assert "unknown validation IDs: invented-command" in errors[0]


def test_writable_block_requires_at_least_one_validation():
    candidate = plan()
    candidate["blocks"][1]["validations"] = []

    valid, errors = validate_mission_plan(candidate)

    assert valid is False
    assert "writable block requires validation IDs" in errors[0]


def test_planning_investigation_and_review_roles_cannot_write():
    for role in ("strategist", "tracker", "razor", "reviewer"):
        candidate = plan()
        candidate["blocks"][1]["role"] = role
        valid, errors = validate_mission_plan(candidate)
        assert valid is False
        assert f"role {role} is review/planning-only" in errors[0]
