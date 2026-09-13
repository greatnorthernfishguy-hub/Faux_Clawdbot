"""Durable plan-then-execute service for Codemine Mission Control."""

# ---- Changelog ----
# [2026-09-12] Codex — Preserve raw Mission Control input in the substrate.
# What: Deposit the exact intent and optional constraints before calling the
#       planner or constructing a proposal envelope.
# Why: Mission Control bypasses Chat, so these turns otherwise never reached the
#      shared NeuroGraph; a validated plan is not a substitute for raw input.
# How: Call the existing on_message pathway before any plan classification.
# [2026-09-12] Codex — Put a reviewable gate between mission planning and work.
# What: Resolve bounded workspaces, load coordinator-owned validations, persist
#       Bethany's proposal, and dispatch only after an explicit execute action.
# Why: The old Mission button planned, mutated, merged, and could push in one
#      opaque request. Josh should approve a concrete plan and inspect the final
#      integration worktree before delivery.
# How: A thin service composes mission_generator and MissionDispatcher while
#      keeping workspace roots and validation argv environment-owned.
# -------------------

from __future__ import annotations

import json
import os
import time
import uuid
import fcntl
from pathlib import Path
from typing import Optional

from mission_dispatcher import MissionDispatcher
from mission_generator import generate_mission_plan


def load_validation_catalog(raw_json: Optional[str] = None) -> tuple[dict, dict]:
    raw = raw_json if raw_json is not None else os.getenv("CODEMINE_VALIDATIONS_JSON", "")
    if not raw.strip():
        raise ValueError("CODEMINE_VALIDATIONS_JSON is unset")
    document = json.loads(raw)
    if document.get("catalog_version") != "1.0.0":
        raise ValueError("Unsupported validation catalog version")
    entries = document.get("validations")
    if not isinstance(entries, dict):
        raise ValueError("Validation catalog must contain a validations object")
    commands = {}
    for validation_id, entry in entries.items():
        argv = entry.get("argv") if isinstance(entry, dict) else None
        if (
            not validation_id
            or not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
        ):
            raise ValueError(f"Invalid validation catalog entry: {validation_id!r}")
        commands[validation_id] = tuple(argv)
    return document, commands


def configured_workspace_roots() -> list[Path]:
    raw = os.getenv("CODEMINE_WORKSPACE_ROOTS", "/workspace")
    return [Path(item).expanduser().resolve() for item in raw.split(os.pathsep) if item]


class MissionService:
    def __init__(
        self,
        *,
        worker_ng,
        validation_catalog_json: Optional[str] = None,
        mission_root: Optional[Path] = None,
        worktree_root: Optional[Path] = None,
        workspace_roots: Optional[list[Path]] = None,
        generator_fn=generate_mission_plan,
        dispatcher_cls=MissionDispatcher,
    ):
        self.ng = worker_ng
        self.catalog_document, self.validation_commands = load_validation_catalog(
            validation_catalog_json
        )
        configured_mission_root = os.getenv(
            "CODEMINE_MISSION_ROOT", "/data/codemine-missions"
        )
        self.mission_root = Path(mission_root or configured_mission_root).resolve()
        configured_worktrees = os.getenv(
            "CODEMINE_WORKTREE_ROOT", "/tmp/codemine-worktrees"
        )
        self.worktree_root = Path(worktree_root or configured_worktrees).resolve()
        self.workspace_roots = [
            Path(root).resolve() for root in (workspace_roots or configured_workspace_roots())
        ]
        self.generator_fn = generator_fn
        self.dispatcher_cls = dispatcher_cls

    def resolve_workspace(self, workspace: Optional[str]) -> Path:
        default = os.getenv("CODEMINE_DEFAULT_WORKSPACE", "/workspace/e-t-systems")
        target = Path(workspace or default).expanduser().resolve()
        if not target.is_dir():
            raise ValueError(f"Workspace does not exist: {target}")
        if not any(target == root or target.is_relative_to(root) for root in self.workspace_roots):
            raise ValueError(f"Workspace is outside CODEMINE_WORKSPACE_ROOTS: {target}")
        return target

    def plan(self, *, intent: str, constraints: Optional[str], workspace: Optional[str]) -> dict:
        if not intent or not intent.strip():
            return {"status": "rejected", "errors": ["Mission intent is required"]}
        try:
            target = self.resolve_workspace(workspace)
        except ValueError as exc:
            return {"status": "rejected", "errors": [str(exc)]}
        self._deposit_raw(intent)
        if constraints and constraints.strip():
            self._deposit_raw(constraints)
        generated = self.generator_fn(
            intent=intent.strip(),
            constraints=constraints.strip() if constraints and constraints.strip() else None,
            workspace=target,
            worker_ng=self.ng,
            validation_catalog=self.validation_commands,
        )
        envelope = {
            "proposal_version": "1.0.0",
            "status": "planned" if generated.get("success") else "rejected",
            "created_at": time.time(),
            "workspace": str(target),
            "validation_catalog_version": self.catalog_document["catalog_version"],
            "generation": generated,
        }
        proposal_dir = self.mission_root / "proposals"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        proposal_id = (
            f"proposal-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        proposal_path = proposal_dir / f"{proposal_id}.json"
        envelope["proposal_path"] = str(proposal_path)
        self._atomic_json(proposal_path, envelope)
        self._save_graph()
        return envelope

    def execute(self, proposal: dict, *, command_approved: bool = False) -> dict:
        proposal_path_value = proposal.get("proposal_path")
        if not proposal_path_value:
            return {"status": "rejected", "errors": ["Proposal has no durable path"]}
        proposal_path = Path(proposal_path_value).resolve()
        proposal_root = (self.mission_root / "proposals").resolve()
        if not proposal_path.is_relative_to(proposal_root):
            return {"status": "rejected", "errors": ["Proposal path escapes mission storage"]}
        lock_path = proposal_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                durable = json.loads(proposal_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                return {"status": "rejected", "errors": [f"Cannot read proposal: {exc}"]}
            if durable.get("status") != "planned":
                return {"status": "rejected", "errors": ["Proposal is not ready to execute"]}
            plan = durable.get("generation", {}).get("plan")
            if not isinstance(plan, dict):
                return {"status": "rejected", "errors": ["Proposal has no MissionPlan"]}
            try:
                target = self.resolve_workspace(durable.get("workspace"))
            except ValueError as exc:
                return {"status": "rejected", "errors": [str(exc)]}
            durable["status"] = "executing"
            durable["execution_started_at"] = time.time()
            self._atomic_json(proposal_path, durable)

        try:
            dispatcher = self.dispatcher_cls(
                repo=target,
                worker_ng=self.ng,
                validation_catalog=self.validation_commands,
                mission_root=self.mission_root,
                worktree_root=self.worktree_root,
            )
            result = dispatcher.dispatch(
                plan, mutations_approved=True, command_approved=command_approved,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        durable["status"] = (
            "planned" if result.get("status") in {
                "awaiting_mutation_approval", "awaiting_human_approval",
            } else "executed"
        )
        durable["execution_finished_at"] = time.time()
        durable["execution"] = result
        self._atomic_json(proposal_path, durable)
        self._save_graph()
        return result

    def _save_graph(self) -> None:
        if self.ng is None:
            return
        try:
            self.ng.save()
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _deposit_raw(self, text: str) -> None:
        if self.ng is None or not text:
            return
        try:
            self.ng.on_message(text)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Raw mission input deposit failed: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _atomic_json(path: Path, value: dict) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
        os.replace(temporary, path)
