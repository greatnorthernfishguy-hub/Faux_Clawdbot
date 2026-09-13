"""Workspace-bound tool surface for the six durable TQB specialists."""

# ---- Changelog ----
# [2026-09-12] Codex — Bind specialist tools to approved block workspaces.
# What: Add role-scoped file, git evidence, and named validation tools with
#       PolicyEngine checks and shared-Graph recall/experience.
# Why: Persona summaries cannot replace inspecting code, diffs, and test output;
#      unrestricted model-authored shell commands are too broad for autonomy.
# How: Workers choose tools and validation IDs while the coordinator owns paths,
#      exact validation argv, mutation authority, and the single shared Graph.
# -------------------

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from policy_engine import check_tool_call
from tools.filesystem_tool import FilesystemTool

logger = logging.getLogger("specialist_tools")

MAX_RESULT_CHARS = 50_000

READ_TOOLS = frozenset({
    "read_file", "list_files", "search_code", "search_conversations",
    "search_testament", "git_status", "git_diff", "run_validation",
})
WRITE_TOOLS = frozenset({"write_file", "edit_file"})

ROLE_TOOLS = {role: READ_TOOLS for role in (
    "strategist", "tracker", "wrench", "forge", "razor", "reviewer",
)}

SPECIALIST_TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read a current file from this block's isolated workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List current files below a workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_depth": {"type": "integer"},
            },
            "required": [],
        },
    },
    {
        "name": "write_file",
        "description": "Write a complete file inside an approved writable block.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace one exact, unique text span in an approved writable block.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "search_code",
        "description": "Recall code-related experience from the shared NeuroGraph; verify it against files.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_conversations",
        "description": "Recall prior conversation experience from the shared NeuroGraph.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_testament",
        "description": "Recall architecture/document experience; verify it against current sources.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "git_status",
        "description": "Show the current branch and worktree status without changing it.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "git_diff",
        "description": "Read the actual current git diff. Optional base must be a revision, never an option.",
        "input_schema": {
            "type": "object",
            "properties": {"base": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "run_validation",
        "description": "Run one coordinator-approved validation by its declared ID.",
        "input_schema": {
            "type": "object",
            "properties": {"validation_id": {"type": "string"}},
            "required": ["validation_id"],
        },
    },
]


def tool_definitions_for(role: str, allow_write_tools: bool = False) -> list[dict]:
    allowed = ROLE_TOOLS.get(role.strip().lower())
    if allowed is None:
        raise ValueError(f"Unknown TQB role: {role}")
    if allow_write_tools:
        allowed = allowed | WRITE_TOOLS
    return [item for item in SPECIALIST_TOOL_DEFINITIONS if item["name"] in allowed]


def _sanitized_subprocess_env() -> dict[str, str]:
    blocked_fragments = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "WEBHOOK", "CREDENTIAL")
    return {
        key: value for key, value in os.environ.items()
        if not any(fragment in key.upper() for fragment in blocked_fragments)
    }


class SpecialistToolRuntime:
    """Tool registry bound to one role and one isolated worktree."""

    def __init__(
        self,
        *,
        role: str,
        workspace: Path,
        worker_ng,
        read_only_paths=None,
        validation_commands=None,
        allow_write_tools: bool = False,
        mutations_approved: bool = False,
    ):
        self.role = role.strip().lower()
        if self.role not in ROLE_TOOLS:
            raise ValueError(f"Unknown TQB role: {role}")
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace does not exist: {self.workspace}")
        self.ng = worker_ng
        self.read_only_paths = [Path(path).resolve() for path in (read_only_paths or [])]
        self.validation_commands = {
            key: tuple(argv) for key, argv in (validation_commands or {}).items()
        }
        self.allow_write_tools = allow_write_tools
        self.mutations_approved = mutations_approved
        self._recall_gate_cache = {}
        self._fs = FilesystemTool(
            self.workspace, policy_engine=True, read_only_paths=self.read_only_paths,
        )

    @property
    def tool_definitions(self) -> list[dict]:
        definitions = tool_definitions_for(self.role, self.allow_write_tools)
        if not self.validation_commands:
            definitions = [item for item in definitions if item["name"] != "run_validation"]
        return definitions

    def execute(self, tool_name: str, args: dict) -> dict:
        allowed_tools = ROLE_TOOLS[self.role]
        if self.allow_write_tools:
            allowed_tools = allowed_tools | WRITE_TOOLS
        if tool_name not in allowed_tools:
            return self._error(tool_name, f"Tool is unavailable to {self.role}")

        allowed, reason = check_tool_call(
            tool_name, args, self.workspace, self.read_only_paths,
        )
        if not allowed:
            return self._error(tool_name, f"Denied: {reason}")
        if tool_name in WRITE_TOOLS and not self.mutations_approved:
            return {
                "status": "staged",
                "tool": tool_name,
                "args": args,
                "description": f"{self.role} requests {tool_name}",
            }

        recall_text = self._recall(tool_name, args)
        try:
            result = self._dispatch(tool_name, args)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        if isinstance(result, dict) and result.get("status") == "error":
            result_text = f"Error: {result.get('error', 'Unknown error')}"
            status = "error"
        else:
            result_text = str(result)
            status = "executed"
        self._deposit(tool_name, args, result_text)
        if recall_text:
            result_text = f"[Substrate recall]\n{recall_text}\n\n[Result]\n{result_text}"
        return {"status": status, "tool": tool_name, "result": result_text}

    def _dispatch(self, tool_name: str, args: dict):
        handlers = {
            "read_file": lambda: self._fs.read_file(
                args.get("path", ""), args.get("start_line"), args.get("end_line")
            ),
            "list_files": lambda: self._fs.list_files(
                args.get("path", "."), args.get("max_depth", 3)
            ),
            "write_file": lambda: self._fs.write_file(
                args.get("path", ""), args.get("content", "")
            ),
            "edit_file": lambda: self._fs.edit_file(
                args.get("path", ""), args.get("old_text", ""), args.get("new_text", "")
            ),
            "search_code": lambda: self._search(args, "code"),
            "search_conversations": lambda: self._search(args, "conversations"),
            "search_testament": lambda: self._search(args, "testament"),
            "git_status": self._git_status,
            "git_diff": lambda: self._git_diff(args.get("base", "")),
            "run_validation": lambda: self._run_validation(args.get("validation_id", "")),
        }
        return handlers[tool_name]()

    def _search(self, args: dict, domain: str) -> str:
        if self.ng is None:
            return "No shared NeuroGraph is available."
        results = self.ng.recall(
            args.get("query", ""), k=args.get("n", 5), threshold=0.3,
        )
        return json.dumps(results, default=str)[:MAX_RESULT_CHARS]

    def _git_status(self) -> str:
        return self._run_fixed(["git", "status", "--short", "--branch"])

    def _git_diff(self, base: str) -> str:
        if base.startswith("-") or any(character.isspace() for character in base):
            raise ValueError("Invalid git base revision")
        argv = ["git", "diff", "--no-ext-diff", "--unified=80"]
        if base:
            argv.append(base)
        return self._run_fixed(argv)

    def _run_validation(self, validation_id: str) -> str:
        argv = self.validation_commands.get(validation_id)
        if argv is None:
            raise ValueError(f"Unknown validation ID: {validation_id}")
        if not argv or not all(isinstance(value, str) and value for value in argv):
            raise ValueError(f"Invalid validation argv for: {validation_id}")
        return self._run_fixed(list(argv))

    def _run_fixed(self, argv: list[str]) -> str:
        result = subprocess.run(
            argv,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("CODEMINE_VALIDATION_TIMEOUT", "300")),
            env=_sanitized_subprocess_env(),
        )
        output = (
            f"EXIT_CODE: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        return output[:MAX_RESULT_CHARS]

    def _recall(self, tool_name: str, args: dict) -> str:
        if self.ng is None:
            return ""
        try:
            from worker_ng import (
                filter_recall_results, gate_repeat_recall, recall_context,
                surfacing_context,
            )
            target = args.get("path") or args.get("query") or args.get("validation_id") or ""
            key = f"{tool_name}:{target}"
            if gate_repeat_recall(self._recall_gate_cache, key, time.time()):
                raw = recall_context(self.ng, tool_name, json.dumps(args, default=str)[:500])
                filtered = filter_recall_results(self.ng, raw)
            else:
                filtered = []
            return surfacing_context(self.ng, filtered)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Specialist recall failed: %s", exc)
            return ""

    def _deposit(self, tool_name: str, args: dict, result: str) -> None:
        if self.ng is None:
            return
        try:
            from worker_ng import ingest_tool_result
            ingest_tool_result(self.ng, tool_name, args, result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Specialist experience deposit failed: %s", exc)

    @staticmethod
    def _error(tool_name: str, reason: str) -> dict:
        return {"status": "error", "tool": tool_name, "result": reason}
