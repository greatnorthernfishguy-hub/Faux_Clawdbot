# ---- Changelog ----
# [2026-03-29] Anvil (TQB) — Initial creation of policy_engine.py
# What: Cricket-shaped gating layer with Rim (immutable) and Mesh (evolving) components
# Why: PRD Block B — all tool calls must route through policy checks before execution
# How: Rim = hardcoded constitutional constraints (path, shell, content).
#       Mesh = static bootstrap gating table (mutating tools require review).
#       Audit = JSONL append log for every tool call.
# -------------------

"""
Policy Engine — Cricket-shaped gating layer for Faux_Clawdbot.

Rim: immutable constitutional constraints (code, not config).
Mesh: evolving gating decisions (static bootstrap for now).
Audit: persistent JSONL log of every tool call and its outcome.
"""

import json
import os
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

_AUDIT_DIR = Path(__file__).resolve().parent / "data" / "audit"


def _ensure_audit_dir() -> None:
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def _audit_log(
    tool_name: str,
    args: dict,
    allowed: bool,
    reason: str,
) -> None:
    """Append one JSON line to the audit log.

    Content values longer than 200 characters are truncated in the log entry
    to avoid dumping huge payloads into the audit trail.
    """
    _ensure_audit_dir()

    sanitized_args = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 200:
            sanitized_args[k] = v[:200] + "...[truncated]"
        else:
            sanitized_args[k] = v

    entry = {
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": tool_name,
        "args": sanitized_args,
        "allowed": allowed,
        "reason": reason,
    }

    log_path = _AUDIT_DIR / "policy.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Rim — immutable constitutional constraints
# ---------------------------------------------------------------------------
# These are CODE. They cannot be toggled, overridden, or reconfigured at
# runtime. Changing them requires a code change, a commit, and a review.
# ---------------------------------------------------------------------------

# Sensitive filename patterns (blocked for write)
_SENSITIVE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\.env$", re.IGNORECASE),
    re.compile(r"\.env\.", re.IGNORECASE),
    re.compile(r".*\.key$", re.IGNORECASE),
    re.compile(r"^credentials\..*", re.IGNORECASE),
)

# Shell command allowlist — only these prefixes are permitted
_SHELL_ALLOWLIST: tuple[str, ...] = (
    "python",
    "pip",
    "pytest",
    "git",
    "npm",
    "node",
    "ls",
    "grep",
    "find",
    "wc",
    "head",
    "tail",
    "diff",
)

# Secret patterns in content (partial matches are enough to deny)
_SECRET_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),         # OpenAI-style keys
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),         # GitHub personal access tokens
    re.compile(r"gho_[A-Za-z0-9]{36,}"),         # GitHub OAuth tokens
    re.compile(r"ghs_[A-Za-z0-9]{36,}"),         # GitHub server tokens
    re.compile(r"ghr_[A-Za-z0-9]{36,}"),         # GitHub refresh tokens
    re.compile(r"AKIA[0-9A-Z]{16}"),             # AWS access key IDs
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),  # Bearer tokens
    re.compile(r"xox[bpras]-[A-Za-z0-9\-]+"),   # Slack tokens
    re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"),   # Anthropic API keys
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."),  # JWTs
)


def can_access_path(path: str, mode: str, workspace: Path) -> tuple[bool, str]:
    """Check whether *path* is allowed for the given *mode* within *workspace*.

    Parameters
    ----------
    path : str
        The filesystem path the tool wants to access.
    mode : str
        ``"read"`` or ``"write"``.
    workspace : Path
        The root directory the bot is allowed to operate within.

    Returns
    -------
    tuple[bool, str]
        (allowed, reason)
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError) as exc:
        return False, f"Path resolution failed: {exc}"

    workspace_resolved = workspace.resolve()

    # Path must be within the workspace
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        return False, (
            f"Path escapes workspace. Resolved path '{resolved}' "
            f"is not inside '{workspace_resolved}'."
        )

    # Write-mode: block sensitive files
    if mode == "write":
        filename = resolved.name
        for pat in _SENSITIVE_PATTERNS:
            if pat.search(filename):
                return False, (
                    f"Write denied — '{filename}' matches sensitive file "
                    f"pattern '{pat.pattern}'."
                )

    return True, "Path access permitted."


def can_execute_shell(command: str) -> tuple[bool, str]:
    """Check whether *command* is on the shell allowlist.

    Only the first token (the binary name) is checked against the allowlist.

    Returns
    -------
    tuple[bool, str]
        (allowed, reason)
    """
    stripped = command.strip()
    if not stripped:
        return False, "Empty command denied."

    # Extract the first token (the binary / command name)
    first_token = stripped.split()[0]
    # Strip any path prefix so `/usr/bin/python` matches `python`
    binary = os.path.basename(first_token)

    for allowed_prefix in _SHELL_ALLOWLIST:
        # Match exactly or as a prefix (e.g. `python3` matches `python`)
        if binary == allowed_prefix or binary.startswith(allowed_prefix):
            return True, f"Shell command permitted (matched '{allowed_prefix}')."

    return False, (
        f"Shell command denied — '{binary}' is not on the allowlist. "
        f"Allowed: {', '.join(_SHELL_ALLOWLIST)}."
    )


def can_write_content(path: str, content: str) -> tuple[bool, str]:
    """Check whether *content* to be written to *path* contains secrets.

    Returns
    -------
    tuple[bool, str]
        (allowed, reason)
    """
    for pat in _SECRET_PATTERNS:
        match = pat.search(content)
        if match:
            # Don't leak the actual secret into the deny reason
            snippet_start = max(0, match.start() - 10)
            snippet_end = min(len(content), match.end() + 5)
            context_hint = content[snippet_start:match.start()] + "***REDACTED***"
            return False, (
                f"Content contains what appears to be a secret or token "
                f"(pattern: '{pat.pattern}'). Context: ...{context_hint}..."
            )

    return True, "Content approved — no secret patterns detected."


# ---------------------------------------------------------------------------
# Mesh — evolving gating decisions (static bootstrap)
# ---------------------------------------------------------------------------
# Bootstrap: all mutating tools are gated for review. Read-only tools
# auto-execute. This will graduate to substrate-informed gating later
# (Competence Model: Apprentice stage).
# ---------------------------------------------------------------------------

_GATED_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "shell_execute",
    "push_to_github",
    "pull_from_github",
    "notebook_add",
    "notebook_delete",
    "create_shadow_branch",
})


# QB_DISPATCHED — when True, mesh gating is bypassed because QB's hooks
# are the enforcement layer. Set via environment variable by QB's cron.
_QB_DISPATCHED = os.getenv("QB_DISPATCHED", "").lower() in ("1", "true", "yes")


def should_gate_for_review(tool_name: str, args: dict) -> bool:  # noqa: ARG001
    """Return ``True`` if *tool_name* should be held for review.

    Under QB authority: auto-execute everything (QB's hooks enforce).
    Standalone: mutating tools are staged for human review.
    """
    if _QB_DISPATCHED:
        return False
    return tool_name in _GATED_TOOLS


# ---------------------------------------------------------------------------
# Public entry point — run all applicable checks and audit-log the result
# ---------------------------------------------------------------------------

def check_tool_call(
    tool_name: str,
    args: dict,
    workspace: Path,
) -> tuple[bool, str]:
    """Run Rim checks applicable to *tool_name* and log the result.

    This is the main entry point callers should use.  It dispatches to the
    appropriate Rim checks based on tool semantics, logs the outcome, and
    returns the verdict.

    Returns
    -------
    tuple[bool, str]
        (allowed, reason)
    """
    allowed = True
    reason = "Permitted."

    # --- Path access checks ---
    if tool_name in ("write_file", "read_file", "notebook_add", "notebook_delete"):
        path = args.get("path", args.get("file_path", ""))
        mode = "write" if tool_name != "read_file" else "read"
        allowed, reason = can_access_path(path, mode, workspace)

    # --- Content checks (write operations) ---
    if allowed and tool_name == "write_file":
        content = args.get("content", "")
        path = args.get("path", args.get("file_path", ""))
        allowed, reason = can_write_content(path, content)

    # --- Shell command checks ---
    if allowed and tool_name == "shell_execute":
        command = args.get("command", "")
        allowed, reason = can_execute_shell(command)

    _audit_log(tool_name, args, allowed, reason)
    return allowed, reason
