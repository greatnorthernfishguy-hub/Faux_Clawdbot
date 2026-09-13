# ---- Changelog ----
# [2026-09-13] Codex — Apply sensitive-name denial after external allowlisting.
# What: Preserve the matched read-only root without returning before Rim checks.
# Why: An allowlisted sibling path could otherwise expose its .env or key files.
# How: Defer the external-read success return until sensitive checks complete.
# [2026-09-13] Codex — Enforce the vendored-file boundary at tool execution.
# What: Deny writes to all current vendored substrate files and the retired peer bridge.
# Why: Plan validation alone cannot be the final constitutional boundary.
# How: Apply the immutable filename deny set in write-mode path authorization.
# [2026-09-13] Codex — Close secret-read and process/Git authority gaps.
# What: Deny sensitive reads, dangerous allowlisted subcommands, environment
#       assignments, and direct Git mutation tools at the constitutional Rim.
# Why: Binary-name allowlisting still admitted Python code, find -exec, Git push,
#      and child-process secret exposure outside Mission Control.
# How: Validate argv semantics and keep Git mutation authority inside Mission's
#      deterministic coordinator.
# [2026-09-12] Codex — Make shell allowlisting structural
# What: Parse commands with shlex, reject shell grammar, validate exact binaries,
#       and expose the parsed argv/environment for shell=False execution.
# Why: Checking only the first word while executing with shell=True allowed a
#      permitted command to smuggle arbitrary follow-on commands.
# How: One canonical parser now serves PolicyEngine, ShellTool, and SpecExecutor.
# [2026-04-17] Claude (Sonnet 4.6) — read_only_paths: ecosystem sibling repo read access (#168)
# What: can_access_path + check_tool_call gain optional read_only_paths param (list[Path]).
#   Paths outside workspace are permitted for read_file if they fall under a read_only_paths root.
# Why: QB workspace boundary (HF Space = /home/josh/Faux_Clawdbot) blocks reads of NeuroGraph
#   and other sibling repos. Spec execution on VPS workers needs cross-repo read for verification.
# How: ValueError branch in can_access_path walks read_only_paths before denying. All callers
#   default to None (backward-compat). app.py passes _ECOSYSTEM_READ_PATHS; spec_executor passes
#   constraints["read_only_paths"] from the spec.
# [2026-04-06] Josh + Claude — Add edit_file to policy checks and gating
# What: edit_file path+content checks in check_tool_call, added to _GATED_TOOLS
# Why: New edit_file tool needs the same security gates as write_file
# How: Same path access (write mode) and content secret scan as write_file
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
import shlex
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

# Sensitive filename patterns (blocked for reads and writes)
_SENSITIVE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\.env$", re.IGNORECASE),
    re.compile(r"\.env\.", re.IGNORECASE),
    re.compile(r".*\.key$", re.IGNORECASE),
    re.compile(r"^credentials\..*", re.IGNORECASE),
    re.compile(r"^\.git-credentials$", re.IGNORECASE),
    re.compile(r"^\.netrc$", re.IGNORECASE),
    re.compile(r"^\.npmrc$", re.IGNORECASE),
    re.compile(r"^\.pypirc$", re.IGNORECASE),
    re.compile(r"^id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?$", re.IGNORECASE),
)

_VENDORED_FILES = frozenset({
    "ng_lite.py",
    "ng_tract_bridge.py",
    "ng_ecosystem.py",
    "openclaw_adapter.py",
    "ng_autonomic.py",
    "ng_embed.py",
    "ng_salience_gate.py",
    "ng_updater.py",
    # Retired and constitutionally prohibited from restoration.
    "ng_peer_bridge.py",
})

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


def can_access_path(
    path: str,
    mode: str,
    workspace: Path,
    read_only_paths: list[Path] | None = None,
) -> tuple[bool, str]:
    """Check whether *path* is allowed for the given *mode* within *workspace*.

    Parameters
    ----------
    path : str
        The filesystem path the tool wants to access.
    mode : str
        ``"read"`` or ``"write"``.
    workspace : Path
        The root directory the bot is allowed to operate within.
    read_only_paths : list[Path] | None
        Additional directories that are readable but not writable (e.g. sibling
        ecosystem repos).  Only consulted when the path falls outside *workspace*
        and mode is ``"read"``.

    Returns
    -------
    tuple[bool, str]
        (allowed, reason)
    """
    try:
        p = Path(path)
        # Resolve relative paths against workspace, not CWD
        resolved = (workspace / p).resolve() if not p.is_absolute() else p.resolve()
    except (OSError, ValueError) as exc:
        return False, f"Path resolution failed: {exc}"

    workspace_resolved = workspace.resolve()

    external_read_root: Path | None = None

    # Path must be within the workspace unless it is an explicitly read-only root.
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        # Outside workspace — check read-only external allowlist
        if mode == "read" and read_only_paths:
            for ro_root in read_only_paths:
                try:
                    resolved.relative_to(Path(ro_root).resolve())
                    external_read_root = Path(ro_root).resolve()
                    break
                except ValueError:
                    continue
        if external_read_root is None:
            return False, (
                f"Path escapes workspace. Resolved path '{resolved}' "
                f"is not inside '{workspace_resolved}'."
            )

    filename = resolved.name
    for pat in _SENSITIVE_PATTERNS:
        if pat.search(filename):
            return False, (
                f"{mode.capitalize()} denied — '{filename}' matches sensitive file "
                f"pattern '{pat.pattern}'."
            )

    # Write-mode: additionally block vendored files
    if mode == "write":
        if filename in _VENDORED_FILES:
            return False, (
                f"Write denied — '{filename}' is vendored or retired; repair the "
                "canonical source through its required gates."
            )
    if external_read_root is not None:
        return True, f"Read permitted via external allowlist: {external_read_root}"
    return True, "Path access permitted."


_SHELL_CONTROL_TOKENS = frozenset({";", "&&", "||", "|", "&", ">", ">>", "<", "<<"})
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(.*)$", re.DOTALL)


def parse_shell_command(command: str) -> tuple[list[str], dict[str, str]]:
    """Parse one command into argv and leading environment assignments.

    Shell grammar and command substitution are refused because callers execute
    the returned argv with ``shell=False``.
    """
    if "`" in command or "$(" in command:
        raise ValueError("Shell command substitution is not permitted.")
    if "\n" in command or "\r" in command:
        raise ValueError("Multiline shell commands are not permitted.")
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as exc:
        raise ValueError(f"Invalid shell quoting: {exc}") from exc
    if not tokens:
        raise ValueError("Empty command denied.")
    forbidden = [
        token for token in tokens
        if token in _SHELL_CONTROL_TOKENS
        or (token and all(character in ";&|<>" for character in token))
    ]
    if forbidden:
        raise ValueError(f"Shell control token is not permitted: {forbidden[0]}")

    environment: dict[str, str] = {}
    while tokens:
        match = _ENV_ASSIGNMENT.fullmatch(tokens[0])
        if not match:
            break
        key, value = tokens.pop(0).split("=", 1)
        environment[key] = value
    if not tokens:
        raise ValueError("Command is only environment assignments — no binary found.")
    return tokens, environment


def shell_binary_allowed(binary: str, allowlist: tuple[str, ...] | list[str]) -> bool:
    """Match exact binary names, allowing only numeric Python/pip suffixes."""
    if binary in allowlist:
        return True
    for base in ("python", "pip"):
        if base in allowlist and re.fullmatch(rf"{base}[0-9]+(?:\.[0-9]+)?", binary):
            return True
    return False


def can_execute_shell(command: str) -> tuple[bool, str]:
    """Check whether *command* is on the shell allowlist.

    Only the first token (the binary name) is checked against the allowlist.

    Returns
    -------
    tuple[bool, str]
        (allowed, reason)
    """
    try:
        tokens, environment = parse_shell_command(command)
    except ValueError as exc:
        return False, str(exc)

    # Extract the first token (the binary / command name)
    first_token = tokens[0]
    # Strip any path prefix so `/usr/bin/python` matches `python`
    binary = os.path.basename(first_token)

    if not shell_binary_allowed(binary, _SHELL_ALLOWLIST):
        return False, (
            f"Shell command denied — '{binary}' is not on the allowlist. "
            f"Allowed: {', '.join(_SHELL_ALLOWLIST)}."
        )

    if environment:
        return False, "Leading environment assignments are not permitted."

    arguments = tokens[1:]
    if re.fullmatch(r"python[0-9]+(?:\.[0-9]+)?", binary) or binary == "python":
        if "-c" in arguments:
            return False, "Inline Python execution is not permitted."
        if "-m" in arguments:
            index = arguments.index("-m")
            module = arguments[index + 1] if index + 1 < len(arguments) else ""
            if module not in {"pytest", "compileall", "py_compile"}:
                return False, f"Python module execution is not permitted: {module or '(missing)'}"

    if binary == "git":
        subcommand = next((item for item in arguments if not item.startswith("-")), "")
        if subcommand not in {"diff", "status", "show", "log", "rev-parse", "ls-files"}:
            return False, f"Git subcommand is not read-only: {subcommand or '(missing)'}"

    if binary == "find" and any(
        item in {"-exec", "-execdir", "-ok", "-okdir", "-delete"}
        for item in arguments
    ):
        return False, "Mutating or executable find actions are not permitted."

    if binary.startswith("pip") or binary == "npm":
        if arguments not in (["--version"], ["-v"], ["--help"], ["-h"]):
            return False, f"Package-manager mutation is not permitted through {binary}."

    return True, f"Shell command permitted ('{binary}')."


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
    "edit_file",
    "shell_execute",
    "push_to_github",
    "pull_from_github",
    "notebook_add",
    "notebook_delete",
    "create_shadow_branch",
})


def should_gate_for_review(tool_name: str, args: dict) -> bool:  # noqa: ARG001
    """Return ``True`` if *tool_name* should be held for review.

    Interactive Chat always stages mutations. HF does not run Claude Code's
    local hooks, so no environment flag may claim that those hooks approved it.
    """
    return tool_name in _GATED_TOOLS


# ---------------------------------------------------------------------------
# Public entry point — run all applicable checks and audit-log the result
# ---------------------------------------------------------------------------

def check_tool_call(
    tool_name: str,
    args: dict,
    workspace: Path,
    read_only_paths: list[Path] | None = None,
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
    if tool_name in (
        "write_file", "edit_file", "read_file", "list_files",
        "notebook_add", "notebook_delete",
    ):
        path = args.get("path", args.get("file_path", ""))
        mode = "read" if tool_name in ("read_file", "list_files") else "write"
        allowed, reason = can_access_path(path, mode, workspace, read_only_paths)

    # --- Content checks (write operations) ---
    if allowed and tool_name == "write_file":
        content = args.get("content", "")
        path = args.get("path", args.get("file_path", ""))
        allowed, reason = can_write_content(path, content)

    if allowed and tool_name == "edit_file":
        new_text = args.get("new_text", "")
        path = args.get("path", args.get("file_path", ""))
        allowed, reason = can_write_content(path, new_text)

    # --- Shell command checks ---
    if allowed and tool_name == "shell_execute":
        command = args.get("command", "")
        allowed, reason = can_execute_shell(command)

    if allowed and tool_name in {
        "push_to_github", "pull_from_github", "create_shadow_branch",
    }:
        allowed = False
        reason = "Direct Git mutation is prohibited; use Mission Control."

    _audit_log(tool_name, args, allowed, reason)
    return allowed, reason
