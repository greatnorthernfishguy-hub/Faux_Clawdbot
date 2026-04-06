# ---- Changelog ----
# [2026-03-29] Chisel/TQB — Block C: ShellTool
# What: shell_execute extracted from RecursiveContextManager
# Why: PRD Block C — single-responsibility tool classes
# [2026-03-29] Razor/TQB — Block A: Security Hardening
# What: PolicyEngine gating replaces joke blocklist, shell=False, output size cap
# Why: PRD Block A — structural security, not advisory string matching
# How: Command split via shlex, subprocess shell=False, PolicyEngine allowlist gate
# -------------------

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("tools.shell")

# Maximum output size returned from shell commands (chars)
MAX_OUTPUT_SIZE = 50_000

# Shell timeout — configurable via env
SHELL_TIMEOUT = int(os.getenv("CODEMINE_SHELL_TIMEOUT", "60"))


class ShellTool:
    """Shell command execution scoped to repo_path, gated by PolicyEngine."""

    def __init__(self, repo_path: Path, policy_engine=None):
        self.repo_path = repo_path
        self.policy_engine = policy_engine

    def shell_execute(self, command: str) -> str:
        # PolicyEngine gate — structural security
        if self.policy_engine:
            from policy_engine import check_tool_call
            allowed, reason = check_tool_call("shell_execute", {"command": command}, self.repo_path)
            if not allowed:
                logger.warning("shell_execute denied: %s — %s", command[:100], reason)
                return {"status": "error", "tool": "shell", "error": reason, "type": "PermissionError"}

        try:
            # shell=True — pipes and redirects work. Safety is enforced by
            # PolicyEngine's command allowlist, not by shell=False.
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=SHELL_TIMEOUT,
            )
            stdout = result.stdout
            stderr = result.stderr

            # Output size cap
            if len(stdout) > MAX_OUTPUT_SIZE:
                stdout = stdout[:MAX_OUTPUT_SIZE] + "\n...[truncated]"
            if len(stderr) > MAX_OUTPUT_SIZE:
                stderr = stderr[:MAX_OUTPUT_SIZE] + "\n...[truncated]"

            return f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        except subprocess.TimeoutExpired:
            logger.warning("shell_execute timeout: %s", command[:100])
            return {"status": "error", "tool": "shell", "error": f"Command timed out ({SHELL_TIMEOUT}s limit)", "type": "TimeoutError"}
        except FileNotFoundError as e:
            return {"status": "error", "tool": "shell", "error": f"Command not found: {e}", "type": "FileNotFoundError"}
        except OSError as e:
            return {"status": "error", "tool": "shell", "error": str(e), "type": type(e).__name__}
