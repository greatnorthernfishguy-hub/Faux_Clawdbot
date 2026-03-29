# ---- Changelog ----
# [2026-03-29] Chisel/TQB — Block C: GitTool
# What: push_to_github, pull_from_github, create_shadow_branch extracted from RecursiveContextManager
# Why: PRD Block C — single-responsibility tool classes
# How: Git subprocess calls scoped to repo_path; policy_engine gating for shell execution
# [2026-03-29] Razor/TQB — Block A: Security Hardening
# What: Fixed PolicyEngine integration (check_tool_call returns tuple, not bool)
# Why: Block A/B interface alignment
# How: Use check_tool_call() dispatcher with proper (bool, reason) handling
# -------------------

import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("tools.git")


class GitTool:
    """Git operations scoped to repo_path."""

    def __init__(self, repo_path: Path, policy_engine=None):
        self.repo_path = repo_path
        self.policy_engine = policy_engine

    def _check_policy(self, tool_name: str, args: dict) -> tuple:
        """Check PolicyEngine if present. Returns (allowed, reason)."""
        if not self.policy_engine:
            return True, "No policy engine configured."
        from policy_engine import check_tool_call
        return check_tool_call(tool_name, args, self.repo_path)

    def push_to_github(self, message: str) -> str:
        allowed, reason = self._check_policy("push_to_github", {"message": message})
        if not allowed:
            return {"status": "error", "tool": "git", "error": reason, "type": "PermissionError"}

        try:
            cwd = str(self.repo_path)
            subprocess.run(["git", "config", "user.email", "clawdbot@system.local"], check=False, cwd=cwd)
            subprocess.run(["git", "config", "user.name", "Clawdbot"], check=False, cwd=cwd)
            subprocess.run(["git", "add", "."], check=True, cwd=cwd)
            subprocess.run(["git", "commit", "-m", message], check=True, cwd=cwd)
            return "Changes committed (Push requires configured remote with token)."
        except subprocess.CalledProcessError as e:
            logger.error("Git push failed: %s", e)
            return {"status": "error", "tool": "git", "error": f"Git error: {e}", "type": "CalledProcessError"}
        except OSError as e:
            return {"status": "error", "tool": "git", "error": str(e), "type": type(e).__name__}

    def pull_from_github(self, branch: str) -> str:
        allowed, reason = self._check_policy("pull_from_github", {"branch": branch})
        if not allowed:
            return {"status": "error", "tool": "git", "error": reason, "type": "PermissionError"}

        try:
            subprocess.run(["git", "pull", "origin", branch], check=True, cwd=str(self.repo_path))
            return f"Pulled {branch}"
        except subprocess.CalledProcessError as e:
            logger.error("Git pull failed: %s", e)
            return {"status": "error", "tool": "git", "error": f"Git pull error: {e}", "type": "CalledProcessError"}
        except OSError as e:
            return {"status": "error", "tool": "git", "error": str(e), "type": type(e).__name__}

    def create_shadow_branch(self) -> str:
        allowed, reason = self._check_policy("create_shadow_branch", {})
        if not allowed:
            return {"status": "error", "tool": "git", "error": reason, "type": "PermissionError"}

        ts = int(time.time())
        try:
            subprocess.run(["git", "checkout", "-b", f"shadow_{ts}"], check=True, cwd=str(self.repo_path))
            return f"Created branch shadow_{ts}"
        except subprocess.CalledProcessError as e:
            logger.error("Shadow branch creation failed: %s", e)
            return {"status": "error", "tool": "git", "error": f"Git error: {e}", "type": "CalledProcessError"}
        except OSError as e:
            return {"status": "error", "tool": "git", "error": str(e), "type": type(e).__name__}
