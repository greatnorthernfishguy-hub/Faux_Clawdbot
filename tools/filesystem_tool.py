# ---- Changelog ----
# [2026-03-29] Chisel/TQB — Block C: FilesystemTool
# What: read_file, write_file, list_files extracted from RecursiveContextManager
# Why: PRD Block C — single-responsibility tool classes
# How: Each method gates through policy_engine when present; list_files now respects max_depth
# [2026-03-29] Razor/TQB — Block A: Security Hardening
# What: Path traversal protection, file size guard, proper PolicyEngine integration
# Why: PRD Block A — is_relative_to() enforced, file size check before read, content secret scan
# How: Resolve + is_relative_to on every op; 10MB read guard; PolicyEngine returns (bool, reason)
# -------------------

import logging
from pathlib import Path

logger = logging.getLogger("tools.filesystem")

# Maximum file size to read into memory (bytes)
MAX_READ_SIZE = 10 * 1024 * 1024  # 10MB


class FilesystemTool:
    """Filesystem read/write/list operations scoped to repo_path."""

    def __init__(self, repo_path: Path, policy_engine=None):
        self.repo_path = repo_path
        self.policy_engine = policy_engine

    def _check_path(self, path: str, mode: str) -> tuple:
        """Resolve path and enforce workspace boundary.

        Returns (resolved_path, error_dict_or_None).
        """
        target = (self.repo_path / path).resolve()
        if not target.is_relative_to(self.repo_path.resolve()):
            msg = f"Path outside workspace boundary: {path}"
            logger.warning(msg)
            return None, {"status": "error", "tool": "filesystem", "error": msg, "type": "PermissionError"}

        if self.policy_engine:
            from policy_engine import check_tool_call
            tool_name = "read_file" if mode == "read" else "write_file"
            args = {"path": str(target)}
            allowed, reason = check_tool_call(tool_name, args, self.repo_path)
            if not allowed:
                return None, {"status": "error", "tool": "filesystem", "error": reason, "type": "PermissionError"}

        return target, None

    def read_file(self, path: str, start_line: int = None, end_line: int = None) -> str:
        target, err = self._check_path(path, "read")
        if err:
            return err

        try:
            # File size guard — check BEFORE reading into memory
            if target.exists() and target.stat().st_size > MAX_READ_SIZE:
                msg = f"File too large ({target.stat().st_size:,} bytes). Max: {MAX_READ_SIZE:,} bytes."
                logger.warning("read_file rejected: %s — %s", path, msg)
                return {"status": "error", "tool": "filesystem", "error": msg, "type": "ValueError"}

            content = target.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            if start_line is not None and end_line is not None:
                lines = lines[start_line:end_line]
            return "\n".join(lines)
        except FileNotFoundError:
            return {"status": "error", "tool": "filesystem", "error": f"File not found: {path}", "type": "FileNotFoundError"}
        except PermissionError as e:
            return {"status": "error", "tool": "filesystem", "error": str(e), "type": "PermissionError"}
        except OSError as e:
            return {"status": "error", "tool": "filesystem", "error": str(e), "type": type(e).__name__}

    def write_file(self, path: str, content: str) -> str:
        target, err = self._check_path(path, "write")
        if err:
            return err

        # Content secret scan via PolicyEngine
        if self.policy_engine:
            from policy_engine import can_write_content
            allowed, reason = can_write_content(path, content)
            if not allowed:
                return {"status": "error", "tool": "filesystem", "error": reason, "type": "PermissionError"}

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
            size = target.stat().st_size
            return f"Written to {target} ({size:,} bytes)"
        except PermissionError as e:
            return {"status": "error", "tool": "filesystem", "error": str(e), "type": "PermissionError"}
        except OSError as e:
            return {"status": "error", "tool": "filesystem", "error": str(e), "type": type(e).__name__}

    def list_files(self, path: str = ".", max_depth: int = 3) -> str:
        target, err = self._check_path(path, "read")
        if err:
            return err

        try:
            if not target.exists():
                return "Path not found."
            files = []
            for p in target.rglob("*"):
                if not p.is_file():
                    continue
                if any(part.startswith(".") for part in p.parts):
                    continue
                try:
                    rel = p.relative_to(target)
                except ValueError:
                    continue
                if len(rel.parts) > max_depth:
                    continue
                files.append(str(p.relative_to(self.repo_path)))
            return "\n".join(files[:50])
        except PermissionError as e:
            return {"status": "error", "tool": "filesystem", "error": str(e), "type": "PermissionError"}
        except OSError as e:
            return {"status": "error", "tool": "filesystem", "error": str(e), "type": type(e).__name__}
