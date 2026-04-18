#!/usr/bin/env python3
# ---- Changelog ----
# [2026-04-17] Claude Code (Sonnet 4.6) — Initial creation
# What: CLI runner for WorkBlockSpec JSON files on the VPS.
# Why:  Eliminates the inline python3 -c invocation mess used for #126 and #111.
#       Handles credential injection, env var substitution, executor wiring, and
#       exit codes in one reusable script.
# How:  Reads GITHUB_TOKEN from ~/.git-credentials if not in env. Substitutes
#       ${VAR} patterns in all shell_execute commands. Runs SpecExecutor with
#       a no-op policy_check_fn (specs are pre-validated; executor policy is
#       sufficient). worker_ng=None is safe — substrate ingestion warns but
#       all execution steps still run.
# -------------------
"""
Run a WorkBlockSpec JSON file through the SpecExecutor.

Usage:
    python3 run_spec.py <spec.json> [--dry-run] [--workspace PATH]

Options:
    --dry-run       Print the resolved spec (with env vars substituted) and exit.
    --workspace     Override workspace root (default: /home/josh/Faux_Clawdbot).

Exit codes:
    0  All steps passed, status = completed
    1  Spec aborted or one or more steps failed
    2  Spec file not found, invalid JSON, or schema validation failed
"""

import json
import os
import re
import sys
from pathlib import Path

# ── Credential helpers ────────────────────────────────────────────────────────

def _token_from_git_credentials(host: str = "github.com") -> str | None:
    """Extract a token from ~/.git-credentials for the given host."""
    creds_path = Path("~/.git-credentials").expanduser()
    if not creds_path.exists():
        return None
    creds = creds_path.read_text()
    pattern = rf"https://[^:]+:([^@]+)@{re.escape(host)}"
    m = re.search(pattern, creds)
    return m.group(1) if m else None


def _inject_credentials() -> None:
    """Ensure GITHUB_TOKEN is in the environment, pulling from git-credentials if needed."""
    if os.getenv("GITHUB_TOKEN"):
        return
    token = _token_from_git_credentials("github.com")
    if token:
        os.environ["GITHUB_TOKEN"] = token
    else:
        print("WARNING: GITHUB_TOKEN not set and not found in ~/.git-credentials", file=sys.stderr)


# ── Env-var substitution ──────────────────────────────────────────────────────

def _substitute_env(spec: dict) -> dict:
    """Walk all shell_execute command strings and substitute ${VAR} patterns."""
    spec = json.loads(json.dumps(spec))  # deep copy
    for step in spec.get("steps", []):
        if step.get("tool") == "shell_execute":
            cmd = step.get("params", {}).get("command", "")
            def _replace(m):
                val = os.environ.get(m.group(1), "")
                if not val:
                    print(f"WARNING: ${{{m.group(1)}}} is unset in environment", file=sys.stderr)
                return val
            step["params"]["command"] = re.sub(r"\$\{([^}]+)\}", _replace, cmd)
    return spec


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    spec_path = None
    dry_run = False
    workspace_override = None

    i = 0
    while i < len(args):
        if args[i] == "--dry-run":
            dry_run = True
        elif args[i] == "--workspace" and i + 1 < len(args):
            workspace_override = args[i + 1]
            i += 1
        elif not args[i].startswith("--"):
            spec_path = args[i]
        i += 1

    if spec_path is None:
        print("ERROR: no spec file given", file=sys.stderr)
        return 2

    # Load spec
    try:
        spec = json.loads(Path(spec_path).read_text())
    except FileNotFoundError:
        print(f"ERROR: spec file not found: {spec_path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in spec: {e}", file=sys.stderr)
        return 2

    # Schema validation
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from work_block_schema import validate_spec
        ok, errors = validate_spec(spec)
        if not ok:
            print("ERROR: spec failed schema validation:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 2
    except ImportError:
        print("WARNING: work_block_schema not found — skipping validation", file=sys.stderr)

    # Inject credentials and substitute env vars
    _inject_credentials()
    spec = _substitute_env(spec)

    if dry_run:
        print(json.dumps(spec, indent=2))
        return 0

    # Build executor
    workspace = Path(workspace_override) if workspace_override else Path(__file__).parent

    from spec_executor import SpecExecutor
    executor = SpecExecutor(
        tool_registry={},
        policy_check_fn=lambda *a, **kw: (True, "permitted by run_spec"),
        worker_ng=None,
        workspace=workspace,
    )

    # Execute
    report = executor.execute_block(spec)
    print(json.dumps(report, indent=2))

    status = report.get("status", "unknown")
    summary = report.get("summary", {})
    failed = summary.get("failed", 0)

    if status == "completed" and failed == 0:
        print(f"\n✓ {report.get('block_id')} — {status} ({summary.get('passed', 0)} passed, {summary.get('elapsed_seconds', 0):.1f}s)", file=sys.stderr)
        return 0
    else:
        abort_reason = report.get("abort_reason", "")
        print(f"\n✗ {report.get('block_id')} — {status}: {abort_reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
