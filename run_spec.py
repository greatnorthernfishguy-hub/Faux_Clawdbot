#!/usr/bin/env python3
# ---- Changelog ----
# [2026-04-22] Claude Code (Sonnet 4.6) — Wire QB Reviewer evaluation into run_spec
#   What: After executor finishes, call evaluate_report() with use_persona=True so
#         QB's Reviewer persona reads the report and gives a verbal checkpoint
#         response (DONE / ITERATE / ESCALATE). Her response and decision are printed
#         to stdout and included in the Discord notification.
#   Why:  run_spec.py was a bypass shortcut — it ran specs but skipped the
#         orchestrator's Reviewer loop entirely. qb_checkpoint gates just logged
#         "pending" with no QB response. Josh remembers getting QB's checkpoints
#         when specs ran through the orchestrator; this restores that behaviour.
#   How:  Import evaluate_report from report_evaluator. Call after executor.execute_block().
#         Pass spec + report for full context. Print evaluation to stdout. Pass to
#         _notify_discord() which appends QB's decision + first 300 chars of her
#         response. evaluate_report() has its own exception fallback so it can never
#         crash the run.
# [2026-04-20] Claude Code (Sonnet 4.6) — Wire Codemine's NeuroGraph into spec executor (#191)
#   What: Replaced worker_ng=None with get_worker_ng() — Codemine's full NeuroGraph
#         now receives every spec step result via dual-pass ingestion.
#   Why:  #191 — substrate was getting nothing from spec runs. First run after this
#         change is the bootstrap: she wires herself in blind, then she's present.
#   How:  Import get_worker_ng from worker_ng.py; singleton pattern handles init.
# [2026-04-17] Claude Code (Sonnet 4.6) — Discord webhook notification (#15)
#   What: Added _notify_discord() — posts pass/fail summary to QB_DISCORD_WEBHOOK after
#         every spec run. Added load_dotenv() so .env in Faux_Clawdbot is picked up when
#         .bashrc is not sourced (SSH MCP, systemd, etc.).
#   Why:  QB_DISCORD_WEBHOOK was set in .bashrc but never reached the running process.
#         QB First Blood produced zero notifications (#15). Silent on failure — a dead
#         webhook must never interrupt a spec run.
#   How:  dotenv loads .env before env-var substitution. _notify_discord() called before
#         final exit in main(). Matches orchestrator.py notification style.
# [2026-04-17] Claude Code (Sonnet 4.6) — Initial creation
# What: CLI runner for WorkBlockSpec JSON files on the VPS.
# Why:  Eliminates the inline python3 -c invocation mess used for #126 and #111.
#       Handles credential injection, env var substitution, executor wiring, and
#       exit codes in one reusable spec runner.
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

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

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


# ── Discord notification ─────────────────────────────────────────────────────

def _notify_discord(report: dict, evaluation: dict | None = None) -> None:
    """Post spec completion summary + QB evaluation to Discord webhook. Silent on failure."""
    try:
        import requests as _requests
    except ImportError:
        return
    webhook_url = os.environ.get("QB_DISCORD_WEBHOOK", "").strip()
    if not webhook_url:
        return
    try:
        status = report.get("status", "unknown")
        summary = report.get("summary", {})
        block_id = report.get("block_id", "unknown")
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        elapsed = summary.get("elapsed_seconds", 0)
        abort_reason = report.get("abort_reason", "")
        status_emoji = {"completed": "✅", "aborted": "🚨", "rejected": "❌"}.get(status, "⚠️")
        lines = [
            f"{status_emoji} **Spec {status.upper()}** — `{block_id}`",
            f"Steps: {passed} passed, {failed} failed | Elapsed: {elapsed:.1f}s",
        ]
        if abort_reason:
            lines.append(f"Abort: {abort_reason}")

        if evaluation is not None:
            decision = evaluation.get("decision", "unknown").upper()
            decision_emoji = {"DONE": "✅", "ITERATE": "🔄", "ESCALATE": "⬆️"}.get(decision, "❓")
            qualitative = evaluation.get("qualitative") or {}
            qb_error = qualitative.get("error")
            if qb_error:
                lines.append(f"\n**QB Reviewer:** ⚠️ Evaluation unavailable — {qb_error}")
            else:
                response_text = qualitative.get("response", "").strip()
                preview = response_text[:300] + ("…" if len(response_text) > 300 else "")
                lines.append(f"\n**QB Reviewer:** {decision_emoji} {decision}")
                if preview:
                    lines.append(f"> {preview}")
                hints = evaluation.get("iteration_hints", [])
                if hints:
                    lines.append("**Hints:** " + " | ".join(hints[:3]))

        payload = {"content": "\n".join(lines), "username": "Codemine"}
        _requests.post(webhook_url, json=payload, timeout=5)
    except Exception as exc:
        import logging as _logging
        _logging.getLogger("run_spec").warning("Discord webhook failed: %s", exc)


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
    from worker_ng import get_worker_ng
    executor = SpecExecutor(
        tool_registry={},
        policy_check_fn=lambda *a, **kw: (True, "permitted by run_spec"),
        worker_ng=get_worker_ng(),
        workspace=workspace,
    )

    # Execute
    report = executor.execute_block(spec)
    print(json.dumps(report, indent=2))

    status = report.get("status", "unknown")
    summary = report.get("summary", {})
    failed = summary.get("failed", 0)

    # QB Reviewer evaluation — runs when spec completed with all steps passing.
    # evaluate_report() has its own exception fallback; it can never crash the run.
    # Her decision is advisory — a passing spec stays passing regardless of decision.
    evaluation = None
    if status in ("completed", "complete") and failed == 0:
        try:
            from report_evaluator import evaluate_report
            print("\n--- QB Reviewer evaluating... ---", file=sys.stderr)
            evaluation = evaluate_report(report, spec=spec, use_persona=True)
            print(json.dumps(evaluation, indent=2))
            decision = evaluation.get("decision", "unknown").upper()
            qb_response = (evaluation.get("qualitative") or {}).get("response", "")
            print(f"\n--- QB decision: {decision} ---", file=sys.stderr)
            if qb_response:
                print(qb_response, file=sys.stderr)
        except Exception as exc:
            print(f"WARNING: QB Reviewer evaluation failed: {exc}", file=sys.stderr)

    _notify_discord(report, evaluation=evaluation)

    if status in ("completed", "complete") and failed == 0:
        print(f"\n✓ {report.get('block_id')} — {status} ({summary.get('passed', 0)} passed, {summary.get('elapsed_seconds', 0):.1f}s)", file=sys.stderr)
        return 0
    else:
        abort_reason = report.get("abort_reason", "")
        print(f"\n✗ {report.get('block_id')} — {status}: {abort_reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
