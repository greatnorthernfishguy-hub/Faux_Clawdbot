# ---- Changelog ----
# [2026-04-07] Josh + Claude — Chain of command orchestrator (Phase 6)
# What: The full loop — intent → spec → execute → evaluate → iterate/done
# Why: "Say go and walk away." This is the top-level entry point.
# How: Strategist generates spec → risk classified → executor runs → Reviewer evaluates
#      → done (ingest to Graph) / iterate (Strategist generates follow-up) / escalate (up the chain)
# -------------------

"""Orchestrator — the chain of command in action.

This is the "say go and walk away" entry point. Give it intent and
constraints, it handles everything: spec generation, risk classification,
execution, evaluation, iteration, and escalation.

Every spec, report, and evaluation is dual-pass ingested to the Graph.
The system gets smarter with every mission.
"""

import json
import requests
import os
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orchestrator")

# Worktree base directory — where mission worktrees get created
WORKTREE_BASE = Path("/tmp/tqb-worktrees")


class Orchestrator:
    """Chain of command orchestrator.

    Ties together: Strategist (spec gen) → Risk classifier → Executor →
    Reviewer (evaluation) → iteration loop with authority-bounded retries.
    """

    def __init__(self, spec_executor, worker_ng, workspace: Path):
        """
        Args:
            spec_executor: SpecExecutor instance (from spec_executor.py)
            worker_ng: NeuroGraphMemory instance for Graph context and ingestion
            workspace: Default workspace path
        """
        self.executor = spec_executor
        self.ng = worker_ng
        self.workspace = workspace

    def orchestrate(
        self,
        intent: str,
        constraints: Optional[str] = None,
        workspace: Optional[str] = None,
        graph_query: Optional[str] = None,
        max_iterations: int = 5,
        use_worktree: bool = True,
        cleanup_on_fail: bool = False,
    ) -> dict:
        """Run a full mission from intent to completion.

        Args:
            intent: What to build/fix/audit (natural language)
            constraints: Project-specific rules (natural language)
            workspace: Override workspace for cross-repo work
            graph_query: Custom query for Graph recall (defaults to intent)
            max_iterations: Max spec→execute→evaluate cycles before mandatory escalation
            use_worktree: If True and workspace is a git repo, create an isolated worktree
            cleanup_on_fail: If True, remove worktree on failure. If False, leave for inspection.

        Returns:
            {
                status: "complete" | "escalated" | "failed",
                iterations: int,
                final_report: dict | None,
                final_evaluation: dict | None,
                history: [{spec, report, evaluation}, ...],
                escalation: {to, reason} | None,
                worktree: {path, branch, source_branch, merged} | None,
                elapsed_seconds: float,
            }
        """
        from spec_generator import generate_spec, generate_followup_spec
        from report_evaluator import evaluate_report
        from risk_authority import classify_risk, get_authority, get_retry_budget, get_escalation_target
        from worker_ng import ingest_tool_result

        start_time = time.time()
        history = []
        ws = workspace or str(self.workspace)
        worktree_info = None

        # Create isolated worktree if the target is a git repo
        if use_worktree:
            worktree_info = self._create_worktree(ws, intent)
            if worktree_info:
                ws = worktree_info["path"]
                logger.info("Working in worktree: %s (branch: %s)", ws, worktree_info["branch"])

        # Query Graph for relevant context
        graph_context = self._recall(graph_query or intent)

        logger.info("=== MISSION START ===")
        logger.info("Intent: %s", intent[:200])

        for iteration in range(max_iterations):
            logger.info("--- Iteration %d/%d ---", iteration + 1, max_iterations)

            # Step 1: Generate spec (first time) or follow-up spec (retry)
            if iteration == 0:
                gen_result = generate_spec(
                    intent=intent,
                    constraints=constraints,
                    graph_context=graph_context,
                    workspace=ws,
                )
            else:
                prev = history[-1]
                gen_result = generate_followup_spec(
                    original_spec=prev["spec"],
                    failed_report=prev["report"],
                    evaluation=prev["evaluation"],
                    graph_context=graph_context,
                )

            if not gen_result["success"]:
                logger.error("Spec generation failed: %s", gen_result.get("errors"))
                if worktree_info and cleanup_on_fail:
                    self._cleanup_worktree(worktree_info)
                    worktree_info["cleaned_up"] = True
                return self._build_result(
                    "failed", iteration + 1, None, None, history,
                    {"to": "josh", "reason": f"Spec generation failed: {gen_result.get('errors')}"},
                    start_time, worktree_info,
                )

            spec = gen_result["spec"]
            logger.info("Spec generated: %s", spec.get("block", {}).get("name", "?"))

            # Step 2: Classify risk
            risk = classify_risk(spec, graph_context=graph_context)
            authority = get_authority(risk)
            retry_budget = get_retry_budget(risk)
            logger.info("Risk: %s | Approver: %s | Retries: %d",
                       risk, authority["approver"], retry_budget)

            # Step 3: Execute spec
            report = self.executor.execute_block(spec)
            logger.info("Execution status: %s | Steps: %d passed, %d failed",
                       report.get("status"),
                       report.get("summary", {}).get("passed", 0),
                       report.get("summary", {}).get("failed", 0))

            # Step 4: Evaluate report
            evaluation = evaluate_report(
                report=report,
                spec=spec,
                graph_context=graph_context,
                use_persona=True,
            )
            decision = evaluation["decision"]
            logger.info("Evaluation decision: %s", decision)

            # Record this iteration
            entry = {
                "iteration": iteration + 1,
                "spec": spec,
                "report": report,
                "evaluation": evaluation,
                "risk": risk,
            }
            history.append(entry)

            # Ingest everything to the Graph
            self._ingest_iteration(entry, intent)

            # Step 5: Act on decision
            if decision == "done":
                logger.info("=== MISSION COMPLETE === (%d iterations)", iteration + 1)
                # Merge worktree back to source branch on success
                if worktree_info:
                    merged = self._merge_worktree(worktree_info)
                    worktree_info["merged"] = merged
                return self._build_result(
                    "complete", iteration + 1, report, evaluation, history, None,
                    start_time, worktree_info,
                )

            elif decision == "escalate":
                escalate_to = evaluation.get("escalate_to", "josh")
                reason = evaluation.get("escalate_reason", "Reviewer escalation")
                logger.warning("=== ESCALATED to %s: %s ===", escalate_to, reason)
                if worktree_info and cleanup_on_fail:
                    self._cleanup_worktree(worktree_info)
                    worktree_info["cleaned_up"] = True
                return self._build_result(
                    "escalated", iteration + 1, report, evaluation, history,
                    {"to": escalate_to, "reason": reason},
                    start_time, worktree_info,
                )

            elif decision == "iterate":
                # Check authority-bounded retry budget
                iteration_count_at_this_risk = sum(
                    1 for h in history if h.get("risk") == risk
                )
                if iteration_count_at_this_risk >= retry_budget:
                    escalate_to = get_escalation_target(risk)
                    reason = (f"Retry budget exhausted ({retry_budget} retries at {risk} level). "
                              f"Hints: {evaluation.get('iteration_hints', [])}")
                    logger.warning("=== RETRY BUDGET EXHAUSTED — escalating to %s ===", escalate_to)
                    return self._build_result(
                        "escalated", iteration + 1, report, evaluation, history,
                        {"to": escalate_to, "reason": reason},
                        start_time,
                    )

                logger.info("Iterating. Hints: %s", evaluation.get("iteration_hints", []))
                # Refresh Graph context with latest learnings
                graph_context = self._recall(graph_query or intent)
                continue

        # Max iterations hit
        logger.warning("=== MAX ITERATIONS (%d) — escalating to josh ===", max_iterations)
        if worktree_info and cleanup_on_fail:
            self._cleanup_worktree(worktree_info)
            worktree_info["cleaned_up"] = True
        return self._build_result(
            "escalated", max_iterations,
            history[-1]["report"] if history else None,
            history[-1]["evaluation"] if history else None,
            history,
            {"to": "josh", "reason": f"Max iterations ({max_iterations}) reached without completion"},
            start_time, worktree_info,
        )

    # ------------------------------------------------------------------
    # Git worktree lifecycle
    # ------------------------------------------------------------------

    def _is_git_repo(self, path: str) -> bool:
        """Check if a path is inside a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=path, capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _get_current_branch(self, repo_path: str) -> str:
        """Get the current branch name of a repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path, capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() or "main"
        except (subprocess.TimeoutExpired, OSError):
            return "main"

    def _create_worktree(self, repo_path: str, intent: str) -> Optional[dict]:
        """Create an isolated git worktree for this mission.

        Returns worktree info dict or None if not a git repo.
        """
        if not self._is_git_repo(repo_path):
            logger.info("Workspace %s is not a git repo — skipping worktree", repo_path)
            return None

        source_branch = self._get_current_branch(repo_path)

        # Generate a unique branch name from intent
        slug = intent[:40].lower()
        slug = "".join(c if c.isalnum() or c == "-" else "-" for c in slug)
        slug = slug.strip("-")[:30]
        timestamp = int(time.time())
        branch = f"tqb/{slug}-{timestamp}"

        worktree_path = WORKTREE_BASE / f"{slug}-{timestamp}"
        WORKTREE_BASE.mkdir(parents=True, exist_ok=True)

        try:
            # Create the worktree with a new branch from current HEAD
            subprocess.run(
                ["git", "worktree", "add", "-b", branch, str(worktree_path)],
                cwd=repo_path, capture_output=True, text=True,
                check=True, timeout=30,
            )
            logger.info("Created worktree: %s (branch: %s)", worktree_path, branch)
            return {
                "path": str(worktree_path),
                "branch": branch,
                "source_branch": source_branch,
                "repo_path": repo_path,
                "merged": False,
                "cleaned_up": False,
            }
        except subprocess.CalledProcessError as e:
            logger.error("Failed to create worktree: %s", e.stderr)
            return None
        except subprocess.TimeoutExpired:
            logger.error("Worktree creation timed out")
            return None

    def _merge_worktree(self, info: dict) -> bool:
        """Merge the worktree's branch back to the source branch.

        Returns True if merge succeeded.
        """
        repo_path = info["repo_path"]
        branch = info["branch"]
        source = info["source_branch"]
        worktree_path = info["path"]

        try:
            # Check if there are any changes to merge
            result = subprocess.run(
                ["git", "diff", "--stat", f"{source}...{branch}"],
                cwd=repo_path, capture_output=True, text=True, timeout=10,
            )
            if not result.stdout.strip():
                logger.info("No changes to merge from %s", branch)
                self._cleanup_worktree(info)
                return True

            # Commit any uncommitted work in the worktree first
            subprocess.run(
                ["git", "add", "-A"],
                cwd=worktree_path, capture_output=True, text=True, timeout=10,
            )
            commit_result = subprocess.run(
                ["git", "commit", "-m", f"TQB mission: {branch}"],
                cwd=worktree_path, capture_output=True, text=True, timeout=10,
            )
            if commit_result.returncode == 0:
                logger.info("Committed pending changes in worktree")

            # Merge into source branch
            merge_result = subprocess.run(
                ["git", "merge", "--no-ff", branch, "-m", f"Merge TQB mission: {branch}"],
                cwd=repo_path, capture_output=True, text=True, timeout=30,
            )

            if merge_result.returncode == 0:
                logger.info("Merged %s into %s", branch, source)
                self._cleanup_worktree(info)
                return True
            else:
                logger.error("Merge failed: %s", merge_result.stderr)
                # Leave worktree for manual resolution
                return False

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            logger.error("Merge error: %s", e)
            return False

    def _cleanup_worktree(self, info: dict):
        """Remove a worktree and optionally delete its branch."""
        worktree_path = info["path"]
        branch = info["branch"]
        repo_path = info["repo_path"]

        try:
            # Remove the worktree
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=repo_path, capture_output=True, text=True, timeout=15,
            )
            logger.info("Removed worktree: %s", worktree_path)

            # Delete the branch if it was merged
            if info.get("merged"):
                subprocess.run(
                    ["git", "branch", "-d", branch],
                    cwd=repo_path, capture_output=True, text=True, timeout=5,
                )
                logger.info("Deleted branch: %s", branch)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Worktree cleanup error: %s", e)
            # Fallback: try to remove the directory
            try:
                if Path(worktree_path).exists():
                    shutil.rmtree(worktree_path)
                    logger.info("Removed worktree directory via shutil: %s", worktree_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Graph integration
    # ------------------------------------------------------------------

    def _recall(self, query: str, k: int = 5) -> list:
        """Recall relevant context from the Graph."""
        try:
            return self.ng.recall(query, k=k, threshold=0.35)
        except Exception as e:
            logger.warning("Graph recall failed: %s", e)
            return []

    def _ingest_iteration(self, entry: dict, intent: str):
        """Dual-pass ingest an iteration's results into the Graph."""
        try:
            from worker_ng import ingest_tool_result

            # Forest: the full iteration as gestalt
            iteration_summary = (
                f"Mission: {intent[:200]}\n"
                f"Iteration {entry['iteration']}: "
                f"Risk={entry['risk']}, "
                f"Status={entry['report'].get('status')}, "
                f"Decision={entry['evaluation']['decision']}, "
                f"Steps passed={entry['report'].get('summary', {}).get('passed', 0)}, "
                f"Steps failed={entry['report'].get('summary', {}).get('failed', 0)}"
            )
            self.ng.on_message(iteration_summary)

            # Trees: individual findings
            hints = entry["evaluation"].get("iteration_hints", [])
            for hint in hints:
                self.ng.on_message(f"iteration_hint: {hint}")

            # Ingest failed step reasons as individual concepts
            for sid, r in entry["report"].get("step_results", {}).items():
                if r.get("status") == "fail":
                    self.ng.on_message(f"step_failure: {sid} — {r.get('reason', 'unknown')}")

        except Exception as e:
            logger.warning("Graph ingestion of iteration failed: %s", e)

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _notify_discord(self, result: dict) -> None:
        """Post mission completion notification to Discord via webhook.

        Reads QB_DISCORD_WEBHOOK from environment.  Silent on failure —
        a dead webhook must never interrupt or surface as a mission error.
        """
        webhook_url = os.environ.get("QB_DISCORD_WEBHOOK", "").strip()
        if not webhook_url:
            return
        try:
            status = result.get("status", "unknown")
            iterations = result.get("iterations", 0)
            elapsed = result.get("elapsed_seconds", 0)
            # Pull block name from last history entry if available
            history = result.get("history", [])
            block_name = history[-1].get("block_name", "unknown") if history else "unknown"
            steps_passed = history[-1].get("steps_passed", 0) if history else 0
            steps_failed = history[-1].get("steps_failed", 0) if history else 0
            escalation = result.get("escalation")
            status_emoji = {"complete": "✅", "escalated": "🚨"}.get(status, "⚠️")
            lines = [
                f"{status_emoji} **QB Mission {status.upper()}**",
                f"Block: `{block_name}`",
                f"Iterations: {iterations} | Steps: {steps_passed} passed, {steps_failed} failed | Elapsed: {elapsed}s",
            ]
            if escalation:
                reason = escalation.get("reason", "")
                to = escalation.get("to", "josh")
                lines.append(f"Escalated to **{to}**: {reason}")
            payload = {"content": "\n".join(lines), "username": "Queen Bitch"}
            requests.post(webhook_url, json=payload, timeout=5)
        except Exception as exc:
            logger.warning("Discord webhook notification failed: %s", exc)

    def _build_result(self, status, iterations, final_report, final_evaluation,
                      history, escalation, start_time, worktree_info=None):
        elapsed = round(time.time() - start_time, 2)

        result = {
            "status": status,
            "iterations": iterations,
            "final_report": final_report,
            "final_evaluation": final_evaluation,
            "history": [
                {
                    "iteration": h["iteration"],
                    "risk": h["risk"],
                    "block_name": h["spec"].get("block", {}).get("name", "?"),
                    "exec_status": h["report"].get("status"),
                    "eval_decision": h["evaluation"]["decision"],
                    "steps_passed": h["report"].get("summary", {}).get("passed", 0),
                    "steps_failed": h["report"].get("summary", {}).get("failed", 0),
                }
                for h in history
            ],
            "escalation": escalation,
            "worktree": worktree_info,
            "elapsed_seconds": elapsed,
        }

        # Log the audit trail
        try:
            audit_dir = self.workspace / "data" / "audit" if isinstance(self.workspace, Path) else Path(self.workspace) / "data" / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            with open(audit_dir / "missions.jsonl", "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
        except OSError as e:
            logger.warning("Failed to write mission audit: %s", e)

        self._notify_discord(result)
        return result
