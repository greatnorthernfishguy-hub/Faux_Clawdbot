# ---- Changelog ----
# [2026-04-05] Josh + Claude — Structured spec execution engine
# What: Deterministic execution of WorkBlockSpec JSON specs
# Why: No LLM in the loop — mechanical step execution, mandatory validation, zero improvisation
# How: State machine over step types (action/gate/condition/loop/group), variable bindings,
#      uses existing TOOL_REGISTRY and PolicyEngine, produces structured execution reports
# -------------------

"""Spec Executor — mechanical execution of structured work block specs.

The spec IS the plan. The executor does not interpret, reason, or improvise.
It executes steps top-to-bottom, validates after every action, and reports results.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from work_block_schema import validate_spec

logger = logging.getLogger("spec_executor")


@dataclass
class ExecutionContext:
    """Tracks state across a block's execution."""
    block_id: str
    bindings: dict = field(default_factory=dict)       # $var -> result string
    step_results: dict = field(default_factory=dict)    # step_id -> {status, ...}
    tool_call_count: int = 0
    start_time: float = field(default_factory=time.time)
    aborted: bool = False
    abort_reason: str = ""


class SpecExecutor:
    """Deterministic executor for WorkBlockSpec JSON specs.

    Uses the same TOOL_REGISTRY and PolicyEngine as the conversational agent loop.
    No LLM calls — this is a mechanical state machine.
    """

    def __init__(self, tool_registry: dict, policy_check_fn, worker_ng, workspace: Path):
        """
        Args:
            tool_registry: The TOOL_REGISTRY dict from app.py (name -> lambda handler)
            policy_check_fn: The check_tool_call function from policy_engine
            worker_ng: NeuroGraphMemory instance for substrate ingestion
            workspace: Path to the workspace root
        """
        self.tools = tool_registry
        self.policy_check = policy_check_fn
        self.ng = worker_ng
        self.workspace = workspace

    def execute_block(self, spec: dict) -> dict:
        """Execute a work block spec. Returns a structured execution report."""

        # 1. Schema validation
        valid, errors = validate_spec(spec)
        if not valid:
            return {
                "report_version": "1.0.0",
                "block_id": spec.get("block", {}).get("id", "UNKNOWN"),
                "status": "rejected",
                "errors": errors,
            }

        block = spec["block"]
        constraints = spec["constraints"]
        ctx = ExecutionContext(block_id=block["id"])

        logger.info("Executing block %s: %s", block["id"], block["name"])

        # 2. Execute steps
        for step in spec["steps"]:
            if ctx.aborted:
                break
            if ctx.tool_call_count >= constraints.get("max_iterations", 15):
                ctx.aborted = True
                ctx.abort_reason = f"Max iterations ({constraints.get('max_iterations', 15)}) exceeded"
                break
            elapsed = time.time() - ctx.start_time
            if elapsed > constraints.get("timeout_seconds", 300):
                ctx.aborted = True
                ctx.abort_reason = f"Block timeout ({constraints.get('timeout_seconds', 300)}s) exceeded"
                break

            self._execute_step(step, ctx, constraints)

        # 3. Build report
        return self._build_report(spec, ctx)

    # ------------------------------------------------------------------
    # Step dispatch
    # ------------------------------------------------------------------

    def _execute_step(self, step: dict, ctx: ExecutionContext, constraints: dict):
        step_type = step["type"]
        dispatch = {
            "action": self._execute_action,
            "gate": self._execute_gate,
            "condition": self._execute_condition,
            "loop": self._execute_loop,
            "group": self._execute_group,
        }
        handler = dispatch.get(step_type)
        if handler:
            handler(step, ctx, constraints)
        else:
            ctx.step_results[step["id"]] = {"status": "fail", "reason": f"Unknown step type: {step_type}"}

    # ------------------------------------------------------------------
    # Action steps
    # ------------------------------------------------------------------

    def _execute_action(self, step: dict, ctx: ExecutionContext, constraints: dict):
        tool_name = step["tool"]
        step_id = step["id"]
        desc = step.get("description", tool_name)

        logger.info("  [%s] %s → %s", step_id, desc, tool_name)

        # Tool allowlist check
        allowlist = constraints.get("tool_allowlist")
        if allowlist and tool_name not in allowlist:
            self._record_failure(ctx, step_id, f"Tool '{tool_name}' not in block allowlist")
            self._handle_failure(step["on_failure"], step_id, ctx)
            return

        # Resolve $var references in params
        params = self._resolve_bindings(step["params"], ctx)

        # PolicyEngine rim check
        allowed, reason = self.policy_check(tool_name, params, self.workspace)
        if not allowed:
            self._record_failure(ctx, step_id, f"PolicyEngine denied: {reason}")
            self._handle_failure(step["on_failure"], step_id, ctx)
            return

        # Execute tool
        ctx.tool_call_count += 1
        handler = self.tools.get(tool_name)
        if not handler:
            self._record_failure(ctx, step_id, f"Unknown tool: {tool_name}")
            self._handle_failure(step["on_failure"], step_id, ctx)
            return

        try:
            result = handler(params)
        except Exception as e:
            logger.error("  [%s] Tool exception: %s", step_id, e, exc_info=True)
            self._record_failure(ctx, step_id, f"Tool exception: {type(e).__name__}: {e}")
            self._handle_failure(step["on_failure"], step_id, ctx)
            return

        # Coerce result to string
        if isinstance(result, dict):
            if result.get("status") == "error":
                self._record_failure(ctx, step_id, f"Tool error: {result.get('error', 'Unknown')}")
                self._handle_failure(step["on_failure"], step_id, ctx)
                return
            result_str = json.dumps(result, default=str)
        else:
            result_str = str(result)

        # Bind result
        if "bind_result" in step:
            ctx.bindings[step["bind_result"]] = result_str

        # Run validation checks
        passed = self._run_validation(step["validation"], result_str, ctx)

        ctx.step_results[step_id] = {
            "status": "pass" if passed else "fail",
            "tool": tool_name,
            "result_preview": result_str[:500],
            "validation_passed": passed,
        }

        if not passed:
            logger.warning("  [%s] Validation failed", step_id)
            self._handle_failure(step["on_failure"], step_id, ctx)

        # Ingest into worker substrate
        try:
            from worker_ng import ingest_tool_result
            ingest_tool_result(self.ng, tool_name, params, result_str)
        except Exception as e:
            logger.warning("  [%s] Substrate ingestion failed: %s", step_id, e)

    # ------------------------------------------------------------------
    # Gate steps
    # ------------------------------------------------------------------

    def _execute_gate(self, step: dict, ctx: ExecutionContext, _constraints: dict):
        gate_type = step["gate_type"]
        step_id = step["id"]

        logger.info("  [%s] Gate: %s (%s)", step_id, step["description"], gate_type)

        if gate_type == "auto_approve":
            ctx.step_results[step_id] = {"status": "auto_approved", "gate_type": gate_type}
            return

        # For human_review and qb_checkpoint, record as pending.
        # The execution report captures the checkpoint state.
        # QB or human resumes by re-submitting from the next step.
        ctx.step_results[step_id] = {
            "status": f"pending_{gate_type}",
            "gate_type": gate_type,
            "staged_actions": step.get("staged_actions", []),
            "description": step["description"],
        }

        # Gates don't abort — they pause. QB decides whether to continue.
        # For now, we continue execution past gates (QB reviews the report after).

    # ------------------------------------------------------------------
    # Condition steps
    # ------------------------------------------------------------------

    def _execute_condition(self, step: dict, ctx: ExecutionContext, constraints: dict):
        step_id = step["id"]
        check_result = self._evaluate_check(step["check"], None, ctx)
        branch = step["if_true"] if check_result else step["if_false"]

        logger.info("  [%s] Condition → %s", step_id, "if_true" if check_result else "if_false")

        ctx.step_results[step_id] = {
            "status": "evaluated",
            "branch_taken": "if_true" if check_result else "if_false",
        }

        for sub_step in branch:
            if ctx.aborted:
                break
            self._execute_step(sub_step, ctx, constraints)

    # ------------------------------------------------------------------
    # Loop steps
    # ------------------------------------------------------------------

    def _execute_loop(self, step: dict, ctx: ExecutionContext, constraints: dict):
        step_id = step["id"]
        over = step["over"]
        bind_item = step.get("bind_item", "$item")
        max_iter = step.get("max_iterations", 20)

        # Resolve items
        if "items" in over:
            items = over["items"]
        elif "from_result" in over:
            raw = ctx.bindings.get(over["from_result"], "")
            delimiter = over.get("split_on", "\n")
            items = [i.strip() for i in raw.split(delimiter) if i.strip()]
        else:
            items = []

        items = items[:max_iter]
        logger.info("  [%s] Loop over %d items", step_id, len(items))

        for item in items:
            if ctx.aborted:
                break
            ctx.bindings[bind_item] = item
            for sub_step in step["body"]:
                if ctx.aborted:
                    break
                self._execute_step(sub_step, ctx, constraints)

        ctx.step_results[step_id] = {"status": "completed", "iterations": len(items)}

    # ------------------------------------------------------------------
    # Group steps
    # ------------------------------------------------------------------

    def _execute_group(self, step: dict, ctx: ExecutionContext, constraints: dict):
        step_id = step["id"]
        logger.info("  [%s] Group: %s", step_id, step.get("description", ""))

        for sub_step in step["steps"]:
            if ctx.aborted:
                break
            self._execute_step(sub_step, ctx, constraints)

        group_ok = all(
            ctx.step_results.get(s["id"], {}).get("status") != "fail"
            for s in step["steps"]
            if s["id"] in ctx.step_results
        )
        ctx.step_results[step_id] = {"status": "pass" if group_ok else "fail"}

        if not group_ok and "on_failure" in step:
            self._handle_failure(step["on_failure"], step_id, ctx)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self, validation: dict, result: str, ctx: ExecutionContext) -> bool:
        for check in validation["checks"]:
            if not self._evaluate_check(check, result, ctx):
                desc = check.get("description", check["operator"])
                logger.warning("    Validation check failed: %s", desc)
                return False
        return True

    def _evaluate_check(self, check: dict, default_target: Optional[str], ctx: ExecutionContext) -> bool:
        # Resolve target
        target_ref = check.get("target")
        if target_ref and target_ref.startswith("$"):
            target = ctx.bindings.get(target_ref, "")
        else:
            target = default_target or ""

        op = check["operator"]
        value = check.get("value", "")

        if op == "contains":
            return str(value) in target
        elif op == "not_contains":
            return str(value) not in target
        elif op == "equals":
            return target.strip() == str(value).strip()
        elif op == "not_equals":
            return target.strip() != str(value).strip()
        elif op == "matches_regex":
            return bool(re.search(str(value), target))
        elif op == "result_is_string":
            return isinstance(target, str)
        elif op == "result_is_not_error":
            return "Error:" not in target and '"status": "error"' not in target
        elif op == "file_exists":
            return (self.workspace / str(value)).exists()
        elif op == "file_contains":
            # value format: "path::substring"
            parts = str(value).split("::", 1)
            if len(parts) == 2:
                try:
                    content = (self.workspace / parts[0]).read_text(errors="ignore")
                    return parts[1] in content
                except OSError:
                    return False
            return False
        elif op == "output_length_gt":
            return len(target) > int(value)
        elif op == "output_length_lt":
            return len(target) < int(value)

        logger.warning("    Unknown operator: %s — failing safe", op)
        return False

    # ------------------------------------------------------------------
    # Bindings
    # ------------------------------------------------------------------

    def _resolve_bindings(self, params: dict, ctx: ExecutionContext) -> dict:
        """Replace $var references in param values with bound results."""
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("$"):
                resolved[k] = ctx.bindings.get(v, v)
            else:
                resolved[k] = v
        return resolved

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def _handle_failure(self, handler: dict, step_id: str, ctx: ExecutionContext):
        action = handler["action"]
        msg = handler.get("message", f"Step {step_id} failed")

        if action == "abort_block":
            ctx.aborted = True
            ctx.abort_reason = msg
            logger.error("  ABORT: %s", msg)
        elif action == "retry":
            # Retry is handled by the caller re-invoking _execute_action
            # For now, log it — real retry needs a loop wrapper
            logger.info("  RETRY requested for %s (not yet implemented in executor)", step_id)
        elif action == "skip":
            logger.info("  SKIP: %s — continuing", msg)
        elif action == "goto":
            # goto requires step index lookup — deferred
            logger.info("  GOTO %s requested (not yet implemented)", handler.get("goto_step"))
        elif action == "escalate_to_qb":
            ctx.aborted = True
            ctx.abort_reason = f"ESCALATED: {msg}"
            logger.warning("  ESCALATE TO QB: %s", msg)

    def _record_failure(self, ctx: ExecutionContext, step_id: str, reason: str):
        ctx.step_results[step_id] = {"status": "fail", "reason": reason}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _build_report(self, spec: dict, ctx: ExecutionContext) -> dict:
        """Build the execution report for QB."""
        total = len(ctx.step_results)
        passed = sum(1 for r in ctx.step_results.values() if r.get("status") == "pass")
        failed = sum(1 for r in ctx.step_results.values() if r.get("status") == "fail")
        pending = sum(1 for r in ctx.step_results.values() if "pending" in r.get("status", ""))

        return {
            "report_version": "1.0.0",
            "block_id": spec["block"]["id"],
            "block_name": spec["block"]["name"],
            "agent": spec["block"].get("agent", "unnamed"),
            "status": "aborted" if ctx.aborted else ("complete" if failed == 0 else "partial_failure"),
            "abort_reason": ctx.abort_reason if ctx.aborted else None,
            "summary": {
                "total_steps": total,
                "passed": passed,
                "failed": failed,
                "pending_review": pending,
                "tool_calls": ctx.tool_call_count,
                "elapsed_seconds": round(time.time() - ctx.start_time, 2),
            },
            "step_results": ctx.step_results,
            "bindings": {k: v[:200] for k, v in ctx.bindings.items()},
            "acceptance_criteria": spec["block"]["acceptance_criteria"],
        }
