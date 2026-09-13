"""Reusable model/tool loop for Codemine chat and TQB specialists."""

# ---- Changelog ----
# [2026-09-12] Codex — Extract the existing Codemine model/tool loop.
# What: Run model responses, tool calls, staged actions, and final summaries
#       through one injectable function shared by Chat and specialist roles.
# Why: TQB specialists need the same proven tool loop against their own isolated
#      workspaces; a second implementation would drift from the Chat behavior.
# How: Preserve Anthropic-shaped content blocks and delegate policy/tool behavior
#      to the caller-provided executor.
# -------------------

from __future__ import annotations

import time
import uuid
from typing import Callable


def _model_receipt(response) -> dict:
    usage = getattr(response, "usage", None)
    return {
        "provider": getattr(response, "provider", None),
        "model": getattr(response, "model", None),
        "input_tokens": (
            getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0
            if usage is not None else 0
        ),
        "output_tokens": (
            getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0
            if usage is not None else 0
        ),
        "cost": getattr(usage, "cost", None) if usage is not None else None,
    }


def run_tool_loop(
    *,
    client,
    system_prompt: str,
    messages: list,
    tools: list,
    call_model_fn: Callable,
    execute_tool_fn: Callable[[str, dict], dict],
    truncate_result_fn: Callable[[str], str],
    max_iterations: int,
) -> dict:
    """Run one model turn until it answers, stages work, or hits its hard cap."""
    api_messages = list(messages)
    accumulated_text = ""
    staged_actions = []
    tool_events = []
    model_calls = []
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1
        response = call_model_fn(client, system_prompt, api_messages, tools)
        model_calls.append(_model_receipt(response))
        assistant_content = []
        tool_blocks = []

        for block in response.content:
            if block.type == "text":
                accumulated_text += ("\n\n" if accumulated_text else "") + block.text
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_blocks.append(block)
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        if not tool_blocks or iteration >= max_iterations - 1:
            break

        api_messages.append({"role": "assistant", "content": assistant_content})
        tool_results = []

        for block in tool_blocks:
            result = execute_tool_fn(block.name, block.input)
            status = result.get("status", "error")
            event = {"tool": block.name, "status": status}

            if status == "executed":
                result_text = truncate_result_fn(str(result.get("result", "")))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )
                event["result_preview"] = str(result.get("result", ""))[:100]
            elif status == "staged":
                proposal = {
                    "id": f"p_{uuid.uuid4().hex}",
                    "tool": block.name,
                    "args": result.get("args", block.input),
                    "description": result.get("description", f"Staged: {block.name}"),
                    "timestamp": time.strftime("%H:%M:%S"),
                }
                staged_actions.append(proposal)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "This tool has been staged for review. Awaiting approval.",
                    }
                )
            else:
                error_text = str(result.get("result", "Unknown error"))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {error_text}",
                        "is_error": True,
                    }
                )
                event["result_preview"] = error_text[:100]

            tool_events.append(event)

        api_messages.append({"role": "user", "content": tool_results})

    if not accumulated_text.strip() and tool_events:
        api_messages.append(
            {
                "role": "user",
                "content": "You executed tools but produced no explanation. Summarize the results.",
            }
        )
        try:
            response = call_model_fn(client, system_prompt, api_messages, [])
            model_calls.append(_model_receipt(response))
            for block in response.content:
                if block.type == "text":
                    accumulated_text = block.text
                    break
        except (OSError, ConnectionError, ValueError):
            accumulated_text = "Actions completed."

    return {
        "text": accumulated_text,
        "staged_actions": staged_actions,
        "tool_events": tool_events,
        "model_calls": model_calls,
        "messages": api_messages,
        "iterations": iterations,
    }
