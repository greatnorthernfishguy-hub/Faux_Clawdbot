"""Behavior tests for the shared Codemine model/tool loop."""

# ---- Changelog ----
# [2026-09-12] Codex — Pin shared agent-runner behavior.
# What: Cover executed tools, staged gates, errors, and tool-only summaries.
# Why: Chat and every TQB specialist must use identical loop semantics.
# How: Deterministic fake response blocks and injected tool executors.
# -------------------

from types import SimpleNamespace

from agent_runner import run_tool_loop


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool(name="read_file", tool_id="tool-1", **arguments):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=arguments)


def response(*blocks):
    return SimpleNamespace(content=list(blocks))


def run(responses, executor):
    calls = []

    def model_call(_client, _system, messages, tools):
        calls.append({"messages": list(messages), "tools": list(tools)})
        return responses.pop(0)

    result = run_tool_loop(
        client=object(),
        system_prompt="system",
        messages=[{"role": "user", "content": "task"}],
        tools=[{"name": "read_file"}],
        call_model_fn=model_call,
        execute_tool_fn=executor,
        truncate_result_fn=lambda value: value[:20],
        max_iterations=5,
    )
    return result, calls


def test_executes_tool_and_returns_followup_text():
    result, calls = run(
        [response(tool(path="README.md")), response(text("Done with evidence."))],
        lambda name, args: {"status": "executed", "result": "file contents"},
    )

    assert result["text"] == "Done with evidence."
    assert result["tool_events"] == [
        {"tool": "read_file", "status": "executed", "result_preview": "file contents"}
    ]
    assert calls[1]["messages"][-1]["content"][0]["content"] == "file contents"


def test_staged_tool_is_returned_without_execution_claim():
    result, _calls = run(
        [response(tool(name="write_file", path="x")), response(text("Awaiting review."))],
        lambda name, args: {
            "status": "staged",
            "args": args,
            "description": "Write x",
        },
    )

    assert result["text"] == "Awaiting review."
    assert len(result["staged_actions"]) == 1
    assert result["staged_actions"][0]["tool"] == "write_file"
    assert result["tool_events"][0]["status"] == "staged"


def test_tool_error_is_returned_to_model():
    result, calls = run(
        [response(tool()), response(text("The read failed."))],
        lambda name, args: {"status": "error", "result": "permission denied"},
    )

    assert result["text"] == "The read failed."
    tool_result = calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert tool_result["content"] == "Error: permission denied"


def test_tool_only_run_requests_final_summary():
    result, calls = run(
        [response(tool()), response(), response(text("Observed the file."))],
        lambda name, args: {"status": "executed", "result": "observed"},
    )

    assert result["text"] == "Observed the file."
    assert calls[-1]["tools"] == []
    assert "Summarize the results" in calls[-1]["messages"][-1]["content"]
