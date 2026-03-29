# ---- Changelog ----
# [2026-02-04] Josh — PLATINUM COPY: Original Clawdbot Unified Command Center
# [2026-03-29] Hammer (TQB) — Block G: Agent Loop Hardening + Assembly
# What: Complete rewrite — Claude native tool_use, PolicyEngine gates, tool registry,
#        context window management, ZIP bomb protection, structured error handling
# Why: PRD §7.G — zero regex in tool parsing, all mutating tools gated, hard max iterations
# How: Anthropic SDK messages.create() with tools param; tool_use/tool_result block loop;
#        PolicyEngine.check_tool_call() on every tool; TOOL_REGISTRY dict dispatch
# -------------------

import gradio as gr
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path

from recursive_context import RecursiveContextManager
from policy_engine import check_tool_call, should_gate_for_review
from model_client import get_client, call_model
from system_prompt import build_system_prompt
from tool_definitions import TOOL_DEFINITIONS
from worker_ng import get_worker_ng, ingest_tool_result, recall_context

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clawdbot_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Clawdbot")

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

REPO_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
ctx = RecursiveContextManager(str(REPO_PATH))
client = get_client()
worker_ng = get_worker_ng()

TEXT_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml',
    '.md', '.txt', '.rst', '.html', '.css', '.scss', '.sh', '.bash',
    '.sql', '.toml', '.cfg', '.ini', '.conf', '.xml', '.csv',
    '.env', '.gitignore', '.dockerfile'
}

# Maximum extracted ZIP size (bytes) — ZIP bomb protection
MAX_ZIP_EXTRACT_SIZE = 100 * 1024 * 1024  # 100MB

# Maximum context messages (by estimated token count)
MAX_CONTEXT_TOKENS = 100_000
AVG_CHARS_PER_TOKEN = 4

# Maximum tool result size before summarization (chars)
MAX_TOOL_RESULT_SIZE = 10_000

MAX_ITERATIONS = 15

# ---------------------------------------------------------------------------
# Tool Registry — dispatch by name, no if/elif chain
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "read_file": lambda args: ctx.read_file(
        args.get("path", ""), args.get("start_line"), args.get("end_line")
    ),
    "write_file": lambda args: ctx.write_file(
        args.get("path", ""), args.get("content", "")
    ),
    "list_files": lambda args: ctx.list_files(
        args.get("path", "."), args.get("max_depth", 3)
    ),
    "search_code": lambda args: _format_code_results(
        ctx.search_code(args.get("query", ""), args.get("n", 5))
    ),
    "search_conversations": lambda args: _format_conversation_results(
        ctx.search_conversations(args.get("query", ""), args.get("n", 5))
    ),
    "search_testament": lambda args: _format_testament_results(
        ctx.search_testament(args.get("query", ""), args.get("n", 5))
    ),
    "ingest_workspace": lambda args: ctx.ingest_workspace(),
    "shell_execute": lambda args: ctx.shell_execute(args.get("command", "")),
    "push_to_github": lambda args: ctx.push_to_github(
        args.get("message", "Manual Backup")
    ),
    "pull_from_github": lambda args: ctx.pull_from_github(
        args.get("branch", "main")
    ),
    "create_shadow_branch": lambda args: ctx.create_shadow_branch(),
    "notebook_read": lambda args: ctx.notebook_read(),
    "notebook_add": lambda args: ctx.notebook_add(args.get("content", "")),
    "notebook_delete": lambda args: ctx.notebook_delete(args.get("index", 0)),
    "map_repository_structure": lambda args: ctx.map_repository_structure(),
    "get_stats": lambda args: ctx.get_stats(),
}


def _format_code_results(results: list) -> str:
    if not results:
        return "No matches found."
    return "\n".join(
        f"{r['file']}\n```\n{r['snippet']}\n```" for r in results
        if isinstance(r, dict) and "file" in r
    )


def _format_conversation_results(results: list) -> str:
    if not results:
        return "No matches found."
    return "\n---\n".join(
        r.get("content", str(r)) for r in results
        if isinstance(r, dict)
    )


def _format_testament_results(results: list) -> str:
    if not results:
        return "No matches found."
    return "\n\n".join(
        f"**{r['file']}**\n{r['snippet']}" for r in results
        if isinstance(r, dict) and "file" in r
    )


# ---------------------------------------------------------------------------
# Tool Execution — PolicyEngine gate + worker NG integration
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a tool call through PolicyEngine gate and worker NG.

    Returns dict with status, tool, result keys.
    """
    # PolicyEngine Rim check
    allowed, reason = check_tool_call(tool_name, args, REPO_PATH)
    if not allowed:
        logger.warning("Tool denied by PolicyEngine: %s — %s", tool_name, reason)
        return {"status": "error", "tool": tool_name, "result": f"Denied: {reason}"}

    # Mesh check — should this be staged for review?
    if should_gate_for_review(tool_name, args):
        return {
            "status": "staged",
            "tool": tool_name,
            "args": args,
            "description": f"Staged for review: {tool_name}",
        }

    # Recall past experience from worker substrate
    context_str = json.dumps(args, default=str)[:500]
    recalls = recall_context(worker_ng, tool_name, context_str)

    # Execute
    handler = TOOL_REGISTRY.get(tool_name)
    if not handler:
        return {"status": "error", "tool": tool_name, "result": f"Unknown tool: {tool_name}"}

    try:
        result = handler(args)
        # Coerce dict results (error dicts from tools) to string
        if isinstance(result, dict):
            if result.get("status") == "error":
                result_str = f"Error: {result.get('error', 'Unknown error')}"
            else:
                result_str = json.dumps(result, default=str)
        else:
            result_str = str(result)

        # Ingest tool result as raw experience into worker substrate (Law 7)
        ingest_tool_result(worker_ng, tool_name, args, result_str)

        return {"status": "executed", "tool": tool_name, "result": result_str}
    except Exception as e:
        logger.error("[%s] execution failed: %s: %s", tool_name, type(e).__name__, e, exc_info=True)
        error_result = f"Error: {type(e).__name__}: {e}"
        ingest_tool_result(worker_ng, tool_name, args, error_result)
        return {"status": "error", "tool": tool_name, "result": error_result}


def execute_staged_tool(tool_name: str, args: dict) -> str:
    """Execute a previously staged tool (user approved via gate)."""
    handler = TOOL_REGISTRY.get(tool_name)
    if not handler:
        return f"Unknown tool: {tool_name}"
    try:
        result = handler(args)
        if isinstance(result, dict) and result.get("status") == "error":
            return f"Error: {result.get('error', 'Unknown error')}"
        return str(result)
    except Exception as e:
        logger.error("[%s] staged execution failed: %s", tool_name, e, exc_info=True)
        return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# File Upload Processing (with ZIP bomb protection)
# ---------------------------------------------------------------------------

def process_uploaded_file(file) -> str:
    if file is None:
        return ""
    if isinstance(file, list):
        file = file[0] if len(file) > 0 else None
    if file is None:
        return ""

    file_path = file.name if hasattr(file, 'name') else str(file)
    file_name = os.path.basename(file_path)
    suffix = os.path.splitext(file_name)[1].lower()

    if suffix == '.zip':
        return _process_zip(file_path, file_name)

    if suffix in TEXT_EXTENSIONS or suffix == '':
        try:
            # File size guard
            size = os.path.getsize(file_path)
            if size > 10 * 1024 * 1024:
                return f"Uploaded: {file_name} (too large: {size:,} bytes, max 10MB)"

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if len(content) > 50000:
                content = content[:50000] + "\n...(truncated)"
            return f"**Uploaded: {file_name}**\n```\n{content}\n```"
        except OSError as e:
            return f"**Uploaded: {file_name}** (error reading: {e})"

    try:
        size = os.path.getsize(file_path)
    except OSError:
        size = 0
    return f"**Uploaded: {file_name}** (binary file, {size:,} bytes)"


def _process_zip(file_path: str, file_name: str) -> str:
    """Extract ZIP with bomb protection — check total size before extracting."""
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            # ZIP bomb protection: check total extracted size
            total_size = sum(info.file_size for info in z.infolist())
            if total_size > MAX_ZIP_EXTRACT_SIZE:
                return (
                    f"ZIP rejected: {file_name} — total extracted size "
                    f"({total_size:,} bytes) exceeds {MAX_ZIP_EXTRACT_SIZE:,} byte limit."
                )

            extract_to = REPO_PATH / "uploaded_assets" / file_name.replace(".zip", "")
            if extract_to.exists():
                shutil.rmtree(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)
            z.extractall(extract_to)

        file_list = [f.name for f in extract_to.glob('*')]
        preview = ", ".join(file_list[:10])
        return (
            f"**Unzipped: {file_name}**\n"
            f"Location: `{extract_to}`\nContents: {preview}\n"
            f"SYSTEM NOTE: Files extracted. Use list_files('{extract_to.name}') to explore."
        )
    except zipfile.BadZipFile:
        return f"Failed: {file_name} is not a valid ZIP file."
    except OSError as e:
        return f"Failed to unzip {file_name}: {e}"


# ---------------------------------------------------------------------------
# Context Window Management
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    return len(text) // AVG_CHARS_PER_TOKEN


def _truncate_tool_result(result: str) -> str:
    """Summarize oversized tool results."""
    if len(result) <= MAX_TOOL_RESULT_SIZE:
        return result
    return result[:MAX_TOOL_RESULT_SIZE] + f"\n...[truncated — {len(result):,} chars total]"


def _build_api_messages(history: list, system_prompt: str) -> list:
    """Build API message list with token-aware windowing.

    System prompt is always present. Oldest messages dropped first.
    """
    messages = []
    budget = MAX_CONTEXT_TOKENS - _estimate_tokens(system_prompt)

    # Walk backwards from most recent, adding messages until budget exhausted
    for msg in reversed(history):
        tokens = _estimate_tokens(msg.get("content", ""))
        if budget - tokens < 0 and messages:
            break
        budget -= tokens
        messages.append(msg)

    messages.reverse()
    return messages


# ---------------------------------------------------------------------------
# Agent Loop — Claude native tool_use
# ---------------------------------------------------------------------------

def agent_loop(message: str, history: list, pending_proposals: list, uploaded_file) -> tuple:
    safe_hist = history or []
    safe_props = pending_proposals or []

    if not message.strip() and uploaded_file is None:
        return (safe_hist, "", safe_props, _format_gate_choices(safe_props),
                _stats_label_files(), _stats_label_convos())

    full_message = message.strip()
    if uploaded_file:
        full_message = f"{process_uploaded_file(uploaded_file)}\n\n{full_message}"

    safe_hist = safe_hist + [{"role": "user", "content": full_message}]

    # Ingest user message into NeuroGraph
    try:
        ctx.ng.on_message(full_message)
    except (OSError, ValueError) as e:
        logger.warning("NG ingestion failed: %s", e)

    # Inject semantically relevant memories
    memory_context = ""
    try:
        recalls = ctx.ng.recall(full_message, k=5, threshold=0.35)
        if recalls:
            snippets = "\n---\n".join(r.get("content", "")[:300] for r in recalls)
            memory_context = f"\n\n## Relevant Memory (NeuroGraph):\n{snippets}\n"
    except (OSError, ValueError) as e:
        logger.warning("NG recall failed: %s", e)

    # Build system prompt
    stats = ctx.get_stats()
    notebook_text = ctx.notebook_read()
    system_prompt = build_system_prompt(stats, notebook_text, TOOL_DEFINITIONS) + memory_context

    # Build token-aware message window
    api_messages = _build_api_messages(safe_hist, system_prompt)

    accumulated_text = ""
    staged_this_turn = []
    tool_results_buffer = []

    for iteration in range(MAX_ITERATIONS):
        try:
            resp = call_model(client, system_prompt, api_messages, TOOL_DEFINITIONS)
        except Exception as e:
            logger.error("API call failed: %s", e, exc_info=True)
            safe_hist.append({"role": "assistant", "content": f"API Error: {e}"})
            return (safe_hist, "", safe_props, _format_gate_choices(safe_props),
                    _stats_label_files(), _stats_label_convos())

        # Process response content blocks
        assistant_content = []
        has_tool_use = False

        for block in resp.content:
            if block.type == "text":
                accumulated_text += ("\n\n" if accumulated_text else "") + block.text
                assistant_content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                has_tool_use = True
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # If no tool calls, we're done
        if not has_tool_use:
            break

        # Hard stop at max iterations — no advisory, just stop
        if iteration >= MAX_ITERATIONS - 1:
            break

        # Append assistant message with all content blocks
        api_messages.append({"role": "assistant", "content": assistant_content})

        # Process tool calls and build tool results
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            res = execute_tool(block.name, block.input)

            if res["status"] == "executed":
                result_text = _truncate_tool_result(str(res["result"]))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
                tool_results_buffer.append(f"Used {block.name}: {str(res['result'])[:100]}...")

            elif res["status"] == "staged":
                p_id = f"p_{int(time.time())}_{block.name}"
                staged_this_turn.append({
                    "id": p_id,
                    "tool": block.name,
                    "args": res.get("args", block.input),
                    "description": res.get("description", f"Staged: {block.name}"),
                    "timestamp": time.strftime("%H:%M:%S"),
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "This tool has been staged for review. Awaiting approval.",
                })

            elif res["status"] == "error":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {res.get('result', 'Unknown error')}",
                    "is_error": True,
                })

        # Append tool results as user message
        api_messages.append({"role": "user", "content": tool_results})

    # If no text accumulated but tools ran, request summary
    if not accumulated_text.strip() and tool_results_buffer:
        try:
            api_messages.append({
                "role": "user",
                "content": "You executed tools but produced no explanation. Summarize the results."
            })
            final_resp = call_model(client, system_prompt, api_messages, [])
            for block in final_resp.content:
                if block.type == "text":
                    accumulated_text = block.text
                    break
        except Exception as e:
            logger.warning("Summary request failed: %s", e)
            accumulated_text = "Actions completed."

    final = accumulated_text
    if staged_this_turn:
        final += "\n\n**Proposals Staged.** Check the Gate tab."
        safe_props += staged_this_turn

    if not final:
        final = "Processed request but have no text response."

    safe_hist.append({"role": "assistant", "content": final})

    try:
        ctx.save_conversation_turn(full_message, final, len(safe_hist))
    except (OSError, ValueError) as e:
        logger.warning("Conversation save failed: %s", e)

    return (safe_hist, "", safe_props, _format_gate_choices(safe_props),
            _stats_label_files(), _stats_label_convos())


# ---------------------------------------------------------------------------
# Gate UI Helpers
# ---------------------------------------------------------------------------

def _format_gate_choices(proposals):
    return gr.CheckboxGroup(
        choices=[(f"[{p['timestamp']}] {p['description']}", p['id']) for p in proposals],
        value=[]
    )


def execute_approved_proposals(ids, proposals, history):
    if not ids:
        return "No selection.", proposals, _format_gate_choices(proposals), history
    results, remaining = [], []
    for p in proposals:
        if p['id'] in ids:
            out = execute_staged_tool(p['tool'], p['args'])
            results.append(f"**{p['tool']}**: {out}")
        else:
            remaining.append(p)
    if results:
        history.append({"role": "assistant", "content": "**Executed:**\n" + "\n".join(results)})
    return "Done.", remaining, _format_gate_choices(remaining), history


def auto_continue_after_approval(history, proposals):
    last = history[-1].get("content", "") if history else ""
    if "**Executed:**" in str(last):
        return agent_loop("[System: Tools executed. Continue.]", history, proposals, None)
    return (history, "", proposals, _format_gate_choices(proposals),
            _stats_label_files(), _stats_label_convos())


def _stats_label_files():
    return f"Files: {ctx.get_stats().get('total_files', 0)}"


def _stats_label_convos():
    return f"Convos: {ctx.get_stats().get('conversations', 0)}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="TQB Worker") as demo:
    state_proposals = gr.State([])
    gr.Markdown("# TQB Worker Command Center")
    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    stat_f = gr.Markdown(_stats_label_files())
                    stat_c = gr.Markdown(_stats_label_convos())
                    btn_ref = gr.Button("Refresh")
                    file_in = gr.File(label="Upload", file_count="multiple")
                with gr.Column(scale=4):
                    chat = gr.Chatbot(height=600)
                    with gr.Row():
                        txt = gr.Textbox(scale=6, placeholder="Prompt...")
                        btn_send = gr.Button("Send", scale=1)
        with gr.Tab("Gate"):
            gate = gr.CheckboxGroup(label="Proposals", interactive=True)
            with gr.Row():
                btn_exec = gr.Button("Execute", variant="primary")
                btn_clear = gr.Button("Clear")
            res_md = gr.Markdown()

    inputs = [txt, chat, state_proposals, file_in]
    outputs = [chat, txt, state_proposals, gate, stat_f, stat_c]

    txt.submit(agent_loop, inputs, outputs)
    btn_send.click(agent_loop, inputs, outputs)
    btn_ref.click(
        lambda: (_stats_label_files(), _stats_label_convos()),
        None, [stat_f, stat_c]
    )

    btn_exec.click(
        execute_approved_proposals,
        [gate, state_proposals, chat],
        [res_md, state_proposals, gate, chat]
    ).then(
        auto_continue_after_approval,
        [chat, state_proposals],
        outputs
    )
    btn_clear.click(
        lambda p: ("Cleared.", [], _format_gate_choices([])),
        state_proposals,
        [res_md, state_proposals, gate]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
