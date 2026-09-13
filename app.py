# ---- Changelog ----
# [2026-09-13] Codex — Retire the pre-callosum topology sync at startup.
# What: Remove the Git/JSONL sync import and boot call; all TQB lenses continue
#       through Codemine's one shared substrate instance.
# Why: The old transport predates Corpus Callosum and must not mutate or export
#      the sleeping third-hemisphere Graph during modernization.
# How: Preserve its implementation under Defunct-Historical for provenance.
# [2026-09-12] Codex — Replace one-click mission mutation with plan and execute gates.
# What: Mission Control first persists Bethany's concrete MissionPlan, then an
#       explicit second action dispatches it into isolated worktrees.
# Why: The legacy button could generate, mutate, merge, and push without a
#      reviewable boundary or external integration review.
# How: Compose MissionService and retain the proposal in Gradio state.
# [2026-09-12] Codex — Reuse one model/tool loop across Chat and TQB roles
# What: Chat delegates response/tool iteration to agent_runner.run_tool_loop().
# Why: Specialists need the same proven execution behavior without a parallel
#       implementation that can drift from the interactive agent.
# How: Keep Chat's prompt, context, policy executor, gate UI, and persistence;
#      extract only the provider-neutral response/tool iteration.
# [2026-07-06] Weft (TQB) — Wire pattern-completion recall filter/compose/gate into call sites
# What: (1) agent_loop()'s per-turn substrate context now runs .recall() output through
#       filter_recall_results() + surfacing_context() instead of joining the raw vdb shards.
#       (2) execute_tool()'s per-tool-call recall is gated per tool+target via
#       gate_repeat_recall() (module-level _recall_gate_cache) — a repeat touch within the
#       TTL window skips the filtered-recall pass and falls back to SurfacingMonitor-only
#       content via surfacing_context(worker_ng, []).
# Why: PRD §2-4 (2026-07-06-codemine-pattern-completion-recall) — .recall() was surfacing
#      raw vdb shards (short "tree concept" fragments) instead of her actual lived turns,
#      and every tool call re-ran the full pattern-completion pass with no dedup.
# How: worker_ng.filter_recall_results/surfacing_context/gate_repeat_recall (sub-step 1,
#      unchanged here). Outer except (OSError, ValueError) narrow-catch convention preserved
#      at both sites — the new helpers already fail soft internally.
# [2026-04-20] Codemine (BLK-NG-192) — Wire full substrate signal loop into Chat tab
# What: Replaced single-pass on_message + flat recall with dual-pass ingest +
#       worker_ng.step() + spreading activation recall + response deposit.
# Why:  #192 — Duck Ethics / non-zero emergence. Signal loop was open:
#       single-pass only, step() never called, response never deposited back.
# How:  ingest_tool_result() for user messages (dual-pass). step() propagates
#       topology before recall. Substrate context section replaces flat recall.
#       Response deposited after turn — loop closes. Codemine built this herself.
# [2026-04-06] Josh + Claude — Add edit_file to TOOL_REGISTRY
# What: Wire edit_file tool through ctx facade and add to registry
# Why: Gap 3 — targeted find-and-replace is safer than full overwrite for cross-repo work
# How: New TOOL_REGISTRY entry delegates to ctx.edit_file()
# [2026-02-04] Josh — PLATINUM COPY: Original Clawdbot Unified Command Center
# [2026-03-29] Hammer (TQB) — Block G: Agent Loop Hardening + Assembly
# What: Complete rewrite — Claude native tool_use, PolicyEngine gates, tool registry,
#        context window management, ZIP bomb protection, structured error handling
# Why: PRD §7.G — zero regex in tool parsing, all mutating tools gated, hard max iterations
# How: Anthropic SDK messages.create() with tools param; tool_use/tool_result block loop;
#        PolicyEngine.check_tool_call() on every tool; TOOL_REGISTRY dict dispatch
# -------------------

from dotenv import load_dotenv
load_dotenv()

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
from agent_runner import run_tool_loop
from system_prompt import build_system_prompt
from tool_definitions import TOOL_DEFINITIONS
from worker_ng import get_worker_ng, ingest_tool_result, recall_context, filter_recall_results, surfacing_context, gate_repeat_recall
from spec_executor import SpecExecutor
from mission_service import MissionService

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

APP_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = Path(os.getenv("REPO_PATH", str(APP_PATH))).expanduser().resolve()

# Ecosystem sibling repos that specs are allowed to READ (but not write).
# These paths only exist on VPS; on HF Space they resolve but won't be found
# by the OS — policy permits them, actual file access governs availability.
_ECOSYSTEM_READ_PATHS: list[Path] = [
    Path("/home/josh/NeuroGraph"),
    Path("/home/josh/The-Inference-Difference"),
    Path("/home/josh/TrollGuard"),
    Path("/home/josh/The-Healing-Collective"),
    Path("/home/josh/Immunis"),
    Path("/home/josh/Elmer"),
    Path("/home/josh/Bunyan"),
    Path("/home/josh/Darwin"),
    Path("/home/josh/QuantumGraph"),
    Path("/home/josh/Praxis"),
    Path("/home/josh/agent-zero"),
    Path("/home/josh/Condensate"),
    Path("/home/josh/UniAI"),
    Path("/home/josh/Morphogenesis"),
    Path("/home/josh/UniOS"),
    Path("/home/josh/portal-v2"),
    Path("/home/josh/docs"),
]
worker_ng = get_worker_ng()          # one shared Codemine substrate for every lens
_recall_gate_cache: dict = {}        # per-target TTL dedup for execute_tool()'s recall gate
ctx = RecursiveContextManager(str(REPO_PATH), ng=worker_ng)  # facade shares the same instance
client = get_client()

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
        args.get("path") or "", args.get("start_line"), args.get("end_line")
    ),
    "write_file": lambda args: ctx.write_file(
        args.get("path") or "", args.get("content") or ""
    ),
    "edit_file": lambda args: ctx.edit_file(
        args.get("path") or "", args.get("old_text") or "", args.get("new_text") or ""
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
    "shell_execute": lambda args: ctx.shell_execute(args.get("command") or ""),
    "push_to_github": lambda args: ctx.push_to_github(
        args.get("message") or "Manual Backup"
    ),
    "pull_from_github": lambda args: ctx.pull_from_github(
        args.get("branch") or "main"
    ),
    "create_shadow_branch": lambda args: ctx.create_shadow_branch(),
    "notebook_read": lambda args: ctx.notebook_read(),
    "notebook_add": lambda args: ctx.notebook_add(args.get("content") or ""),
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
    allowed, reason = check_tool_call(tool_name, args, REPO_PATH, _ECOSYSTEM_READ_PATHS)
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

    # Recall past experience from worker substrate — gated per-target: a repeat
    # touch to the same tool+target within the TTL window skips the full
    # filtered-recall pass and falls back to SurfacingMonitor-only content.
    target = args.get("path") or args.get("command") or args.get("query") or args.get("message") or args.get("branch") or ""
    gate_key = f"{tool_name}:{target}"
    recall_text = ""
    try:
        if gate_repeat_recall(_recall_gate_cache, gate_key, time.time()):
            context_str = json.dumps(args, default=str)[:500]
            raw_recalls = recall_context(worker_ng, tool_name, context_str)
            filtered = filter_recall_results(worker_ng, raw_recalls)
        else:
            filtered = []
        recall_text = surfacing_context(worker_ng, filtered)
    except (OSError, ValueError) as e:
        logger.warning("Recall gating/filtering failed: %s", e)

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

        # Deposit the untouched result before adding recalled context for the
        # model. The synthetic display string is not raw experience (Law 7).
        ingest_tool_result(worker_ng, tool_name, args, result_str)

        # Prepend substrate recall if available — agent sees what it learned before
        if recall_text:
            result_str = f"[Substrate recall for {tool_name}]\n{recall_text}\n\n[Result]\n{result_str}"

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
        content = msg.get("content", "")
        # Skip empty messages — API rejects them
        if not content:
            continue
        # Content can be a string or a list (tool_result blocks)
        # For string content, estimate tokens normally
        # For list content (tool results), stringify for estimation
        if isinstance(content, list):
            token_est = sum(_estimate_tokens(str(c)) for c in content)
        else:
            token_est = _estimate_tokens(str(content))
        if budget - token_est < 0 and messages:
            break
        budget -= token_est
        # Sanitize — only pass role and content
        clean = {"role": msg["role"], "content": content}
        messages.append(clean)

    messages.reverse()
    return messages


# ---------------------------------------------------------------------------
# Agent Loop — Claude native tool_use
# ---------------------------------------------------------------------------

def agent_loop(message: str, history: list, pending_proposals: list, uploaded_file) -> tuple:
    safe_hist = list(history or [])
    safe_props = list(pending_proposals or [])

    if not message.strip() and uploaded_file is None:
        return (safe_hist, "", safe_props, _format_gate_choices(safe_props),
                _stats_label_files(), _stats_label_convos())

    full_message = message.strip()
    if uploaded_file:
        full_message = f"{process_uploaded_file(uploaded_file)}\n\n{full_message}"

    safe_hist = safe_hist + [{"role": "user", "content": full_message}]

    # Ingest user message as raw experience — dual-pass (gestalt + concepts, Law 7)
    try:
        ingest_tool_result(worker_ng, "user_message", {}, full_message)
        worker_ng.step()  # propagate activation through topology before recall
    except (OSError, ValueError) as e:
        logger.warning("NG ingestion failed: %s", e)

    # Assemble substrate context — spreading activation recall, substrate-first
    # content resolution, composed with SurfacingMonitor's recency signal.
    substrate_context = ""
    try:
        recalls = worker_ng.recall(full_message, k=8, threshold=0.30)
        filtered = filter_recall_results(worker_ng, recalls)
        recall_text = surfacing_context(worker_ng, filtered)
        if recall_text:
            substrate_context = (
                "\n\n## Substrate Context (What I have learned and experienced):\n"
                + recall_text + "\n"
            )
    except (OSError, ValueError) as e:
        logger.warning("NG recall failed: %s", e)

    # Build system prompt
    stats = ctx.get_stats()
    notebook_text = ctx.notebook_read()
    system_prompt = build_system_prompt(stats, notebook_text, TOOL_DEFINITIONS) + substrate_context

    # Build token-aware message window
    api_messages = _build_api_messages(safe_hist, system_prompt)

    try:
        run = run_tool_loop(
            client=client,
            system_prompt=system_prompt,
            messages=api_messages,
            tools=TOOL_DEFINITIONS,
            call_model_fn=call_model,
            execute_tool_fn=execute_tool,
            truncate_result_fn=_truncate_tool_result,
            max_iterations=MAX_ITERATIONS,
        )
    except (OSError, ConnectionError, ValueError) as e:
        logger.error("API call failed: %s: %s", type(e).__name__, e, exc_info=True)
        safe_hist.append({"role": "assistant", "content": f"API Error: {e}"})
        return (safe_hist, "", safe_props, _format_gate_choices(safe_props),
                _stats_label_files(), _stats_label_convos())

    accumulated_text = run["text"]
    staged_this_turn = run["staged_actions"]

    final = accumulated_text
    if staged_this_turn:
        final += "\n\n**Proposals Staged.** Check the Gate tab."
        safe_props += staged_this_turn

    if not final:
        final = "Processed request but have no text response."

    safe_hist.append({"role": "assistant", "content": final})

    # Deposit response as raw experience — closes the signal loop (#192)
    try:
        ingest_tool_result(worker_ng, "assistant_response", {}, final)
    except (OSError, ValueError) as e:
        logger.warning("NG response deposit failed: %s", e)

    try:
        ctx.save_conversation_turn(full_message, final, len(safe_hist))
        # Explicit checkpoint at end of turn — auto_save_interval is high to avoid
        # mid-turn I/O thrashing, so we save once when the turn is complete
        worker_ng.save()
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
    new_history = list(history) if history else []
    if results:
        new_history.append({"role": "assistant", "content": "**Executed:**\n" + "\n".join(results)})
    return "Done.", remaining, _format_gate_choices(remaining), new_history


def auto_continue_after_approval(history, proposals):
    # Don't auto-continue — it causes infinite gate loops.
    # User sends a new message to continue after approval.
    return (history, "", proposals, _format_gate_choices(proposals),
            _stats_label_files(), _stats_label_convos())


def _stats_label_files():
    return f"Files: {ctx.get_stats().get('total_files', 0)}"


def _stats_label_convos():
    return f"Convos: {ctx.get_stats().get('conversations', 0)}"


# ---------------------------------------------------------------------------
# Spec Executor — structured work block execution
# ---------------------------------------------------------------------------

_spec_executor = SpecExecutor(
    tool_registry=TOOL_REGISTRY,
    policy_check_fn=check_tool_call,
    worker_ng=worker_ng,
    workspace=REPO_PATH,
)


def execute_spec(spec_json: str) -> str:
    """Execute a structured work block spec. Returns JSON execution report."""
    try:
        spec = json.loads(spec_json)
    except json.JSONDecodeError as e:
        return json.dumps({"status": "rejected", "errors": [f"Invalid JSON: {e}"]}, indent=2)

    report = _spec_executor.execute_block(spec)

    # Log report to audit trail
    try:
        audit_dir = REPO_PATH / "data" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        with open(audit_dir / "blocks.jsonl", "a") as f:
            f.write(json.dumps(report, default=str) + "\n")
    except OSError as e:
        logger.warning("Failed to write block audit: %s", e)

    # Save NG checkpoint after spec execution
    try:
        worker_ng.save()
    except (OSError, ValueError) as e:
        logger.warning("NG checkpoint after spec execution failed: %s", e)

    return json.dumps(report, indent=2, default=str)


# ---------------------------------------------------------------------------
# Mission Control — durable plan, then approved isolated execution
# ---------------------------------------------------------------------------

_mission_service = MissionService(
    worker_ng=worker_ng,
)


def plan_mission(intent: str, constraints: str, workspace: str):
    """Create and persist a concrete MissionPlan without mutating the target."""
    ws = workspace.strip() if workspace.strip() else None
    cs = constraints.strip() if constraints.strip() else None
    result = _mission_service.plan(
        intent=intent, constraints=cs, workspace=ws,
    )
    return json.dumps(result, indent=2, default=str), result


def execute_planned_mission(proposal: dict, command_approved: bool):
    """Execute the displayed proposal and stop at external integration review."""
    result = _mission_service.execute(
        proposal or {}, command_approved=bool(command_approved),
    )
    if result.get("status") not in {"awaiting_mutation_approval", "awaiting_human_approval"}:
        proposal = {**(proposal or {}), "status": "executed"}
    return json.dumps(result, indent=2, default=str), proposal


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="TQB Worker") as demo:
    state_proposals = gr.State([])
    state_mission_plan = gr.State({})
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
        with gr.Tab("Spec"):
            gr.Markdown("### Work Block Spec Executor\nPaste a JSON spec and execute it mechanically.")
            spec_input = gr.Textbox(
                label="JSON Spec", lines=15,
                placeholder='{"spec_version": "1.0.0", "block": {...}, ...}'
            )
            btn_spec = gr.Button("Execute Spec", variant="primary")
            spec_output = gr.Textbox(label="Execution Report", lines=20, interactive=False)
        with gr.Tab("Mission"):
            gr.Markdown(
                "### Mission Control\nBethany inspects and proposes a plan first. "
                "Execution uses isolated worktrees and stops before source merge or push."
            )
            mission_intent = gr.Textbox(
                label="Intent", lines=3,
                placeholder="What do you want built, fixed, or audited?"
            )
            mission_constraints = gr.Textbox(
                label="Constraints (optional)", lines=3,
                placeholder="Project rules, standards, things to avoid..."
            )
            mission_workspace = gr.Textbox(
                label="Workspace (optional)", lines=1,
                placeholder="/home/josh (leave blank for default)"
            )
            btn_plan_mission = gr.Button("Create Plan", variant="primary")
            command_approval = gr.Checkbox(
                label="Approve command-level / constitutional scope if the plan contains it",
                value=False,
            )
            btn_execute_mission = gr.Button("Execute Displayed Plan")
            mission_output = gr.Textbox(label="Mission Plan / Result", lines=25, interactive=False)

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
    btn_spec.click(execute_spec, [spec_input], [spec_output])
    btn_plan_mission.click(
        plan_mission,
        [mission_intent, mission_constraints, mission_workspace],
        [mission_output, state_mission_plan],
    )
    btn_execute_mission.click(
        execute_planned_mission,
        [state_mission_plan, command_approval],
        [mission_output, state_mission_plan],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
