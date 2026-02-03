import zipfile
import shutil

"""
Clawdbot Unified Command Center

CHANGELOG [2026-02-02 - Gemini]
RESTORED: search_conversations & search_testament tools (previously deleted by mistake).
PRESERVED: ZIP extraction, Gradio 6 fixes, and UI layout.
"""

import gradio as gr
from huggingface_hub import InferenceClient
from recursive_context import RecursiveContextManager
from pathlib import Path
import os
import json
import re
import time
import traceback
import zipfile
import shutil

# =============================================================================
# INITIALIZATION
# =============================================================================
client = InferenceClient("https://router.huggingface.co/v1", token=os.getenv("HF_TOKEN"))
ET_SYSTEMS_SPACE = os.getenv("ET_SYSTEMS_SPACE", "")
REPO_PATH = os.getenv("REPO_PATH", "/workspace/e-t-systems")

def sync_from_space(space_id: str, local_path: Path):
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token: return
    try:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem(token=token)
        space_path = f"spaces/{space_id}"
        all_files = fs.glob(f"{space_path}/**")
        local_path.mkdir(parents=True, exist_ok=True)
        for file_path in all_files:
            rel = file_path.replace(f"{space_path}/", "", 1)
            if any(p.startswith('.') for p in rel.split('/')) or '__pycache__' in rel: continue
            try:
                if fs.info(file_path)['type'] == 'directory': continue
            except: continue
            dest = local_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with fs.open(file_path, "rb") as f: dest.write_bytes(f.read())
    except Exception: pass

def _resolve_repo_path() -> str:
    repo_path = Path(REPO_PATH)
    if ET_SYSTEMS_SPACE: sync_from_space(ET_SYSTEMS_SPACE, repo_path)
    if repo_path.exists() and any(repo_path.iterdir()): return str(repo_path)
    return os.path.dirname(os.path.abspath(__file__))

ctx = RecursiveContextManager(_resolve_repo_path())
MODEL_ID = "moonshotai/Kimi-K2.5"

# =============================================================================
# TOOL DEFINITIONS (RESTORED MEMORY TOOLS)
# =============================================================================
TOOL_DEFINITIONS = """
## Available Tools

### Tools you can use freely (no approval needed):
- **search_code(query, n=5)** — Semantic search across the E-T Systems codebase.
- **read_file(path, start_line, end_line)** — Read a specific file or line range.
- **list_files(path, max_depth)** — List directory contents as a tree.
- **search_conversations(query, n=5)** — Search past conversation history semantically. USE THIS to recall what we were working on.
- **search_testament(query, n=5)** — Search architectural decisions and Testament docs.

### Tools that get staged for Josh to approve:
- **write_file(path, content)** — Write content to a file. REQUIRES CHANGELOG header.
- **shell_execute(command)** — Run a shell command.
- **create_shadow_branch()** — Create a timestamped backup branch.
"""

def build_system_prompt() -> str:
    stats = ctx.get_stats()
    return f"""You are Clawdbot 🦞.

## System Stats
- 📂 Files: {stats.get('total_files', 0)}
- 💾 Conversations: {stats.get('conversations', 0)}

{TOOL_DEFINITIONS}
"""

def parse_tool_calls(content: str) -> list:
    calls = []
    for match in re.finditer(r'<\|tool_call_begin\|>\s*functions\.(\w+):\d+\s*\n(.*?)<\|tool_call_end\|>', content, re.DOTALL):
        try: calls.append((match.group(1), json.loads(match.group(2).strip())))
        except: calls.append((match.group(1), {"raw": match.group(2).strip()}))
    for block in re.finditer(r'<function_calls>(.*?)</function_calls>', content, re.DOTALL):
        for invoke in re.finditer(r'<invoke\s+name="(\w+)">(.*?)</invoke>', block.group(1), re.DOTALL):
            args = {}
            for p in re.finditer(r'<parameter\s+name="(\w+)">(.*?)</parameter>', invoke.group(2), re.DOTALL):
                try: args[p.group(1)] = json.loads(p.group(2).strip())
                except: args[p.group(1)] = p.group(2).strip()
            calls.append((invoke.group(1), args))
    return calls

def extract_conversational_text(content: str) -> str:
    cleaned = re.sub(r'<\|tool_call_begin\|>.*?<\|tool_call_end\|>', '', content, flags=re.DOTALL)
    return re.sub(r'<function_calls>.*?</function_calls>', '', cleaned, flags=re.DOTALL).strip()

def execute_tool(tool_name: str, args: dict) -> dict:
    try:
        if tool_name == 'search_code':
            res = ctx.search_code(args.get('query', ''), args.get('n', 5))
            return {"status": "executed", "tool": tool_name, "result": "\n".join([f"📄 {r['file']}\n```{r['snippet']}```" for r in res])}
        
        elif tool_name == 'read_file':
            return {"status": "executed", "tool": tool_name, "result": ctx.read_file(args.get('path', ''), args.get('start_line'), args.get('end_line'))}
        
        elif tool_name == 'list_files':
            return {"status": "executed", "tool": tool_name, "result": ctx.list_files(args.get('path', ''), args.get('max_depth', 3))}
        
        # RESTORED: Memory Tools
        elif tool_name == 'search_conversations':
            res = ctx.search_conversations(args.get('query', ''), args.get('n', 5))
            formatted = "\n---\n".join([f"{r['content']}" for r in res]) if res else "No matches found."
            return {"status": "executed", "tool": tool_name, "result": formatted}

        # RESTORED: Testament Tools
        elif tool_name == 'search_testament':
            res = ctx.search_testament(args.get('query', ''), args.get('n', 5))
            formatted = "\n\n".join([f"📜 **{r['file']}**\n{r['snippet']}" for r in res]) if res else "No matches found."
            return {"status": "executed", "tool": tool_name, "result": formatted}

        elif tool_name == 'write_file':
            return {"status": "staged", "tool": tool_name, "args": args, "description": f"✏️ Write to `{args.get('path')}`"}
        
        elif tool_name == 'shell_execute':
            return {"status": "staged", "tool": tool_name, "args": args, "description": f"🖥️ Execute: `{args.get('command')}`"}
        
        elif tool_name == 'create_shadow_branch':
            return {"status": "staged", "tool": tool_name, "args": args, "description": "🛡️ Create shadow branch"}
        
        return {"status": "error", "result": f"Unknown tool: {tool_name}"}
    except Exception as e: return {"status": "error", "result": str(e)}

def execute_staged_tool(tool_name: str, args: dict) -> str:
    try:
        if tool_name == 'write_file': return ctx.write_file(args.get('path', ''), args.get('content', ''))
        if tool_name == 'shell_execute': return ctx.shell_execute(args.get('command', ''))
        if tool_name == 'create_shadow_branch': return ctx.create_shadow_branch()
    except Exception as e: return f"Error: {e}"
    return "Unknown tool"

# --- FIXED FILE UPLOAD HANDLER ---
TEXT_EXTENSIONS = {'.py', '.js', '.ts', '.json', '.md', '.txt', '.yaml', '.yml', '.html', '.css', '.sh', '.toml', '.sql', '.env', '.dockerfile'}

def process_uploaded_file(file) -> str:
    if file is None: return ""
    file = file[0] if isinstance(file, list) else file
    file_path = file.name if hasattr(file, 'name') else str(file)
    file_name = os.path.basename(file_path)
    
    upload_dir = Path("/workspace/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    if file_name.lower().endswith('.zip'):
        extract_to = upload_dir / file_name.replace('.zip', '')
        if extract_to.exists(): shutil.rmtree(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(file_path, 'r') as z: z.extractall(extract_to)
            return f"📦 **Unzipped:** `{extract_to}`\nFiles available for tools."
        except Exception as e: return f"❌ Zip Error: {e}"
    
    if os.path.splitext(file_name)[1].lower() in TEXT_EXTENSIONS:
        try:
            with open(file_path, 'r', errors='ignore') as f: return f"📎 **{file_name}**\n```\n{f.read()[:50000]}\n```"
        except Exception as e: return f"Error reading {file_name}: {e}"
    return f"📎 **{file_name}** (Binary ignored)"

# --- AGENT LOOP ---
def agent_loop(message: str, history: list, pending_proposals: list, uploaded_file) -> tuple:
    if not message.strip() and uploaded_file is None:
        return (history, "", pending_proposals, _format_gate_choices(pending_proposals), _stats_label_files(), _stats_label_convos())

    full_message = message.strip()
    if uploaded_file: full_message = f"{process_uploaded_file(uploaded_file)}\n\n{full_message}"

    history = history + [{"role": "user", "content": full_message}]
    api_messages = [{"role": "system", "content": build_system_prompt()}] + [{"role": h["role"], "content": h["content"]} for h in history[-20:]]

    accumulated_text = ""
    staged_this_turn = []

    for _ in range(5):
        try:
            resp = client.chat_completion(model=MODEL_ID, messages=api_messages, max_tokens=2048)
            content = resp.choices[0].message.content or ""
        except Exception as e:
            history.append({"role": "assistant", "content": f"API Error: {e}"})
            return (history, "", pending_proposals, _format_gate_choices(pending_proposals), _stats_label_files(), _stats_label_convos())

        calls = parse_tool_calls(content)
        text = extract_conversational_text(content)
        if text: accumulated_text += ("\n\n" if accumulated_text else "") + text
        
        if not calls: break

        results = []
        for name, args in calls:
            res = execute_tool(name, args)
            if res["status"] == "executed": results.append(f"Result: {res['result']}")
            elif res["status"] == "staged":
                staged_this_turn.append({"id": f"p_{int(time.time())}_{name}", "tool": name, "args": res["args"], "description": res["description"], "timestamp": time.strftime("%H:%M:%S")})
                results.append(f"STAGED: {name}")
        
        api_messages += [{"role": "assistant", "content": content}, {"role": "user", "content": "\n".join(results)}]

    final = accumulated_text + ("\n\n🛡️ Check Gate." if staged_this_turn else "")
    history.append({"role": "assistant", "content": final or "Thinking..."})
    ctx.save_conversation_turn(full_message, final, len(history))
    
    return (history, "", pending_proposals + staged_this_turn, _format_gate_choices(pending_proposals + staged_this_turn), _stats_label_files(), _stats_label_convos())

# --- UI COMPONENTS ---
def _format_gate_choices(proposals):
    return gr.CheckboxGroup(choices=[(f"[{p['timestamp']}] {p['description']}", p['id']) for p in proposals], value=[])

def execute_approved_proposals(ids, proposals, history):
    if not ids: return "No selection.", proposals, _format_gate_choices(proposals), history
    results, remaining = [], []
    for p in proposals:
        if p['id'] in ids: results.append(f"**{p['tool']}**: {execute_staged_tool(p['tool'], p['args'])}")
        else: remaining.append(p)
    if results: history.append({"role": "assistant", "content": "✅ **Executed:**\n" + "\n".join(results)})
    return "Done.", remaining, _format_gate_choices(remaining), history

def auto_continue_after_approval(history, proposals):
    last = history[-1].get("content", "")
    text = last[0].get("text", "") if isinstance(last, list) else str(last)
    if not text.startswith("✅"): return history, "", proposals, _format_gate_choices(proposals), _stats_label_files(), _stats_label_convos()
    return agent_loop("[Approved. Continue.]", history, proposals, None)

def _stats_label_files(): return f"📂 Files: {ctx.get_stats().get('total_files', 0)}"
def _stats_label_convos(): return f"💾 Convos: {ctx.get_stats().get('conversations', 0)}"

# --- UI LAYOUT ---
with gr.Blocks(title="🦞 Clawdbot") as demo:
    state_proposals = gr.State([])
    gr.Markdown("# 🦞 Clawdbot Command Center")
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    stat_f = gr.Markdown(_stats_label_files())
                    stat_c = gr.Markdown(_stats_label_convos())
                    btn_ref = gr.Button("🔄")
                    file_in = gr.File(label="Upload", file_count="multiple", file_types=['.py', '.js', '.json', '.md', '.txt', '.yaml', '.sh', '.zip', '.env', '.toml', '.sql'])
                with gr.Column(scale=4):
                    chat = gr.Chatbot(height=600, avatar_images=(None, "https://em-content.zobj.net/source/twitter/408/lobster_1f99e.png"))
                    with gr.Row():
                        txt = gr.Textbox(scale=6, placeholder="Prompt...")
                        btn_send = gr.Button("Send", scale=1)
        with gr.Tab("🛡️ Gate"):
            gate = gr.CheckboxGroup(label="Proposals", interactive=True)
            with gr.Row():
                btn_exec = gr.Button("✅ Execute", variant="primary")
                btn_clear = gr.Button("🗑️ Clear")
            res_md = gr.Markdown()

    inputs = [txt, chat, state_proposals, file_in]
    outputs = [chat, txt, state_proposals, gate, stat_f, stat_c]
    
    txt.submit(agent_loop, inputs, outputs)
    btn_send.click(agent_loop, inputs, outputs)
    btn_ref.click(lambda: (_stats_label_files(), _stats_label_convos()), None, [stat_f, stat_c])
    
    btn_exec.click(execute_approved_proposals, [gate, state_proposals, chat], [res_md, state_proposals, gate, chat]).then(
        auto_continue_after_approval, [chat, state_proposals], outputs
    )
    btn_clear.click(lambda p: ("Cleared.", [], _format_gate_choices([])), state_proposals, [res_md, state_proposals, gate])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
