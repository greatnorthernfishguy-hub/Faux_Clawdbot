import gradio as gr
from huggingface_hub import InferenceClient
from recursive_context import RecursiveContextManager
from pathlib import Path
import os
import json
import re
import time
import zipfile
import shutil
import traceback
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clawdbot_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Clawdbot")

def log_action(action: str, details: str):
    logger.info(f"ACTION: {action} | DETAILS: {details}")

"""
Clawdbot Unified Command Center
DIAMOND COPY [2026-02-04]
FIXED: Syntax Error in agent_loop (missing except block).
FIXED: Anti-'Ok' Logic (Recursion loop).
FIXED: Data persistence warnings.
"""

# =============================================================================
# CONFIGURATION & INIT
# =============================================================================

AVAILABLE_TOOLS = {
    "list_files", "read_file", "search_code", "write_file", 
    "create_shadow_branch", "shell_execute", "get_stats",
    "search_conversations", "search_testament", "push_to_github",
    "pull_from_github", "notebook_add", "notebook_delete", "notebook_read",
    "map_repository_structure", "map_repository_structure"
}

TEXT_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml',
    '.md', '.txt', '.rst', '.html', '.css', '.scss', '.sh', '.bash',
    '.sql', '.toml', '.cfg', '.ini', '.conf', '.xml', '.csv',
    '.env', '.gitignore', '.dockerfile'
}

client = InferenceClient("https://router.huggingface.co/v1", token=os.getenv("HF_TOKEN"))
ET_SYSTEMS_SPACE = os.getenv("ET_SYSTEMS_SPACE", "")
REPO_PATH = os.getenv("REPO_PATH", "/workspace/e-t-systems")
MODEL_ID = "moonshotai/Kimi-K2.5"

# =============================================================================
# REPO SYNC
# =============================================================================
def _resolve_repo_path() -> str:
    # FORCE: Use the directory where this app.py file actually lives
    return os.path.dirname(os.path.abspath(__file__))

# Initialize Memory
ctx = RecursiveContextManager(_resolve_repo_path())


# =============================================================================
# TOOL PARSERS & EXECUTION
# =============================================================================

def build_system_prompt() -> str:
    stats = ctx.get_stats()
    
    # READ NOTEBOOK
    nb_text = ctx.notebook_read()
    notebook_section = f"\n## 🧠 WORKING MEMORY (Notebook):\n{nb_text}\n" if nb_text else ""

    tools_doc = """
## Available Tools
- **search_code(query, n=5)**: Semantic search codebase.
- **read_file(path, start_line, end_line)**: Read file content.
- **list_files(path, max_depth)**: Explore directory tree.
- **search_conversations(query, n=5)**: Search persistent memory.
- **search_testament(query, n=5)**: Search docs/plans.
- **write_file(path, content)**: Create/Update file (REQUIRES CHANGELOG).
- **shell_execute(command)**: Run shell command.
- **create_shadow_branch()**: Backup repository.
- **push_to_github(message)**: Save current state to GitHub.
- **pull_from_github(branch)**: Hard reset state from GitHub.
- **notebook_read()**: Read your working memory.
- **notebook_add(content)**: Add a note (max 25).
- **notebook_delete(index)**: Delete a note.
- **map_repository_structure()**: Analyze code structure (files/functions).
"""
    return f"""You are Clawdbot 🦞. ... {tools_doc} ...

System Stats: {stats.get('total_files', 0)} files, {stats.get('conversations', 0)} memories.
{notebook_section} 
{tools_doc}
Output Format: Use [TOOL: tool_name(arg="value")] for tools.

## CRITICAL PROTOCOLS:
1. **DIRECT ACTION**: Do not say what you are *going* to do. JUST DO IT. Do not say "I will now search for the file." and stop. Output the `[TOOL: ...]` command immediately in the same response.
2. **RECURSIVE MEMORY FIRST**: If the user asks about past context (e.g., "the new UI"), you MUST use `search_conversations` BEFORE you answer.
3. **THINK OUT LOUD**: When writing code, output the full code block in the chat BEFORE calling `write_file`.
4. **CHECK BEFORE WRITE**: Before writing code, use `read_file` or `list_files` to ensure you aren't overwriting good code with bad.
5. **NO SILENCE**: If you perform an action, report the result.
"""

def parse_tool_calls(text: str) -> list:
    calls = []
    # 1. Bracket Format
    bracket_pattern = r"\[TOOL:\s*(\w+)\((.*?)\)\]"
    for match in re.finditer(bracket_pattern, text, re.DOTALL):
        tool_name = match.group(1)
        args_str = match.group(2)
        args = parse_tool_args(args_str)
        calls.append((tool_name, args))

    # 2. XML Format
    if "<|tool_calls" in text:
        clean_text = re.sub(r"<\|tool_calls_section_begin\|>", "", text)
        clean_text = re.sub(r"<\|tool_calls_section_end\|>", "", clean_text)
        clean_text = re.sub(r"<tool_code>", "", clean_text)
        clean_text = re.sub(r"</tool_code>", "", clean_text)
        xml_matches = re.finditer(r"(\w+)\s*\((.*?)\)", clean_text, re.DOTALL)
        for match in xml_matches:
            tool_name = match.group(1)
            if tool_name in AVAILABLE_TOOLS:
                calls.append((tool_name, parse_tool_args(match.group(2))))

    return calls

def parse_tool_args(args_str: str) -> dict:
    args = {}
    try:
        if args_str.strip().startswith('{'): return json.loads(args_str)
        pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]+))'
        for match in re.finditer(pattern, args_str):
            key = match.group(1)
            val = match.group(2) or match.group(3) or match.group(4)
            if val and val.isdigit(): val = int(val)
            args[key] = val
    except: pass
    return args

def extract_conversational_text(content: str) -> str:
    cleaned = re.sub(r'\[TOOL:.*?\]', '', content, flags=re.DOTALL)
    cleaned = re.sub(r'<\|tool_calls.*?<\|tool_calls.*?\|>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()

def execute_tool(tool_name: str, args: dict) -> dict:
    try:
        # --- SEARCH & READ ---
        if tool_name == 'search_code':
            res = ctx.search_code(args.get('query', ''), args.get('n', 5))
            return {"status": "executed", "tool": tool_name, "result": "\n".join([f"📄 {r['file']}\n```{r['snippet']}```" for r in res])}
        
        elif tool_name == 'read_file':
            # FIX: Mapped 'start_line' -> 'start', 'end_line' -> 'end' to match RecursiveContext
            return {"status": "executed", "tool": tool_name, "result": ctx.read_file(args.get('path', ''), args.get('start_line'), args.get('end_line'))}
        
        elif tool_name == 'list_files':
            return {"status": "executed", "tool": tool_name, "result": ctx.list_files(args.get('path', ''), args.get('max_depth', 3))}
        
        elif tool_name == 'search_conversations':
            res = ctx.search_conversations(args.get('query', ''), args.get('n', 5))
            # FIX: Handle Xet results (metadata/id) vs Mock results
            formatted_res = []
            for r in res:
                if 'metadata' in r: # Xet Result
                    meta = r['metadata']
                    formatted_res.append(f"[{meta.get('timestamp','?')}] {meta.get('role','?')}: {meta.get('content','')}")
                else: # Fallback
                    formatted_res.append(str(r))
            
            formatted = "\n---\n".join(formatted_res) if formatted_res else "No matches found."
            return {"status": "executed", "tool": tool_name, "result": formatted}
            
        elif tool_name == 'search_testament':
            res = ctx.search_testament(args.get('query', ''), args.get('n', 5))
            formatted = "\n\n".join([f"📜 **{r['file']}**\n{r['snippet']}" for r in res]) if res else "No matches found."
            return {"status": "executed", "tool": tool_name, "result": formatted}

        # --- WRITE & OPS ---
        elif tool_name == 'write_file':
            result = ctx.write_file(args.get('path', ''), args.get('content', ''))
            return {"status": "executed", "tool": tool_name, "result": result}
            
        elif tool_name == 'shell_execute':
            result = ctx.shell_execute(args.get('command', ''))
            return {"status": "executed", "tool": tool_name, "result": result}
            
        elif tool_name == 'push_to_github':
            result = ctx.push_to_github(args.get('message', 'Manual Backup'))
            return {"status": "executed", "tool": tool_name, "result": result}
            
        elif tool_name == 'pull_from_github':
            result = ctx.pull_from_github(args.get('branch', 'main'))
            return {"status": "executed", "tool": tool_name, "result": result}    
            
        elif tool_name == 'map_repository_structure':
            # FIX: Added status key
            result = ctx.map_repository_structure()
            return {"status": "executed", "tool": tool_name, "result": result}
            
        elif tool_name == 'create_shadow_branch':
            return {"status": "staged", "tool": tool_name, "args": args, "description": "🛡️ Create shadow branch"}

        # --- NOTEBOOK ---
        elif tool_name == 'notebook_add':
             # FIX: Added status key
             return {"status": "executed", "tool": tool_name, "result": ctx.notebook_add(args.get('content', ''))}
        elif tool_name == 'notebook_read':
             return {"status": "executed", "tool": tool_name, "result": ctx.notebook_read()}
        elif tool_name == 'notebook_delete':
             return {"status": "executed", "tool": tool_name, "result": ctx.notebook_delete(args.get('index', 0))}
             
        return {"status": "error", "result": f"Unknown tool: {tool_name}"}
    except Exception as e: return {"status": "error", "result": str(e)}

def execute_staged_tool(tool_name: str, args: dict) -> str:
    try:
        if tool_name == 'write_file': return ctx.write_file(args.get('path', ''), args.get('content', ''))
        if tool_name == 'shell_execute': return ctx.shell_execute(args.get('command', ''))
        if tool_name == 'create_shadow_branch': return ctx.create_shadow_branch()
    except Exception as e: return f"Error: {e}"
    return "Unknown tool"

# =============================================================================
# ROBUST HELPERS
# =============================================================================

def process_uploaded_file(file) -> str:
    if file is None: return ""
    if isinstance(file, list): file = file[0] if len(file) > 0 else None
    if file is None: return ""
    
    file_path = file.name if hasattr(file, 'name') else str(file)
    file_name = os.path.basename(file_path)
    suffix = os.path.splitext(file_name)[1].lower()

    if suffix == '.zip':
        try:
            extract_to = Path(REPO_PATH) / "uploaded_assets" / file_name.replace(".zip", "")
            if extract_to.exists(): shutil.rmtree(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(file_path, 'r') as z: z.extractall(extract_to)
            file_list = [f.name for f in extract_to.glob('*')]
            preview = ", ".join(file_list[:10])
            return (f"📦 **Unzipped: {file_name}**\nLocation: `{extract_to}`\nContents: {preview}\n"
                    f"SYSTEM NOTE: The files are extracted. Use list_files('{extract_to.name}') to explore them.")
        except Exception as e: return f"⚠️ Failed to unzip {file_name}: {e}"

    if suffix in TEXT_EXTENSIONS or suffix == '':
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if len(content) > 50000: content = content[:50000] + "\n...(truncated)"
            return f"📎 **Uploaded: {file_name}**\n```\n{content}\n```"
        except Exception as e: return f"📎 **Uploaded: {file_name}** (error reading: {e})"
    return f"📎 **Uploaded: {file_name}** (binary file, {os.path.getsize(file_path):,} bytes)"

def call_model_with_retry(messages, model_id, max_retries=4):
    for attempt in range(max_retries):
        try:
            return client.chat_completion(model=model_id, messages=messages, max_tokens=8192, temperature=0.7)
        except Exception as e:
            error_str = str(e)
            if "504" in error_str or "503" in error_str or "timeout" in error_str.lower():
                if attempt == max_retries - 1: raise e
                time.sleep(2 * (2 ** attempt))
            else:
                raise e

# =============================================================================
# AGENT LOOP
# =============================================================================

def agent_loop(message: str, history: list, pending_proposals: list, uploaded_file) -> tuple:
    safe_hist = history or []
    safe_props = pending_proposals or []
    
    try:
        # 1. Handle Empty Input
        if not message.strip() and uploaded_file is None:
            return (safe_hist, "", safe_props, _format_gate_choices(safe_props), _stats_label_files(), _stats_label_convos())

        full_message = message.strip()
        if uploaded_file:
            full_message = f"{process_uploaded_file(uploaded_file)}\n\n{full_message}"

        # 2. Add User Message to History
        safe_hist = safe_hist + [{"role": "user", "content": full_message}]
        
        # 3. Build API Context
        system_prompt = build_system_prompt()
        api_messages = [{"role": "system", "content": system_prompt}]
        for h in safe_hist[-40:]: # Context Window Protection
            api_messages.append({"role": h["role"], "content": h["content"]})

        accumulated_text = ""
        staged_this_turn = []
        MAX_ITERATIONS = 15 
        tool_results_buffer = [] 

        # 4. The "Anti-Ok" Recursion Loop
        for iteration in range(MAX_ITERATIONS):
            try:
                # Force a conclusion if we are running too long
                if iteration == MAX_ITERATIONS - 1:
                    api_messages.append({"role": "system", "content": "SYSTEM ALERT: Max steps reached. Stop using tools. Summarize findings immediately."})

                # Call Model
                resp = call_model_with_retry(api_messages, MODEL_ID)
                content = resp.choices[0].message.content or ""
            except Exception as e:
                safe_hist.append({"role": "assistant", "content": f"⚠️ API Error: {e}"})
                return (safe_hist, "", safe_props, _format_gate_choices(safe_props), _stats_label_files(), _stats_label_convos())

            # Parse Content
            calls = parse_tool_calls(content)
            text = extract_conversational_text(content)
            
            # Accumulate text
            if text: 
                accumulated_text += ("\n\n" if accumulated_text else "") + text
            
            # No tools? We are done.
            if not calls: 
                break

            # Execute Tools
            results = []
            for name, args in calls:
                res = execute_tool(name, args)
                if res["status"] == "executed":
                    output = f"[Tool Result: {name}]\n{res['result']}"
                    results.append(output)
                    tool_results_buffer.append(f"Used {name}: {str(res['result'])[:100]}...")
                elif res["status"] == "staged":
                    p_id = f"p_{int(time.time())}_{name}"
                    staged_this_turn.append({
                        "id": p_id, "tool": name, "args": res["args"], 
                        "description": res["description"], "timestamp": time.strftime("%H:%M:%S")
                    })
                    results.append(f"[STAGED: {name}]")
            
            # Loop Back: Feed results into the model so it can "see" them
            if results:
                api_messages += [
                    {"role": "assistant", "content": content}, 
                    {"role": "user", "content": "\n".join(results)}
                ]
            else:
                break

        # 5. THE SAFETY NET (Fixing the "No Text Response" bug)
        if not accumulated_text.strip() and tool_results_buffer:
            try:
                summary_prompt = api_messages + [{"role": "system", "content": "You have executed tools but produced no text explanation. Summarize the tool results for the user now."}]
                final_resp = call_model_with_retry(summary_prompt, MODEL_ID)
                accumulated_text = final_resp.choices[0].message.content or "Task completed (See logs)."
            except:
                accumulated_text = "✅ Actions completed."

        # 6. Finalize Output
        final = accumulated_text
        if staged_this_turn:
            final += "\n\n🛡️ **Proposals Staged.** Check the Gate tab."
            safe_props += staged_this_turn
        
        if not final: final = "🤔 I processed that but have no text response."
        
        safe_hist.append({"role": "assistant", "content": final})
        
        try: ctx.save_conversation_turn(full_message, final, len(safe_hist))
        except: pass

        return (safe_hist, "", safe_props, _format_gate_choices(safe_props), _stats_label_files(), _stats_label_convos())

    except Exception as e:
        safe_hist.append({"role": "assistant", "content": f"💥 Critical Error: {e}"})
        return (safe_hist, "", safe_props, _format_gate_choices(safe_props), _stats_label_files(), _stats_label_convos())

# =============================================================================
# UI COMPONENTS
# =============================================================================

def _format_gate_choices(proposals):
    return gr.CheckboxGroup(choices=[(f"[{p['timestamp']}] {p['description']}", p['id']) for p in proposals], value=[])

def execute_approved_proposals(ids, proposals, history):
    if not ids: return "No selection.", proposals, _format_gate_choices(proposals), history
    results, remaining = [], []
    for p in proposals:
        if p['id'] in ids:
            out = execute_staged_tool(p['tool'], p['args'])
            results.append(f"**{p['tool']}**: {out}")
        else: remaining.append(p)
    if results: history.append({"role": "assistant", "content": "✅ **Executed:**\n" + "\n".join(results)})
    return "Done.", remaining, _format_gate_choices(remaining), history

def auto_continue_after_approval(history, proposals):
    last = history[-1].get("content", "") if history else ""
    if "✅ **Executed:**" in str(last):
        return agent_loop("[System: Tools executed. Continue.]", history, proposals, None)
    return history, "", proposals, _format_gate_choices(proposals), _stats_label_files(), _stats_label_convos()

def _stats_label_files(): return f"📂 Files: {ctx.get_stats().get('total_files', 0)}"
def _stats_label_convos(): return f"💾 Convos: {ctx.get_stats().get('conversations', 0)}"

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

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
                    file_in = gr.File(label="Upload", file_count="multiple")
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
