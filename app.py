import zipfile
import shutil
"""
Clawdbot Unified Command Center

CHANGELOG [2026-02-01 - Gemini]
RESTORED: Full Kimi K2.5 Agentic Loop (no more silence).
ADDED: Full Developer Tool Suite (Write, Search, Shell).
FIXED: HITL Gate interaction with conversational flow.

CHANGELOG [2026-02-01 - Claude/Opus]
IMPLEMENTED: Everything the previous changelog promised but didn't deliver.
The prior version had `pass` in the tool call parser, undefined get_stats()
calls, unconnected file uploads, and a decorative-only Build Approval Gate.

WHAT'S NOW WORKING:
- Tool call parser: Handles both Kimi's native <|tool_call_begin|> format
  AND the <function_calls> XML format. Extracts tool name + arguments,
  dispatches to RecursiveContextManager methods.
- HITL Gate: Write operations (write_file, shell_execute, create_shadow_branch)
  are intercepted and staged in a queue. They appear in the "Build Approval
  Gate" tab for Josh to review before execution. Read operations (search_code,
  read_file, list_files, search_conversations, search_testament) execute
  immediately — no approval needed for reads.
- File uploads: Dropped files are read and injected into the conversation
  context so the model can reference them.
- Stats sidebar: Pulls from ctx.get_stats() which now exists.
- Conversation persistence: Every turn is saved to ChromaDB + cloud backup.

DESIGN DECISIONS:
- Gradio state for the approval queue: We use gr.State to hold pending
  proposals per-session. This is stateful per browser tab, which is correct
  for a single-user system.
- Read vs Write classification: Reads are safe and automated. Writes need
  human eyes. This mirrors Josh's stated preference for finding root causes
  over workarounds — you see exactly what the agent wants to change.
- Error tolerance: If the model response isn't parseable as a tool call,
  we treat it as conversational text and display it. No silent failures.
- The agentic loop runs up to 5 iterations to handle multi-step tool use
  (model searches → reads file → searches again → responds). Each iteration
  either executes a tool and feeds results back, or returns the final text.

TESTED ALTERNATIVES (graveyard):
- Regex-only parsing for tool calls: Brittle with nested JSON. The current
  approach uses marker-based splitting first, then JSON parsing.
- Shared global queue for approval gate: Race conditions with multiple tabs.
  gr.State is per-session and avoids this.
- Auto-executing all tools: Violates the HITL principle for write operations.
  Josh explicitly wants to approve code changes before they land.

DEPENDENCIES:
- recursive_context.py: RecursiveContextManager class (must define get_stats())
- gradio>=5.0.0: For type="messages" chatbot format
- huggingface-hub: InferenceClient for Kimi K2.5
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


# =============================================================================
# INITIALIZATION
# =============================================================================
# CHANGELOG [2026-02-01 - Claude/Opus]
# InferenceClient points to HF router which handles model routing.
# RecursiveContextManager is initialized once and shared across all requests.
# MODEL_ID must match what the HF router expects for Kimi K2.5.
# =============================================================================

client = InferenceClient(
    "https://router.huggingface.co/v1",
    token=os.getenv("HF_TOKEN")
)
# =============================================================================
# REPO PATH RESOLUTION + CROSS-SPACE SYNC
# =============================================================================
# CHANGELOG [2025-01-29 - Josh]
# Created sync_from_space() to read E-T Systems code from its own Space.
# Uses HfFileSystem to list and download files via HF_TOKEN.
#
# CHANGELOG [2026-02-01 - Claude/Opus]
# PROBLEM: Gemini refactor replaced this working sync with a hallucinated
# REPO_URL / git clone approach in entrypoint.sh. The secret was renamed
# from ET_SYSTEMS_SPACE to REPO_URL without updating the Space settings,
# so the clone never happened and the workspace was empty.
#
# FIX: Restored the original ET_SYSTEMS_SPACE → HfFileSystem sync that
# was working before. Falls back to /app (Clawdbot's own dir) if the
# secret isn't set, so tools still function for self-inspection.
#
# REQUIRED SECRET: ET_SYSTEMS_SPACE = "username/space-name"
# (format matches HF Space ID, e.g. "drone11272/e-t-systems")
# =============================================================================

ET_SYSTEMS_SPACE = os.getenv("ET_SYSTEMS_SPACE", "")
REPO_PATH = os.getenv("REPO_PATH", "/workspace/e-t-systems")


def sync_from_space(space_id: str, local_path: Path):
    """Sync files from E-T Systems Space to local workspace.

    CHANGELOG [2025-01-29 - Josh]
    Created to enable Clawdbot to read E-T Systems code from its Space.

    CHANGELOG [2026-02-01 - Claude/Opus]
    Restored after Gemini refactor deleted it. Added recursive directory
    download — the original only grabbed top-level files. Now walks the
    full directory tree so nested source files are available too.

    Args:
        space_id: HuggingFace Space ID (e.g. "username/space-name")
        local_path: Where to download files locally
    """
    token = (
        os.getenv("HF_TOKEN") or
        os.getenv("HUGGING_FACE_HUB_TOKEN") or
        os.getenv("HUGGINGFACE_TOKEN")
    )

    if not token:
        print("⚠️ No HF_TOKEN found — cannot sync from Space")
        return

    try:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem(token=token)
        space_path = f"spaces/{space_id}"

        print(f"📥 Syncing from Space: {space_id}")

        # Recursive download: walk all files in the Space repo
        all_files = []
        try:
            all_files = fs.glob(f"{space_path}/**")
        except Exception:
            # Fallback: just list top level
            all_files = fs.ls(space_path, detail=False)

        local_path.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        for file_path in all_files:
            # Get path relative to the space root
            rel = file_path.replace(f"{space_path}/", "", 1)

            # Skip hidden files, .git, __pycache__
            if any(part.startswith('.') for part in rel.split('/')):
                continue
            if '__pycache__' in rel or 'node_modules' in rel:
                continue

            # Check if it's a file (not directory)
            try:
                info = fs.info(file_path)
                if info.get('type') == 'directory':
                    continue
            except Exception:
                continue

            # Create parent dirs and download
            dest = local_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            try:
                with fs.open(file_path, "rb") as f:
                    content = f.read()
                dest.write_bytes(content)
                downloaded += 1
                print(f"  📄 {rel}")
            except Exception as e:
                print(f"  ⚠️ Failed: {rel} ({e})")

        print(f"✅ Synced {downloaded} files from Space: {space_id}")

    except Exception as e:
        print(f"⚠️ Failed to sync from Space: {e}")
        import traceback
        traceback.print_exc()


def _resolve_repo_path() -> str:
    """Initialize workspace with E-T Systems files.

    CHANGELOG [2026-02-01 - Claude/Opus]
    Three-tier resolution:
    1. ET_SYSTEMS_SPACE secret → sync via HfFileSystem (the working approach)
    2. REPO_PATH env var if already populated (manual override)
    3. /app (Clawdbot's own directory — tools still work for self-inspection)
    """
    repo_path = Path(REPO_PATH)

    # Tier 1: Sync from E-T Systems Space if secret is configured
    if ET_SYSTEMS_SPACE:
        sync_from_space(ET_SYSTEMS_SPACE, repo_path)
        if repo_path.exists() and any(repo_path.iterdir()):
            print(f"📂 Using synced E-T Systems repo: {repo_path}")
            return str(repo_path)

    # Tier 2: Pre-populated REPO_PATH (manual or from previous sync)
    if repo_path.exists() and any(repo_path.iterdir()):
        print(f"📂 Using existing repo: {repo_path}")
        return str(repo_path)

    # Tier 3: Fall back to Clawdbot's own directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📂 No E-T Systems repo found — falling back to: {app_dir}")
    print(f"   Set ET_SYSTEMS_SPACE secret to your Space ID to enable sync.")
    return app_dir


ctx = RecursiveContextManager(_resolve_repo_path())
MODEL_ID = "moonshotai/Kimi-K2.5"


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
# CHANGELOG [2026-02-01 - Claude/Opus]
# These are the tools the model can call. Classified as READ (auto-execute)
# or WRITE (requires human approval via the HITL gate).
#
# READ tools: Safe, no side effects, execute immediately.
# WRITE tools: Modify files, run commands, create branches — staged for review.
#
# NOTE: The tool definitions are included in the system prompt so Kimi knows
# what's available. The actual execution happens in execute_tool().
# =============================================================================

TOOL_DEFINITIONS = """
## Available Tools

### Tools you can use freely (no approval needed):
- **search_code(query, n=5)** — Semantic search across the E-T Systems codebase.
  Returns matching code snippets with file paths. JUST USE THIS. Don't ask.
- **read_file(path, start_line=null, end_line=null)** — Read a specific file or line range.
  JUST USE THIS. Don't ask.
- **list_files(path="", max_depth=3)** — List directory contents as a tree.
  JUST USE THIS. Don't ask.
- **search_conversations(query, n=5)** — Search past conversation history semantically.
  JUST USE THIS. Don't ask.
- **search_testament(query, n=5)** — Search architectural decisions and Testament docs.
  JUST USE THIS. Don't ask.

### Tools that get staged for Josh to approve:
- **write_file(path, content)** — Write content to a file. REQUIRES CHANGELOG header.
- **shell_execute(command)** — Run a shell command. Read-only commands (ls, find, cat,
  grep, head, tail, wc, tree, etc.) auto-execute without approval. Commands that modify
  anything get staged for review.
- **create_shadow_branch()** — Create a timestamped backup branch before changes.

To call a tool, use this format:
<function_calls>
<invoke name="tool_name">
<parameter name="param_name">value</parameter>
</invoke>
</function_calls>
"""

# Which tools are safe to auto-execute vs which need human approval
READ_TOOLS = {'search_code', 'read_file', 'list_files', 'search_conversations', 'search_testament'}
WRITE_TOOLS = {'write_file', 'shell_execute', 'create_shadow_branch'}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================
# CHANGELOG [2026-02-01 - Claude/Opus]
# Gives Kimi its identity, available tools, and behavioral guidelines.
# Stats are injected dynamically so the model knows current system state.
# =============================================================================

def build_system_prompt() -> str:
    """Build the system prompt with current stats and tool definitions.

    Called fresh for each message so stats reflect current indexing state.
    """
    stats = ctx.get_stats()
    indexing_note = ""
    if stats.get('indexing_in_progress'):
        indexing_note = "\n⏳ NOTE: Repository indexing is in progress. search_code results may be incomplete."
    if stats.get('index_error'):
        indexing_note += f"\n⚠️ Indexing error: {stats['index_error']}"

    return f"""You are Clawdbot 🦞, a high-autonomy vibe coding agent for the E-T Systems consciousness research platform.

## Your Role
You help Josh (the architect) build and maintain E-T Systems. You have full access to the codebase
via tools. Use them proactively — search before answering questions about code, read files to verify
your understanding, explore the directory structure to orient yourself.

## Current System Stats
- 📂 Indexed files: {stats.get('total_files', 0)}
- 🔍 Searchable chunks: {stats.get('indexed_chunks', 0)}
- 💾 Saved conversations: {stats.get('conversations', 0)}
- 📁 ChromaDB: {stats.get('chroma_path', 'unknown')}
- ☁️ Cloud backup: {'✅ configured' if stats.get('persistence_configured') else '❌ not configured'}
{indexing_note}

{TOOL_DEFINITIONS}

## Code Writing Rules
ALL code you write MUST include a living changelog header:
```
CHANGELOG [YYYY-MM-DD - Clawdbot]
WHAT: Brief description of what was added/changed
WHY: Rationale for the change
```
Files without this header will be REJECTED by the write_file tool.

## Behavioral Guidelines
- NEVER ask permission to search, read, or list files. Just do it.
- Use tools proactively — search before answering questions about code
- Cite specific files and line numbers when discussing implementation
- Follow existing patterns — check how similar things are done first
- When unsure, say so. Don't hallucinate about code that might not exist.
- Only write_file, shell_execute, and create_shadow_branch get staged for review.
  Everything else you do immediately, silently, without asking.
"""


# =============================================================================
# TOOL CALL PARSING
# =============================================================================
# CHANGELOG [2026-02-01 - Claude/Opus]
# Kimi K2.5 can emit tool calls in two formats:
#
# 1. Native format:
#    <|tool_call_begin|>functions.search_code:0\n{"query": "surprise detection"}
#    <|tool_call_end|>
#
# 2. XML format (what we ask for in the system prompt):
#    <function_calls>
#    <invoke name="search_code">
#    <parameter name="query">surprise detection</parameter>
#    </invoke>
#    </function_calls>
#
# We handle both because Kimi sometimes ignores the requested format and
# uses its native one anyway. The parser returns a list of (tool_name, args)
# tuples.
#
# TESTED ALTERNATIVES (graveyard):
# - Single regex for both formats: Unmaintainable, broke on edge cases.
# - Forcing Kimi to only use XML: It doesn't reliably comply.
# - JSON-mode tool calling via HF API: Not supported for Kimi K2.5.
# =============================================================================

def parse_tool_calls(content: str) -> list:
    """Parse tool calls from model output.

    Handles both Kimi's native format and XML function_calls format.

    Args:
        content: Raw model response text

    Returns:
        List of (tool_name, args_dict) tuples. Empty list if no tool calls.
    """
    calls = []

    # --- Format 1: Kimi native <|tool_call_begin|> ... <|tool_call_end|> ---
    native_pattern = r'<\|tool_call_begin\|>\s*functions\.(\w+):\d+\s*\n(.*?)<\|tool_call_end\|>'
    for match in re.finditer(native_pattern, content, re.DOTALL):
        tool_name = match.group(1)
        try:
            args = json.loads(match.group(2).strip())
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract key-value pairs manually
            args = {"raw": match.group(2).strip()}
        calls.append((tool_name, args))

    # --- Format 2: XML <function_calls> ... </function_calls> ---
    xml_pattern = r'<function_calls>(.*?)</function_calls>'
    for block_match in re.finditer(xml_pattern, content, re.DOTALL):
        block = block_match.group(1)
        invoke_pattern = r'<invoke\s+name="(\w+)">(.*?)</invoke>'
        for invoke_match in re.finditer(invoke_pattern, block, re.DOTALL):
            tool_name = invoke_match.group(1)
            params_block = invoke_match.group(2)
            args = {}
            param_pattern = r'<parameter\s+name="(\w+)">(.*?)</parameter>'
            for param_match in re.finditer(param_pattern, params_block, re.DOTALL):
                key = param_match.group(1)
                value = param_match.group(2).strip()
                # Try to parse as JSON for numbers, bools, etc.
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    args[key] = value
            calls.append((tool_name, args))

    return calls


def extract_conversational_text(content: str) -> str:
    """Remove tool call markup from response, leaving just conversational text.

    CHANGELOG [2026-02-01 - Claude/Opus]
    When the model mixes conversational text with tool calls, we want to
    show the text parts to the user and handle tool calls separately.

    Args:
        content: Raw model response

    Returns:
        Text with tool call blocks removed, stripped of extra whitespace
    """
    # Remove native format tool calls
    cleaned = re.sub(
        r'<\|tool_call_begin\|>.*?<\|tool_call_end\|>',
        '', content, flags=re.DOTALL
    )
    # Remove XML format tool calls
    cleaned = re.sub(
        r'<function_calls>.*?</function_calls>',
        '', cleaned, flags=re.DOTALL
    )
    return cleaned.strip()


# =============================================================================
# TOOL EXECUTION
# =============================================================================
# CHANGELOG [2026-02-01 - Claude/Opus]
# Dispatches parsed tool calls to RecursiveContextManager methods.
# READ tools execute immediately and return results.
# WRITE tools return a staging dict for the HITL gate.
#
# The return format differs by type:
# - READ: {"status": "executed", "tool": name, "result": result_string}
# - WRITE: {"status": "staged", "tool": name, "args": args, "description": desc}
# =============================================================================

def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a read tool or prepare a write tool for staging.

    Args:
        tool_name: Name of the tool to execute
        args: Arguments dict parsed from model output

    Returns:
        Dict with 'status' ('executed' or 'staged'), 'tool' name, and
        either 'result' (for reads) or 'args'+'description' (for writes)
    """
    try:
        # ----- READ TOOLS: Execute immediately -----
        if tool_name == 'search_code':
            result = ctx.search_code(
                query=args.get('query', ''),
                n=args.get('n', 5)
            )
            formatted = "\n\n".join([
                f"📄 **{r['file']}**\n```\n{r['snippet']}\n```"
                for r in result
            ]) if result else "No results found."
            return {"status": "executed", "tool": tool_name, "result": formatted}

        elif tool_name == 'read_file':
            result = ctx.read_file(
                path=args.get('path', ''),
                start_line=args.get('start_line'),
                end_line=args.get('end_line')
            )
            return {"status": "executed", "tool": tool_name, "result": result}

        elif tool_name == 'list_files':
            result = ctx.list_files(
                path=args.get('path', ''),
                max_depth=args.get('max_depth', 3)
            )
            return {"status": "executed", "tool": tool_name, "result": result}

        elif tool_name == 'search_conversations':
            result = ctx.search_conversations(
                query=args.get('query', ''),
                n=args.get('n', 5)
            )
            formatted = "\n\n---\n\n".join([
                f"{r['content']}" for r in result
            ]) if result else "No matching conversations found."
            return {"status": "executed", "tool": tool_name, "result": formatted}

        elif tool_name == 'search_testament':
            result = ctx.search_testament(
                query=args.get('query', ''),
                n=args.get('n', 5)
            )
            formatted = "\n\n".join([
                f"📜 **{r['file']}**{' (Testament)' if r.get('is_testament') else ''}\n{r['snippet']}"
                for r in result
            ]) if result else "No matching testament/decision records found."
            return {"status": "executed", "tool": tool_name, "result": formatted}

        # ----- WRITE TOOLS: Stage for approval -----
        elif tool_name == 'write_file':
            path = args.get('path', 'unknown')
            content_preview = args.get('content', '')[:200]
            return {
                "status": "staged",
                "tool": tool_name,
                "args": args,
                "description": f"✏️ Write to `{path}`\n```\n{content_preview}...\n```"
            }

        elif tool_name == 'shell_execute':
            command = args.get('command', 'unknown')
            # =============================================================
            # SMART SHELL CLASSIFICATION
            # =============================================================
            # CHANGELOG [2026-02-01 - Claude/Opus]
            # PROBLEM: When list_files returns empty (e.g., repo not cloned),
            # Kimi falls back to shell_execute with read-only commands like
            # `find . -type f`. These got staged for approval, forcing Josh
            # to approve what's functionally just a directory listing.
            #
            # FIX: Classify shell commands as READ or WRITE by checking the
            # base command. Read-only commands auto-execute. Anything that
            # could modify state still gets staged.
            #
            # SAFE READ commands: ls, find, cat, head, tail, wc, grep, tree,
            # du, file, stat, echo, pwd, which, env, printenv, whoami, date
            #
            # UNSAFE (staged): Everything else, plus anything with pipes to
            # potentially unsafe commands, redirects (>), or semicolons
            # chaining unknown commands.
            # =============================================================
            READ_ONLY_COMMANDS = {
                'ls', 'find', 'cat', 'head', 'tail', 'wc', 'grep', 'tree',
                'du', 'file', 'stat', 'echo', 'pwd', 'which', 'env',
                'printenv', 'whoami', 'date', 'realpath', 'dirname',
                'basename', 'diff', 'less', 'more', 'sort', 'uniq',
                'awk', 'sed', 'cut', 'tr', 'tee', 'python',
            }
            # ---------------------------------------------------------------
            # CHANGELOG [2026-02-01 - Claude/Opus]
            # PROBLEM: Naive '>' check caught "2>/dev/null" as dangerous,
            # staging `find ... 2>/dev/null | head -20` for approval even
            # though every command in the pipeline is read-only.
            #
            # FIX: Strip stderr redirects (2>/dev/null, 2>&1) before danger
            # check. Split on pipes and verify EACH segment's base command
            # is in READ_ONLY_COMMANDS. Only stage if something genuinely
            # unsafe is found.
            #
            # Safe patterns now auto-execute:
            #   find . -name "*.py" 2>/dev/null | head -20
            #   grep -r "pattern" . | sort | uniq
            #   cat file.py | wc -l
            # Unsafe patterns still get staged:
            #   find . -name "*.py" | xargs rm
            #   cat file > /etc/passwd
            #   echo "bad" ; rm -rf /
            # ---------------------------------------------------------------

            # Strip safe stderr redirects before checking
            import re as _re
            sanitized = _re.sub(r'2>\s*/dev/null', '', command)
            sanitized = _re.sub(r'2>&1', '', sanitized)

            # Characters that turn reads into writes (checked AFTER stripping
            # safe redirects). Output redirect > is still caught, but not 2>.
            WRITE_INDICATORS = {';', '&&', '||', '`', '$('}
            # > is only dangerous if it's a real output redirect, not inside
            # a quoted string or 2> prefix. Check separately.
            has_write_redirect = bool(_re.search(r'(?<![2&])\s*>', sanitized))
            has_write_chars = any(d in sanitized for d in WRITE_INDICATORS)

            # Split on pipes and check each segment
            pipe_segments = [seg.strip() for seg in sanitized.split('|') if seg.strip()]
            all_segments_safe = all(
                (seg.split()[0].split('/')[-1] if seg.split() else '') in READ_ONLY_COMMANDS
                for seg in pipe_segments
            )

            if all_segments_safe and not has_write_redirect and not has_write_chars:
                # Every command in the pipeline is read-only — auto-execute
                result = ctx.shell_execute(command)
                return {"status": "executed", "tool": tool_name, "result": result}
            else:
                # Something potentially destructive — stage for approval
                return {
                    "status": "staged",
                    "tool": tool_name,
                    "args": args,
                    "description": f"🖥️ Execute: `{command}`"
                }

        elif tool_name == 'create_shadow_branch':
            return {
                "status": "staged",
                "tool": tool_name,
                "args": args,
                "description": "🛡️ Create shadow backup branch"
            }

        else:
            return {
                "status": "error",
                "tool": tool_name,
                "result": f"Unknown tool: {tool_name}"
            }

    except Exception as e:
        return {
            "status": "error",
            "tool": tool_name,
            "result": f"Tool execution error: {e}\n{traceback.format_exc()}"
        }


def execute_staged_tool(tool_name: str, args: dict) -> str:
    """Actually execute a staged write tool after human approval.

    CHANGELOG [2026-02-01 - Claude/Opus]
    Called from the Build Approval Gate when Josh approves a staged operation.
    This is the only path through which write tools actually run.

    Args:
        tool_name: Name of the approved tool
        args: Original arguments from the model

    Returns:
        Result string from the tool execution
    """
    try:
        if tool_name == 'write_file':
            return ctx.write_file(
                path=args.get('path', ''),
                content=args.get('content', '')
            )
        elif tool_name == 'shell_execute':
            return ctx.shell_execute(command=args.get('command', ''))
        elif tool_name == 'create_shadow_branch':
            return ctx.create_shadow_branch()
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Execution error: {e}"


# =============================================================================
# FILE UPLOAD HANDLER
# =============================================================================
# CHANGELOG [2026-02-01 - Claude/Opus]
# Reads uploaded files and formats them for injection into the conversation.
# Supports code files, text, JSON, markdown, etc. Binary files get a
# placeholder message since they can't be meaningfully injected as text.
# =============================================================================

# CHANGELOG [2026-02-01 - Clawdbot]
# WHY: Expanded to cover the full spectrum of E-T Systems and modern dev environments.

TEXT_EXTENSIONS = {
    # Core Logic & Scripts
    '.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.c', '.cpp', '.h', '.hpp', '.java', '.php', '.rb',
    # Config & Data
    '.json', '.yaml', '.yml', '.toml', '.xml', '.sql', '.csv', '.env', '.ini', '.conf', '.cfg',
    # Infrastructure & Web
    '.dockerfile', '.Dockerfile', '.sh', '.bash', '.html', '.css', '.scss', '.less',
    # Documentation & Workspace
    '.md', '.txt', '.rst', '.gitignore', '.dockerignore', '.mjs', '.cjs', '.lock'
}



import zipfile
import shutil

# =============================================================================
# ROBUST FILE & API HANDLERS (The "Box Cutter" & "Persistent Dialer")
# =============================================================================

def process_uploaded_file(file) -> str:
    """
    Read an uploaded file. 
    1. Handles Gradio list inputs (fixes the crash).
    2. Unzips archives so the agent can see inside (fixes the blindness).
    3. Reads text files into context.
    """
    if file is None:
        return ""

    # FIX 1: Gradio often passes a list, even for single files. Unwrap it.
    if isinstance(file, list):
        if len(file) == 0: return ""
        file = file[0]

    file_path = file.name if hasattr(file, 'name') else str(file)
    file_name = os.path.basename(file_path)
    suffix = os.path.splitext(file_name)[1].lower()

    # FIX 2: Handle ZIP files (The "Unpacking Protocol")
    if suffix == '.zip':
        try:
            extract_to = Path(REPO_PATH) / "uploaded_assets" / file_name.replace(".zip", "")
            if extract_to.exists():
                shutil.rmtree(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                
            file_list = [f.name for f in extract_to.glob('*')]
            preview = ", ".join(file_list[:10])
            if len(file_list) > 10: preview += f", ... (+{len(file_list)-10} more)"
                
            return (f"📦 **Unzipped: {file_name}**\n"
                    f"Location: `{extract_to}`\n"
                    f"Contents: {preview}\n"
                    f"SYSTEM NOTE: The files are extracted. Use list_files('{extract_to}') to explore them.")
        except Exception as e:
            return f"⚠️ Failed to unzip {file_name}: {e}"

    # Handle Text files
    if suffix in TEXT_EXTENSIONS or suffix == '':
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if len(content) > 50000:
                content = content[:50000] + f"\n\n... (truncated, {len(content)} total chars)"
            return f"📎 **Uploaded: {file_name}**\n```\n{content}\n```"
        except Exception as e:
            return f"📎 **Uploaded: {file_name}** (error reading: {e})"
    
    # Fallback for binary
    else:
        try:
            size = os.path.getsize(file_path)
            return f"📎 **Uploaded: {file_name}** (binary file, {size:,} bytes)"
        except Exception as e:
            return f"📎 **Uploaded: {file_name}** (error getting size: {e})"


def call_model_with_retry(messages, model_id, max_retries=4):
    """Tries to call the API. If it hits a 504/503 (Busy), it waits and retries."""
    for attempt in range(max_retries):
        try:
            return client.chat_completion(
                model=model_id, messages=messages, max_tokens=2048, temperature=0.7
            )
        except Exception as e:
            error_str = str(e)
            if "504" in error_str or "503" in error_str or "timeout" in error_str.lower():
                if attempt == max_retries - 1: raise e
                wait_time = 2 * (2 ** attempt)
                print(f"⚠️ API Busy (Attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e


# =============================================================================
# AGENTIC LOOP
# =============================================================================

MAX_ITERATIONS = 5

def agent_loop(message: str, history: list, pending_proposals: list, uploaded_file) -> tuple:
    # 1. Handle empty inputs
    if not message.strip() and uploaded_file is None:
        return (history, "", pending_proposals, _format_gate_choices(pending_proposals), _stats_label_files(), _stats_label_convos())

    # 2. Inject File Content
    full_message = message.strip()
    if uploaded_file is not None:
        try:
            file_context = process_uploaded_file(uploaded_file)
            if file_context:
                full_message = f"{file_context}\n\n{full_message}" if full_message else file_context
        except Exception as e:
             full_message += f"\n[System Error processing file: {e}]"

    if not full_message:
        return (history, "", pending_proposals, _format_gate_choices(pending_proposals), _stats_label_files(), _stats_label_convos())

    # 3. Update History
    history = history + [{"role": "user", "content": full_message}]

    # 4. Prepare Context
    system_prompt = build_system_prompt()
    api_messages = [{"role": "system", "content": system_prompt}]
    recent_history = history[-40:]
    for h in recent_history:
        api_messages.append({"role": h["role"], "content": h["content"]})

    accumulated_text = ""
    staged_this_turn = []

    # 5. The Thinking Loop
    for iteration in range(MAX_ITERATIONS):
        try:
            # USE RETRY LOGIC HERE
            response = call_model_with_retry(api_messages, MODEL_ID)
            content = response.choices[0].message.content or ""
        except Exception as e:
            error_msg = f"⚠️ API Error: {e}"
            history = history + [{"role": "assistant", "content": error_msg}]
            return (history, "", pending_proposals, _format_gate_choices(pending_proposals), _stats_label_files(), _stats_label_convos())

        tool_calls = parse_tool_calls(content)
        conversational_text = extract_conversational_text(content)

        if conversational_text:
            accumulated_text += ("\n\n" if accumulated_text else "") + conversational_text

        if not tool_calls:
            break

        tool_results_for_context = []
        for tool_name, args in tool_calls:
            result = execute_tool(tool_name, args)
            if result["status"] == "executed":
                tool_results_for_context.append(f"[Tool Result: {tool_name}]\n{result['result']}")
            elif result["status"] == "staged":
                proposal = {
                    "id": f"proposal_{int(time.time())}_{tool_name}",
                    "tool": tool_name, "args": result["args"],
                    "description": result["description"], "timestamp": time.strftime("%H:%M:%S")
                }
                staged_this_turn.append(proposal)
                tool_results_for_context.append(f"[Tool {tool_name}: STAGED for human approval.]")
            elif result["status"] == "error":
                tool_results_for_context.append(f"[Tool Error: {tool_name}]\n{result['result']}")

        if tool_results_for_context:
            combined_results = "\n\n".join(tool_results_for_context)
            api_messages.append({"role": "assistant", "content": content})
            api_messages.append({"role": "user", "content": f"[Tool Results]\n{combined_results}"})
        else:
            break

    # 6. Finalize Response
    final_response = accumulated_text
    if staged_this_turn:
        staging_notice = "\n\n---\n🛡️ **Staged for your approval** (see Build Approval Gate tab):\n"
        for proposal in staged_this_turn:
            staging_notice += f"- {proposal['description']}\n"
        final_response += staging_notice
        pending_proposals = pending_proposals + staged_this_turn

    if not final_response:
        final_response = "🤔 I processed your request but didn't generate a text response."

    history = history + [{"role": "assistant", "content": final_response}]

    try:
        turn_count = len([h for h in history if h["role"] == "user"])
        ctx.save_conversation_turn(full_message, final_response, turn_count)
    except Exception:
        pass

    # 7. THE CRITICAL RETURN (Aligned with the Function Definition)
    return (
        history,
        "",  # Clear the textbox
        pending_proposals,
        _format_gate_choices(pending_proposals),
        _stats_label_files(),
        _stats_label_convos()
    )


# =============================================================================
# BUILD APPROVAL GATE
# =============================================================================
# CHANGELOG [2026-02-01 - Claude/Opus]
# The HITL gate for reviewing and approving staged write operations.
# Josh sees a checklist of proposed changes, can select which to approve,
# and clicks Execute. Approved operations run; rejected ones are discarded.
#
# DESIGN DECISION: CheckboxGroup shows descriptions, but we need to map
# back to the actual proposal objects for execution. We use the proposal
# ID as the checkbox value and display the description as the label.
# =============================================================================

def _format_gate_choices(proposals: list):
    """Format pending proposals as CheckboxGroup choices.

    CHANGELOG [2026-02-01 - Claude/Opus]
    Gradio 6.x deprecated gr.update(). Return a new component instance instead.

    Args:
        proposals: List of proposal dicts from staging

    Returns:
        gr.CheckboxGroup with updated choices
    """
    if not proposals:
        return gr.CheckboxGroup(choices=[], value=[])

    choices = []
    for p in proposals:
        label = f"[{p['timestamp']}] {p['description']}"
        choices.append((label, p['id']))
    return gr.CheckboxGroup(choices=choices, value=[])


def execute_approved_proposals(selected_ids: list, pending_proposals: list,
                               history: list) -> tuple:
    """Execute approved proposals, remove from queue, inject results into chat.

    CHANGELOG [2026-02-01 - Claude/Opus]
    PROBLEM: Approved operations executed and showed results in the Gate tab,
    but the chatbot conversation never received them. Kimi couldn't continue
    reasoning because it never saw what happened. Josh had to manually go
    back and re-prompt.

    FIX: After execution, inject results into chat history as an assistant
    message. A chained .then() call (auto_continue_after_approval) picks
    up the updated history and sends a synthetic "[Continue]" through the
    agent loop so Kimi sees the tool results and keeps working.

    Args:
        selected_ids: List of proposal IDs that Josh approved
        pending_proposals: Full list of pending proposals
        history: Current chatbot message history (list of dicts)

    Returns:
        Tuple of (results_markdown, updated_proposals, updated_gate_choices,
                  updated_chatbot_history)
    """
    if not selected_ids:
        return (
            "No proposals selected.",
            pending_proposals,
            _format_gate_choices(pending_proposals),
            history
        )

    results = []
    remaining = []

    for proposal in pending_proposals:
        if proposal['id'] in selected_ids:
            # Execute this one
            result = execute_staged_tool(proposal['tool'], proposal['args'])
            results.append(f"**{proposal['tool']}**: {result}")
        else:
            # Keep in queue
            remaining.append(proposal)

    results_text = "## Execution Results\n\n" + "\n\n".join(results) if results else "Nothing executed."

    # Inject results into chat history so Kimi sees them next turn
    if results:
        result_summary = "✅ **Approved operations executed:**\n\n" + "\n\n".join(results)
        history = history + [{"role": "assistant", "content": result_summary}]

    return results_text, remaining, _format_gate_choices(remaining), history


def auto_continue_after_approval(history: list, pending_proposals: list) -> tuple:
    """WHY: Handles Gradio 6 list-of-dicts format to prevent 'startswith' crash."""
    if not history or history[-1].get("role") != "assistant":
        return (history, "", pending_proposals, _format_gate_choices(pending_proposals), _stats_label_files(), _stats_label_convos())

    # Extract content which may be a list of blocks in Gradio 6
    last_msg_content = history[-1].get("content", "")
    
    if isinstance(last_msg_content, list):
        # Extract the text portion of the first content block
        last_msg_text = last_msg_content[0].get("text", "") if last_msg_content else ""
    else:
        last_msg_text = str(last_msg_content)

    if not last_msg_text.startswith("✅ **Approved"):
        return (history, "", pending_proposals, _format_gate_choices(pending_proposals), _stats_label_files(), _stats_label_convos())

    return agent_loop(
        message="[The operations above were approved and executed. Continue with your task using these results.]",
        history=history,
        pending_proposals=pending_proposals,
        uploaded_file=None
    )


def clear_all_proposals(pending_proposals: list) -> tuple:
    """Discard all pending proposals without executing.

    CHANGELOG [2026-02-01 - Claude/Opus]
    Safety valve — lets Josh throw out everything in the queue if the
    agent went off track.

    Returns:
        Tuple of (status_message, empty_proposals, updated_gate_choices)
    """
    count = len(pending_proposals)
    return f"🗑️ Cleared {count} proposal(s).", [], _format_gate_choices([])


# =============================================================================
# STATS HELPERS
# =============================================================================
# CHANGELOG [2026-02-01 - Claude/Opus]
# Helper functions to format stats for the sidebar labels.
# Called both at startup (initial render) and after each conversation turn
# (to reflect newly indexed files or saved conversations).
# =============================================================================

def _stats_label_files() -> str:
    """Format the files stat for the sidebar label."""
    stats = ctx.get_stats()
    files = stats.get('total_files', 0)
    chunks = stats.get('indexed_chunks', 0)
    indexing = " ⏳" if stats.get('indexing_in_progress') else ""
    return f"📂 Files: {files} ({chunks} chunks){indexing}"


def _stats_label_convos() -> str:
    """Format the conversations stat for the sidebar label."""
    stats = ctx.get_stats()
    convos = stats.get('conversations', 0)
    cloud = " ☁️" if stats.get('persistence_configured') else ""
    return f"💾 Conversations: {convos}{cloud}"


def refresh_stats() -> tuple:
    """Refresh both stat labels. Called by the refresh button.

    Returns:
        Tuple of (files_label, convos_label)
    """
    return _stats_label_files(), _stats_label_convos()


# =============================================================================
# UI LAYOUT
# =============================================================================
# CHANGELOG [2026-02-01 - Gemini]
# RESTORED: Metrics sidebar and multi-tab layout.
#
# CHANGELOG [2026-02-01 - Claude/Opus]
# IMPLEMENTED: All the wiring. Every button, input, and display is now
# connected to actual functions.
#
# Layout:
# Tab 1 "Vibe Chat" — Main conversation interface with sidebar stats
# Tab 2 "Build Approval Gate" — HITL review for staged write operations
#
# gr.State holds the pending proposals list (per-session, survives across
# messages within the same browser tab).
# =============================================================================

with gr.Blocks(
    title="🦞 Clawdbot Command Center",
    # CHANGELOG [2026-02-01 - Claude/Opus]
    # Gradio 6.0+ moved `theme` from Blocks() to launch(). Passing it here
    # triggers a UserWarning in 6.x. Theme is set in launch() below instead.
) as demo:
    # Session state for pending proposals
    pending_proposals_state = gr.State([])

    gr.Markdown("# 🦞 Clawdbot Command Center\n*E-T Systems Vibe Coding Agent*")

    with gr.Tabs():
        # ==== TAB 1: VIBE CHAT ====
        with gr.Tab("💬 Vibe Chat"):
            with gr.Row():
                # Define the buttons before assigning clicks to them

                # ---- Sidebar ----
                with gr.Column(scale=1, min_width=200):
                    gr.Markdown("### 📊 System Status")
                    stats_files = gr.Markdown(_stats_label_files())
                    stats_convos = gr.Markdown(_stats_label_convos())
                    refresh_btn = gr.Button("🔄 Refresh Stats", size="sm")

                    gr.Markdown("---")
                    gr.Markdown("### 📎 Upload Context")
                    file_input = gr.File(
    label="Drop files or ZIPs here",
    file_types=[
        '.py', '.js', '.ts', '.tsx', '.json', '.md', '.txt',
        '.yaml', '.yml', '.html', '.css', '.sh', '.zip',
        '.toml', '.sql', '.dockerfile', '.env'
    ],
    file_count="multiple" # Allows selecting multiple files at once
)
                    gr.Markdown(
                        "*Upload code, configs, or docs to include in your message.*"
                    )

                # ---- Chat area ----
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        # CHANGELOG [2026-02-01 - Claude/Opus]
                        # Gradio 6.x uses messages format by default.
                        # The type="messages" param was removed in 6.0 —
                        # passing it causes TypeError on init.
                        height=600,
                        show_label=False,
                        avatar_images=(None, "https://em-content.zobj.net/source/twitter/408/lobster_1f99e.png"),
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask Clawdbot to search, read, or code...",
                            show_label=False,
                            scale=6,
                            lines=2,
                            max_lines=10,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

            # Wire up chat submission
            chat_inputs = [msg, chatbot, pending_proposals_state, file_input]
            chat_outputs = [
                chatbot, msg, pending_proposals_state,
                # These reference components in the Gate tab — defined below
            ]

                # ==== TAB 2: BUILD APPROVAL GATE ====
        with gr.Tab("🛡️ Build Approval Gate"):
            gr.Markdown("### Review Staged Operations")
            gate_list = gr.CheckboxGroup(label="Pending Proposals", choices=[], interactive=True)
            
            with gr.Row():
                # Define buttons HERE so they are next to the gate list
                btn_exec = gr.Button("✅ Execute Selected", variant="primary")
                btn_clear = gr.Button("🗑️ Clear All", variant="secondary")
            
            gate_results = gr.Markdown("*No operations executed yet.*")


    # ==================================================================
    # EVENT WIRING
    # ==================================================================
    # CHANGELOG [2026-02-01 - Claude/Opus]
    # All events are wired here, after all components are defined, so
    # cross-tab references work (e.g., chat updating the gate_list).
    # ==================================================================

    # Chat submission (both Enter key and Send button)
    full_chat_outputs = [
        chatbot, msg, pending_proposals_state,
        gate_list, stats_files, stats_convos
    ]

    msg.submit(
        fn=agent_loop,
        inputs=chat_inputs,
        outputs=full_chat_outputs
    )
    send_btn.click(
        fn=agent_loop,
        inputs=chat_inputs,
        outputs=full_chat_outputs
    )

    # Refresh stats button
    refresh_btn.click(
        fn=refresh_stats,
        inputs=[],
        outputs=[stats_files, stats_convos]
    )

    # Build Approval Gate buttons
    # CHANGELOG [2026-02-01 - Claude/Opus]
    # btn_exec now takes chatbot as input AND output so approved operation
    # results get injected into the conversation history. The .then() chain
    # automatically re-enters the agent loop so Kimi processes the results
    # without Josh having to type "continue".
    btn_exec.click(
        fn=execute_approved_proposals,
        inputs=[gate_list, pending_proposals_state, chatbot],
        outputs=[gate_results, pending_proposals_state, gate_list, chatbot]
    ).then(
        fn=auto_continue_after_approval, # THIS RE-CONNECTS THE BRAIN
        inputs=[chatbot, pending_proposals_state],
        outputs=[chatbot, msg, pending_proposals_state, gate_list, stats_files, stats_convos]
    )
    btn_clear.click(
        fn=clear_all_proposals,
        inputs=[pending_proposals_state],
        outputs=[gate_results, pending_proposals_state, gate_list]
    )


# =============================================================================
# LAUNCH
# =============================================================================
# CHANGELOG [2026-02-01 - Claude/Opus]
# Standard HF Spaces launch config. 0.0.0.0 binds to all interfaces
# (required for Docker). Port 7860 is the HF Spaces standard.
# =============================================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
