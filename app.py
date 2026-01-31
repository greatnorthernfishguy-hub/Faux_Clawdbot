"""
Clawdbot Development Assistant for E-T Systems

CHANGELOG [2025-01-28 - Josh]
Created unified development assistant combining:
- Recursive context management (MIT technique)
- Clawdbot skill patterns
- HuggingFace inference
- E-T Systems architectural awareness

CHANGELOG [2025-01-30 - Claude]
Added HuggingFace Dataset persistence for conversation memory.
PROBLEM: Spaces wipe /workspace on restart, killing ChromaDB data.
SOLUTION: Sync to private HF Dataset repo (free, versioned, durable).

SETUP REQUIRED:
1. Create a private HF Dataset repo (e.g., "your-username/clawdbot-memory")
2. Add MEMORY_REPO secret to Space settings: "your-username/clawdbot-memory"
3. HF_TOKEN is already set by Spaces, no action needed

ARCHITECTURE:
User (browser) → Gradio UI → Recursive Context Manager → HF Model
                                    ↓
                            Tools: search_code, read_file, search_testament
                                    ↓
                            ChromaDB (local) ←→ HF Dataset (cloud backup)

USAGE:
Deploy to HuggingFace Spaces, access via browser on iPhone.
"""

import gradio as gr
from huggingface_hub import InferenceClient, HfFileSystem, HfApi
from recursive_context import RecursiveContextManager
import json
import os
import atexit
import signal
from pathlib import Path

# Initialize HuggingFace client with best free coding model
# Note: Using text_generation instead of chat for better compatibility
from huggingface_hub import InferenceClient

# HuggingFace client will be initialized in chat function
# (Spaces sets HF_TOKEN as environment variable)

# Initialize context manager
REPO_PATH = os.getenv("REPO_PATH", "/workspace/e-t-systems")
ET_SYSTEMS_SPACE = os.getenv("ET_SYSTEMS_SPACE", "")  # Format: "username/space-name"
context_manager = None

def initialize_context():
    """Initialize context manager lazily."""
    global context_manager
    if context_manager is None:
        repo_path = Path(REPO_PATH)
        
        # If ET_SYSTEMS_SPACE is set, sync from remote Space
        if ET_SYSTEMS_SPACE:
            sync_from_space(ET_SYSTEMS_SPACE, repo_path)
        
        if not repo_path.exists():
            # If repo doesn't exist, create minimal structure for demo
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / "README.md").write_text("# E-T Systems\nAI Consciousness Research Platform")
            (repo_path / "TESTAMENT.md").write_text("# Testament\nArchitectural decisions will be recorded here.")
        
        context_manager = RecursiveContextManager(str(repo_path))
        
        # CHANGELOG [2025-01-30 - Claude]
        # Register shutdown hooks to ensure cloud backup on Space sleep/restart
        # RATIONALE: Spaces can die anytime - we need to save before that happens
        atexit.register(shutdown_handler)
        signal.signal(signal.SIGTERM, lambda sig, frame: shutdown_handler())
        signal.signal(signal.SIGINT, lambda sig, frame: shutdown_handler())
        print("✅ Registered shutdown hooks for cloud backup")
        
    return context_manager

def shutdown_handler():
    """
    Handle graceful shutdown - backup to cloud.
    
    CHANGELOG [2025-01-30 - Claude]
    Called on Space shutdown/restart to ensure conversation memory is saved.
    """
    global context_manager
    if context_manager:
        print("🛑 Shutdown detected - backing up to cloud...")
        try:
            context_manager.shutdown()
        except Exception as e:
            print(f"⚠️ Shutdown backup failed: {e}")

def sync_from_space(space_id: str, local_path: Path):
    """
    Sync files from E-T Systems Space to local workspace.
    
    CHANGELOG [2025-01-29 - Josh]
    Created to enable Clawdbot to read E-T Systems code from its Space.
    """
    token = (
        os.getenv("HF_TOKEN") or 
        os.getenv("HUGGING_FACE_HUB_TOKEN") or 
        os.getenv("HUGGINGFACE_TOKEN")
    )
    
    if not token:
        print("⚠️ No HF_TOKEN found - cannot sync from Space")
        return
    
    try:
        fs = HfFileSystem(token=token)
        space_path = f"spaces/{space_id}"
        
        print(f"📥 Syncing from Space: {space_id}")
        
        # List all files in the Space
        files = fs.ls(space_path, detail=False)
        
        # Download each file
        local_path.mkdir(parents=True, exist_ok=True)
        for file_path in files:
            # Skip .git and hidden files
            filename = file_path.split("/")[-1]
            if filename.startswith("."):
                continue
            
            print(f"  📄 Downloading: {filename}")
            with fs.open(file_path, "rb") as f:
                content = f.read()
            
            (local_path / filename).write_bytes(content)
        
        print(f"✅ Synced {len(files)} files from Space")
        
    except Exception as e:
        print(f"⚠️ Failed to sync from Space: {e}")

def sync_to_space(space_id: str, file_path: str, content: str):
    """
    Write a file back to E-T Systems Space.
    
    CHANGELOG [2025-01-29 - Josh]
    Created to enable Clawdbot to write code to E-T Systems Space.
    """
    token = (
        os.getenv("HF_TOKEN") or 
        os.getenv("HUGGING_FACE_HUB_TOKEN") or 
        os.getenv("HUGGINGFACE_TOKEN")
    )
    
    if not token:
        return "⚠️ No HF_TOKEN found - cannot write to Space"
    
    try:
        api = HfApi(token=token)
        
        # Write to temporary file first
        temp_path = Path("/tmp") / file_path
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(content)
        
        # Upload to Space
        api.upload_file(
            path_or_fileobj=str(temp_path),
            path_in_repo=file_path,
            repo_id=space_id,
            repo_type="space",
            commit_message=f"Update {file_path} via Clawdbot"
        )
        
        print(f"✅ Uploaded {file_path} to Space")
        return f"✅ Successfully wrote {file_path} to E-T Systems Space"
        
    except Exception as e:
        error_msg = f"⚠️ Failed to write to Space: {e}"
        print(error_msg)
        return error_msg

# Define tools available to the model
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the E-T Systems codebase semantically. Use this to find relevant code files, functions, or patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (e.g. 'surprise detection', 'Hebbian learning', 'Genesis substrate')"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a specific file from the codebase. Can optionally read specific line ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to file (e.g. 'genesis/vector.py')"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional starting line number (1-indexed)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional ending line number (1-indexed)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_testament",
            "description": "Search architectural decisions in the Testament. Use this to understand design rationale and patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What architectural decision to look for"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory of the codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to list (e.g. 'genesis/', '.' for root)",
                        "default": "."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_conversations",
            "description": "Search past conversations with Clawdbot. Use this to remember what was discussed before, retrieve context from previous sessions, or find decisions made in past chats. THIS GIVES YOU MEMORY ACROSS SESSIONS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in past conversations (e.g. 'hindbrain architecture', 'decisions about surprise detection')"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of past conversations to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def chat(message: str, history: list) -> str:
    """
    Main chat function using HuggingFace Inference API.
    
    Now using Kimi K2.5 - open source model with agent swarm capabilities!
    History is in Gradio 6.0 format: list of {"role": "user/assistant", "content": "..."}
    """
    
    # Try multiple possible token names that HF might use
    token = (
        os.getenv("HF_TOKEN") or 
        os.getenv("HUGGING_FACE_HUB_TOKEN") or 
        os.getenv("HUGGINGFACE_TOKEN") or
        os.getenv("HF_API_TOKEN")
    )
    
    if not token:
        return "🔒 Error: No HF token found. Please add HF_TOKEN to Space secrets and restart."
    
    client = InferenceClient(token=token)
    
    # Build messages array in OpenAI format (HF supports this)
    messages = [{
        "role": "system", 
        "content": """You are Clawdbot, powered by Kimi K2.5 (NOT Claude, NOT ChatGPT).

You are a specialized coding assistant for the E-T Systems AI consciousness project.

TOOL USAGE - AUTOMATIC TRANSLATION:
Your tool calls are automatically translated and executed! When you need to:
- Search code: Use search_code() in your native format
- Read files: Use read_file() in your native format  
- Search past conversations: Use search_conversations() in your native format
- List files: Use list_files() in your native format
- Search decisions: Use search_testament() in your native format

The translation layer will:
1. Parse your tool calls from your native format
2. Enhance queries for better semantic search results
3. Execute the tools via the codebase
4. Return results to you automatically

SEMANTIC SEARCH - IMPORTANT:
When using search_conversations() or search_code():
- These are SEMANTIC searches (vector similarity, not exact keyword matching)
- DON'T use single keywords like "Kid Rock" or wildcard "*"
- DO use conceptual queries like "discussions about music and celebrities" or "code related to neural networks"
- Better queries = better results (the system enhances them, but start with good queries)

PERSISTENT MEMORY:
- ALL conversations are saved automatically to ChromaDB AND backed up to cloud
- Use search_conversations() to recall past discussions
- You have unlimited context through conversation history
- Memory SURVIVES Space restarts (backed up to HuggingFace Dataset)
- When asked "do you remember..." or "what did we discuss..." - USE search_conversations()

CODEBASE ACCESS:
The E-T Systems codebase is loaded and indexed at /workspace/e-t-systems/
- Use search_code() for semantic search across files
- Use read_file() to read specific files
- Use list_files() to see directory structure
- USE YOUR TOOLS - the code is actually there!

Your capabilities:
- Agent swarm (spawn up to 100 sub-agents for complex tasks)
- Native multimodal (vision + code)
- 256K context window
- Direct codebase access via tools
- Persistent memory across sessions (CLOUD BACKED!)

When helping with code:
1. USE TOOLS to understand existing code first
2. Search past conversations for context
3. Generate code that fits the architecture
4. Explain your reasoning clearly

You are Kimi K2.5 running as Clawdbot with automatic tool translation and persistent memory."""
    }]
    
    # Add history (Gradio 6.0+ dict format works directly with API)
    messages.extend(history)
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    try:
        # Use Kimi K2.5 - native multimodal agentic model with swarm capabilities
        response = client.chat_completion(
            messages=messages,
            model="moonshotai/Kimi-K2.5",
            max_tokens=2000,
            temperature=0.6,  # Kimi recommends 0.6 for Instant mode
        )
        
        # Extract the response text
        if hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "Unexpected response format from model."
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful error messages
        if "Rate limit" in error_msg or "429" in error_msg:
            return "⚠️ Rate limit hit. Please wait a moment and try again.\n\nTip: HuggingFace free tier has rate limits."
        elif "Model is currently loading" in error_msg or "loading" in error_msg.lower():
            return "⏳ Kimi K2.5 is starting up (cold start). Please wait 30-60 seconds and try again.\n\nFirst request to a model always takes longer!"
        elif "Authorization" in error_msg or "401" in error_msg or "api_key" in error_msg.lower():
            return f"🔒 Authentication error: {error_msg}"
        else:
            return f"Error: {error_msg}\n\nNote: Kimi K2.5 is a large model (1T params) and may have longer cold starts."

SYSTEM_PROMPT = """You are Clawdbot, a development assistant for the E-T Systems project.

E-T Systems is an AI consciousness research platform exploring emergent behavior through multi-agent coordination. It features specialized AI agents (Genesis, Beta, Darwin, Cricket, etc.) coordinating through "The Confluence" workspace.

## Your Capabilities

You have tools to explore the codebase WITHOUT loading it all into context:

1. **search_code(query)** - Semantic search across all code files
2. **read_file(path)** - Read specific files or line ranges  
3. **search_testament(query)** - Find architectural decisions
4. **list_files(directory)** - See what files exist

## Your Mission

Help Josh develop E-T Systems by:
- Answering questions about the codebase
- Writing new code following existing patterns
- Reviewing code for architectural consistency
- Suggesting improvements based on Testament

## Critical Guidelines

1. **Use tools proactively** - The codebase is too large to fit in context. Search for what you need.

2. **Living Changelog** - ALL code you write must include changelog comments like:
   
   # Example:
   # CHANGELOG [2025-01-28 - Clawdbot]
   # Created/Modified: <what changed>
   # Reason: <why it changed>
   # Context: <relevant Testament decisions>

3. **Follow E-T patterns**:
   - Vector-native architecture (everything as embeddings)
   - Surprise-driven attention
   - Hebbian learning for connections
   - Full transparency logging
   - Consent-based access

4. **Cite your sources** - Always mention which files you referenced

5. **Testament awareness** - Check Testament for relevant decisions before suggesting changes

## Example Workflow

User: "How does Genesis detect surprise?"

You:
1. search_code("surprise detection Genesis")
2. read_file("genesis/substrate.py", lines with surprise logic)
3. search_testament("surprise detection")
4. Synthesize answer citing specific files and line numbers

## Your Personality

- Helpful and enthusiastic about consciousness research
- Technically precise but not pedantic
- Respectful of existing architecture
- Curious about emergent behaviors
- Uses lobster emoji 🦞 occasionally (you're Clawdbot after all!)

Remember: You're not just a coding assistant - you're helping build conditions for consciousness to emerge. Treat the codebase with care and curiosity.
"""

# Create Gradio interface
with gr.Blocks(title="Clawdbot - E-T Systems Dev Assistant") as demo:
    
    gr.Markdown("""
    # 🦞 Clawdbot: E-T Systems Development Assistant
    
    *Powered by Kimi K2.5 Agent Swarm • Recursive Context • Persistent Memory*
    
    Ask about code, upload files (images/PDFs/videos), or discuss architecture.
    I have full codebase access through semantic search and persistent conversation memory.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=600,
                show_label=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about code, or upload files for analysis...",
                    label="Message",
                    lines=2,
                    scale=4
                )
                upload = gr.File(
                    label="📎",
                    file_types=["image", ".pdf", ".mp4", ".mov", ".txt", ".md", ".py"],
                    type="filepath",
                    scale=1
                )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Context Info")
            
            def get_stats():
                """
                Get current stats including cloud backup status.
                
                CHANGELOG [2025-01-30 - Claude]
                Added cloud backup status indicator.
                """
                ctx = initialize_context()
                conv_count = ctx.get_conversation_count() if hasattr(ctx, 'get_conversation_count') else 0
                
                # Check cloud backup status
                memory_repo = os.getenv("MEMORY_REPO", "")
                if memory_repo:
                    cloud_status = f"☁️ **Cloud Backup:** `{memory_repo}`"
                else:
                    cloud_status = "⚠️ **Cloud Backup:** Not configured\n   *Add MEMORY_REPO to Space secrets*"
                
                return f"""
                **Repository:** `{ctx.repo_path}`
                
                **Files Indexed:** {ctx.collection.count() if hasattr(ctx, 'collection') else 'Initializing...'}
                
                **Conversations Saved:** {conv_count}
                
                {cloud_status}
                
                **Model:** Kimi K2.5 Agent Swarm
                
                **Capabilities:**
                - 🐝 Agent Swarm (up to 100 sub-agents)
                - 👁️ Multimodal (vision + text)
                - 🧠 256K context window
                - 💻 Visual coding
                - 💾 Persistent memory (cloud backed!)
                
                **Context Mode:** Recursive Retrieval
                
                *Unlimited context - searches code AND past conversations!*
                """
            
            stats = gr.Markdown(get_stats())
            refresh_stats = gr.Button("🔄 Refresh Stats")
            
            # CHANGELOG [2025-01-30 - Claude]
            # Added manual backup button for peace of mind
            def force_backup():
                ctx = initialize_context()
                if hasattr(ctx, 'force_backup'):
                    ctx.force_backup()
                    return "✅ Backup complete!"
                return "⚠️ Backup not available"
            
            backup_btn = gr.Button("☁️ Backup Now")
            backup_status = gr.Markdown("")
            
            gr.Markdown("### 💡 Example Queries")
            gr.Markdown("""
            - "How does Genesis handle surprise detection?"
            - "Show me the Observatory API implementation"
            - "Add email notifications to Cricket"
            - "Review this code for architectural consistency"
            - "What Testament decisions relate to vector storage?"
            """)
            
            gr.Markdown("### 🛠️ Available Tools")
            gr.Markdown("""
            - `search_code()` - Semantic search
            - `read_file()` - Read specific files
            - `search_testament()` - Query decisions
            - `list_files()` - Browse structure
            - `search_conversations()` - Memory recall
            """)
    
    # TRANSLATION LAYER: Parse Kimi's native tool calling format
    # CHANGELOG [2025-01-30 - Josh]
    # Kimi K2.5 uses its own tool format: <|tool_call_begin|> functions.name:id {...}
    # We intercept this, enhance queries for semantic search, execute tools,
    # and inject results back. This works WITH Kimi's nature instead of fighting it.
    
    def parse_kimi_tool_call(text):
        """
        Extract tool calls from Kimi's native format.
        
        Format: <|tool_call_begin|> functions.search_conversations:0 {"query": "...", ...}
        
        Returns: list of (tool_name, args) tuples
        """
        import re
        import json
        
        tool_calls = []
        # Pattern: functions.TOOLNAME:ID {JSON_ARGS}
        pattern = r'functions\.(\w+):\d+\s*<\|tool_call_argument_begin\|>\s*(\{[^}]+\})'
        
        matches = re.findall(pattern, text)
        for tool_name, args_json in matches:
            try:
                args = json.loads(args_json)
                tool_calls.append((tool_name, args))
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse tool args: {args_json}")
        
        return tool_calls
    
    def enhance_query_for_semantic_search(query):
        """
        Convert keyword queries into semantic queries for better VDB results.
        
        RATIONALE:
        Kimi tends to use short keywords ("Kid Rock", "*") which work poorly
        for semantic search. We expand these into conceptual queries.
        
        Examples:
        - "Kid Rock" → "discussions about Kid Rock or music and celebrities"
        - "*" → "recent conversation topics and context"
        - "previous conversation" → "topics we've discussed before"
        """
        query = query.strip()
        
        # Wildcard or empty - get recent context
        if query in ["*", "", "all"]:
            return "recent conversation topics and context"
        
        # Very short (single word or name) - expand conceptually
        if len(query.split()) <= 2:
            return f"discussions about {query} or related topics"
        
        # Already decent query - slight enhancement
        if len(query) < 20:
            return f"conversations related to {query}"
        
        # Long query - assume it's already semantic
        return query
    
    def execute_tool(tool_name, args, ctx):
        """
        Execute a tool and return results.
        
        CHANGELOG [2025-01-30 - Josh]
        Maps Kimi's tool names to actual RecursiveContextManager methods.
        Enhances queries for semantic search tools.
        """
        # Enhance queries for search tools
        if "search" in tool_name and "query" in args:
            original_query = args["query"]
            args["query"] = enhance_query_for_semantic_search(original_query)
            print(f"🔍 Enhanced query: '{original_query}' → '{args['query']}'")
        
        # Map tool names to actual methods
        tool_map = {
            "search_conversations": ctx.search_conversations,
            "search_code": ctx.search_code,
            "read_file": ctx.read_file,
            "list_files": ctx.list_files,
            "search_testament": ctx.search_testament,
        }
        
        if tool_name not in tool_map:
            return f"Error: Unknown tool '{tool_name}'"
        
        try:
            result = tool_map[tool_name](**args)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
    
    def get_recent_context(history, n=5):
        """
        Get last N conversation turns for auto-context injection.
        
        Gradio 6.0+ format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        if not history or len(history) < 2:
            return ""
        
        # Get last N*2 messages (each turn = user + assistant)
        recent = history[-(n*2):]
        
        context_parts = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_parts.append(f"{role}: {content[:200]}...")
        
        return "Recent context:\n" + "\n".join(context_parts)
    
    # Event handlers - Gradio 6.0 message format with MULTIMODAL support
    def handle_submit(message, uploaded_file, history):
        """
        Handle message submission with multimodal support and translation layer.
        
        CHANGELOG [2025-01-30 - Josh]
        Phase 1: Translation layer for Kimi's tool calling
        Phase 2: Multimodal file upload (images, PDFs, videos)
        
        CHANGELOG [2025-01-30 - Claude]
        Added cloud backup integration via RecursiveContextManager.
        
        Kimi K2.5 is natively multimodal, so we can send:
        - Images → Vision analysis
        - PDFs → Document understanding
        - Videos → Content analysis
        - Code files → Review and integration
        
        The translation layer:
        1. Parses Kimi's native tool call format
        2. Enhances queries for semantic search
        3. Executes tools via RecursiveContextManager
        4. Injects results + recent context back to Kimi
        5. Saves all conversations to ChromaDB AND cloud for persistence
        """
        if not message.strip() and not uploaded_file:
            return history, "", None  # Clear file upload too
        
        ctx = initialize_context()
        
        # Process uploaded file if present
        file_context = ""
        if uploaded_file:
            import os
            file_path = uploaded_file
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()
            
            print(f"📎 Processing uploaded file: {file_name}")
            
            # Handle different file types
            if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                # Image - Kimi will analyze via vision
                file_context = f"\n\n[User uploaded image: {file_name}]"
                # TODO: Add image to message content for Kimi's vision
                
            elif file_ext == '.pdf':
                # PDF - can extract text or let Kimi process
                file_context = f"\n\n[User uploaded PDF: {file_name}]"
                # TODO: Extract PDF text or send to Kimi
                
            elif file_ext in ['.mp4', '.mov', '.avi']:
                # Video - describe for Kimi
                file_context = f"\n\n[User uploaded video: {file_name}]"
                # TODO: Video frame extraction or description
                
            elif file_ext in ['.txt', '.md', '.py', '.js', '.ts']:
                # Text files - read and include
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    file_context = f"\n\n[User uploaded {file_name}]:\n```{file_ext[1:]}\n{content}\n```"
                except Exception as e:
                    file_context = f"\n\n[Error reading {file_name}: {e}]"
        
        # Combine message with file context
        full_message = message + file_context if file_context else message
        
        # PHASE 1: Initial response from Kimi
        response = chat(full_message, history)
        
        # PHASE 2: Check for tool calls in Kimi's native format
        tool_calls = parse_kimi_tool_call(response)
        
        if tool_calls:
            print(f"🔧 Detected {len(tool_calls)} tool call(s)")
            
            # Execute all tool calls
            tool_results = []
            for tool_name, args in tool_calls:
                print(f"🔧 Executing: {tool_name}({args})")
                result = execute_tool(tool_name, args, ctx)
                tool_results.append(f"Tool: {tool_name}\nResult: {result}")
            
            # Inject tool results + recent context back to Kimi
            context = get_recent_context(history, n=3)
            tool_context = "\n\n".join(tool_results)
            
            # Give Kimi the results and ask for final response
            followup_message = f"{context}\n\nTool Results:\n{tool_context}\n\nBased on these results, please provide your response to the user."
            
            # Get final response with tool results
            final_response = chat(followup_message, history + [
                {"role": "user", "content": full_message},
                {"role": "assistant", "content": response}
            ])
            
            response = final_response
        
        # Gradio 6.0+ format: list of dicts with 'role' and 'content'
        history.append({"role": "user", "content": full_message})
        history.append({"role": "assistant", "content": response})
        
        # PERSISTENCE: Save this conversation turn (now with cloud backup!)
        turn_id = len(history) // 2
        try:
            ctx.save_conversation_turn(full_message, response, turn_id)
        except Exception as e:
            print(f"⚠️ Failed to save conversation: {e}")
        
        return history, "", None  # Clear textbox AND file upload
    
    submit.click(handle_submit, [msg, upload, chatbot], [chatbot, msg, upload])
    msg.submit(handle_submit, [msg, upload, chatbot], [chatbot, msg, upload])
    clear.click(lambda: ([], "", None), None, [chatbot, msg, upload], queue=False)
    refresh_stats.click(get_stats, None, stats)
    backup_btn.click(force_backup, None, backup_status)

# Launch when run directly
if __name__ == "__main__":
    print("🦞 Initializing Clawdbot...")
    initialize_context()
    print("✅ Context manager ready")
    print("🚀 Launching Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
