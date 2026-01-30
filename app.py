"""
Clawdbot Development Assistant for E-T Systems

CHANGELOG [2025-01-28 - Josh]
Created unified development assistant combining:
- Recursive context management (MIT technique)
- Clawdbot skill patterns
- HuggingFace inference
- E-T Systems architectural awareness

ARCHITECTURE:
User (browser) → Gradio UI → Recursive Context Manager → HF Model
                                    ↓
                            Tools: search_code, read_file, search_testament

USAGE:
Deploy to HuggingFace Spaces, access via browser on iPhone.
"""

import gradio as gr
from huggingface_hub import InferenceClient, HfFileSystem, HfApi
from recursive_context import RecursiveContextManager
import json
import os
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
    return context_manager

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

def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Execute tool calls from the model.
    
    This is where the recursive context magic happens -
    the model can search and read only what it needs.
    """
    ctx = initialize_context()
    
    try:
        if tool_name == "search_code":
            results = ctx.search_code(
                arguments['query'],
                n_results=arguments.get('n_results', 5)
            )
            return json.dumps(results, indent=2)
        
        elif tool_name == "read_file":
            lines = None
            if 'start_line' in arguments and 'end_line' in arguments:
                lines = (arguments['start_line'], arguments['end_line'])
            content = ctx.read_file(arguments['path'], lines)
            return content
        
        elif tool_name == "search_testament":
            result = ctx.search_testament(arguments['query'])
            return result
        
        elif tool_name == "list_files":
            directory = arguments.get('directory', '.')
            files = ctx.list_files(directory)
            return json.dumps(files, indent=2)
        
        elif tool_name == "search_conversations":
            # CHANGELOG [2025-01-30 - Josh]: Added conversation search
            # Enables Kimi to remember past discussions across sessions
            results = ctx.search_conversations(
                arguments['query'],
                n_results=arguments.get('n_results', 5)
            )
            if not results:
                return "No past conversations found. This might be a fresh session."
            
            formatted = "=== PAST CONVERSATIONS ===\n\n"
            for conv in results:
                formatted += f"Turn {conv['turn']}:\n"
                formatted += f"USER: {conv['user']}\n"
                formatted += f"ASSISTANT: {conv['assistant']}\n"
                formatted += "---\n"
            return formatted
        
        else:
            return f"Unknown tool: {tool_name}"
    
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

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

IMPORTANT - YOU HAVE ACCESS TO THE CODEBASE:
- You can search code with search_code()
- You can read files with read_file()
- You can list files with list_files()
- You can search architectural decisions with search_testament()

CRITICAL - YOU HAVE PERSISTENT MEMORY:
- Use search_conversations() to remember past discussions
- ALL conversations are saved automatically and persist across sessions
- When asked "do you remember..." or "what did we discuss..." - USE search_conversations()
- You have UNLIMITED CONTEXT through conversation history

The E-T Systems codebase is already loaded and indexed. When asked about the codebase, USE YOUR TOOLS to actually read it.

Your capabilities:
- Agent swarm (spawn up to 100 sub-agents for complex tasks)
- Native multimodal (vision + code)
- 256K context window
- Direct codebase access via tools
- Persistent memory across sessions via search_conversations()

When helping with code:
1. First USE TOOLS to understand existing code
2. Search past conversations if context might be there
3. Generate code that fits the existing architecture
4. Explain your reasoning clearly

You are NOT Claude. You are Kimi K2.5 running as Clawdbot with persistent memory."""
    }]
    
    # Add history (already in correct format from Gradio 6.0)
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
    
    *Powered by Recursive Context Retrieval (MIT) + Qwen2.5-Coder-32B*
    
    Ask me anything about the E-T Systems codebase, request new features, 
    review code, or discuss architecture. I have access to the full repository
    through semantic search and can retrieve exactly what I need.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=600,
                show_label=False,
                avatar_images=(None, "🦞")
            )
            
            msg = gr.Textbox(
                placeholder="Ask about the codebase, request features, or paste code for review...",
                label="Message",
                lines=3
            )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Context Info")
            
            def get_stats():
                ctx = initialize_context()
                conv_count = ctx.get_conversation_count() if hasattr(ctx, 'get_conversation_count') else 0
                return f"""
                **Repository:** `{ctx.repo_path}`
                
                **Files Indexed:** {ctx.collection.count() if hasattr(ctx, 'collection') else 'Initializing...'}
                
                **Conversations Saved:** {conv_count}
                
                **Model:** Kimi K2.5
                
                **Capabilities:**
                - 🐝 Agent Swarm (up to 100 sub-agents)
                - 👁️ Multimodal (vision + text)
                - 🧠 256K context window
                - 💻 Visual coding
                - 💾 Persistent memory across sessions
                
                **Context Mode:** Recursive Retrieval
                
                *Unlimited context - searches code AND past conversations!*
                """
            
            stats = gr.Markdown(get_stats())
            refresh_stats = gr.Button("🔄 Refresh Stats")
            
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
            """)
    
    # Event handlers - Gradio 6.0 message format
    def handle_submit(message, history):
        """
        Handle message submission and update chat history.
        
        CHANGELOG [2025-01-30 - Josh]
        Added conversation persistence - every turn is saved to ChromaDB
        for unlimited context across sessions.
        """
        if not message.strip():
            return history, ""
        
        # Get response
        response = chat(message, history)
        
        # Gradio 6.0 format: list of dicts with 'role' and 'content'
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # PERSISTENCE: Save this conversation turn
        # Turn ID = current history length (monotonic, unique)
        turn_id = len(history) // 2  # Divide by 2 since each turn has user + assistant
        ctx = initialize_context()
        try:
            ctx.save_conversation_turn(message, response, turn_id)
        except Exception as e:
            print(f"⚠️ Failed to save conversation: {e}")
        
        return history, ""  # Return empty string to clear textbox
    
    submit.click(handle_submit, [msg, chatbot], [chatbot, msg])
    msg.submit(handle_submit, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)
    refresh_stats.click(get_stats, None, stats)

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
