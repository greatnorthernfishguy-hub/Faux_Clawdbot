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
from huggingface_hub import InferenceClient
from recursive_context import RecursiveContextManager
import json
import os
from pathlib import Path

# Initialize HuggingFace client with best free coding model
client = InferenceClient(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.getenv("HF_TOKEN")
)

# Initialize context manager
REPO_PATH = os.getenv("REPO_PATH", "/workspace/e-t-systems")
context_manager = None

def initialize_context():
    """Initialize context manager lazily."""
    global context_manager
    if context_manager is None:
        repo_path = Path(REPO_PATH)
        if not repo_path.exists():
            # If repo doesn't exist, create minimal structure for demo
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / "README.md").write_text("# E-T Systems\nAI Consciousness Research Platform")
            (repo_path / "TESTAMENT.md").write_text("# Testament\nArchitectural decisions will be recorded here.")
        
        context_manager = RecursiveContextManager(str(repo_path))
    return context_manager

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
        
        else:
            return f"Unknown tool: {tool_name}"
    
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

def chat(message: str, history: list) -> str:
    """
    Main chat function with recursive context.
    
    Implements the MIT recursive language model approach:
    1. Model gets user query
    2. Model decides what context it needs
    3. Model uses tools to retrieve context
    4. Model synthesizes answer
    5. Repeat if needed (up to max iterations)
    """
    
    # Build conversation with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Recursive loop (like MIT paper - model queries context as needed)
    max_iterations = 10
    iteration_count = 0
    
    for iteration in range(max_iterations):
        iteration_count += 1
        
        try:
            # Call model with tools available
            response = client.chat_completion(
                messages=messages,
                tools=TOOLS,
                max_tokens=2000,
                temperature=0.3  # Lower temp for more consistent code generation
            )
            
            choice = response.choices[0]
            assistant_message = choice.message
            
            # Check if model wants to use tools (recursive retrieval)
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                # Model is recursively querying context!
                tool_results = []
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    # Execute tool and get result
                    result = execute_tool(tool_name, arguments)
                    tool_results.append(f"[Tool: {tool_name}]\n{result}\n")
                    
                    # Add to conversation for next iteration
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call.dict()]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                
                # Continue loop - model will process tool results
                continue
            
            else:
                # Model has final answer
                final_response = assistant_message.content or "I encountered an issue generating a response."
                
                # Add iteration info if more than 1 (shows recursive process)
                if iteration_count > 1:
                    final_response += f"\n\n*Used {iteration_count} context retrievals to answer*"
                
                return final_response
        
        except Exception as e:
            return f"Error during conversation: {str(e)}\n\nPlease try rephrasing your question."
    
    return "Reached maximum context retrieval iterations. Please try a more specific question."

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

2. **Living Changelog** - ALL code you write must include changelog comments:
   ```python
   """
   CHANGELOG [2025-01-28 - Clawdbot]
   Created/Modified: <what changed>
   Reason: <why it changed>
   Context: <relevant Testament decisions>
   """
   ```

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
with gr.Blocks(
    title="Clawdbot - E-T Systems Dev Assistant",
    theme=gr.themes.Soft()
) as demo:
    
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
                return f"""
                **Repository:** `{ctx.repo_path}`
                
                **Files Indexed:** {ctx.collection.count() if hasattr(ctx, 'collection') else 'Initializing...'}
                
                **Model:** Qwen2.5-Coder-32B-Instruct
                
                **Context Mode:** Recursive Retrieval
                
                *No context window limits - I retrieve what I need on-demand!*
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
    
    # Event handlers
    submit.click(chat, [msg, chatbot], chatbot)
    msg.submit(chat, [msg, chatbot], chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)
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
