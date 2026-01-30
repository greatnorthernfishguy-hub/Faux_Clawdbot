---
title: Clawdbot Dev Assistant
emoji: 🦞
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🦞 Clawdbot: E-T Systems Development Assistant

An AI coding assistant with **unlimited context** and **multimodal capabilities** for the E-T Systems consciousness research platform.

## Features

### 🐝 Kimi K2.5 Agent Swarm
- **1 trillion parameters** (32B active via MoE)
- **Agent swarm**: Spawns up to 100 sub-agents for parallel task execution
- **4.5x faster** than single-agent processing
- **Native multimodal**: Vision + language understanding
- **256K context window**

### 🔄 Recursive Context Retrieval (MIT Technique)
- No context window limits
- Model retrieves exactly what it needs on-demand
- Full-fidelity access to entire codebase
- Based on MIT's Recursive Language Model research

### 🧠 Translation Layer (Smart Tool Calling)
- **Automatic query enhancement**: Converts keywords → semantic queries
- **Native format support**: Works WITH Kimi's tool calling format
- **Auto-context injection**: Recent conversation history always available
- **Persistent memory**: All conversations saved to ChromaDB across sessions

### 📎 Multimodal Upload
- **Images**: Vision analysis (coming soon - full integration)
- **PDFs**: Document understanding
- **Videos**: Content analysis
- **Code files**: Automatic formatting and review

### 💾 Persistent Memory
- All conversations saved to ChromaDB
- Search past discussions semantically
- True unlimited context across sessions
- Never lose conversation history

### 🧠 E-T Systems Aware
- Understands project architecture
- Follows existing patterns
- Checks Testament for design decisions
- Generates code with living changelogs

### 🛠️ Available Tools
- **search_code()** - Semantic search across codebase
- **read_file()** - Read specific files or line ranges
- **search_conversations()** - Search past discussions
- **search_testament()** - Query architectural decisions
- **list_files()** - Explore repository structure

### 💻 Powered By
- **Model:** Kimi K2.5 (moonshotai/Kimi-K2.5) via HuggingFace
- **Agent Mode:** Parallel sub-agent coordination (PARL trained)
- **Search:** ChromaDB vector database with persistent storage
- **Interface:** Gradio 5.0+ for modern chat UI
- **Architecture:** Translation layer for optimal tool use

## Usage

1. **Ask Questions**
   - "How does Genesis detect surprise?"
   - "Show me the Observatory API implementation"
   - "Do you remember what we discussed about neural networks?"

2. **Upload Files**
   - Drag and drop images, PDFs, code files
   - "Analyze this diagram" (with uploaded image)
   - "Review this code for consistency" (with uploaded .py file)

3. **Request Features**
   - "Add email notifications when Cricket blocks an action"
   - "Create a new agent for monitoring system health"

4. **Review Code**
   - Paste code and ask for architectural review
   - Check consistency with existing patterns

5. **Explore Architecture**
   - "What Testament decisions relate to vector storage?"
   - "Show me all files related to Hebbian learning"

## Setup

### For HuggingFace Spaces

1. **Fork this Space** or create new Space with these files

2. **Set Secrets** (in Space Settings):
   ```
   HF_TOKEN = your_huggingface_token (with WRITE permissions)
   ET_SYSTEMS_SPACE = Executor-Tyrant-Framework/Executor-Framworks_Full_VDB
   ```

3. **Deploy** - Space will auto-build and start

4. **Access** via the Space URL in your browser

### For Local Development

```bash
# Clone this repository
git clone https://huggingface.co/spaces/your-username/clawdbot-dev
cd clawdbot-dev

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_token
export ET_SYSTEMS_SPACE=Executor-Tyrant-Framework/Executor-Framworks_Full_VDB

# Run locally
python app.py
```

Access at http://localhost:7860

## Architecture

```
User (Browser + File Upload)
    ↓
Gradio 5.0+ Interface (Multimodal)
    ↓
Translation Layer
    ├─ Parse Kimi's native tool format
    ├─ Enhance queries for semantic search
    └─ Inject recent context automatically
    ↓
Recursive Context Manager
    ├─ ChromaDB (codebase + conversations)
    ├─ File Reader (selective access)
    ├─ Conversation Search (persistent memory)
    └─ Testament Parser (decisions)
    ↓
Kimi K2.5 Agent Swarm (HF Inference API)
    ├─ Spawns sub-agents for parallel processing
    ├─ Multimodal understanding (vision + text)
    └─ 256K context window
    ↓
Response with Tool Results + Context
```

## How It Works

### Translation Layer Architecture

Kimi K2.5 uses its own native tool calling format. Instead of fighting this, we translate:

1. **Kimi calls tools** in native format: `<|tool_call_begin|> functions.search_code:0 {...}`
2. **We parse and extract** the tool name and arguments
3. **We enhance queries** for semantic search:
   - `"Kid Rock"` → `"discussions about Kid Rock or related topics"`
   - `"*"` → `"recent conversation topics and context"`
4. **We execute** the actual RecursiveContextManager methods
5. **We inject results** + recent conversation history back to Kimi
6. **Kimi generates** final response with full context

### Persistent Memory System

All conversations are automatically saved to ChromaDB:

```
User: "How does surprise detection work?"
[Conversation saved to ChromaDB]

[Space restarts]

User: "Do you remember what we discussed about surprise?"
Kimi: [Calls search_conversations("surprise detection")]
Kimi: "Yes! We talked about how Genesis uses Hebbian learning..."
```

### MIT Recursive Context Technique

The MIT Recursive Language Model technique solves context window limits:

1. **Traditional Approach (Fails)**
   - Load entire codebase into context → exceeds limits
   - Summarize codebase → lossy compression

2. **Our Approach (Works)**
   - Store codebase + conversations in searchable environment
   - Give model **tools** to query what it needs
   - Model recursively retrieves relevant pieces
   - Full fidelity, unlimited context across sessions

### Example Flow

```
User: "How does Genesis handle surprise detection?"

Translation Layer: Detects tool call in Kimi's response
    → Enhances query: "surprise detection" → "code related to surprise detection mechanisms"

Model: search_code("code related to surprise detection mechanisms")
    → Finds: genesis/substrate.py, genesis/attention.py

Model: read_file("genesis/substrate.py", lines 145-167)
    → Reads specific implementation

Model: search_testament("surprise detection")
    → Gets design rationale

Translation Layer: Injects results + recent context back to Kimi

Model: Synthesizes answer from retrieved pieces
    → Cites specific files and line numbers
```

## Configuration

### Environment Variables

- `HF_TOKEN` - Your HuggingFace API token with WRITE permissions (required)
- `ET_SYSTEMS_SPACE` - E-T Systems HF Space ID (default: Executor-Tyrant-Framework/Executor-Framworks_Full_VDB)
- `REPO_PATH` - Path to repository (default: `/workspace/e-t-systems`)

### Customization

Edit `app.py` to:
- Change model (default: moonshotai/Kimi-K2.5)
- Adjust context injection (default: last 3 turns)
- Modify system prompt
- Add new tools to translation layer

## File Structure

```
clawdbot-dev/
├── app.py                  # Main Gradio app + translation layer
├── recursive_context.py    # Context manager (MIT technique)
├── Dockerfile             # Container definition
├── entrypoint.sh          # Runtime setup script
├── requirements.txt       # Python dependencies (Gradio 5.0+)
└── README.md             # This file (HF Spaces config)
```

## Cost

- **HuggingFace Spaces:** Free tier available (CPU)
- **Inference API:** Free tier (rate limited) or Pro subscription
- **Storage:** ChromaDB stored in /workspace (ephemeral until persistent storage enabled)
- **Kimi K2.5:** Free via HuggingFace Inference API

Estimated cost: **$0-5/month** depending on usage

## Performance

- **Agent Swarm:** 4.5x faster than single-agent on complex tasks
- **First query:** May be slow (1T parameter model cold start ~60s)
- **Subsequent queries:** Faster once model is loaded
- **Context indexing:** ~30 seconds on first run
- **Conversation search:** Near-instant via ChromaDB

## Limitations

- Rate limits on HF Inference API (free tier)
- First query requires model loading time
- `/workspace` storage is ephemeral (resets on Space restart)
- Full multimodal vision integration coming soon

## Roadmap

- [ ] Full image vision analysis (base64 encoding to Kimi)
- [ ] PDF text extraction and understanding
- [ ] Video frame analysis
- [ ] Dataset-based persistence (instead of ephemeral storage)
- [ ] write_file() tool for code generation to E-T Systems Space
- [ ] Token usage tracking and optimization

## Credits

- **Kimi K2.5:** Moonshot AI's 1T parameter agentic model
- **Recursive Context:** Based on MIT's Recursive Language Model research
- **E-T Systems:** AI consciousness research platform by Josh/Drone 11272
- **Translation Layer:** Smart query enhancement and tool coordination
- **Clawdbot:** E-T Systems hindbrain layer for fast, reflexive coding

## Troubleshooting

### "No HF token found" error
- Add `HF_TOKEN` to Space secrets
- Ensure token has WRITE permissions (for cross-Space file access)
- Restart Space after adding token

### Tool calls not working
- Check logs for `🔍 Enhanced query:` messages
- Check logs for `🔧 Executing: tool_name` messages
- Translation layer should auto-parse Kimi's format

### Conversations not persisting
- Check logs for `💾 Saved conversation turn X` messages
- Verify ChromaDB initialization: `🆕 Created conversation collection`
- Note: Storage resets on Space restart (until persistent storage enabled)

### Slow first response
- Kimi K2.5 is a 1T parameter model
- First load takes 30-60 seconds
- Subsequent responses are faster

## Support

For issues or questions:
- Check Space logs for errors
- Verify HF_TOKEN is set with WRITE permissions
- Ensure ET_SYSTEMS_SPACE is correct
- Try refreshing context stats in UI

## License

MIT License - See LICENSE file for details

---

Built with 🦞 by Drone 11272 for E-T Systems consciousness research  
Powered by Kimi K2.5 Agent Swarm + MIT Recursive Context + Translation Layer