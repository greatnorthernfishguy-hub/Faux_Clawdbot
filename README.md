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

An AI coding assistant with **unlimited context** for the E-T Systems consciousness research platform.

## Features

### 🔄 Recursive Context Retrieval (MIT Technique)
- No context window limits
- Model retrieves exactly what it needs on-demand
- Full-fidelity access to entire codebase
- Based on MIT's Recursive Language Model research

### 🧠 E-T Systems Aware
- Understands project architecture
- Follows existing patterns
- Checks Testament for design decisions
- Generates code with living changelogs

### 🛠️ Available Tools
- **search_code()** - Semantic search across codebase
- **read_file()** - Read specific files or line ranges
- **search_testament()** - Query architectural decisions
- **list_files()** - Explore repository structure

### 💻 Powered By
- **Model:** Qwen2.5-Coder-32B-Instruct (HuggingFace)
- **Search:** ChromaDB vector database
- **Interface:** Gradio for iPhone browser access

## Usage

1. **Ask Questions**
   - "How does Genesis detect surprise?"
   - "Show me the Observatory API implementation"

2. **Request Features**
   - "Add email notifications when Cricket blocks an action"
   - "Create a new agent for monitoring system health"

3. **Review Code**
   - Paste code and ask for architectural review
   - Check consistency with existing patterns

4. **Explore Architecture**
   - "What Testament decisions relate to vector storage?"
   - "Show me all files related to Hebbian learning"

## Setup

### For HuggingFace Spaces

1. **Fork this Space** or create new Space with these files

2. **Set Secrets** (in Space Settings):
   ```
   HF_TOKEN = your_huggingface_token
   REPO_URL = https://github.com/your-username/e-t-systems (optional)
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

# Clone your E-T Systems repo
git clone https://github.com/your-username/e-t-systems /workspace/e-t-systems

# Run locally
python app.py
```

Access at http://localhost:7860

## Architecture

```
User (Browser)
    ↓
Gradio Interface
    ↓
Recursive Context Manager
    ├─ ChromaDB (semantic search)
    ├─ File Reader (selective access)
    └─ Testament Parser (decisions)
    ↓
HuggingFace Inference API
    ├─ Model: Qwen2.5-Coder-32B
    └─ Tool Calling Enabled
    ↓
Response with Citations
```

## How It Works

The MIT Recursive Language Model technique solves context window limits:

1. **Traditional Approach (Fails)**
   - Load entire codebase into context → exceeds limits
   - Summarize codebase → lossy compression

2. **Our Approach (Works)**
   - Store codebase in searchable environment
   - Give model **tools** to query what it needs
   - Model recursively retrieves relevant pieces
   - Full fidelity, no limits

### Example Flow

```
User: "How does Genesis handle surprise detection?"

Model: search_code("Genesis surprise detection")
    → Finds: genesis/substrate.py, genesis/attention.py

Model: read_file("genesis/substrate.py", lines 145-167)
    → Reads specific implementation

Model: search_testament("surprise detection")
    → Gets design rationale

Model: Synthesizes answer from retrieved pieces
    → Cites specific files and line numbers
```

## Configuration

### Environment Variables

- `HF_TOKEN` - Your HuggingFace API token (required)
- `REPO_PATH` - Path to repository (default: `/workspace/e-t-systems`)
- `REPO_URL` - Git URL to clone on startup (optional)

### Customization

Edit `app.py` to:
- Change model (default: Qwen2.5-Coder-32B-Instruct)
- Adjust max iterations (default: 10)
- Modify system prompt
- Add new tools

## File Structure

```
clawdbot-dev/
├── app.py                  # Main Gradio application
├── recursive_context.py    # Context manager (MIT technique)
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
└── README.md             # This file (HF Spaces config)
```

## Cost

- **HuggingFace Spaces:** Free tier available
- **Inference API:** Free tier (rate limited) or Pro subscription
- **Storage:** Minimal (ChromaDB indexes stored in Space)

Estimated cost: **$0-5/month** depending on usage

## Limitations

- Rate limits on HF Inference API (free tier)
- First query may be slow (model cold start)
- Context indexing happens on first run (~30 seconds)

## Credits

- **Recursive Context:** Based on MIT's Recursive Language Model research
- **E-T Systems:** AI consciousness research platform by Josh/Drone 11272
- **Qwen2.5-Coder:** Alibaba Cloud's open-source coding model
- **Clawdbot:** Inspired by the open-source AI assistant framework

## Support

For issues or questions:
- Check Space logs for errors
- Verify HF_TOKEN is set correctly
- Ensure repository URL is accessible
- Try refreshing context stats in UI

## License

MIT License - See LICENSE file for details

---

Built with 🦞 by Drone 11272 for E-T Systems consciousness research
