# Deployment Guide: Clawdbot to HuggingFace Spaces

## Quick Start (5 minutes)

### Step 1: Create HuggingFace Account
1. Go to https://huggingface.co
2. Sign up (free tier available)
3. Generate API token:
   - Settings → Access Tokens
   - Create "Read" token
   - Copy token (you'll need it)

### Step 2: Create New Space
1. Click "+ New" → "Space"
2. Configure:
   - **Space name:** `clawdbot-dev` (or your choice)
   - **License:** MIT
   - **SDK:** Docker
   - **Hardware:** CPU Basic (free) or upgrade for faster inference
3. Click "Create Space"

### Step 3: Upload Files
Upload these files to your Space:
- `app.py`
- `recursive_context.py`
- `Dockerfile`
- `requirements.txt`
- `README.md`
- `.gitignore`

**Via Git (Recommended):**
```bash
# Clone your new Space
git clone https://huggingface.co/spaces/your-username/clawdbot-dev
cd clawdbot-dev

# Copy all files from this directory
cp /path/to/clawdbot-dev/* .

# Commit and push
git add .
git commit -m "Initial deployment of Clawdbot"
git push
```

**Via Web Interface:**
- Click "Files" tab
- Click "Add file" → "Upload files"
- Drag and drop all files
- Commit changes

### Step 4: Configure Secrets
1. Go to Space Settings → Repository Secrets
2. Add secrets:
   ```
   Name: HF_TOKEN
   Value: [your HuggingFace API token from Step 1]
   ```
   
   Optional - if you have E-T Systems on GitHub:
   ```
   Name: REPO_URL
   Value: https://github.com/your-username/e-t-systems
   ```

### Step 5: Wait for Build
- Space will automatically build (takes ~5-10 minutes)
- Watch "Logs" tab for progress
- Build complete when you see: "Running on local URL: http://0.0.0.0:7860"

### Step 6: Access Your Assistant
- Click "App" tab
- Your Clawdbot is live!
- Access from iPhone browser: `https://your-username-clawdbot-dev.hf.space`

## Troubleshooting

### Build Fails
**Check logs for:**
- Missing dependencies → Verify requirements.txt
- Docker errors → Check Dockerfile syntax
- Out of memory → Upgrade to paid tier or reduce context size

**Common fixes:**
```bash
# View build logs
# Settings → Logs

# Restart build
# Settings → Factory Reboot
```

### No Repository Access
**If you see "No files indexed":**

1. **Option A: Mount via Secret**
   - Add `REPO_URL` secret with your GitHub repo
   - Restart Space
   - Repository will be cloned on startup

2. **Option B: Direct Upload**
   ```bash
   # In your Space's git clone
   mkdir -p workspace/e-t-systems
   cp -r /path/to/your/e-t-systems/* workspace/e-t-systems/
   git add workspace/
   git commit -m "Add E-T Systems codebase"
   git push
   ```

3. **Option C: Demo Mode**
   - Space creates minimal demo structure
   - Upload files via chat interface
   - Good for testing

### Slow Responses
**Qwen2.5-Coder-32B on free tier has cold starts.**

Solutions:
- Upgrade to GPU (paid tier) for faster inference
- Switch to smaller model (edit app.py):
  ```python
  client = InferenceClient(
      model="bigcode/starcoder2-15b",  # Smaller, faster
      token=os.getenv("HF_TOKEN")
  )
  ```
- Use HF Pro subscription for priority access

### Rate Limits
**Free tier has inference limits.**

Solutions:
- Upgrade to HF Pro ($9/month)
- Add delays between requests
- Use local model (requires GPU tier)

## Advanced Configuration

### Custom Model
Edit `app.py` line 20:
```python
client = InferenceClient(
    model="YOUR_MODEL_HERE",  # e.g., "codellama/CodeLlama-34b-Instruct-hf"
    token=os.getenv("HF_TOKEN")
)
```

### Adjust Recursion Depth
Edit `app.py` line 121:
```python
max_iterations = 10  # Increase for more complex queries
```

### Add New Tools
In `recursive_context.py`, add method:
```python
def your_new_tool(self, arg1, arg2):
    """Your tool description."""
    # Implementation
    return result
```

Then in `app.py`, add to TOOLS list:
```python
{
    "type": "function",
    "function": {
        "name": "your_new_tool",
        "description": "What it does",
        "parameters": {
            # Parameter schema
        }
    }
}
```

And add to execute_tool():
```python
elif tool_name == "your_new_tool":
    return ctx.your_new_tool(arguments['arg1'], arguments['arg2'])
```

## Cost Optimization

### Free Tier Strategy
- Use CPU Basic (free)
- HF Inference free tier (rate limited)
- Only index essential files
- **Total: $0/month**

### Minimal Paid Tier
- CPU Basic (free)
- HF Pro subscription ($9/month)
- Unlimited inference
- **Total: $9/month**

### Performance Tier
- GPU T4 Small ($0.60/hour, pause when not using)
- HF Pro ($9/month)
- Fast inference, local models
- **Total: ~$15-30/month** depending on usage

## iPhone Access

### Bookmark for Easy Access
1. Open Space URL in Safari
2. Tap Share → Add to Home Screen
3. Now appears as app icon

### Shortcuts Integration
Create iOS Shortcut:
```
1. Get text from input
2. Get contents of URL:
   https://your-username-clawdbot-dev.hf.space/api/chat
   Method: POST
   Body: {"message": [text from step 1]}
3. Show result
```

## Monitoring

### Check Health
```
https://your-username-clawdbot-dev.hf.space/health
```

### View Logs
- Settings → Logs (real-time)
- Download for analysis

### Stats
- Check "Context Info" panel in UI
- Shows files indexed, model status

## Updates

### Update Code
```bash
cd clawdbot-dev
# Make changes
git add .
git commit -m "Update: [what changed]"
git push
# Space rebuilds automatically
```

### Update Dependencies
Edit requirements.txt, commit, push.

### Update Repository
If using REPO_URL secret:
- Space pulls latest on restart
- Or: Settings → Factory Reboot

## Security

### Secrets Management
- Never commit API tokens
- Use Space secrets only
- Rotate tokens periodically

### Access Control
- Spaces are public by default
- For private: Settings → Change visibility to "Private"
- Requires HF Pro subscription

## Support Resources

- **HuggingFace Docs:** https://huggingface.co/docs/hub/spaces
- **Gradio Docs:** https://www.gradio.app/docs
- **Issues:** Post in Space "Community" tab

## Next Steps

1. ✅ Deploy Space
2. ✅ Test with simple queries
3. ✅ Upload your E-T Systems code
4. ✅ Try coding requests
5. 🎯 Integrate with E-T Systems workflow
6. 🎯 Add custom tools for your needs
7. 🎯 Connect to Observatory API
8. 🎯 Enable autonomous coding

---

Need help? Check Space logs or create discussion in Community tab.

Happy coding! 🦞
