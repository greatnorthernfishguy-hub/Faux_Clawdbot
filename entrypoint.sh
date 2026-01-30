#!/bin/bash
#
# Entrypoint script for Clawdbot
# 
# CHANGELOG [2025-01-29 - Josh]
# Created to handle runtime repo cloning with authentication
#
# This script:
# 1. Clones E-T Systems repo if REPO_URL is provided
# 2. Uses GITHUB_TOKEN for authentication
# 3. Starts the Gradio app

set -e

echo "===== Application Startup at $(date -u +"%Y-%m-%d %H:%M:%S") ====="
echo ""
echo "🦞 Clawdbot Entrypoint Starting..."

# DEBUG: Check installed Gradio version
echo ""
echo "🔍 DEBUG: Checking Gradio installation..."
python -c "import gradio; print(f'✓ Gradio version: {gradio.__version__}')" || echo "✗ Gradio import failed"
python -c "import gradio, inspect; print(f'✓ Gradio path: {inspect.getfile(gradio)}')" || true
echo ""

# Clone repository if URL provided
if [ -n "$REPO_URL" ]; then
    echo "📦 Repository URL detected: $REPO_URL"
    
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "🔑 GitHub token found, cloning with authentication..."
        # Insert token into URL for authentication
        AUTH_URL=$(echo "$REPO_URL" | sed "s|https://|https://${GITHUB_TOKEN}@|")
        git clone "$AUTH_URL" /workspace/e-t-systems 2>&1 || echo "⚠️ Clone failed or repo already exists"
    else
        echo "⚠️ No GITHUB_TOKEN found, attempting public clone..."
        git clone "$REPO_URL" /workspace/e-t-systems 2>&1 || echo "⚠️ Clone failed or repo already exists"
    fi
else
    echo "ℹ️ No REPO_URL provided, using demo repository"
fi

# Check what got cloned
echo "📂 Repository contents:"
ls -la /workspace/e-t-systems/ || echo "⚠️ Repository directory doesn't exist yet"

# Start the application
echo "🚀 Starting Gradio application..."
exec python app.py