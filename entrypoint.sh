#!/bin/bash
# Clawdbot Development Assistant Entrypoint
#
# CHANGELOG [2025-01-29 - Josh]
# Created for HuggingFace Spaces deployment
# Handles runtime setup before starting the Gradio application
#
# CHANGELOG [2025-01-31 - Claude]
# Added cache directory fallback for ChromaDB embedding model downloads.
# PROBLEM: ChromaDB tries to download ONNX MiniLM to /.cache on first use.
# Running as USER 1000 means /.cache (owned by root) is not writable.
# Dockerfile sets ENV vars to redirect to /data/.cache, but if /data
# isn't available (persistent storage not enabled), we need a fallback.
# FIX: Test /data writability at runtime. If not writable, redirect
# cache env vars to /tmp/.cache (ephemeral but at least writable).
# Also added startup timestamp and diagnostic logging.
#
# CHANGELOG [2026-09-12 - Codex]
# Added an explicit preservation-only boot before every write probe, import,
# repository update, and NeuroGraph/Tonic startup operation.
#
# CHANGELOG [2026-09-13 - Codex]
# Fail closed to preservation mode when CODEMINE_BOOT_MODE is absent.
#
# CHANGELOG [2026-09-13 - Codex]
# Reject every noncanonical mode and require the Python normal-boot guard before
# cache writes, imports, repository pulls, or application startup.

echo ""
echo "===== Application Startup at $(date -u '+%Y-%m-%d %H:%M:%S') ====="
echo ""
echo "🦞 Clawdbot Entrypoint Starting..."

# Preservation mode must run before cache probes, mkdir, imports, repository
# updates, or application startup. It inspects and copies the existing graph,
# then exits without starting NeuroGraph or Tonic.
case "${CODEMINE_BOOT_MODE:-preserve}" in
    preserve)
        echo "🔒 Preservation-only boot: application and NeuroGraph will remain stopped"
        exec python3 /app/preserve_codemine.py "$@"
        ;;
    normal)
        python3 /app/boot_mode.py --check-normal || exit $?
        ;;
    *)
        echo "❌ Invalid CODEMINE_BOOT_MODE; expected 'preserve' or 'normal'" >&2
        exit 2
        ;;
esac

# =========================================================================
# CACHE DIRECTORY SETUP
# =========================================================================
# CHANGELOG [2025-01-31 - Claude]
# Ensure model cache directories exist and are writable BEFORE Python starts.
# =========================================================================

# Test if /data is writable (persistent storage enabled)
if touch /data/.write_test 2>/dev/null; then
    rm -f /data/.write_test
    echo "✅ /data is writable (persistent storage enabled)"
    # Ensure cache subdirectories exist
    mkdir -p /data/.cache/huggingface
else
    echo "⚠️  /data is NOT writable - redirecting cache to /tmp"
    echo "   (Enable persistent storage in Space Settings for durability)"
    # Redirect cache env vars to /tmp (writable but ephemeral)
    export HF_HOME=/tmp/.cache/huggingface
    export TRANSFORMERS_CACHE=/tmp/.cache/huggingface
    export XDG_CACHE_HOME=/tmp/.cache
    mkdir -p /tmp/.cache/huggingface
fi

echo "📁 Cache directory: ${XDG_CACHE_HOME}"

# =========================================================================
# GRADIO VERSION CHECK
# =========================================================================

echo ""
echo "🔍 DEBUG: Checking Gradio installation..."
python3 -c "
import gradio as gr
print(f'✓ Gradio version: {gr.__version__}')
print(f'✓ Gradio path: {gr.__file__}')
" 2>&1 || echo "❌ Gradio import failed!"

# =========================================================================
# REPOSITORY SETUP
# =========================================================================

# Check if a repository URL was provided
if [ -n "$REPO_URL" ]; then
    echo "📦 Cloning repository: $REPO_URL"
    if [ -d "/workspace/e-t-systems/.git" ]; then
        echo "  Repository already cloned, pulling latest..."
        cd /workspace/e-t-systems && git pull
    else
        git clone "$REPO_URL" /workspace/e-t-systems
    fi
else
    echo "ℹ️ No REPO_URL provided, using demo repository"
fi

# Show what's in the repository
echo "📂 Repository contents:"
ls -la /workspace/e-t-systems/ 2>/dev/null || echo "  (empty or not found)"

# =========================================================================
# LAUNCH APPLICATION
# =========================================================================

echo "🚀 Starting Gradio application..."
exec python3 /app/app.py
