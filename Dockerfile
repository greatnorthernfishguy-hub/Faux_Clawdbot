# Dockerfile for Clawdbot Dev Assistant on HuggingFace Spaces
#
# CHANGELOG [2025-01-30 - Josh]
# REBUILD: Updated to Gradio 5.0+ for type="messages" support
# Added translation layer for Kimi K2.5 tool calling
# Added multimodal file upload support
#
# CHANGELOG [2025-01-31 - Claude]
# FIXED: Permissions for HF Spaces runtime user (UID 1000).
# PROBLEM: HF Spaces run containers as user 1000, not root. Directories
#   created during build (as root) weren't writable at runtime, causing
#   ChromaDB to silently fail when trying to create SQLite files.
# FIX: chown all writable directories to 1000:1000, then switch to USER 1000.
#
# CHANGELOG [2025-01-31 - Claude + Gemini]
# FIXED: /.cache PermissionError for ChromaDB ONNX embedding model download.
# PROBLEM: ChromaDB's ONNXMiniLM_L6_V2 ignores XDG_CACHE_HOME and tries to
#   write to ~/.cache. In containers, HOME=/ so it writes to /.cache (root-owned).
# FIX: Set HOME=/tmp so fallback cache paths resolve to /tmp/.cache (writable).
#   Also create /tmp/.cache subdirs during build and chown to UID 1000.
#   The actual fix is in recursive_context.py (DOWNLOAD_PATH override), but
#   HOME=/tmp catches any other library that might try the same trick.
#
# FEATURES:
# - Python 3.11 for Gradio 6.5+
# - ChromaDB with ONNX MiniLM for vector search
# - Git for repo cloning
# - Correct permissions for HF Spaces (UID 1000)
# - Cache dirs pre-created and owned by runtime user

FROM python:3.11-slim

# CACHE BUSTER: Update this date to invalidate Docker cache for everything below
ENV REBUILD_DATE=2025-01-31-v3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# FORCE CLEAN: Uninstall cached Gradio versions to avoid version conflicts
RUN pip uninstall -y gradio gradio-client 2>/dev/null; \
    pip install --no-cache-dir -r requirements.txt

# Create all directories the app needs to write to at runtime
# /workspace/e-t-systems - repo clone target
# /workspace/chroma_db - ChromaDB fallback if /data isn't available
# /data/chroma_db - ChromaDB primary (persistent if storage enabled)
# /tmp/.cache/* - embedding model downloads and HF cache
RUN mkdir -p /workspace/e-t-systems \
             /workspace/chroma_db \
             /data/chroma_db \
             /tmp/.cache/huggingface \
             /tmp/.cache/chroma

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
# CHANGELOG [2025-01-31 - Claude + Gemini]
# HOME=/tmp is the critical one. Many Python libraries (including ChromaDB's
# ONNX embedding function) use ~ as fallback for cache paths. In Docker
# containers, HOME defaults to / if not explicitly set, so ~/.cache becomes
# /.cache which is root-owned. Setting HOME=/tmp ensures any library we
# didn't explicitly configure still has a writable fallback path.
#
# The HF_HOME and XDG_CACHE_HOME vars are belt-and-suspenders for HuggingFace
# Hub downloads (model weights, tokenizers, etc).
#
# NOTE: The actual ChromaDB embedding model path is overridden in Python code
# (recursive_context.py) via DOWNLOAD_PATH attribute. These env vars catch
# everything else.
# =============================================================================
ENV HF_HOME=/tmp/.cache/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV CHROMA_CACHE_DIR=/tmp/.cache/chroma
ENV HOME=/tmp
ENV PYTHONUNBUFFERED=1
ENV REPO_PATH=/workspace/e-t-systems

# Copy application files
COPY recursive_context.py .
COPY app.py .
COPY entrypoint.sh .

# =============================================================================
# PERMISSIONS FOR HF SPACES (UID 1000)
# =============================================================================
# CHANGELOG [2025-01-31 - Claude + Gemini]
# HF Spaces run as UID 1000, not root. All directories the app writes to
# must be owned by 1000:1000, otherwise operations fail silently.
# This includes:
#   /app - application directory (runtime-generated files)
#   /workspace - repo clones and ephemeral ChromaDB fallback
#   /tmp/.cache - embedding model downloads, HF cache
# =============================================================================
RUN chmod +x entrypoint.sh && \
    chown -R 1000:1000 /app /workspace /tmp/.cache

# Expose Gradio port (HF Spaces standard)
EXPOSE 7860

# Switch to non-root user
# CHANGELOG [2025-01-31 - Claude]
# HF Spaces expect UID 1000 at runtime. Setting this explicitly ensures
# consistent behavior between local testing and deployed Spaces.
USER 1000

# Launch via entrypoint script
CMD ["./entrypoint.sh"]
