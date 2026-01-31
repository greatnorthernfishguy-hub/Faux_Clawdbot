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
# ALSO: Added /data directory for HF persistent storage.
# /data is the ONLY path that survives container restarts on HF Spaces.
# Must enable "Persistent Storage" in Space Settings for /data to exist.
# Falls back to /workspace (ephemeral) if /data isn't available.
#
# FEATURES:
# - Python 3.11 for Gradio
# - Gradio 5.0+ for modern chat interface
# - ChromaDB for vector search
# - Git for repo cloning
# - Optimized layer caching
# - Correct permissions for HF Spaces (UID 1000)

FROM python:3.11-slim

# CACHE BUSTER: Force rebuild when dependencies change
# Update this date to invalidate Docker cache for everything below
ENV REBUILD_DATE=2025-01-31-v1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# FORCE CLEAN INSTALL: Uninstall any cached Gradio, then install fresh
RUN pip uninstall -y gradio gradio-client 2>/dev/null; \
    pip install --no-cache-dir -r requirements.txt

# Create workspace directory for repository
RUN mkdir -p /workspace

# Clone E-T Systems repository (if URL provided via build arg)
ARG REPO_URL=""
RUN if [ -n "$REPO_URL" ]; then \
        git clone $REPO_URL /workspace/e-t-systems; \
    else \
        mkdir -p /workspace/e-t-systems && \
        echo "# E-T Systems" > /workspace/e-t-systems/README.md && \
        echo "Repository will be cloned on first run or mounted via Space secrets."; \
    fi

# Copy application code and entrypoint
COPY recursive_context.py .
COPY app.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# =============================================================================
# PERMISSIONS FIX FOR HF SPACES
# =============================================================================
# CHANGELOG [2025-01-31 - Claude]
# HF Spaces run as UID 1000, not root. All directories that the app needs
# to write to must be owned by 1000:1000, otherwise ChromaDB, conversation
# saves, and file downloads will silently fail.
#
# /workspace - ephemeral storage (wiped on restart, but works within session)
# /workspace/chroma_db - ChromaDB fallback if /data isn't available
# /data - HF persistent storage (survives restarts, created by HF at runtime)
#   NOTE: /data may not exist at build time. We create it here so the chown
#   works, but HF may mount over it at runtime. That's fine - HF sets correct
#   permissions on their mount. This is belt-and-suspenders.
# /tmp - needed for temporary files during cloud backup
# /app - the application directory itself (for any runtime-generated files)
# =============================================================================
RUN mkdir -p /workspace/chroma_db /data/chroma_db /tmp && \
    chown -R 1000:1000 /workspace /data /tmp /app

# Expose port for Gradio (HF Spaces uses 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV REPO_PATH=/workspace/e-t-systems

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# =============================================================================
# SWITCH TO NON-ROOT USER
# =============================================================================
# CHANGELOG [2025-01-31 - Claude]
# HF Spaces expect the container to run as UID 1000. Setting this explicitly
# ensures consistent behavior between local testing and deployed Spaces.
# Without this, the process runs as root during build but HF forces UID 1000
# at runtime, causing permission mismatches on files created during build.
# =============================================================================
USER 1000

# Run via entrypoint script (handles repo cloning at runtime)
CMD ["./entrypoint.sh"]
