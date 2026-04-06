# Dockerfile for TQB Worker on HuggingFace Spaces
#
# CHANGELOG [2025-01-30 - Josh]
# REBUILD: Updated to Gradio 5.0+ for type="messages" support
#
# CHANGELOG [2025-01-31 - Claude]
# FIXED: Permissions for HF Spaces runtime user (UID 1000).
#
# CHANGELOG [2025-01-31 - Claude + Gemini]
# FIXED: /.cache PermissionError for embedding model download.
#
# CHANGELOG [2026-03-29 - Hammer/TQB — Block G Assembly]
# UPDATED: Added tools/ directory, new module files, anthropic SDK.
# Swapped from Kimi K2.5 to Claude. Added ANTHROPIC_API_KEY and CLAUDE_MODEL_ID env vars.
# Added /data/ persistent directory for worker NeuroGraph.

FROM python:3.11-slim

# CACHE BUSTER: Update this date to invalidate Docker cache
ENV REBUILD_DATE=2026-04-05-chassis-dialin

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
RUN pip uninstall -y gradio gradio-client 2>/dev/null; \
    pip install --no-cache-dir -r requirements.txt

# Create all directories the app needs to write to at runtime
RUN mkdir -p /workspace/e-t-systems \
             /tmp/.cache/huggingface \
             /tmp/.cache/torch \
             /tmp/.neurograph/checkpoints \
             /data/neurograph_worker/checkpoints \
             /data/audit

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
ENV HF_HOME=/tmp/.cache/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV TORCH_HOME=/tmp/.cache/torch
ENV NEUROGRAPH_WORKSPACE_DIR=/data/neurograph_worker
ENV HOME=/tmp
ENV PYTHONUNBUFFERED=1
ENV REPO_PATH=/workspace/e-t-systems

# Copy application files
COPY recursive_context.py .
COPY app.py .
COPY entrypoint.sh .
COPY neuro_foundation.py .
COPY universal_ingestor.py .
COPY openclaw_hook.py .
COPY neurograph_migrate.py .
COPY policy_engine.py .
COPY model_client.py .
COPY system_prompt.py .
COPY tool_definitions.py .
COPY worker_ng.py .
COPY ng_embed.py .

# Copy tools directory
COPY tools/ ./tools/

# =============================================================================
# PERMISSIONS FOR HF SPACES (UID 1000)
# =============================================================================
RUN chmod +x entrypoint.sh && \
    chown -R 1000:1000 /app /workspace /tmp/.cache /tmp/.neurograph /data

# Expose Gradio port (HF Spaces standard)
EXPOSE 7860

# Switch to non-root user
USER 1000

# Launch via entrypoint script
CMD ["./entrypoint.sh"]
