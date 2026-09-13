# Dockerfile for TQB Worker on HuggingFace Spaces
#
# CHANGELOG [2026-09-12 - Codex]
# SHIPPED: Complete six-person TQB personality set and unit disciplines.
# ADDED: Fail-closed preservation-only boot script and environment-owned paths.
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
    bash \
    git \
    git-lfs \
    build-essential \
    curl \
    wget \
    procps \
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
             /data/codemine-missions \
             /data/audit

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
ENV HF_HOME=/tmp/.cache/huggingface
ENV XDG_CACHE_HOME=/tmp/.cache
ENV TORCH_HOME=/tmp/.cache/torch
ENV NEUROGRAPH_WORKSPACE_DIR=/data/neurograph_worker
ENV CODEMINE_PRESERVATION_DIR=/data/codemine-preservation
ENV CODEMINE_PERSISTENT_ROOT=/data
ENV CODEMINE_BOOT_MODE=normal
ENV CODEMINE_MISSION_ROOT=/data/codemine-missions
ENV CODEMINE_WORKTREE_ROOT=/tmp/codemine-worktrees
ENV CODEMINE_DEFAULT_WORKSPACE=/workspace/e-t-systems
ENV CODEMINE_WORKSPACE_ROOTS=/workspace
ENV CODEMINE_VALIDATIONS_JSON={"catalog_version":"1.0.0","validations":{"git-diff-check":{"description":"Reject_whitespace_errors_in_the_current_diff","argv":["git","diff","--check"]},"python-tests":{"description":"Run_the_Python_pytest_suite_quietly","argv":["python3","-m","pytest","-q"]}}}
ENV CODEMINE_MAX_PARALLEL=3
ENV CODEMINE_VALIDATION_TIMEOUT=300
ENV CODEMINE_PROVIDER=openrouter
ENV CODEMINE_MODEL_ID=moonshotai/kimi-k3
ENV CODEMINE_LEADER_PROVIDER=openrouter
ENV CODEMINE_LEADER_MODEL_ID=moonshotai/kimi-k3
ENV CODEMINE_STRATEGIST_PROVIDER=openrouter
ENV CODEMINE_STRATEGIST_MODEL_ID=moonshotai/kimi-k3
ENV CODEMINE_TRACKER_PROVIDER=openrouter
ENV CODEMINE_TRACKER_MODEL_ID=z-ai/glm-5.3-flash
ENV CODEMINE_WRENCH_PROVIDER=openrouter
ENV CODEMINE_WRENCH_MODEL_ID=moonshotai/kimi-k3
ENV CODEMINE_FORGE_PROVIDER=openrouter
ENV CODEMINE_FORGE_MODEL_ID=z-ai/glm-5.3-flash
ENV CODEMINE_RAZOR_PROVIDER=openrouter
ENV CODEMINE_RAZOR_MODEL_ID=moonshotai/kimi-k3
ENV CODEMINE_REVIEWER_PROVIDER=openrouter
ENV CODEMINE_REVIEWER_MODEL_ID=moonshotai/kimi-k3
ENV CODEMINE_FALLBACK_MODEL_ID=z-ai/glm-5.3-flash
ENV HOME=/tmp
ENV PYTHONUNBUFFERED=1
ENV REPO_PATH=/workspace/e-t-systems
ENV TQB_PERSONALITY_DIR=/app/tqb-personalities

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
COPY agent_runner.py .
COPY specialist_tools.py .
COPY mission_plan.py .
COPY mission_generator.py .
COPY mission_dispatcher.py .
COPY mission_service.py .
COPY system_prompt.py .
COPY tool_definitions.py .
COPY worker_ng.py .
COPY surface_resolver.py .
COPY ng_embed.py .
COPY work_block_schema.py .
COPY spec_executor.py .
COPY persona_client.py .
COPY report_evaluator.py .
COPY risk_authority.py .
COPY ng_topology_sync.py .
COPY preserve_codemine.py .
COPY tqb-personalities/ ./tqb-personalities/

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
