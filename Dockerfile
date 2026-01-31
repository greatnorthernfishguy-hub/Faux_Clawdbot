# Dockerfile for Clawdbot Dev Assistant
FROM python:3.11-slim

# Force rebuild
ENV REBUILD_DATE=2025-01-31-v2
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential curl && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create workspace and storage directories
RUN mkdir -p /workspace/e-t-systems /workspace/chroma_db /data/chroma_db /tmp/.cache/huggingface /tmp/.cache/chroma

# Set environment variables for writable cache locations
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

# Ensure permissions for the non-root Space user (UID 1000)
RUN chmod +x entrypoint.sh && \
    chown -R 1000:1000 /app /workspace /tmp/.cache

# Correctly expose port
EXPOSE 7860

# Switch to the Hugging Face Space user
USER 1000

# Launch
CMD ["./entrypoint.sh"]
