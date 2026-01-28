# Dockerfile for Clawdbot Dev Assistant on HuggingFace Spaces
#
# CHANGELOG [2025-01-28 - Josh]
# Created containerized deployment for HF Spaces
# 
# FEATURES:
# - Python 3.11 for Gradio
# - ChromaDB for vector search
# - Git for repo cloning
# - Optimized layer caching

FROM python:3.11-slim

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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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

# Copy application code
COPY recursive_context.py .
COPY app.py .

# Create directory for ChromaDB persistence
RUN mkdir -p /workspace/chroma_db

# Expose port for Gradio (HF Spaces uses 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV REPO_PATH=/workspace/e-t-systems

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app.py"]
