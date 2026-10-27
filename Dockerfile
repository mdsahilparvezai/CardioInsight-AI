# CardioInsight AI — Dockerfile
# Multi-stage build for a lean production image.
#
# Build:  docker build -t cardioinsight-ai .
# Run:    docker run -p 8501:8501 cardioinsight-ai
# Or with model volume: docker run -p 8501:8501 -v $(pwd)/models:/app/models cardioinsight-ai

# ---------------------------------------------------------------------------
# Stage 1: Build environment (install dependencies, compile if needed)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: install MLops extras if needed
# RUN pip install --no-cache-dir mlflow dvc

# ---------------------------------------------------------------------------
# Stage 2: Runtime environment (copy only what's needed)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser .env.example .

# Create models directory (will be populated at runtime or via volume)
RUN mkdir -p /app/models && chown appuser:appuser /app/models

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# ---------------------------------------------------------------------------
# Alternative: Development stage (for local development with hot reload)
# ---------------------------------------------------------------------------
# FROM runtime AS dev
# USER root
# RUN pip install --no-cache-dir watchdog
# USER appuser
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]