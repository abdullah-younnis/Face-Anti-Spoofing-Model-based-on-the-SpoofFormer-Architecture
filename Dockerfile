# =============================================================================
# SpoofFormer Docker Image
# Multi-stage build for training and inference
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim as base

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Builder - install Python dependencies
# -----------------------------------------------------------------------------
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements (CPU version - remove CUDA index)
COPY requirements.txt .
RUN sed '/extra-index-url/d' requirements.txt > requirements_cpu.txt && \
    pip install --user -r requirements_cpu.txt

# -----------------------------------------------------------------------------
# Stage 3: Training image (with full dependencies)
# -----------------------------------------------------------------------------
FROM base as training

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Ensure scripts are in PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser . .

# Install the package
RUN pip install --user -e .

# Switch to non-root user
USER appuser

# Default command
CMD ["python", "train.py", "--help"]

# -----------------------------------------------------------------------------
# Stage 4: Inference image (lightweight)
# -----------------------------------------------------------------------------
FROM base as inference

# Copy only necessary packages (exclude training-only deps)
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy only inference-related code
COPY --chown=appuser:appuser src/spoofformer /app/src/spoofformer
COPY --chown=appuser:appuser inference.py /app/
COPY --chown=appuser:appuser configs /app/configs
COPY --chown=appuser:appuser pyproject.toml /app/

# Install the package
RUN pip install --user -e .

# Switch to non-root user
USER appuser

# Expose port for potential API server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default command
CMD ["python", "inference.py", "--help"]
