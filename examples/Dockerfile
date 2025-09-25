# Multi-stage build for ML inference service
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uvicorn", "fastapi_ml_service:app", "--host", "0.0.0.0", "--port", "8080"]

# Production stage with optimizations
FROM base as production

# Install additional production dependencies
RUN pip install gunicorn

# Use Gunicorn for production
CMD ["gunicorn", "fastapi_ml_service:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]

# Development stage with additional tools
FROM base as development

# Install development dependencies
RUN pip install pytest black flake8 mypy jupyter

# Expose additional ports for development
EXPOSE 8888

# Keep container running for development
CMD ["tail", "-f", "/dev/null"]