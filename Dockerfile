# AgentMesh-STM Docker Image
# Multi-stage build for smaller final image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .


# Runtime stage
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 agentmesh && \
    chown -R agentmesh:agentmesh /app
USER agentmesh

# Create directories for data persistence
RUN mkdir -p /app/data /app/wal /app/plugins

# Environment variables
ENV AGENTMESH_WORKING_DIR=/app/workspace
ENV AGENTMESH_STORAGE_PATH=/app/data
ENV AGENTMESH_LOG_LEVEL=INFO

# Expose port for API server (if implemented)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import agentmesh_stm; print('ok')" || exit 1

# Default command - run the CLI
ENTRYPOINT ["agentmesh-stm"]
CMD ["--help"]
