# Multi-stage Dockerfile for Causal Uplift Engine
# Supports both API and Dashboard services

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
# Install uv for faster package management
RUN pip install uv

# Install uv for faster package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Copy application code
COPY src/ src/
COPY main.py .
COPY start.sh .
RUN chmod +x start.sh

# Create directories for data and models
RUN mkdir -p data/processed models outputs/plots

# Copy pre-trained artifacts (if available)
COPY --chown=1000:1000 models/ models/
COPY --chown=1000:1000 outputs/ outputs/
COPY --chown=1000:1000 data/ data/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables for integration
ENV API_URL="http://localhost:8000"

# Expose ports (7860 is standard for HF Spaces)
EXPOSE 7860 8000

# Start both services
CMD ["./start.sh"]
