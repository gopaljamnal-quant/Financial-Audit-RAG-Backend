# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1: Builder -- installs build dependencies and compiles wheels
# =============================================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System build dependencies required to compile wheels for asyncpg,
# tiktoken, and other C-extension packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Build wheels for all dependencies into /build/wheels so the runtime
# stage can install them without needing any compiler toolchain.
RUN pip install --upgrade pip setuptools wheel \
    && pip wheel --wheel-dir=/build/wheels -r requirements.txt

# =============================================================================
# Stage 2: Runner -- secure, non-root, slim production image
# =============================================================================
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:${PATH}" \
    APP_HOME=/app \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache

# Minimal runtime system dependencies (libpq for asyncpg's C bindings).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR ${APP_HOME}

# Install pre-built wheels from the builder stage -- no compiler present
# in the final image, minimising attack surface for a financial-audit
# deployment target.
COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

# Copy application source with correct, non-root ownership.
COPY --chown=appuser:appuser . ${APP_HOME}

# Pre-warm the tiktoken BPE cache at build time so the runtime container
# never needs outbound internet access to tokenise text -- required for
# deployment into network-isolated, audit-controlled environments.
RUN mkdir -p ${TIKTOKEN_CACHE_DIR} \
    && chown appuser:appuser ${TIKTOKEN_CACHE_DIR} \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Multi-core production execution: workers scale with available CPU
# cores, keep-alive tuned for typical API-gateway timeouts, and access
# logs disabled in favour of structured application logging.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--timeout-keep-alive", "30", \
     "--no-access-log", \
     "--proxy-headers"]
