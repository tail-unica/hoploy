FROM python:3.10-slim

# Install git, ca-certificates and procps (pgrep for py-spy helper)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

# Create non-root user matching the host UID/GID (passed as build args)
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" app \
    && useradd -u "${UID}" -g "${GID}" -m -d /home/app app

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY . .

# Install py-spy as system-wide tool (root) so it can attach via SYS_PTRACE
RUN pip install --no-cache-dir py-spy

# Sync, pre-create writable directories, then hand ownership only to what's needed at runtime
# /app/plugin, /app/dataset and /app/checkpoints are replaced at runtime by bind-mounts;
# the directories must exist so the container starts even without the mounts.
RUN uv sync --frozen \
    && mkdir -p /app/log /app/plugin /app/dataset /app/checkpoints \
    && chown -R app:app /app/log /app/plugin /app/dataset /app/checkpoints \
    && chown app:app /app \
    && chmod +x /app/scripts/pyspy.sh

USER app

EXPOSE 8100

# Log level configurable via LOG_LEVEL env var (default: info)
CMD ["uv", "run", "python", "-O", "-m", "uvicorn", "hoploy.main:app", "--log-level", "info", "--host", "0.0.0.0", "--port", "8100"]