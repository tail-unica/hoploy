FROM python:3.10-slim

# Install git and ca-certificates
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY . .

RUN uv sync --frozen

EXPOSE 8100

# Log level configurable via LOG_LEVEL env var (default: info)
CMD ["uv", "run", "python", "-O", "-m", "uvicorn", "hoploy.main:app", "--log-level", "info", "--host", "0.0.0.0", "--port", "8100"]