FROM python:3.10-slim

# Install git and ca-certificates
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv from Astral's UV image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --no-install-project

COPY . .

RUN uv sync

# Run through the project CLI entrypoint
CMD ["uv", "run", "python", "-O", "-m", "hoploy.cli"]
# CMD ["uv", "run", "python", "-O", "-m", "uvicorn", "hoploy.api.app:app", "--log-level", "debug", "--host", "0.0.0.0", "--port", "8100"]