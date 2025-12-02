# BUILD
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# dependency files
COPY pyproject.toml ./

# install dependencies (will create uv.lock automatically)
RUN uv sync --no-dev

# copy src files
COPY . .

RUN . .venv/bin/activate

EXPOSE 8000
