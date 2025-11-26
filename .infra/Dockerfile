# BUILD
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# dependency files
COPY pyproject.toml ./

# install dependencies
RUN uv sync --no-dev

# copy src files
COPY . .

# pull models
RUN .venv/bin/dvc pull models/celebrity_matcher.pt models/attractiveness_classifier.pt datasets/celebrities_pretty

EXPOSE 8000
