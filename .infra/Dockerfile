# BUILD
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# dependency files
COPY pyproject.toml ./

# install dependencies
RUN uv sync --no-dev

# pull models
RUN uv run dvc pull models/celebrity_matcher.pt models/attractiveness_classifier.pt datasets/celebrities_pretty

# copy src files
COPY . .

EXPOSE 8000
