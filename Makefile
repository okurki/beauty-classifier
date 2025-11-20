help:
	@echo "Avaliable commands:"
	@echo "==================================================================================="
	@echo "help                 -  show this message"
	@echo "setup                -  setup project (sync, run-migrations, pull-data)"
	@echo "dev                  -  run project in dev mode (debug messages, sqlite db)"
	@echo "run                  -  run project in production mode (info messages, postgres db)"
	@echo "test                 -  run tests"
	@echo "format               -  format code"
	@echo "mlflow-compose-up    -  launch docker compose with mlflow server"
	@echo "compose-up           -  launch docker compose"
	@echo "train-attractiveness -  train attractiveness model"
	@echo "eval-attractiveness  -  evaluate attractiveness model"
	@echo "sync                 -  sync dependencies"
	@echo "run-migrations       -  run migrations"
	@echo "pull-data            -  pull data from remote DVC repository"

sync:
	uv sync

run-migrations:
	uv run --no-sync alembic upgrade head

pull-data:
	uv run --no-sync dvc pull

setup: sync run-migrations pull-data
	pre-commit install

run:
	uv run -m src.interfaces.api.v1

run-prod:
	python -m src.interfaces.api.v1

migrate:
	uv run alembic revision --autogenerate -m "$(m)"
	@echo Please edit the generated migration file and then run 'uv run alembic upgrade head'

format:
	uv run ruff check --fix-only && uv run ruff format

test:
	uv run pytest

compose-up:
	docker compose up -d

mlflow-compose-up:
	docker compose -f docker-compose-mlflow.yml up -d

train-attractiveness:
	uv run -m src.interfaces.cli attractiveness --train

eval-attractiveness:
	uv run -m src.interfaces.cli attractiveness --eval
