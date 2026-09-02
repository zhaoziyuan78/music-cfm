.PHONY: install format lint typecheck test smoke check
install:
	uv sync --extra dev --extra ot
format:
	uv run ruff format .
lint:
	uv run ruff check .
typecheck:
	uv run mypy src
test:
	uv run pytest -q
smoke:
	uv run pytest -q tests/test_smoke_pipeline.py
check: format lint typecheck test
