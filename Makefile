# PySSL Makefile for common development tasks

.PHONY: help test docs clean install lint

help:
	@echo "PySSL Development Commands:"
	@echo "  install     Install package in development mode"
	@echo "  test        Run all tests"
	@echo "  docs        Build documentation"
	@echo "  clean       Clean build artifacts"
	@echo "  lint        Run code linting (placeholder)"
	@echo "  serve-docs  Serve documentation locally"

install:
	uv pip install -e ".[test,docs]"

test:
	uv run pytest tests/ -v

docs:
	uv run sphinx-build -b html docs/source docs/build

clean:
	rm -rf docs/build/
	rm -rf build/
	rm -rf dist/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

serve-docs: docs
	@echo "Starting local documentation server..."
	@echo "Visit: http://localhost:8000"
	python3 -m http.server -d docs/build 8000

lint:
	@echo "Linting placeholder - add your preferred linter here"
	@echo "Example: uv run ruff check ssl_framework/"