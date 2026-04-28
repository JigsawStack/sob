.PHONY: install format lint

PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || command -v py 2>/dev/null)

install:
	@if command -v uv >/dev/null 2>&1; then \
		uv sync; \
	else \
		$(PYTHON) -m pip install -r requirements.txt; \
	fi

format: install
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff format .; \
	else \
		$(PYTHON) -m ruff format .; \
	fi

lint: install
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check .; \
	else \
		$(PYTHON) -m ruff check .; \
	fi