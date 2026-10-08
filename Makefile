.PHONY: install test lint run mcp demo clean

install:
	pip install -e ".[dev,mcp]"

test:
	pytest --cov=preauth --cov-report=term-missing

lint:
	ruff check preauth tests

run:
	uvicorn preauth.main:app --reload --port 8000

mcp:            ## Run the MCP server (stdio) exposing the tools
	python -m preauth.mcp_server

demo:
	python -m preauth.cli demo

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
