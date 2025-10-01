.PHONY: help install install-dev clean lint format typecheck test test-cov docs docs-serve build publish

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint: ## Run linting with ruff
	ruff check .

format: ## Format code with ruff
	ruff format .
	ruff check --fix .

typecheck: ## Run type checking with mypy
	mypy src/histo_omics_lite

test: ## Run tests with pytest
	pytest tests/

test-cov: ## Run tests with coverage
	pytest --cov=histo_omics_lite --cov-report=term-missing --cov-report=html tests/

test-fast: ## Run fast tests only
	pytest tests/ -m "not slow"

docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	mkdocs gh-deploy

build: ## Build package
	python -m build

publish: ## Publish to PyPI
	python -m twine upload dist/*

publish-test: ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

quality: lint typecheck test ## Run all quality checks

ci: clean quality docs build ## Run full CI pipeline

# Development workflow commands
dev-setup: install-dev ## Set up development environment
	@echo "Development environment set up successfully!"
	@echo "Run 'make quality' to run all checks"

pre-commit: ## Run pre-commit hooks manually
	pre-commit run --all-files

update-deps: ## Update dependencies
	pip-compile --upgrade pyproject.toml
	pip-compile --upgrade --extra dev pyproject.toml

# Docker commands (if needed)
docker-build: ## Build Docker image
	docker build -t histo-omics-lite .

docker-run: ## Run in Docker container
	docker run -it --rm histo-omics-lite

# Benchmarking
benchmark: ## Run performance benchmarks
	pytest tests/benchmarks/ -v

# Release commands
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major

release: clean quality build ## Prepare release
	@echo "Release ready! Run 'make publish' to upload to PyPI"