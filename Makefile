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

typecheck: ## Run strict type checking with mypy
	mypy --strict src/histo_omics_lite

test: ## Run tests with pytest
	pytest tests/

test-cov: ## Run tests with coverage
	pytest --cov=histo_omics_lite --cov-report=term-missing --cov-report=html tests/

test-fast: ## Run fast tests only
	pytest tests/ -m "not slow"

smoke: ## Run end-to-end smoke test on synthetic data
	mkdir -p results
	python -m histo_omics_lite.cli data --make --num-patients 24 --tiles-per-patient 2 --seed 1337 --out results/smoke_data
	python -m histo_omics_lite.cli train --config configs/train/cpu_small.yaml --seed 1337 --cpu --epochs 1 --batch-size 16 --json > results/smoke_train.json
	python -m histo_omics_lite.cli eval --ckpt artifacts/checkpoints/best.ckpt --cpu --seed 1337 --num-workers 0 --batch-size 64 --json > results/smoke_eval.json
	python -m histo_omics_lite.cli embed --ckpt artifacts/checkpoints/best.ckpt --cpu --seed 1337 --num-workers 0 --batch-size 64 --out results/smoke_embeddings.parquet --json > results/smoke_embed.json
	python scripts/validate_smoke_results.py --results-dir results

fetch-public: ## Download public demo dataset
	python scripts/fetch_public_data.py

demo-public: fetch-public ## Run full public demo pipeline
	python -m histo_omics_lite.cli train --config configs/train/fast_debug.yaml --data-path data/public --seed 2024 --cpu --epochs 2 --batch-size 16 --json > results/demo_public_train.json
	python -m histo_omics_lite.cli eval --ckpt artifacts/checkpoints/best.ckpt --data-path data/public --cpu --seed 2024 --batch-size 32 --json > results/demo_public_eval.json

demo: ## Generate public demo artifacts and documentation
	python -m histo_omics_lite.cli data --make --num-patients 48 --tiles-per-patient 4 --seed 2024 --out docs/demo_data
	python -m histo_omics_lite.cli train --config configs/train/fast_debug.yaml --seed 2024 --cpu --epochs 2 --batch-size 32 --json > docs/demo_train.json
	python -m histo_omics_lite.cli eval --ckpt artifacts/checkpoints/best.ckpt --cpu --seed 2024 --num-workers 0 --batch-size 64 --json > docs/demo_eval.json
	python -m histo_omics_lite.cli embed --ckpt artifacts/checkpoints/best.ckpt --cpu --seed 2024 --num-workers 0 --batch-size 64 --out docs/demo_embeddings.parquet --json > docs/demo_embed.json

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
benchmark: ## Run performance benchmarks with baseline comparison
	pytest tests/benchmarks/ -v --benchmark-compare --benchmark-compare-fail=min:10% --benchmark-sort=name

benchmark-save: ## Save current performance as baseline
	pytest tests/benchmarks/ -v --benchmark-save=baseline

benchmark-compare: ## Compare current performance to baseline
	pytest tests/benchmarks/ -v --benchmark-compare=baseline --benchmark-compare-fail=min:10%

benchmark-reset: ## Reset performance baselines
	pytest tests/benchmarks/ -v --benchmark-save=baseline --benchmark-save-data

# Release commands
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major

release: clean quality build ## Prepare release
	@echo "Release ready! Run 'make publish' to upload to PyPI"
