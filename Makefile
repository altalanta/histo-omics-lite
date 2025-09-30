.PHONY: setup data smoke demo public-demo lint format type test build clean

# Setup and installation
setup:
	pip install -e .[dev]
	pre-commit install

setup-cpu:
	pip install -e .[cpu]

# Dataset preparation
data:
	python -c "from histo_omics_lite.data.synthetic import make_tiny; make_tiny('data/synthetic')"

public-demo:
	python scripts/fetch_public_demo.py --output-dir data/public_demo

# Smoke test (fast CI check)
smoke: data
	python -m histo_omics_lite.training.train --config-name fast_debug
	python scripts/smoke_test.py

# Full demo pipeline
demo: public-demo
	python scripts/run_end_to_end.py --dataset public_demo --smoke-test

demo-synthetic:
	python scripts/run_end_to_end.py --dataset synthetic --smoke-test

# Code quality
lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

type:
	mypy src

test:
	pytest -v --cov=src/histo_omics_lite --cov-report=term-missing

test-smoke:
	pytest tests/test_smoke.py -v

# Build and packaging
build:
	python -m build

# Cleanup
clean:
	rm -rf build dist *.egg-info
	rm -rf data/synthetic data/public_demo
	rm -rf results mlruns
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
