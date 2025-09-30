.PHONY: setup data smoke lint format type test build

setup:
	pip install -e .[dev]
	pre-commit install

data:
	python -c "from histo_omics_lite.data.synthetic import make_tiny; make_tiny('data/synthetic')"

smoke: data
	python -m histo_omics_lite.training.train --config-name fast_debug
	python scripts/smoke_test.py

lint:
	ruff check src tests

format:
	ruff format src tests

type:
	mypy src

test:
	pytest -q

build:
	python -m build
