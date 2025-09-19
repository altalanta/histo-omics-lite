.PHONY: lint type test format build data

lint:
	ruff check src tests

type:
	mypy src

format:
	ruff format src tests

test:
	pytest --maxfail=1 --disable-warnings

build:
	python -m build

data:
	python -m histo_omics_lite.data.shard_maker synthetic
