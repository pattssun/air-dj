.PHONY: run lint format

run:
	python air_dj.py

lint:
	ruff check .

format:
	ruff format .
