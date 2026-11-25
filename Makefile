PY ?= .venv/bin/python

.PHONY: setup ingest audit features train evaluate value app test all

setup:
	python3 -m venv .venv
	$(PY) -m pip install -e ".[dev]"

ingest:
	$(PY) -m fv.ingest.transfermarkt

audit: ingest
	$(PY) -m fv.evaluation.audit

features:
	@echo "phase 1: not implemented yet"

train:
	@echo "phase 1: not implemented yet"

evaluate:
	@echo "phase 1: not implemented yet"

value:
	@echo "phase 4: not implemented yet"

app:
	@echo "phase 4: not implemented yet"

test:
	$(PY) -m pytest -q

all: ingest audit features train evaluate value
