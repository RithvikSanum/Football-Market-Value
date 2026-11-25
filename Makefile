PY ?= .venv/bin/python

.PHONY: setup ingest audit features train evaluate value app test all

setup:
	python3 -m venv .venv
	$(PY) -m pip install -e ".[model,dev]"

ingest:
	$(PY) -m fv.ingest.transfermarkt
	$(PY) -m fv.ingest.fx
	$(PY) -m fv.features.currency

audit: ingest
	$(PY) -m fv.evaluation.audit

features: ingest
	$(PY) -m fv.features.elo
	$(PY) -m fv.features.build
	$(PY) -m fv.features.market_index

train: features
	$(PY) -m fv.models.pricing

evaluate:
	$(PY) -m fv.evaluation.report

value:
	@echo "phase 4: not implemented yet"

app:
	@echo "phase 4: not implemented yet"

test:
	$(PY) -m pytest -q

all: ingest audit features train evaluate value test
