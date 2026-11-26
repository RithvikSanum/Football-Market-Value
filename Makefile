PY ?= .venv/bin/python

.PHONY: setup ingest audit features train evaluate value site test all

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
	$(PY) -m fv.features.build          # pass 1 fixes the transfer universe (player ids)
	$(PY) -m fv.ingest.wikidata         # TM id -> enwiki article (cached)
	$(PY) -m fv.ingest.wikipedia        # page views for universe + current players (cached)
	$(PY) -m fv.ingest.contracts        # dated players.csv snapshots from Kaggle versions (cached)
	$(PY) -m fv.features.build          # pass 2 with hype
	$(PY) -m fv.features.market_index

train: features
	$(PY) -m fv.models.pricing

evaluate:
	$(PY) -m fv.evaluation.report
	$(PY) -m fv.evaluation.ablation
	$(PY) -m fv.evaluation.phase2_report

value:
	@echo "phase 4: not implemented yet"

site:  ## static website (phase 4): export JSON + render Jinja2 templates into site/
	@echo "phase 4: not implemented yet"

test:
	$(PY) -m pytest -q

all: ingest audit features train evaluate value test
