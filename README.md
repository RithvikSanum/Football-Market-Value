# Football player valuation engine

Values players against **real transfer fees** (not crowd opinion), with price ranges, explanations and
12-month forecasts. Status: **phase 0 (data audit) complete** — see `reports/data_audit.md`.

## Run

```bash
make setup     # venv + editable install
make ingest    # download transfermarkt-datasets (cached), convert to Parquet, archive players snapshot
make audit     # writes reports/data_audit.md
make test
```

## Data sources and terms

| Source | Use | Terms / notes |
|---|---|---|
| transfermarkt-datasets (dcaribou) | transfers, valuations, appearances, lineups, bios | CC0 Public Domain (Kaggle page). **Updates paused; data current to 6 Jul 2026.** |
| Understat, ClubElo, Wikipedia/Wikidata, SoFIFA | phase 1–2 | terms to be checked and recorded when ingested |

Raw data lives in `data/` (gitignored). Scraped sources are cached and rate-limited (≤1 req/s,
descriptive User-Agent; see `config.yaml`).
