# Football player valuation engine

Values players against **real transfer fees** (not crowd opinion), with price ranges, explanations and
12-month forecasts. Status: **phase 1 (core pricing model) complete, at checkpoint 1** — see `reports/model_report.md`
(validation results only; test split untouched) and `reports/data_audit.md`.

## Run

```bash
make setup     # venv + editable install
make ingest    # transfermarkt-datasets + ECB FX (cached), Parquet, fees in EUR/GBP, players snapshot
make audit     # writes reports/data_audit.md
make features  # own club Elo, transfer features, market index
make train     # deflator choice, baselines, ridge, LightGBM (+ quantiles/CQR, SHAP)
make evaluate  # writes reports/model_report.md + reports/figures/
make test
```

## Data sources and terms

| Source | Use | Terms / notes |
|---|---|---|
| transfermarkt-datasets (dcaribou) | transfers, valuations, appearances, lineups, bios | CC0 Public Domain (Kaggle page). **Updates paused; data current to 6 Jul 2026.** |
| ECB euro reference rates (Data Portal) | EUR/GBP on each transfer date | Free, no key. Reuse terms not yet verified (to check before publishing) |
| ClubElo | not used | API (api.clubelo.com) unreachable in Nov 2026; replaced by our own Elo from transfermarkt-datasets games |
| Understat, Wikipedia/Wikidata, SoFIFA | phase 2 | terms to be checked and recorded when ingested |

Raw data lives in `data/` (gitignored). Scraped sources are cached and rate-limited (≤1 req/s,
descriptive User-Agent; see `config.yaml`).
