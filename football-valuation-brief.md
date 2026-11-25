# Project brief: Football player valuation engine ("a more accurate Transfermarkt")

Read this whole brief before writing any code. Work through the phases in order and **stop at each checkpoint** to show me results before moving on.

> **Revision log**
> - **v1.0** — original brief.
> - **v1.1 (2026-11-25, after checkpoint 0)** — training window and splits changed (section 5); contract approach chosen (section 6); current valuations dated 30 June 2026 because the source is frozen (section 9).
> - **v1.2 (2026-11-25)** — added **section 15: market level / football inflation**. Sections 5, 7, 8, 9, 10, 11, 13 and 14 updated to match. Phase 0 work from section 15 moves into Phase 1 because Phase 0 is already done.

---

## 1. What we're building

A system that values football players more accurately than Transfermarkt, gives a **price range** rather than a single number, explains **why** a player is worth what he is, and **forecasts** his value 12 months ahead.

Key idea: Transfermarkt values are crowd opinions. The ground truth is **real transfer fees**. We train on real fees and judge every model, including Transfermarkt's own value, by how well it predicts real fees.

Two headline questions the project must answer honestly:
1. **Can stats alone price players as well as the crowd?** (model *without* Transfermarkt value as an input)
2. **Can we improve on the crowd?** (model *with* Transfermarkt value as an input, correcting its biases)

If a model doesn't beat the Transfermarkt baseline, report that plainly. Honest results matter more than flattering ones.

---

## 2. Constraints

- **Free data only.** No paid APIs, no paid tiers.
- **Main model is LightGBM** (point estimates, quantiles and forecasts). Ridge regression is the simple baseline.
- Python 3.11+. Store data in Parquet and query with DuckDB.
- **No deep learning, betting features, social media scraping or wage data in v1.**
- Scraped sources (Understat, SoFIFA, ClubElo): be polite. Send a descriptive User-Agent, add delays (≤1 request/sec unless the source says more is fine), cache everything locally, never re-download what's cached, and don't commit raw data to git. Check each source's terms and note them in the README.
- **Do not use FBref for advanced stats.** Its xG and other advanced data were removed in January 2026.
- If a source is down, blocked or has changed format, **stop and tell me**. Suggest a free alternative. Never fake or fill in data.
- **No CPI or general inflation anywhere.** Football price levels come only from football data (section 15).

---

## 3. Data sources (all free)

| Purpose | Source | Access notes |
|---|---|---|
| Transfers + fees, Transfermarkt value history, appearances, games, lineups, clubs, player bios, current contract expiry | **transfermarkt-datasets** (github.com/dcaribou/transfermarkt-datasets; Kaggle: `davidcariboo/player-scores`) | CSVs. **Updates paused; data current to 6 July 2026.** Licence CC0. Old Kaggle versions downloadable without login. |
| xG, xA, npxG, xGChain, xGBuildup, key passes, shots | **Understat** (top 5 leagues + Russian league, from 2014/15) | Use the `soccerdata` package (`Understat` reader) or the `understat` package. Cache per league-season. |
| Club and league strength over time | **ClubElo** | CSV API: `http://api.clubelo.com/YYYY-MM-DD` (all clubs on a date) and `http://api.clubelo.com/{ClubName}` (one club's history). `soccerdata.ClubElo` wraps it. |
| Popularity / hype | **Wikipedia Pageviews API** (Wikimedia REST) | `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{Article}/daily/{start}/{end}`. No key; data starts July 2015; CC0. Descriptive User-Agent required. |
| Transfermarkt player ID → Wikipedia article | **Wikidata SPARQL** | Property **P2446** = Transfermarkt player ID. Pull the enwiki sitelink for each ID. |
| Pace and physical/technical attributes (phase 2) | **SoFIFA** (EA FC / FIFA ratings by edition) | `soccerdata.SoFIFA`. These are game ratings, so treat them as a proxy and label them that way. |
| Exchange rates (section 15) | **ECB euro reference rates** via the ECB Data Portal or the Frankfurter API | Daily EUR/GBP (and other currencies if needed) from 1999. Cache locally. |
| League revenues (section 15) | **Deloitte Annual Review of Football Finance**, league annual reports / announcements | Entered by hand in `data/manual/league_revenues.csv` with a source URL for every figure. Blank if unverifiable. |

Known schemas (verify in phase 0, they may have changed):
- `transfers`: player_id, transfer_date, transfer_season, from_club_id, to_club_id, from_club_name, to_club_name, transfer_fee, market_value_in_eur, player_name
- `player_valuations`: player_id, date, market_value_in_eur, current_club_id, player_club_domestic_competition_id
- Also expect: players, appearances, games, club_games, game_lineups, game_events, clubs, competitions

---

## 4. Rules against data leakage (critical)

- Every feature for a transfer must use **only information available before `transfer_date`**.
- **Transfermarkt value at transfer:** don't trust `transfers.market_value_in_eur` blindly. Transfermarkt sometimes updates values right after a deal, which leaks the fee. Use the **last valuation in `player_valuations` dated at least 30 days before the transfer**. Compare both versions in the report.
- **Contract expiry:** `players.contract_expiration_date` is the *current* contract only. **Never use it for historical transfers.** Use it only for valuing current players (see section 6, Contract).
- Models that use other models' outputs (e.g. abnormal hype residuals, market index) must fit those helper models on the training period only.
- **Market index:** the level used for a transfer in window *w* is estimated only from transfers in windows before *w* (plus, for the nowcast, deals already completed earlier in *w*). Revenue figures are used only if published before the forecast date.
- Write **unit tests** asserting that feature timestamps are earlier than the transfer date for a sample of rows.

---

## 5. Target, filtering and time splits

**Training set:** permanent transfers with a known fee ≥ €250k (make the threshold configurable). Exclude loans, end-of-loan moves, free transfers and unknown fees. Report how many rows each filter removes.

**Target:** `log(fee) − log(market_index at transfer date)`, i.e. the player's **real value** in base-date money. The market index is defined in **section 15**: a rolling hedonic index (main), with a repeat-sales index and the Transfermarkt market level as cross-checks, chosen on the validation split. Plot the index; it's a nice chart for the README.

**Report median predictions, not means.** We're predicting log values, so exponentiated predictions are medians. That's fine; don't add bias corrections.

**Time splits** (configurable in `config.yaml`; changed at checkpoint 0 so that everything after mid-2020 is out of sample):
- Training window starts 2012-07-01 (appearance data begins in 2012/13)
- Train: 2012-07-01 ≤ transfer_date < 2020-07-01
- Validation (tuning + early stopping): 2020-07-01 → 2021-06-30
- Calibration (conformal intervals): 2021-07-01 → 2022-06-30
- **Test (touch once, at the end):** ≥ 2022-07-01
- For current valuations: retrain on everything up to 2025-06-30, recalibrate on 2025/26, then value players as of 2026-06-30.

---

## 6. Features (factor groups)

Build features in groups so we can run **ablation tests** (does each group actually help?).

**A. Base**
- Age at transfer, position / sub-position, foot, height
- Selling league, buying league, cross-border flag, Premier League buyer flag
- Selling-club and buying-club ClubElo on the transfer date
- Transfer window (summer/winter); late-window flag (last 3 days of a window)
- Tenure at selling club (a weak proxy for contract length)

**B. Performance (last 365 days and last 2 seasons)**
- Minutes, starts, minutes share of team minutes; goals, assists, cards per 90 (from appearances)
- Understat xG, npxG, xA, xGChain, xGBuildup, key passes and shots per 90 (missing outside Understat leagues; LightGBM handles NaN)
- Trend in minutes share (last 6 months vs the 6 months before)

**C. Experience**
- Career top-flight minutes; **league-adjusted minutes** (each match's minutes weighted by opponent ClubElo)
- European competition minutes; big-game minutes (vs top-quartile Elo opponents)
- International caps if the dataset has them *(checkpoint 0: only current totals exist, so not usable historically)*
- **Experience relative to age** (league-adjusted minutes ÷ years since age 17)
- Prior minutes in the buying club's league ("proven in the league")

**D. Age index (profile-aware aging)**
- Estimate value aging curves using the **delta method**: year-on-year change in log Transfermarkt value for the *same* player, averaged by age, then summed cumulatively into a curve. This avoids survivor bias.
- Build curves per position group in phase 1 and per archetype in phase 2. Peak age = the curve's maximum.
- `age_index = age − peak_age_for_profile`

**E. Hype / popularity** (English Wikipedia page views, mapped via Wikidata P2446)
- **Brand:** mean daily views over the 365 days before the date
- **Hype:** ratio of the 30-day mean to the 365-day mean
- **Abnormal hype:** residual from regressing log brand views on performance + experience features (fit on the training period only)
- Missing before Aug 2015. Leave as NaN.

**F. Availability (injury proxy, free version)**
- Infer **unavailability spells** from lineups and appearances: the player was registered at the club (between his transfer dates), had been a regular (≥50% minutes share in the previous 10 matches), then was absent from the matchday squad for ≥3 consecutive club matches.
- Don't count the match right after a red card (approximate suspension handling).
- Features: availability rate over the last 1/2/3 seasons, number of long spells (≥5 matches) in the last 24 months, months since the last long spell, recurrence flag (≥2 long spells within 12 months)
- Label these everywhere as **"inferred unavailability"**, not confirmed injuries.
- Make the module pluggable so a real injury dataset can replace it later.

**G. Market momentum** (TM-derived, so excluded from the "stats only" model)
- Change in log Transfermarkt value over the prior 6 and 12 months; distance from career-high value

**H. Crowd value** (TM-corrected model only)
- Log Transfermarkt value ≥30 days before the transfer

**I. Pace and profile (phase 2)**
- SoFIFA pace, acceleration, sprint speed, stamina and strength for the edition current at the transfer date
- **Archetypes:** Gaussian Mixture Model per position group on standardised per-90 stats plus SoFIFA attributes. Store soft memberships (e.g. 0.6 winger / 0.4 playmaker), name the clusters by hand afterwards, and show me the cluster profiles.
- Pace-dependence score: pace z-score within position × archetype membership weighted by each archetype's athletic profile. Keep it simple and document it.
- Estimate a **pace decline curve per archetype** (delta method on SoFIFA pace) and use it in forecasting.

**Contract (current valuations only):** years left on the current contract is very important but has no training history. Do two things:
1. Investigate whether past snapshots of the dataset can be retrieved (Kaggle dataset versions or the repo's DVC/git history). If they can, reconstruct contract-at-transfer and add it as a feature. Report feasibility at checkpoint 2.
2. Start archiving a dated copy of the `players` table on every pipeline run so contract history builds up from now on.

*Checkpoint 0 outcome:* Kaggle versions carry `contract_expiration_date` from about late November 2022, so contract-at-transfer can be rebuilt for transfers from December 2022 only. Chosen approach: a **separate contract-adjustment model** fitted on 2023+ transfers, layered on top of the main pricing model. Because 2023+ falls in the test split, the way to fit it without breaking the "touch test once" rule is decided at checkpoint 2.

**Entity matching across sources** (Transfermarkt ↔ Understat ↔ SoFIFA ↔ ClubElo club names):
- Normalise names (unidecode, lowercase, strip punctuation)
- Match on name + date of birth + club/season where possible, using rapidfuzz for fuzzy matching
- Save match tables and **report match rates**; spot-check 20 random matches per source

---

## 7. Models

**Pricing model (price today)**
- Baselines:
  - Transfermarkt value ≥30 days before transfer, adjusted with a single scale factor fit on train
  - Ridge regression
- **LightGBM "stats only"** (groups A–F, I): regression on the real-value target (section 15) with native categorical features
- **LightGBM "TM-corrected"** (all groups including G, H)
- Tuning: Optuna (or a modest grid) with time-ordered cross-validation inside the training period, early stopping on validation. Fixed random seeds.
- Predictions in money = predicted real value × market index on the valuation date (section 15).

**Price ranges**
- LightGBM quantile models (q = 0.1, 0.5, 0.9), then **conformalized quantile regression (CQR)** on the calibration split so 80% intervals really cover about 80%. Implement CQR directly (a few lines) or use MAPIE.

**Explanations**
- SHAP TreeExplainer: global summary plus a per-player waterfall
- Aggregate SHAP by factor group for statements like "£8m of his price comes from hype"

**Evaluators (multiple valuation views)**
1. **Model value:** the LightGBM TM-corrected median and range
2. **Comparables value:** median inflation-adjusted fee of the K=10 nearest past transfers (standardised age, position, minutes share, per-90 output, club Elo, league). Show the comps.
3. **Crowd value:** current Transfermarkt value, rescaled by the baseline factor
4. **Book value (estimate):** fee paid when the player last joined, written off straight-line over 5 years unless the contract length is known. Clearly labelled as an estimate; it's the seller's accounting floor.
- **Consensus:** weighted blend of 1–3, weights from inverse validation error (or a simple stacking model). Show the **spread between evaluators** as a disagreement indicator.

**Forecast (value in 12 months)**
- Dense target: `log(TM value at t+12m / TM value at t)` on twice-yearly snapshot dates (1 Jan, 1 Jul), features as of t
- Add pace, pace × age index, availability, hype-decay (hype ratio) and experience trajectory features
- LightGBM quantiles (0.1/0.5/0.9) + CQR, giving bear / base / bull
- Baselines: no change; position/archetype age curve only
- Forecast fair value = current fair value × predicted ratio. Document this assumption.
- **Combined with the market forecast** (section 15.4): future price = player real-value forecast × market index scenario.

**Undervalued signal backtest**
- On snapshot dates in the test period, flag players where model median ÷ Transfermarkt value is in the top decile
- Compare the next-12-month TM value change (and any sale fees) of flagged vs unflagged players
- Report honestly, including if the signal doesn't work

---

## 8. Evaluation (what the report must show)

- **Median absolute % error (MdAPE)**, MAE on log scale, % of predictions within ±25% and ±50%. Compare Transfermarkt baseline, ridge, LightGBM stats-only and LightGBM TM-corrected.
- Broken down by fee band (<€5m, €5–20m, €20–50m, €50m+), buying league, position group and age band
- **Interval quality:** empirical coverage of the 80% intervals (target 75–85%) and median interval width
- **Ablation:** change in validation MdAPE when each factor group is removed. This answers "do pace / hype / injuries / experience matter?" Show it as a bar chart.
- Forecast metrics vs both forecast baselines
- Undervalued-signal backtest results
- **Market index:** comparison of index methods on validation; errors by window and by segment (no trend expected; flag drift); market-forecast backtest vs no-change (section 15)
- Plots: transfer price index over time (by segment, with uncertainty bands); predicted vs actual (log-log); calibration; SHAP summary; aging curves by archetype; pace decline curves; errors by window
- Write the report to `reports/model_report.md` with PNG charts in `reports/figures/`

---

## 9. App (phase 4)

Streamlit app:
- **Player page:**
  - Consensus fair value with 80% range, in **£ and €** (FX rate set in config)
  - Toggle between **"today's money"** and **"money of the time"** (section 15.6)
  - Transfermarkt value and the % gap ("undervalued by 18%")
  - Evaluator breakdown with spread
  - SHAP waterfall grouped by factor
  - 12-month forecast fan chart (bear/base/bull), split into **player vs market** contributions
  - Comparable transfers
  - Factor scores (see below)
  - **Durability rating** AAA → CCC from inferred availability percentiles
  - Inferred unavailability timeline
- **Factor scores (0–100):** percentile within position group among current players for Quality, Experience, Age Index (closer to or before peak scores higher), Brand, Hype, Durability, Momentum, Pace (when available) and Contract (years left).
- **Screener page:** filter by league, position, age and value; sort by gap to Transfermarkt, forecast upside or factor scores.
- **Methodology page:** data sources, how values are made, **how the market index works**, known limitations.
- Scope for current valuations: start with all current **Premier League** players, configurable to more leagues. Current = as of 30 June 2026 (source frozen at 6 July 2026).

---

## 10. Project structure

```
football-valuation/
  README.md            # overview, headline findings, how to run, sources + terms, limitations
  config.yaml          # dates, thresholds, leagues, FX, rate limits, paths, market index + scenarios
  pyproject.toml       # or requirements.txt
  Makefile             # make ingest / features / train / evaluate / value / app / all / test
  data/                # raw/, interim/, processed/, snapshots/ (gitignored)
    manual/            # league_revenues.csv (hand-entered, sourced; committed to git)
  src/fv/
    ingest/            # transfermarkt.py, fx.py, understat.py, clubelo.py, wikidata.py, wikipedia.py, sofifa.py
    entity/            # matching.py
    features/          # base.py, performance.py, experience.py, aging.py, hype.py,
                       # availability.py, momentum.py, profile.py, market_index.py, build.py
    models/            # pricing.py, intervals.py, comps.py, evaluators.py, forecast.py, archetypes.py,
                       # market_forecast.py
    evaluation/        # metrics.py, backtest.py, ablation.py, report.py
    app/               # streamlit_app.py
  tests/               # leakage tests, feature unit tests, small pipeline smoke test
  reports/
```

Suggested dependencies: pandas, pyarrow, duckdb, numpy, scikit-learn, lightgbm, shap, optuna, mapie (optional), soccerdata, understat (optional), requests, rapidfuzz, unidecode, pyyaml, matplotlib, plotly, streamlit, pytest.

Engineering standards: type hints, logging, one CLI command per stage, idempotent and cached stages, `make all` runs end-to-end from a clean checkout.

---

## 11. Phases and checkpoints

**Phase 0 — Setup and data audit** *(done)*
- Scaffold the project and download transfermarkt-datasets.
- Report:
  - tables and columns, row counts and date ranges
  - fee coverage (known / free / loan / unknown)
  - whether game_lineups and international caps exist
  - leagues covered
  - feasibility of contract history
- **CHECKPOINT 0:** show me the audit before going further. *(done)*

**Phase 1 — Core pricing model**
- *Carried over from Phase 0:* currency check (section 15.2) and ECB FX ingest; re-run the audit.
- Base, performance (appearances-based), experience, momentum and crowd features, plus ClubElo
- Market index (section 15.1): rolling hedonic index, segment indices with partial pooling, repeat-sales and TM-level cross-checks, method choice on validation, nowcast
- Time splits, baselines, both LightGBM models, quantiles + CQR, SHAP, leakage tests
- Diagnostics (section 15.5): errors by window and by segment
- First version of the report
- **CHECKPOINT 1:** show the metrics table (validation only, not test yet), the currency check results, the index chart by segment with uncertainty bands, and the errors-by-window plot.

**Phase 2 — Extra factors**
- Understat, Wikipedia hype (via Wikidata), inferred availability, SoFIFA pace, archetypes, profile-aware age index and pace decline curves
- Match-rate reports; ablation study
- **CHECKPOINT 2:** show ablation results, archetype profiles, aging curves and contract-history feasibility.

**Phase 3 — Evaluators, forecast, signal**
- Comps, crowd, book value and consensus
- 12-month forecast model with baselines; undervalued-signal backtest
- Market forecast (section 15.3): revenue-linked and random-walk models, Monte Carlo scenarios, one-year-ahead backtest
- Combined player forecast (section 15.4) with player vs market breakdown
- Diagnostics repeated (section 15.5)
- **Run the final test-set evaluation once.**
- **CHECKPOINT 3:** show final test results and the market-forecast backtest.

**Phase 4 — App and polish**
- Retrain for current valuations; Streamlit app
- Inflation-adjusted records chart, money toggle, player vs market fan chart, index methodology (section 15.6)
- README with headline findings and charts (including the index chart and the records chart); methodology page
- **CHECKPOINT 4:** demo the app and summarise the findings.

---

## 12. Definition of done (v1)

- `make all` runs the whole pipeline from scratch on a fresh machine.
- The report compares all models against the Transfermarkt baseline on the untouched test set, with breakdowns.
- 80% intervals have 75–85% empirical coverage on the test set (or the report explains why not).
- Leakage tests pass, including for the market index.
- The app shows a full player page for every current Premier League player.
- The README documents sources, terms, methods, findings and limitations.

---

## 13. Known limitations to document

- Reported fees are noisy (add-ons, undisclosed fees, swap deals, currency conversions).
- Only players who moved have fees (selection bias).
- No historical contract length unless reconstructed.
- Injuries are inferred from absences, not confirmed medical data.
- SoFIFA attributes are game ratings, not measured physical data.
- Wikipedia views start in 2015 and are English-language only in v1.
- Very few €100m+ deals, so the top end is uncertain.
- The market index is an estimate with its own uncertainty, and thin segments (e.g. top-end deals in a winter window) lean heavily on the pooled overall index.
- The combined forecast assumes player value and market level move independently.
- League revenue coverage is limited to figures verifiable from public sources.

---

## 14. Roadmap (not v1, but design so these can slot in)

- Factor-projection forecasts with Monte Carlo scenarios (project age, hype decay, contract run-down and availability, then re-price)
- Survival model for injury risk once real injury histories are available
- Cash-flow evaluator (contribution → points → prize money/revenue, minus wages) once wage estimates are sourced
- Buyer-specific value (what a player is worth to a given club)
- Premier League Squad Cost Ratio headroom per club
- More leagues; non-English Wikipedia editions *(repeat-sales index moved into v1 as a cross-check, section 15.1)*

---

## 15. Market level / football inflation

**Goal.** Separate how good a player is from how expensive the market is:

`price = player's real value × market index`

The model predicts **real value** = `log(fee) − log(market index at the transfer date)`. The index converts real value into money at any date: past, present or future. This matters because LightGBM cannot extrapolate beyond price levels it has seen. Never use CPI or general inflation.

### 15.1 Market index (Phase 1)

- **Main method: rolling hedonic index.** Regress log fee on player features plus **window fixed effects** (one per summer / winter window). For each window, fit using only transfers from **earlier windows**, and carry the last known level forward. No leakage.
- **Segment indices:**
  - Premier League buyers vs other buyers
  - Top-end deals (top 10% of fees) vs the rest. *Define "top 10%" from earlier windows only*, so the segment label itself doesn't leak.
- **Partial pooling:** where a segment has few deals, shrink its window effect toward the overall index. Shrinkage strength is set in `config.yaml` (as a prior "pseudo-count" of deals: the segment estimate is a weighted average of its own estimate and the overall one, weighted by deal count vs pseudo-count).
- **Cross-checks:**
  - (a) **Repeat-sales index** from players sold 2+ times, adjusted for change in age and minutes share between sales.
  - (b) **Transfermarkt market level:** median TM value of top-5-league players on each date. Used as a real-time signal and as a fallback.
- **Choose** between the methods on the validation split (which one gives the best pricing-model error once used as the deflator) and report how they compare.
- **Nowcast** the current window: start from the last known level, then update it with this window's deals so far (shrunk toward the carried-forward level) and with the movement in the TM market level since the last known level.
- Output: one index level (with standard error) per window and segment, plus a daily series obtained by carrying levels forward.

### 15.2 Currency check (Phase 0 → moved to Phase 1)

- Find out how the dataset converted fees to EUR. In particular, check whether older fees were converted at a single recent exchange rate. Report the findings (originally for checkpoint 0; now at checkpoint 1, or immediately if found sooner).
- Store each fee in EUR and GBP using the exchange rate **on the transfer date**. Use free ECB reference rates (ECB Data Portal or the Frankfurter API) and cache them.

### 15.3 Market forecast (Phase 3)

- Forecast the index 1–3 years ahead in two ways:
  - (a) **Revenue-linked:** model index growth against league revenue growth. Create `data/manual/league_revenues.csv` with columns `season, league, revenue, currency, source_url, published_date`. Fill it only with figures verifiable from a public source (e.g. Deloitte Annual Review of Football Finance, league announcements). Unverifiable → leave blank and tell the user. **Never invent figures.** Use only revenue data published before each forecast date.
  - (b) **Statistical:** log index as a random walk with drift; estimate drift and volatility.
- **Monte Carlo** simulation for base / bull / bear scenarios. The bear case is set to the **largest historical drop in the index** (check the 2020 COVID windows). Scenario parameters live in `config.yaml`.
- **Backtest:** from each past window, forecast the level one year ahead; compare error with a no-change baseline. Report at checkpoint 3.

### 15.4 Combined player forecast (Phase 3)

- Future price distribution = player's real-value forecast (existing forecast model) × market index scenarios, **combined by sampling both**. Document the assumption that the two are independent.
- Break each forecast into the **player's contribution and the market's contribution** (e.g. "+12% player, +5% market").

### 15.5 Diagnostics (Phase 1, repeated in Phase 3)

- Plot model errors by window and by segment. They should show no trend; flag any drift.
- Show the index chart (by segment, with uncertainty bands) and the errors-by-window plot at checkpoint 1.

### 15.6 Outputs (Phase 4)

- Chart: **"Most expensive transfers of all time, adjusted for football inflation"**: top 20 in today's £ and €, with the original fee alongside.
- App: toggle between "today's money" and "money of the time"; fan chart with the player vs market breakdown; methodology section explaining the index.
- README: add the index chart and the inflation-adjusted records chart.

### 15.7 Config

`config.yaml` holds: segment definitions, shrinkage strength, base date for "today's money", FX source, and scenario parameters.
