# Phase 0 data audit — transfermarkt-datasets


## 0. Source status (important)

transfermarkt-datasets **updates are paused; data is current to 6 July 2026** (README + Kaggle notice,
GitHub discussion #383). `games` ends 2026-07-06, `appearances` 2026-06-28, `player_valuations` 2026-06-12,
and **2026/27 squads are not covered**. Implication: "current" valuations can only be as of June 2026
(end of 2025/26), and summer-2026 transfers lack post-window club assignments. Licence: **CC0 Public Domain**
(Kaggle page). Download: R2 bucket linked from the repo README (zip of gzipped CSVs, last modified 2026-09-05).


## 1. Tables

| table             |      rows | date range              | columns                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:------------------|----------:|:------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| appearances       | 1,894,350 | 2012-07-03 → 2026-06-28 | appearance_id, game_id, player_id, player_club_id, player_current_club_id, date, player_name, competition_id, yellow_cards, red_cards, goals, assists, minutes_played                                                                                                                                                                                                                                                                     |
| club_games        |   177,916 |                         | game_id, club_id, own_goals, own_position, own_manager_name, opponent_id, opponent_goals, opponent_position, opponent_manager_name, hosting, is_win                                                                                                                                                                                                                                                                                       |
| clubs             |       796 |                         | club_id, club_code, name, domestic_competition_id, total_market_value, squad_size, average_age, foreigners_number, foreigners_percentage, national_team_players, stadium_name, stadium_seats, net_transfer_record, coach_name, last_season, filename, url                                                                                                                                                                                 |
| competitions      |        65 |                         | competition_id, competition_code, name, sub_type, type, country_id, country_name, domestic_league_code, confederation, total_clubs, url                                                                                                                                                                                                                                                                                                   |
| countries         |       124 |                         | country_id, country_name, country_code, confederation, total_clubs, total_players, average_age, url                                                                                                                                                                                                                                                                                                                                       |
| game_events       | 1,274,469 | 2006-06-09 → 2026-07-06 | game_event_id, date, game_id, minute, type, club_id, club_name, player_id, description, player_in_id, player_assist_id                                                                                                                                                                                                                                                                                                                    |
| game_lineups      | 3,179,016 | 2013-07-02 → 2026-07-06 | game_lineups_id, date, game_id, player_id, club_id, player_name, type, position, number, team_captain                                                                                                                                                                                                                                                                                                                                     |
| games             |    88,958 | 2006-06-09 → 2026-07-06 | game_id, competition_id, season, round, date, home_club_id, away_club_id, home_club_goals, away_club_goals, home_club_position, away_club_position, home_club_manager_name, away_club_manager_name, stadium, attendance, referee, url, home_club_formation, away_club_formation, home_club_name, away_club_name, aggregate, competition_type                                                                                              |
| national_teams    |       124 |                         | national_team_id, name, team_code, country_id, country_name, country_code, confederation, team_image_url, squad_size, average_age, foreigners_number, foreigners_percentage, total_market_value, coach_name, fifa_ranking, last_season, url                                                                                                                                                                                               |
| player_valuations |   656,301 | 2000-01-20 → 2026-06-12 | player_id, date, market_value_in_eur, current_club_name, current_club_id, player_club_domestic_competition_id                                                                                                                                                                                                                                                                                                                             |
| players           |    50,149 |                         | player_id, first_name, last_name, name, last_season, current_club_id, player_code, country_of_birth, city_of_birth, country_of_citizenship, date_of_birth, sub_position, position, foot, height_in_cm, contract_expiration_date, agent_name, image_url, international_caps, international_goals, current_national_team_id, url, current_club_domestic_competition_id, current_club_name, market_value_in_eur, highest_market_value_in_eur |
| transfers         |   175,165 | 1993-07-01 → 2030-06-30 | player_id, transfer_date, transfer_season, from_club_id, to_club_id, from_club_name, to_club_name, transfer_fee, market_value_in_eur, player_name                                                                                                                                                                                                                                                                                         |

## 2. Fee coverage and training-set funnel

**Encoding finding:** the `transfers` table has no transfer-type column. Loans, end-of-loan moves and free transfers all appear as `transfer_fee = 0` (loan fees are not recorded: e.g. João Félix to Chelsea, Jan 2023, shows 0). `NaN` = unknown/undisclosed fee. Positive fees on out-and-back pairs are mostly permanent sale + loan back (Pulisic, Saliba, Pedri), so `fee > 0` is treated as a permanent transfer. The loan / free split below is a heuristic (out-and-back to the same club within 400 days) and only informational — all zero-fee rows are excluded anyway.

Data cutoff used: 2026-07-06; fee threshold: €250,000.

| category                                   |   rows |   pct |
|:-------------------------------------------|-------:|------:|
| KEEP: permanent, fee >= threshold          |  14830 |   8.5 |
| fee 0: end-of-loan (heuristic: return leg) |  29615 |  16.9 |
| fee 0: free transfer / other               |  39053 |  22.3 |
| fee 0: loan (heuristic: outbound leg)      |  26912 |  15.4 |
| fee > 0 but < threshold                    |   2717 |   1.6 |
| future-dated (after data cutoff)           |    520 |   0.3 |
| unknown fee (NaN)                          |  61518 |  35.1 |

### Kept rows by split (config.yaml dates)

| split                   |    n | first               | last                |   median_fee_m |   <5m |   5-20m |   20-50m |   50m+ |
|:------------------------|-----:|:--------------------|:--------------------|---------------:|------:|--------:|---------:|-------:|
| 0 pre-window (excluded) |  470 | 2002-07-01 00:00:00 | 2012-03-01 00:00:00 |           2    |   348 |     103 |       18 |      1 |
| 1 train                 | 4966 | 2012-07-01 00:00:00 | 2020-06-30 00:00:00 |           2.25 |  3420 |    1183 |      309 |     54 |
| 2 validation            |  883 | 2020-07-01 00:00:00 | 2021-06-30 00:00:00 |           1.95 |   643 |     180 |       51 |      9 |
| 3 calibration           | 1105 | 2021-07-01 00:00:00 | 2022-06-25 00:00:00 |           1.82 |   830 |     226 |       42 |      7 |
| 4 test                  | 7406 | 2022-07-01 00:00:00 | 2026-07-02 00:00:00 |           2    |  5377 |    1510 |      421 |     98 |

### Selling league of kept rows (clubs table, else player's last valuation record at that club)

| selling_league   |    n |   pct |
|:-----------------|-----:|------:|
| (unknown)        | 3198 |  21.6 |
| IT1              | 1076 |   7.3 |
| FR1              | 1006 |   6.8 |
| GB1              |  998 |   6.7 |
| L1               |  854 |   5.8 |
| ES1              |  730 |   4.9 |
| BRA1             |  604 |   4.1 |
| ARG1             |  596 |   4   |
| PO1              |  584 |   3.9 |
| NL1              |  548 |   3.7 |
| BE1              |  529 |   3.6 |
| DK1              |  344 |   2.3 |
| RU1              |  317 |   2.1 |
| TR1              |  287 |   1.9 |
| MLS1             |  285 |   1.9 |
| SE1              |  251 |   1.7 |
| NO1              |  237 |   1.6 |
| MEX1             |  211 |   1.4 |
| C1               |  200 |   1.3 |
| SER1             |  199 |   1.3 |
| KR1              |  190 |   1.3 |
| TS1              |  187 |   1.3 |
| GR1              |  183 |   1.2 |
| A1               |  180 |   1.2 |
| PL1              |  174 |   1.2 |
| UKR1             |  165 |   1.1 |
| COL1             |  150 |   1   |
| SC1              |  133 |   0.9 |
| RO1              |  124 |   0.8 |
| SA1              |   98 |   0.7 |
| JAP1             |   85 |   0.6 |
| RSK1             |   70 |   0.5 |
| AUS1             |   37 |   0.2 |

### Share of kept transfers with any appearance data in the prior 365 days

|   year |    n |   pct_with_apps_last_365d |
|-------:|-----:|--------------------------:|
|   2002 |    2 |                       0   |
|   2003 |    1 |                       0   |
|   2004 |    3 |                       0   |
|   2005 |    8 |                       0   |
|   2006 |   16 |                       0   |
|   2007 |   47 |                       0   |
|   2008 |   43 |                       0   |
|   2009 |   62 |                       0   |
|   2010 |   99 |                       0   |
|   2011 |  154 |                       0   |
|   2012 |  200 |                       7   |
|   2013 |  308 |                      51   |
|   2014 |  348 |                      52.3 |
|   2015 |  502 |                      54.4 |
|   2016 |  613 |                      50.9 |
|   2017 |  789 |                      54.2 |
|   2018 |  857 |                      51.5 |
|   2019 | 1116 |                      48.5 |
|   2020 |  938 |                      45.4 |
|   2021 |  940 |                      51.6 |
|   2022 | 1400 |                      50.1 |
|   2023 | 1704 |                      48.6 |
|   2024 | 1800 |                      47.2 |
|   2025 | 2121 |                      54.3 |
|   2026 |  759 |                      52   |

## 3. Transfermarkt value at transfer: leakage check

`mv_transfers` = `transfers.market_value_in_eur`; `mv_lagged` = last `player_valuations` entry ≥30 days before the transfer; `mv_after` = first valuation within 60 days after (shown only to illustrate post-deal updating). A higher correlation for `mv_transfers` than `mv_lagged` would indicate the transfers-table value has absorbed information from the deal.

| version      | available       |   median fee/TM | MdAPE (unscaled)   |   corr log |
|:-------------|:----------------|----------------:|:-------------------|-----------:|
| mv_transfers | 14,283 / 14,830 |            1    | 43.3%              |      0.794 |
| mv_lagged    | 14,210 / 14,830 |            1.05 | 47.1%              |      0.772 |
| mv_after     | 3,914 / 14,830  |            1    | 26.3%              |      0.878 |

The two pre-transfer versions differ on 23.8% of rows where both exist.

## 4. Lineups, international caps, player bios

`game_lineups` exists (types: starting_lineup, substitutes). Coverage by competition type:

| competition_type          |   games |   games_with_lineups |   pct |
|:--------------------------|--------:|---------------------:|------:|
| domestic_league           |   63382 |                58366 |  92.1 |
| domestic_cup              |   12904 |                11805 |  91.5 |
| other                     |    7599 |                 6819 |  89.7 |
| international_cup         |    3117 |                 2901 |  93.1 |
| nan                       |    1214 |                 1106 |  91.1 |
| national_team_competition |     742 |                  200 |  27   |

**International caps** exist in `players` (`international_caps`, `international_goals`) but only as the **current career total** at scrape time. Using them for historical transfers would leak, so they are usable only for current valuations. National-team appearances in the dataset start only in Dec 2025 (AFCON 2025, World Cup 2026), so they give **no** historical caps proxy; international experience is dropped from historical features in v1:

| competition_id   | first               | last                |   appearances |
|:-----------------|:--------------------|:--------------------|--------------:|
| AFCN             | 2025-12-21 00:00:00 | 2026-01-18 00:00:00 |           776 |
| FIWC             | 2026-06-11 00:00:00 | 2026-06-28 00:00:00 |          2251 |

Player bio completeness:

|   players |   caps_non_null |   caps_positive |   contract_non_null |   height |   foot |   dob |
|----------:|----------------:|----------------:|--------------------:|---------:|-------:|------:|
|     50149 |           19356 |           16946 |               31580 |    45784 |  44256 | 50100 |

## 5. Leagues covered (games / appearances / lineups)

| competition_id   | country_name   |   first_season |   last_season |   games |   players_with_apps |
|:-----------------|:---------------|---------------:|--------------:|--------:|--------------------:|
| ES1              | Spain          |           2012 |          2025 |    5320 |                2677 |
| GB1              | England        |           2012 |          2025 |    5320 |                2466 |
| IT1              | Italy          |           2012 |          2025 |    5320 |                2736 |
| FR1              | France         |           2012 |          2025 |    4997 |                2893 |
| TR1              | Türkiye        |           2012 |          2025 |    4618 |                3107 |
| L1               | Germany        |           2012 |          2025 |    4284 |                2234 |
| NL1              | Netherlands    |           2012 |          2025 |    4210 |                2617 |
| PO1              | Portugal       |           2012 |          2025 |    4152 |                3059 |
| BE1              | Belgium        |           2012 |          2025 |    3550 |                2660 |
| RU1              | Russia         |           2012 |          2025 |    3360 |                2152 |
| GR1              | Greece         |           2012 |          2025 |    3085 |                2698 |
| SC1              | Scotland       |           2012 |          2025 |    2903 |                2049 |
| UKR1             | Ukraine        |           2012 |          2025 |    2700 |                2078 |
| DK1              | Denmark        |           2012 |          2025 |    2312 |                1698 |
| MLS1             | United States  |           2024 |          2025 |     727 |                   0 |
| SA1              | Saudi Arabia   |           2024 |          2025 |     612 |                   0 |
| PL1              | Poland         |           2024 |          2025 |     612 |                   0 |
| BRA1             | Brazil         |           2024 |          2025 |     557 |                   0 |
| SER1             | Serbia         |           2024 |          2025 |     480 |                   0 |
| TS1              | Czech Republic |           2024 |          2025 |     480 |                   0 |
| RO1              | Romania        |           2024 |          2025 |     480 |                   0 |
| C1               | Switzerland    |           2024 |          2025 |     430 |                   0 |
| ARG1             | Argentina      |           2024 |          2025 |     418 |                   0 |
| A1               | Austria        |           2024 |          2025 |     356 |                   0 |
| KR1              | Croatia        |           2024 |          2025 |     348 |                   0 |
| NO1              | Norway         |           2024 |          2025 |     329 |                   0 |
| SE1              | Sweden         |           2024 |          2025 |     329 |                   0 |
| AUS1             | Australia      |           2024 |          2025 |     325 |                   0 |
| JAP1             | Japan          |           2024 |          2024 |     296 |                   0 |
| MEX1             | Mexico         |           2024 |          2025 |     241 |                   0 |
| RSK1             | Korea, South   |           2024 |          2025 |     231 |                   0 |

## 6. Contract history feasibility

`players.contract_expiration_date` is the current contract only. Two sources of past snapshots checked:

1. **Repo DVC history** (`data/raw/transfermarkt-scraper.dvc`, public HTTP remote). Only the current
   manifest's objects are still on the remote; older commits' objects return 404 (garbage-collected).
   The current manifest contains per-season `players.json.gz` files, but they were **re-scraped long
   after the season** (implied scrape year from age + DOB: seasons 2012–2022 ≈ 2023, season 2023 ≈ 2024),
   so their `contract_expires` is not as-of-season. They also carry `joined` and
   `day_of_last_contract_extension` fields.
2. **Kaggle dataset versions** (`davidcariboo/player-scores`, CC0, 679 versions). Old versions are
   downloadable without login via the Kaggle API download endpoint with `datasetVersionNumber`.
   `players.csv` with `contract_expiration_date` first appears at **≈ v193 (late Nov 2022)** and is
   present in every later version through v679 (5 Sep 2026), roughly weekly. Version sizes grow from
   ~23 MB (2022) to ~246 MB (2026); range requests let us pull just `players.csv` from each zip.

**Verdict:** contract-at-transfer can be reconstructed (last snapshot before the transfer where the player
was at the selling club → years remaining) for transfers from **~Dec 2022 onward** only. That covers part of
validation (2022/23), all of calibration and test, but **none of the training period (< 2022-07-01)**.
Options for checkpoint 2: (a) a separate contract-adjustment model fit on 2023+ transfers, or (b) shifting
splits so a contract-aware model trains on 2023–2025. A dated `players` snapshot is now archived on every
ingest run in `data/snapshots/players/`.


## 7. Currency check (brief 15.2)

**Question:** how were fees converted to EUR, and were older fees converted at one recent rate?

**Test 1: known £ deals.** Dataset € fee ÷ reported £ fee vs the ECB rate on the transfer date. A ratio of 1.000 means the fee was converted at the rate of the day; a single recent rate would give ratios drifting with the historical rate. Deviations (Caicedo, Rice) are headline-with-add-ons vs base-fee differences, not FX.

| player            | to        | date       |   fee in dataset (€m) |   reported (£m) |   implied €/£ |   ECB €/£ on date |   ratio |
|:------------------|:----------|:-----------|----------------------:|----------------:|--------------:|------------------:|--------:|
| Ángel Di María    | Man Utd   | 2014-08-26 |                 75    |            59.7 |         1.256 |             1.256 |   1     |
| Kevin De Bruyne   | Man City  | 2015-08-30 |                 76    |            55   |         1.382 |             1.366 |   1.012 |
| Paul Pogba        | Man Utd   | 2016-08-09 |                105    |            89   |         1.18  |             1.171 |   1.008 |
| Romelu Lukaku     | Man Utd   | 2017-07-10 |                 84.7  |            75   |         1.129 |             1.131 |   0.999 |
| Virgil van Dijk   | Liverpool | 2018-01-01 |                 84.65 |            75   |         1.129 |             1.127 |   1.001 |
| Kepa Arrizabalaga | Chelsea   | 2018-08-08 |                 80    |            71   |         1.127 |             1.11  |   1.015 |
| Harry Maguire     | Man Utd   | 2019-08-05 |                 87    |            80   |         1.088 |             1.088 |   0.999 |
| Jack Grealish     | Man City  | 2021-08-05 |                117.5  |           100   |         1.175 |             1.176 |   0.999 |
| Enzo Fernández    | Chelsea   | 2023-01-31 |                121    |           106.8 |         1.133 |             1.135 |   0.998 |
| Declan Rice       | Arsenal   | 2023-07-15 |                116.6  |           105   |         1.11  |             1.168 |   0.95  |
| Moisés Caicedo    | Chelsea   | 2023-08-14 |                116    |           115   |         1.009 |             1.16  |   0.87  |

**Test 2: round numbers.** Share of fees within 0.25% of a round amount (multiple of 250k, or 50k under 1m). Fees agreed in £ should look round in £ under the conversion actually used. 'Best fixed rate' scans 0.70–0.95 £/€; if old fees had been restated at one recent rate, that rate would beat the transfer-date rate.

| deals               | era     |    n |   round in € |   round in £ @ transfer-date rate |   round in £ @ best fixed rate |
|:--------------------|:--------|-----:|-------------:|----------------------------------:|-------------------------------:|
| England → England   | 2012–16 |   91 |        0.451 |                             0.352 |                          0.297 |
| England → England   | 2017–21 |  155 |        0.439 |                             0.484 |                          0.406 |
| England → England   | 2022–26 |  211 |        0.545 |                             0.592 |                          0.436 |
| Eurozone → Eurozone | 2012–16 |  569 |        0.803 |                             0.127 |                          0.453 |
| Eurozone → Eurozone | 2017–21 | 1299 |        0.829 |                             0.121 |                          0.545 |
| Eurozone → Eurozone | 2022–26 | 1624 |        0.782 |                             0.128 |                          0.493 |

**Findings.** (1) Fees are stored natively in EUR; euro-area deals are round in € (~80%). (2) Fees agreed in £ were converted at the rate around the deal date; for headline English deals the dataset € fee matches £ × ECB rate on the day to within ±0.2%. (3) No sign of restatement at a single recent rate: no fixed rate beats the transfer-date rate. (4) Many English fees are also round in €, so Transfermarkt appears to round converted figures, adding roughly ±1–2% noise. **Action:** keep `fee_eur` as given and derive `fee_gbp` with the ECB rate on the transfer date (`data/processed/transfers_fx.parquet`).