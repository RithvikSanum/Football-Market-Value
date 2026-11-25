"""Phase 0 data audit for transfermarkt-datasets. Writes reports/data_audit.md."""
from __future__ import annotations

import logging

import duckdb
import pandas as pd

from fv.config import load_config, path, setup_logging

log = logging.getLogger(__name__)

DATE_COLS = {
    "appearances": "date", "game_events": "date", "game_lineups": "date", "games": "date",
    "player_valuations": "date", "transfers": "transfer_date",
}


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    d = path("interim") / "transfermarkt"
    for p in sorted(d.glob("*.parquet")):
        con.execute(f"CREATE VIEW {p.stem} AS SELECT * FROM read_parquet('{p}')")
    return con


def tables_section(con) -> str:
    rows = []
    for (t,) in con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall():
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        cols = [c[0] for c in con.execute(f"DESCRIBE {t}").fetchall()]
        rng = ""
        if t in DATE_COLS:
            lo, hi = con.execute(f"SELECT min({DATE_COLS[t]}), max({DATE_COLS[t]}) FROM {t}").fetchone()
            rng = f"{lo} → {hi}"
        rows.append({"table": t, "rows": f"{n:,}", "date range": rng, "columns": ", ".join(cols)})
    return "## 1. Tables\n\n" + _md(pd.DataFrame(rows))


def fee_section(con, cutoff: str, min_fee: float) -> str:
    # Classify every transfer row. Loans / end-of-loan / free all have fee = 0 in this dataset
    # (loan fees are not kept), so we split zero-fee rows heuristically by round trips.
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW t_class AS
    WITH t AS (
      SELECT *,
        lead(from_club_id) OVER w AS n_from, lead(to_club_id) OVER w AS n_to,
        lead(transfer_date) OVER w AS n_date,
        lag(from_club_id) OVER w AS p_from, lag(to_club_id) OVER w AS p_to,
        lag(transfer_date) OVER w AS p_date
      FROM transfers WINDOW w AS (PARTITION BY player_id ORDER BY transfer_date)
    )
    SELECT *,
      CASE
        WHEN transfer_date > DATE '{cutoff}' THEN 'future-dated (after data cutoff)'
        WHEN transfer_fee IS NULL THEN 'unknown fee (NaN)'
        WHEN transfer_fee > 0 AND transfer_fee < {min_fee} THEN 'fee > 0 but < threshold'
        WHEN transfer_fee >= {min_fee} THEN 'KEEP: permanent, fee >= threshold'
        WHEN p_from = to_club_id AND p_to = from_club_id AND transfer_date - p_date <= 400
          THEN 'fee 0: end-of-loan (heuristic: return leg)'
        WHEN n_from = to_club_id AND n_to = from_club_id AND n_date - transfer_date <= 400
          THEN 'fee 0: loan (heuristic: outbound leg)'
        ELSE 'fee 0: free transfer / other'
      END AS cls
    FROM t
    """)
    funnel = con.execute("""
      SELECT cls AS category, count(*) AS rows, round(100.0*count(*)/sum(count(*)) OVER (),1) AS pct
      FROM t_class GROUP BY 1 ORDER BY 1""").df()

    s = load_config()["splits"]
    splits = con.execute(f"""
      SELECT CASE WHEN transfer_date < DATE '{s['train_start']}' THEN '0 pre-window (excluded)'
                  WHEN transfer_date < DATE '{s['train_end']}' THEN '1 train'
                  WHEN transfer_date < DATE '{s['val_end']}' THEN '2 validation'
                  WHEN transfer_date < DATE '{s['calib_end']}' THEN '3 calibration'
                  ELSE '4 test' END AS split,
             count(*) n, min(transfer_date) first, max(transfer_date) last,
             round(median(transfer_fee)/1e6,2) AS median_fee_m,
             sum((transfer_fee<5e6)::int) "<5m", sum((transfer_fee>=5e6 AND transfer_fee<20e6)::int) "5-20m",
             sum((transfer_fee>=20e6 AND transfer_fee<50e6)::int) "20-50m", sum((transfer_fee>=50e6)::int) "50m+"
      FROM t_class WHERE cls LIKE 'KEEP%' GROUP BY 1 ORDER BY 1""").df()

    # Selling / buying league: player's club competition at the latest valuation before the move,
    # falling back to the clubs table (current league).
    leagues = con.execute(f"""
      WITH k AS (SELECT * FROM t_class WHERE cls LIKE 'KEEP%'),
      fl AS (
        SELECT k.player_id, k.transfer_date, k.from_club_id, k.to_club_id,
          coalesce(cf.domestic_competition_id,
            (SELECT v.player_club_domestic_competition_id FROM player_valuations v
              WHERE v.player_id = k.player_id AND v.date < k.transfer_date
                AND v.current_club_id = k.from_club_id ORDER BY v.date DESC LIMIT 1)) AS from_comp
        FROM k LEFT JOIN clubs cf ON cf.club_id = k.from_club_id)
      SELECT coalesce(from_comp, '(unknown)') AS selling_league,
             count(*) n, round(100.0*count(*)/sum(count(*)) OVER (),1) pct
      FROM fl GROUP BY 1 ORDER BY 2 DESC""").df()

    perf = con.execute("""
      WITH k AS (SELECT player_id, transfer_date FROM t_class WHERE cls LIKE 'KEEP%'),
      a AS (
        SELECT k.player_id, k.transfer_date, sum(a.minutes_played) AS mins
        FROM k LEFT JOIN appearances a ON a.player_id = k.player_id
          AND a.date < k.transfer_date AND a.date >= k.transfer_date - INTERVAL 365 DAY
        GROUP BY ALL)
      SELECT year(transfer_date) AS year, count(*) n,
             round(100.0*avg((coalesce(mins,0) > 0)::int),1) AS pct_with_apps_last_365d
      FROM a GROUP BY 1 ORDER BY 1""").df()

    return (
        "## 2. Fee coverage and training-set funnel\n\n"
        "**Encoding finding:** the `transfers` table has no transfer-type column. Loans, end-of-loan "
        "moves and free transfers all appear as `transfer_fee = 0` (loan fees are not recorded: e.g. "
        "João Félix to Chelsea, Jan 2023, shows 0). `NaN` = unknown/undisclosed fee. Positive fees on "
        "out-and-back pairs are mostly permanent sale + loan back (Pulisic, Saliba, Pedri), so "
        "`fee > 0` is treated as a permanent transfer. The loan / free split below is a heuristic "
        "(out-and-back to the same club within 400 days) and only informational — all zero-fee rows "
        "are excluded anyway.\n\n"
        f"Data cutoff used: {cutoff}; fee threshold: €{min_fee:,.0f}.\n\n"
        + _md(funnel)
        + "\n\n### Kept rows by split (config.yaml dates)\n\n" + _md(splits)
        + "\n\n### Selling league of kept rows (clubs table, else player's last valuation record at that club)\n\n" + _md(leagues)
        + "\n\n### Share of kept transfers with any appearance data in the prior 365 days\n\n" + _md(perf)
    )


def tm_value_section(con, lag_days: int) -> str:
    df = con.execute(f"""
      WITH k AS (SELECT * FROM t_class WHERE cls LIKE 'KEEP%'),
      pv AS (
        SELECT k.player_id, k.transfer_date, k.transfer_fee, k.market_value_in_eur AS mv_transfers,
          (SELECT v.market_value_in_eur FROM player_valuations v
            WHERE v.player_id = k.player_id AND v.date <= k.transfer_date - INTERVAL {lag_days} DAY
            ORDER BY v.date DESC LIMIT 1) AS mv_lagged,
          (SELECT v.market_value_in_eur FROM player_valuations v
            WHERE v.player_id = k.player_id AND v.date > k.transfer_date
              AND v.date <= k.transfer_date + INTERVAL 60 DAY
            ORDER BY v.date LIMIT 1) AS mv_after
        FROM k)
      SELECT * FROM pv""").df()
    import numpy as np

    def stats(col: str) -> dict:
        d = df[df[col] > 0]
        lr = np.log(d["transfer_fee"] / d[col])
        ape = (d[col] / d["transfer_fee"] - 1).abs()
        return {"version": col, "available": f"{len(d):,} / {len(df):,}",
                "median fee/TM": round(float(np.exp(lr.median())), 2),
                "MdAPE (unscaled)": f"{100 * ape.median():.1f}%",
                "corr log": round(float(np.corrcoef(np.log(d['transfer_fee']), np.log(d[col]))[0, 1]), 3)}

    tab = pd.DataFrame([stats("mv_transfers"), stats("mv_lagged"), stats("mv_after")])
    both = df[(df.mv_transfers > 0) & (df.mv_lagged > 0)]
    differ = (both.mv_transfers != both.mv_lagged).mean()
    return (
        "## 3. Transfermarkt value at transfer: leakage check\n\n"
        f"`mv_transfers` = `transfers.market_value_in_eur`; `mv_lagged` = last `player_valuations` "
        f"entry ≥{lag_days} days before the transfer; `mv_after` = first valuation within 60 days after "
        "(shown only to illustrate post-deal updating). A higher correlation for `mv_transfers` than "
        "`mv_lagged` would indicate the transfers-table value has absorbed information from the deal.\n\n"
        + _md(tab)
        + f"\n\nThe two pre-transfer versions differ on {100 * differ:.1f}% of rows where both exist."
    )


def misc_section(con) -> str:
    lineups = con.execute("""
      SELECT g.competition_type, count(DISTINCT g.game_id) games,
             count(DISTINCT l.game_id) games_with_lineups,
             round(100.0*count(DISTINCT l.game_id)/count(DISTINCT g.game_id),1) pct
      FROM games g LEFT JOIN game_lineups l USING (game_id) GROUP BY 1 ORDER BY 2 DESC""").df()
    leagues = con.execute("""
      SELECT g.competition_id, c.country_name, min(g.season) first_season, max(g.season) last_season,
             count(DISTINCT g.game_id) games, count(DISTINCT a.player_id) players_with_apps
      FROM games g JOIN competitions c USING (competition_id)
      LEFT JOIN appearances a USING (game_id)
      WHERE g.competition_type = 'domestic_league' GROUP BY ALL ORDER BY first_season, games DESC""").df()
    caps = con.execute("""
      SELECT count(*) players, count(international_caps) caps_non_null,
             sum((international_caps > 0)::int) caps_positive,
             count(contract_expiration_date) contract_non_null,
             count(height_in_cm) height, count(foot) foot, count(date_of_birth) dob
      FROM players""").df()
    nt = con.execute("""
      SELECT competition_id, min(date) first, max(date) last, count(*) appearances
      FROM appearances WHERE competition_id IN (SELECT competition_id FROM competitions
        WHERE type = 'national_team_competition') GROUP BY 1 ORDER BY 1""").df()
    return (
        "## 4. Lineups, international caps, player bios\n\n"
        "`game_lineups` exists (types: starting_lineup, substitutes). Coverage by competition type:\n\n"
        + _md(lineups)
        + "\n\n**International caps** exist in `players` (`international_caps`, `international_goals`) "
        "but only as the **current career total** at scrape time. Using them for historical transfers "
        "would leak, so they are usable only for current valuations. National-team appearances in the "
        "dataset start only in Dec 2025 (AFCON 2025, World Cup 2026), so they give **no** historical "
        "caps proxy; international experience is dropped from historical features in v1:\n\n"
        + _md(nt) + "\n\nPlayer bio completeness:\n\n" + _md(caps)
        + "\n\n## 5. Leagues covered (games / appearances / lineups)\n\n" + _md(leagues)
    )


CONTRACT = """## 6. Contract history feasibility

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
"""

FRESHNESS = """## 0. Source status (important)

transfermarkt-datasets **updates are paused; data is current to 6 July 2026** (README + Kaggle notice,
GitHub discussion #383). `games` ends 2026-07-06, `appearances` 2026-06-28, `player_valuations` 2026-06-12,
and **2026/27 squads are not covered**. Implication: "current" valuations can only be as of June 2026
(end of 2025/26), and summer-2026 transfers lack post-window club assignments. Licence: **CC0 Public Domain**
(Kaggle page). Download: R2 bucket linked from the repo README (zip of gzipped CSVs, last modified 2026-09-05).
"""


KNOWN_GBP_DEALS = {  # widely reported headline fees in GBP (English buyers)
    ("Ángel Di María", "Man Utd"): 59.7e6, ("Kevin De Bruyne", "Man City"): 55e6,
    ("Paul Pogba", "Man Utd"): 89e6, ("Romelu Lukaku", "Man Utd"): 75e6,
    ("Virgil van Dijk", "Liverpool"): 75e6, ("Kepa Arrizabalaga", "Chelsea"): 71e6,
    ("Harry Maguire", "Man Utd"): 80e6, ("Jack Grealish", "Man City"): 100e6,
    ("Enzo Fernández", "Chelsea"): 106.8e6, ("Declan Rice", "Arsenal"): 105e6,
    ("Moisés Caicedo", "Chelsea"): 115e6,
}


def currency_section(con) -> str:
    """Brief 15.2: how were fees converted to EUR?"""
    import numpy as np
    from fv.ingest import fx

    t = con.execute("""
      SELECT t.player_name, t.to_club_name, t.transfer_date, t.transfer_fee,
             cf.domestic_competition_id f, ct.domestic_competition_id tt
      FROM transfers t LEFT JOIN clubs cf ON cf.club_id = t.from_club_id
      LEFT JOIN clubs ct ON ct.club_id = t.to_club_id
      WHERE t.transfer_fee >= 250000 AND t.transfer_date <= (SELECT max(date) FROM games)""").df()
    t["gbp_per_eur"] = fx.rate_on(t["transfer_date"]).values

    rows = []
    for (p, club), gbp in KNOWN_GBP_DEALS.items():
        m = t[(t.player_name == p) & (t.to_club_name == club)]
        if m.empty:
            continue
        r = m.iloc[-1]
        implied = r.transfer_fee / gbp
        ecb = 1 / r.gbp_per_eur
        rows.append({"player": p, "to": club, "date": r.transfer_date.date(),
                     "fee in dataset (€m)": r.transfer_fee / 1e6, "reported (£m)": gbp / 1e6,
                     "implied €/£": round(implied, 3), "ECB €/£ on date": round(ecb, 3),
                     "ratio": round(implied / ecb, 3)})
    known = pd.DataFrame(rows).sort_values("date")

    def round_share(x):
        step = np.where(x >= 1e6, 250e3, 50e3)
        return float(((np.abs(x / step - np.round(x / step)) * step / x) < 0.0025).mean())

    ez = ["ES1", "IT1", "L1", "FR1", "NL1", "PO1", "BE1", "GR1"]
    groups = {"England → England": (t.f == "GB1") & (t.tt == "GB1"),
              "Eurozone → Eurozone": t.f.isin(ez) & t.tt.isin(ez)}
    out = []
    for g, m in groups.items():
        for era, lo, hi in [("2012–16", 2012, 2016), ("2017–21", 2017, 2021), ("2022–26", 2022, 2026)]:
            x = t[m & t.transfer_date.dt.year.between(lo, hi)]
            row = {"deals": g, "era": era, "n": len(x), "round in €": round_share(x.transfer_fee),
                   "round in £ @ transfer-date rate": round_share(x.transfer_fee * x.gbp_per_eur)}
            row["round in £ @ best fixed rate"] = max(
                round_share(x.transfer_fee * r) for r in np.arange(0.70, 0.95, 0.01))
            out.append(row)
    rnd = pd.DataFrame(out).round(3)
    return (
        "## 7. Currency check (brief 15.2)\n\n"
        "**Question:** how were fees converted to EUR, and were older fees converted at one recent rate?\n\n"
        "**Test 1: known £ deals.** Dataset € fee ÷ reported £ fee vs the ECB rate on the transfer date. "
        "A ratio of 1.000 means the fee was converted at the rate of the day; a single recent rate would "
        "give ratios drifting with the historical rate. Deviations (Caicedo, Rice) are headline-with-add-ons "
        "vs base-fee differences, not FX.\n\n" + _md(known)
        + "\n\n**Test 2: round numbers.** Share of fees within 0.25% of a round amount (multiple of 250k, "
        "or 50k under 1m). Fees agreed in £ should look round in £ under the conversion actually used. "
        "'Best fixed rate' scans 0.70–0.95 £/€; if old fees had been restated at one recent rate, that rate "
        "would beat the transfer-date rate.\n\n" + _md(rnd)
        + "\n\n**Findings.** (1) Fees are stored natively in EUR; euro-area deals are round in € (~80%). "
        "(2) Fees agreed in £ were converted at the rate around the deal date; for headline English deals the "
        "dataset € fee matches £ × ECB rate on the day to within ±0.2%. (3) No sign of restatement at a single "
        "recent rate: no fixed rate beats the transfer-date rate. (4) Many English fees are also round in €, "
        "so Transfermarkt appears to round converted figures, adding roughly ±1–2% noise. "
        "**Action:** keep `fee_eur` as given and derive `fee_gbp` with the ECB rate on the transfer date "
        "(`data/processed/transfers_fx.parquet`)."
    )


def main() -> None:
    setup_logging()
    cfg = load_config()
    con = connect()
    cutoff = str(con.execute("SELECT max(date) FROM games").fetchone()[0])
    parts = [
        "# Phase 0 data audit — transfermarkt-datasets\n",
        FRESHNESS,
        tables_section(con),
        fee_section(con, cutoff, cfg["training"]["min_fee_eur"]),
        tm_value_section(con, cfg["training"]["tm_value_lag_days"]),
        misc_section(con),
        CONTRACT,
        currency_section(con),
    ]
    out = path("reports") / "data_audit.md"
    out.write_text("\n\n".join(parts))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
