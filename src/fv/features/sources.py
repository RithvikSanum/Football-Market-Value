"""DuckDB views over the source tables, optionally truncated at a cutoff date.

Every feature module reads only these views. The leakage test rebuilds features for a transfer from a
connection whose views are truncated at that transfer's date and checks the results are identical.
"""
from __future__ import annotations

import duckdb

from fv.config import path

TM_TABLES = ["appearances", "game_lineups", "games", "player_valuations", "players", "transfers",
             "clubs", "competitions"]
DATE_COL = {"appearances": "date", "game_lineups": "date", "games": "date",
            "player_valuations": "date", "transfers": "transfer_date"}
PROCESSED = {"elo_club_history": "date", "elo_games": "date"}


def connect(cutoff: str | None = None, max_date: str | None = None) -> duckdb.DuckDBPyConnection:
    """cutoff: keep only rows dated strictly before it (leakage tests).
    max_date: drop rows after the dataset's real cutoff (future-dated scheduled transfers)."""
    con = duckdb.connect()
    tm = path("interim") / "transfermarkt"
    for t in TM_TABLES:
        where = []
        if t in DATE_COL and cutoff:
            where.append(f"{DATE_COL[t]} < DATE '{cutoff}'")
        if t == "transfers" and max_date:
            where.append(f"transfer_date <= DATE '{max_date}'")
        w = f" WHERE {' AND '.join(where)}" if where else ""
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{tm / (t + '.parquet')}'){w}")
    for t, dc in PROCESSED.items():
        w = f" WHERE {dc} < DATE '{cutoff}'" if cutoff else ""
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{path('processed') / (t + '.parquet')}'){w}")
    # optional phase-2 sources (present once ingested)
    wd = path("interim") / "wikidata_tm_enwiki.parquet"
    if wd.exists():   # static id -> article map (no dates)
        con.execute(f"CREATE VIEW wikidata_map AS SELECT * FROM read_parquet('{wd}')")
    pv = path("interim") / "wiki_pageviews.parquet"
    if pv.exists():
        w = f" WHERE date < DATE '{cutoff}'" if cutoff else ""
        con.execute(f"CREATE VIEW wiki_pageviews AS SELECT * FROM read_parquet('{pv}'){w}")
    return con
