"""Group E: popularity from English Wikipedia page views (mapped via Wikidata P2446).

- brand_views: mean daily views over [t-365d, t), counting days with no recorded views as 0, with the
  window clipped to the API start (1 Jul 2015); NaN if fewer than 30 days of window are available.
- log_brand = log(1 + brand_views)
- hype_ratio = log((1 + 30-day mean) / (1 + 365-day mean))
- abnormal_hype (added in build.main): residual of log_brand on performance / experience / age
  features, fitted on training-period transfers only.
Players with a Wikidata item but no English article get 0 views; players missing from Wikidata or
before Aug 2015 get NaN. Article titles are today's titles: views logged under an older title before
a page move are missed (documented limitation).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
HYPE_COLS = ["log_brand", "hype_ratio"]
API_START = "2015-07-01"


def hype_features(con) -> pd.DataFrame:
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    tg = con.sql("SELECT transfer_id FROM targets").df()
    if not {"wikidata_map", "wiki_pageviews"} <= tables:
        log.warning("hype: Wikipedia sources not ingested; group E is NaN")
        return tg.assign(**{c: np.nan for c in HYPE_COLS})
    df = con.sql(f"""
      WITH tg AS (SELECT transfer_id, player_id, t FROM targets),
      w AS (
        SELECT tg.*, m.article, m.qid,
          greatest(tg.t - INTERVAL 365 DAY, DATE '{API_START}') AS lo365,
          greatest(tg.t - INTERVAL 30 DAY, DATE '{API_START}') AS lo30
        FROM tg LEFT JOIN wikidata_map m USING (player_id)),
      v AS (
        SELECT w.transfer_id,
          sum(p.views) FILTER (WHERE p.date >= w.lo365) AS v365,
          sum(p.views) FILTER (WHERE p.date >= w.lo30) AS v30
        FROM w JOIN wiki_pageviews p ON p.article = w.article AND p.date >= w.lo365 AND p.date < w.t
        GROUP BY 1)
      SELECT w.transfer_id, w.qid, w.article,
        date_diff('day', w.lo365, w.t) AS d365, date_diff('day', w.lo30, w.t) AS d30,
        coalesce(v.v365, 0) AS v365, coalesce(v.v30, 0) AS v30
      FROM w LEFT JOIN v USING (transfer_id)""").df()
    ok = df.qid.notna() & (df.d365 >= 30) & (df.d30 >= 30)
    b365 = df.v365 / df.d365.clip(lower=1)
    b30 = df.v30 / df.d30.clip(lower=1)
    df["log_brand"] = np.where(ok, np.log1p(b365), np.nan)
    df["hype_ratio"] = np.where(ok, np.log1p(b30) - np.log1p(b365), np.nan)
    return df[["transfer_id", *HYPE_COLS]]


ABNORMAL_X = ["age", "mins_365", "ga_p90_365", "min_share_365", "career_league_mins", "adj_mins",
              "euro_mins", "big_game_mins", "from_elo", "to_elo"]


def add_abnormal_hype(df: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    """abnormal_hype = log_brand minus OLS prediction from performance/experience (train rows only)."""
    X = df[ABNORMAL_X].astype(float)
    X = X.fillna(X[train_mask].median())
    X = pd.concat([X, np.log1p(X[["mins_365", "career_league_mins", "adj_mins", "euro_mins",
                                   "big_game_mins"]].clip(lower=0)).add_prefix("log_")], axis=1)
    fit = train_mask & df.log_brand.notna()
    mu, sd = X[fit].mean(), X[fit].std().replace(0, 1)
    Z = np.c_[np.ones(len(X)), ((X - mu) / sd).values]
    beta, *_ = np.linalg.lstsq(Z[fit.values], df.log_brand[fit].values, rcond=None)
    pred = Z @ beta
    df = df.assign(abnormal_hype=np.where(df.log_brand.notna(), df.log_brand - pred, np.nan))
    r2 = 1 - np.var(df.abnormal_hype[fit]) / np.var(df.log_brand[fit])
    return df, {"n_fit": int(fit.sum()), "r2": float(r2)}
