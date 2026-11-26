"""Player snapshot panel (phase 3): every player valued >= min_value at a club in a core league on each
1 Jan / 1 Jul, with the same features as transfers, plus the 12-month forecast target.

A snapshot is treated as a hypothetical deal *within the player's current club and league*
(from_club = to_club = current club), so "buying league" = current league, no cross-border move.
Features use only data before t (same builders as transfers). Extra columns:
- tm_now: latest Transfermarkt value dated before t (within 180 days); log_tm_now
- y_fwd12 = log(TM value as of t + 365 days / tm_now)   <- forecast target. "As of" = latest valuation on
  or before t + 365, which must be dated at least 180 days after t (a real revaluation; otherwise NaN,
  e.g. retired players or players who dropped out of TM's update cycle). TM revaluation dates are
  irregular (e.g. the April 2020 COVID cut), so a fixed +-window around t + 365 misses many players.
- sale_fee_12m / sale_date_12m: first permanent transfer with a known fee in (t, t + 365]
- log_index: market index level at t (segmented hedonic nowcast at the window start: carried level
  + PL-buyer deviation + top-end deviation; "top end" = log_tm_lag above the 90th percentile of the
  prior 365 days' core transfers, a proxy for the hedonic predicted price)
- crowd_ratio at t (median log fee / TM over the prior 365 days' transfers), log_tm_adj
Output: data/processed/snapshot_panel.parquet
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fv.config import load_config, path, setup_logging
from fv.features import aging, sources
from fv.features.base import LEAGUE_GAMES, league_coverage
from fv.features.build import build_features
from fv.features.hype import HYPE_COLS, add_abnormal_hype, hype_features
from fv.features.windows import window_idx

log = logging.getLogger(__name__)


def core_leagues() -> list[str]:
    return sorted(k for k, v in league_coverage().items() if v <= pd.Timestamp("2012-12-31"))


def snapshot_dates(start: str = "2013-01-01", end: str = "2025-07-01") -> list[pd.Timestamp]:
    return [d for d in pd.date_range(start, end, freq="MS") if d.month in (1, 7)]


def snapshot_targets(con, dates: list[pd.Timestamp], min_value: float,
                     leagues: list[str] | None = None) -> pd.DataFrame:
    leagues = leagues or core_leagues()
    con.register("snap_dates", pd.DataFrame({"t": pd.to_datetime(dates)}))
    con.register("snap_leagues", pd.DataFrame({"lg": leagues}))
    df = con.execute(f"""
      WITH v AS (
        SELECT d.t, pv.player_id, arg_max(pv.market_value_in_eur, pv.date) AS tm_now,
               arg_max(pv.current_club_id, pv.date) AS club_id, max(pv.date) AS tm_now_date
        FROM snap_dates d JOIN player_valuations pv
          ON pv.date < d.t AND pv.date >= d.t - INTERVAL 180 DAY AND pv.market_value_in_eur > 0
        GROUP BY 1, 2),
      lg AS (
        SELECT v.t, v.club_id, arg_max(g.competition_id, g.date) AS league
        FROM (SELECT DISTINCT t, club_id FROM v) v JOIN club_league_games g
          ON g.club_id = v.club_id AND g.date < v.t AND g.date >= v.t - INTERVAL 365 DAY
        GROUP BY 1, 2)
      SELECT v.*, lg.league FROM v JOIN lg USING (t, club_id)
      WHERE v.tm_now >= {min_value} AND lg.league IN (SELECT lg FROM snap_leagues)""").df()
    df["t"] = pd.to_datetime(df.t)
    df["transfer_id"] = "snap_" + df.player_id.astype(str) + "_" + df.t.dt.strftime("%Y%m%d")
    df["from_club_id"] = df.club_id
    df["to_club_id"] = df.club_id
    return df


def forward_targets(con, s: pd.DataFrame) -> pd.DataFrame:
    """12-month-ahead TM value and first known-fee sale (used only as targets / backtest outcomes)."""
    con.register("snap_s", s[["transfer_id", "player_id", "t"]])
    fwd = con.execute("""
      SELECT s.transfer_id,
        arg_max(pv.market_value_in_eur, pv.date) AS tm_fwd12, max(pv.date) AS tm_fwd12_date
      FROM snap_s s JOIN player_valuations pv ON pv.player_id = s.player_id
        AND pv.date > s.t AND pv.date <= s.t + INTERVAL 365 DAY AND pv.market_value_in_eur > 0
      GROUP BY 1, s.t
      HAVING max(pv.date) >= s.t + INTERVAL 180 DAY""").df()
    uni = pd.read_parquet(path("processed") / "features.parquet",
                          columns=["player_id", "transfer_date", "fee_eur"])
    uni["transfer_date"] = pd.to_datetime(uni.transfer_date)
    m = s[["transfer_id", "player_id", "t"]].merge(uni, on="player_id")
    m = m[(m.transfer_date > m.t) & (m.transfer_date <= m.t + pd.Timedelta(days=365))]
    sale = (m.sort_values("transfer_date").drop_duplicates("transfer_id")
             .rename(columns={"transfer_date": "sale_date_12m", "fee_eur": "sale_fee_12m"})
             [["transfer_id", "sale_date_12m", "sale_fee_12m"]])
    return fwd.merge(sale, on="transfer_id", how="left")


def index_at(t: pd.Series, pl: pd.Series, top: pd.Series) -> pd.Series:
    """Segmented real-time index level at window starts (see module docstring)."""
    win = pd.read_parquet(path("processed") / "market_index_windows.parquet").set_index("window_idx")
    w = window_idx(pd.to_datetime(t))
    lvl = win.level.reindex(w).values
    dpl = np.where(pl.values == 1, win.dev_pl.reindex(w).values, win.dev_nonpl.reindex(w).values)
    dtop = np.where(top.values == 1, win.dev_top.reindex(w).values, win.dev_rest.reindex(w).values)
    return pd.Series(lvl + dpl + dtop, index=t.index)


def realtime_transfer_stats(t: pd.Series, log_tm_lag: pd.Series) -> pd.DataFrame:
    """Crowd ratio and top-end flag at arbitrary dates from transfers strictly before t (365 days)."""
    f = pd.read_parquet(path("processed") / "features.parquet",
                        columns=["transfer_date", "fee_eur", "log_tm_lag", "to_league"])
    f["transfer_date"] = pd.to_datetime(f.transfer_date)
    f = f.sort_values("transfer_date")
    core = f[f.to_league.isin(core_leagues())]
    r = f.dropna(subset=["log_tm_lag"])
    rd, rv = r.transfer_date.values, (np.log(r.fee_eur) - r.log_tm_lag).values
    cd, cv = core.transfer_date.values, core.log_tm_lag.values
    out = pd.DataFrame(index=t.index, columns=["crowd_ratio", "top_thr"], dtype=float)
    for d in pd.to_datetime(t).unique():
        lo, hi = np.searchsorted(rd, d - np.timedelta64(365, "D")), np.searchsorted(rd, d)
        cr = float(np.median(rv[lo:hi])) if hi - lo >= 30 else np.nan
        lo2, hi2 = np.searchsorted(cd, d - np.timedelta64(365, "D")), np.searchsorted(cd, d)
        x = cv[lo2:hi2]
        thr = float(np.nanquantile(x, 0.9)) if np.isfinite(x).sum() >= 30 else np.nan
        rows = (pd.to_datetime(t) == d).values
        out.loc[rows, "crowd_ratio"], out.loc[rows, "top_thr"] = cr, thr
    out["seg_top"] = (log_tm_lag.values > out.top_thr.values).astype(float)
    return out


def add_market_columns(p: pd.DataFrame) -> pd.DataFrame:
    rt = realtime_transfer_stats(p.t, p.log_tm_lag)
    p = p.assign(crowd_ratio=rt.crowd_ratio.values, seg_top=rt.seg_top.values)
    p["log_index"] = index_at(p.t, p.pl_buyer, p.seg_top).values
    p["log_tm_adj"] = p.log_tm_lag + p.crowd_ratio
    p["log_tm_now"] = np.log(p.tm_now)
    return p


def refresh_hype(p: pd.DataFrame) -> pd.DataFrame:
    """Recompute group E (after page views are ingested) incl. abnormal hype fitted on train transfers."""
    cfg = load_config()
    con = sources.connect()
    con.register("tg", p[["transfer_id", "player_id", "t"]])
    con.execute("CREATE OR REPLACE TEMP TABLE targets AS SELECT * FROM tg")
    hy = hype_features(con)
    p = p.drop(columns=[c for c in [*HYPE_COLS, "abnormal_hype"] if c in p]).merge(hy, on="transfer_id",
                                                                                   how="left")
    f = pd.read_parquet(path("processed") / "features.parquet")
    td = pd.to_datetime(f.transfer_date)
    train = (td >= cfg["splits"]["train_start"]) & (td < cfg["splits"]["train_end"])
    both = pd.concat([f.assign(_snap=False), p.assign(_snap=True)], ignore_index=True)
    mask = pd.concat([train, pd.Series(False, index=p.index)], ignore_index=True)
    both, _ = add_abnormal_hype(both, mask)
    p = p.reset_index(drop=True)
    p["abnormal_hype"] = both.loc[both._snap, "abnormal_hype"].values
    return p


def build(dates: list[pd.Timestamp] | None = None, min_value: float | None = None,
          out_name: str = "snapshot_panel.parquet") -> pd.DataFrame:
    cfg = load_config()
    dates = dates or snapshot_dates()
    min_value = min_value or cfg.get("forecast", {}).get("min_value_eur", 1e6)
    con = sources.connect()
    con.execute(LEAGUE_GAMES)
    s = snapshot_targets(con, dates, min_value)
    log.info("snapshot targets: %d rows over %d dates", len(s), s.t.nunique())
    curves = pd.read_csv(path("processed") / "aging_curves.csv")
    feats = []
    for d, g in s.groupby("t"):            # one date at a time keeps the SQL joins small
        con = sources.connect()
        feats.append(build_features(con, g, curves))
        log.info("features %s: %d", d.date(), len(g))
    fe = pd.concat(feats, ignore_index=True)
    con = sources.connect()
    p = s.merge(fe.drop(columns=["t"]), on="transfer_id", how="left")
    p = p.merge(forward_targets(con, s), on="transfer_id", how="left")
    p["y_fwd12"] = np.log(p.tm_fwd12 / p.tm_now)
    p = add_market_columns(p)
    p = refresh_hype(p)
    p.to_parquet(path("processed") / out_name, index=False)
    log.info("wrote %s: %s; with 12m target %.1f%%", out_name, p.shape, 100 * p.y_fwd12.notna().mean())
    return p


if __name__ == "__main__":
    setup_logging()
    build()
