"""Evaluators (brief section 7): several views of a player's price, plus a consensus and its spread.

1. model value  - LightGBM TM-corrected median (from the pricing model; passed in as `log_model`)
2. comparables  - median inflation-adjusted fee of the K = 10 nearest *earlier* transfers in the same
                  position group. Distance on z-scored age, minutes share, goals+assists per 90,
                  minutes (365d), league-adjusted career minutes (log), selling-club Elo, plus a penalty
                  of COMP_LEAGUE_PENALTY when the selling league differs. Comp fees are converted to real
                  value with the comp's index level and back with the target's.
3. crowd value  - Transfermarkt value (>= 30 days before) x real-time fee/TM ratio (log_tm_adj). The
                  brief's fixed train-period factor drifts (fees fell from ~1.25x to ~1.0x TM after 2020).
4. book value   - ESTIMATE: fee paid when the player joined his current club, written off straight-line
                  over 5 years (the seller's accounting floor). Not part of the consensus.
Consensus = weighted geometric mean of 1-3. Weights = a simple stacking fit on validation: the convex
combination (grid step 0.05) with the lowest validation MAE in logs. (Inverse-MAE weights were tried
first and gave a consensus worse than the model alone, because comparables are much weaker.)
Weights are renormalised over the evaluators available for each player; spread = max - min of the available
log values (a disagreement indicator).
"""
from __future__ import annotations

import json
import logging

import duckdb
import numpy as np
import pandas as pd

from fv.config import path

log = logging.getLogger(__name__)
K = 10
COMP_LEAGUE_PENALTY = 0.5
COMP_X = ["age", "min_share_365", "ga_p90_365", "mins_365", "log_adj_mins", "from_elo"]
BOOK_YEARS = 5
EVALS = {"model": "log_model", "comparables": "log_comps", "crowd": "log_crowd"}


def _comp_matrix(df: pd.DataFrame, mu: pd.Series | None = None, sd: pd.Series | None = None):
    X = df.assign(log_adj_mins=np.log1p(df.adj_mins.clip(lower=0)))[COMP_X].astype(float)
    if mu is None:
        mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).fillna(0.0)     # missing -> population mean
    return Z.values, mu, sd


def comparables(targets: pd.DataFrame, pool: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """targets / pool need: position, from_league, COMP_X inputs, t (targets) / transfer_date, fee_eur,
    log_index (both), transfer_id (pool). Returns log_comps, comp_ids (comma list) per target row."""
    pool = pool.sort_values("transfer_date").reset_index(drop=True)
    Zp, mu, sd = _comp_matrix(pool)
    Zt, _, _ = _comp_matrix(targets, mu, sd)
    real = np.log(pool.fee_eur.values) - pool.log_index.values
    pdates = pool.transfer_date.values
    ppos, plg, pid = pool.position.values, pool.from_league.values, pool.transfer_id.values
    out_v, out_ids = np.full(len(targets), np.nan), [""] * len(targets)
    tt, tpos, tlg = targets.t.values, targets.position.values, targets.from_league.values
    tli = targets.log_index.values
    for i in range(len(targets)):
        n = np.searchsorted(pdates, tt[i], side="left")       # strictly earlier transfers
        if n < k:
            continue
        cand = np.flatnonzero(ppos[:n] == tpos[i])
        if len(cand) < k:
            continue
        d = np.sqrt(((Zp[cand] - Zt[i]) ** 2).sum(axis=1)) + COMP_LEAGUE_PENALTY * (plg[cand] != tlg[i])
        nn = cand[np.argpartition(d, k - 1)[:k]]
        out_v[i] = np.median(real[nn]) + tli[i]
        out_ids[i] = ",".join(map(str, pid[nn]))
    return pd.DataFrame({"log_comps": out_v, "comp_ids": out_ids}, index=targets.index)


def book_value(targets: pd.DataFrame, club_col: str = "from_club_id") -> pd.Series:
    """Estimated book value (EUR) at t: last recorded fee into the current club, written down over 5y.
    Free / zero-fee arrivals give 0; unknown fee or no record gives NaN."""
    con = duckdb.connect()
    tm = path("interim") / "transfermarkt"
    con.register("bt", targets[["player_id", "t", club_col]].rename(columns={club_col: "club_id"})
                 .assign(_row=np.arange(len(targets))))
    d = con.execute(f"""
      SELECT bt._row, arg_max(x.transfer_fee, x.transfer_date) AS fee, max(x.transfer_date) AS joined
      FROM bt JOIN read_parquet('{tm / 'transfers.parquet'}') x
        ON x.player_id = bt.player_id AND x.to_club_id = bt.club_id AND x.transfer_date < bt.t
      GROUP BY 1""").df()
    out = pd.Series(np.nan, index=targets.index)
    if d.empty:
        return out
    t = pd.to_datetime(targets.t.values[d._row.values])
    yrs = (t.values - pd.to_datetime(d.joined).values) / np.timedelta64(1, "D") / 365.25
    v = d.fee.astype(float).values * np.clip(1 - yrs / BOOK_YEARS, 0, 1)
    out.iloc[d._row.values] = v
    return out


def fit_weights(val: pd.DataFrame, step: float = 0.05) -> dict:
    """val: log_fee and the EVALS columns. Stacking: convex weights minimising validation MAE (log) on
    the common subset; inverse-MAE weights reported for reference."""
    cols = list(EVALS.values())
    common = val.dropna(subset=cols)
    y = common.log_fee.values
    L = common[cols].values
    mae = {k: float(np.mean(np.abs(common[c] - common.log_fee))) for k, c in EVALS.items()}
    inv = {k: (1 / v) / sum(1 / x for x in mae.values()) for k, v in mae.items()}
    best, best_w = np.inf, None
    grid = np.arange(0, 1 + 1e-9, step)
    for a in grid:
        for b in grid:
            if a + b > 1 + 1e-9:
                continue
            wv = np.array([a, b, 1 - a - b])
            e = float(np.mean(np.abs(L @ wv - y)))
            if e < best:
                best, best_w = e, wv
    return {"weights": dict(zip(EVALS, map(float, best_w))), "inverse_mae_weights": inv,
            "val_mae_log": mae, "stacked_val_mae_log": best, "n_common": len(common)}


def consensus(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    L = np.column_stack([df[c].values for c in EVALS.values()])
    w = np.array([weights[k] for k in EVALS])
    ok = np.isfinite(L)
    ww = np.where(ok, w, 0.0)
    cons = np.where(ww.sum(1) > 0, np.nansum(np.where(ok, L, 0) * ww, 1) / np.maximum(ww.sum(1), 1e-12),
                    np.nan)
    spread = np.where(ok.sum(1) >= 2, np.nanmax(np.where(ok, L, np.nan), 1) - np.nanmin(np.where(ok, L, np.nan), 1),
                      np.nan)
    return pd.DataFrame({"log_consensus": cons, "evaluator_spread": spread,
                         "n_evaluators": ok.sum(1)}, index=df.index)


def transfer_pool() -> pd.DataFrame:
    f = pd.read_parquet(path("processed") / "features.parquet")
    idx = pd.read_parquet(path("processed") / "market_index_transfers.parquet")
    meta = json.loads((path("processed") / "pricing_meta.json").read_text())
    from fv.models.pricing import DEFLATORS
    f = f.merge(idx, on="transfer_id", how="left")
    f["transfer_date"] = pd.to_datetime(f.transfer_date)
    f["log_index"] = f[DEFLATORS[meta["deflator"]]].astype(float)
    f["log_index"] = f.log_index.fillna(f.log_index.dropna().iloc[0] if f.log_index.notna().any() else 0)
    return f
