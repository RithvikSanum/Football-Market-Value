"""Combined 12-month price forecast (brief section 15.4): player real-value forecast x market scenario.

Player part: split-normal through the CQR-calibrated quantiles of the forecast model (q10, q50, q90;
sigma below = (q50 - q10) / 1.2816, above = (q90 - q50) / 1.2816). Market part: the Monte Carlo paths
of the index forecast (market_paths.npz, 1-year horizon). Sampled independently and added in logs.
ASSUMPTION (documented): player value and market level move independently.
Breakdown: player contribution = exp(median player draw) - 1, market = exp(median market draw) - 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fv.config import load_config, path

Z90 = 1.2815515655446004


def player_draws(q10: float, q50: float, q90: float, n: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = max(q50 - q10, 1e-6) / Z90, max(q90 - q50, 1e-6) / Z90
    z = rng.standard_normal(n)
    return q50 + np.where(z < 0, z * lo, z * hi)


def combined_forecast(df: pd.DataFrame, horizon_years: int = 1, n: int | None = None) -> pd.DataFrame:
    """df: log_value_now (log fair value today), pred_q10, pred_lgb, pred_q90 (player log change).
    Returns bear/base/bull price (EUR) and the player / market breakdown per row."""
    cfg = load_config()
    n = n or cfg["market_forecast"]["monte_carlo"]["n_paths"]
    rng = np.random.default_rng(cfg["seed"])
    paths = np.load(path("processed") / "market_paths.npz")[f"h{horizon_years}"]
    fc = pd.read_csv(path("processed") / "market_forecast.csv").set_index("horizon_years")
    bear_mkt = float(fc.loc[horizon_years, "bear"])
    m = rng.choice(paths, size=n, replace=True)
    rows = []
    for r in df.itertuples():
        p = player_draws(r.pred_q10, r.pred_lgb, r.pred_q90, n, rng)
        tot = p + m
        rows.append({"price_bear": float(np.exp(r.log_value_now + np.quantile(p, 0.1) + bear_mkt)),
                     "price_p10": float(np.exp(r.log_value_now + np.quantile(tot, 0.1))),
                     "price_base": float(np.exp(r.log_value_now + np.median(tot))),
                     "price_bull": float(np.exp(r.log_value_now + np.quantile(tot, 0.9))),
                     "player_contribution": float(np.exp(np.median(p)) - 1),
                     "market_contribution": float(np.exp(np.median(m)) - 1)})
    return pd.DataFrame(rows, index=df.index)
