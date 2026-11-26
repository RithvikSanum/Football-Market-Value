"""Market (football price index) forecast, brief section 15.3.

Series: the real-time, seasonally adjusted log index per transfer window (`level_deseason` in
market_index_windows.parquet; two windows per year; each level uses only deals before that window).

Models, forecasting h years (= 2h windows) ahead from origin window j:
- no change:       L_j
- random walk with drift: L_j + 2h * mu_j, mu_j = mean window step up to j (after burn-in)
- revenue-linked:  L_j + h * (a + b * g_j), where g_j = latest annual 'big five' revenue growth
  *available* at the window start (data/manual/league_revenues.csv, `available_from` <= start), and
  (a, b) are fitted on past (annual index change, revenue growth then available) pairs only. Needs >= 5
  pairs; otherwise not produced.
Backtest: every origin with enough history, 1 year ahead, vs no change.
Monte Carlo (latest origin): window steps ~ Normal(mu, sigma) from history, n paths, where the centre mu
is the historical drift if "random walk + drift" has the lower 1-year backtest MAE than "no change", else
0 (config market_forecast.monte_carlo.drift: auto | historical | zero); base / bull =
config quantiles; bear = the largest historical h-year fall of the index (scenario, not a quantile).
Outputs: data/processed/market_backtest.csv, market_forecast.csv, market_paths.npz (1-3y log changes).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fv.config import ROOT, load_config, path, setup_logging
from fv.features.windows import window_start

log = logging.getLogger(__name__)


def series() -> pd.DataFrame:
    cfg = load_config()
    w = pd.read_parquet(path("processed") / "market_index_windows.parquet")
    burn = cfg["market_index"]["min_history_windows"]
    w = w[w.window_idx >= burn].sort_values("window_idx").reset_index(drop=True)
    w["start"] = [window_start(int(j)) for j in w.window_idx]
    return w[["window_idx", "window", "start", "level_deseason", "se"]].rename(columns={"level_deseason": "L"})


def revenue_growth_available(dates: pd.Series, league: str = "big_five") -> pd.Series:
    r = pd.read_csv(ROOT / load_config()["market_forecast"]["revenues_csv"])
    r = r[r.league == league].sort_values("season").reset_index(drop=True)
    r["g"] = np.log(r.revenue).diff()
    r["available_from"] = pd.to_datetime(r.available_from)
    out = []
    for d in pd.to_datetime(dates):
        a = r[(r.available_from <= d) & r.g.notna()]
        out.append(a.g.iloc[-1] if len(a) else np.nan)
    return pd.Series(out, index=dates.index)


def forecast_from(s: pd.DataFrame, j: int, h: int) -> dict:
    """Forecasts of L at position j + 2h made with data up to position j (positional index)."""
    L = s.L.values
    steps = np.diff(L[: j + 1])
    out = {"no change": L[j]}
    if len(steps) >= 4:
        out["random walk + drift"] = L[j] + 2 * h * steps.mean()
    # revenue-linked: past pairs (annual change ending at i, revenue growth available at i - 2)
    g = s.g_avail.values
    pairs = [(L[i] - L[i - 2], g[i - 2]) for i in range(2, j + 1) if np.isfinite(g[i - 2])]
    if len(pairs) >= 5 and np.isfinite(g[j]):
        y, x = map(np.array, zip(*pairs))
        b, a = np.polyfit(x, y, 1)
        out["revenue-linked"] = L[j] + h * (a + b * g[j])
        out["_rev_beta"] = b
    return out


def backtest(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for j in range(len(s) - 2):
        fc = forecast_from(s, j, 1)
        actual = s.L.iloc[j + 2]
        for m, v in fc.items():
            if not m.startswith("_"):
                rows.append({"origin": s.window.iloc[j], "target": s.window.iloc[j + 2], "model": m,
                             "forecast": v, "actual": actual, "error": v - actual})
    return pd.DataFrame(rows)


def monte_carlo(s: pd.DataFrame, cfg: dict, summ: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mc = cfg["market_forecast"]["monte_carlo"]
    sc = cfg["market_forecast"]["scenarios"]
    L = s.L.values
    steps = np.diff(L)
    mu_hist, sd = steps.mean(), steps.std(ddof=1)
    mode = mc.get("drift", "auto")
    if mode == "auto":
        mode = ("historical" if summ.loc["random walk + drift", "MAE_common"] < summ.loc["no change", "MAE_common"]
                else "zero")
    mu = mu_hist if mode == "historical" else 0.0
    rng = np.random.default_rng(mc["seed"])
    rows, paths = [], {}
    for h in cfg["market_forecast"]["horizons_years"]:
        sim = rng.normal(mu, sd, size=(mc["n_paths"], 2 * h)).sum(axis=1)   # log change over h years
        paths[f"h{h}"] = sim
        hist = L[2 * h:] - L[:-2 * h]
        k = int(np.argmin(hist))
        rows.append({"horizon_years": h, "origin": s.window.iloc[-1],
                     "base": float(np.quantile(sim, sc["base"]["quantile"])),
                     "bull": float(np.quantile(sim, sc["bull"]["quantile"])),
                     "p10": float(np.quantile(sim, 0.1)),
                     "bear": float(hist[k]), "bear_from": s.window.iloc[k],
                     "bear_to": s.window.iloc[k + 2 * h], "drift_mode": mode, "mu_per_window": mu,
                     "mu_hist_per_window": mu_hist, "sd_per_window": sd})
    return pd.DataFrame(rows), paths


def main() -> None:
    setup_logging()
    cfg = load_config()
    s = series()
    s["g_avail"] = revenue_growth_available(s.start).values
    bt = backtest(s)
    bt.to_csv(path("processed") / "market_backtest.csv", index=False)
    summ = bt.groupby("model").agg(n=("error", "size"), MAE=("error", lambda e: e.abs().mean()),
                                   RMSE=("error", lambda e: np.sqrt((e ** 2).mean())),
                                   bias=("error", "mean"))
    # same origins for all models (fair comparison)
    common = bt.pivot_table(index="origin", columns="model", values="error").dropna()
    summ["MAE_common"] = common.abs().mean()
    summ["n_common"] = len(common)
    log.info("market backtest (1y ahead, log points):\n%s", summ.round(4).to_string())
    fc, paths = monte_carlo(s, cfg, summ)
    last = forecast_from(s, len(s) - 1, 1)
    fc["revenue_linked_1y"] = last.get("revenue-linked", np.nan) - s.L.iloc[-1]
    fc["revenue_beta"] = last.get("_rev_beta", np.nan)
    fc.to_csv(path("processed") / "market_forecast.csv", index=False)
    summ.reset_index().to_csv(path("processed") / "market_backtest_summary.csv", index=False)
    np.savez(path("processed") / "market_paths.npz", **paths)
    log.info("market forecast from %s:\n%s", s.window.iloc[-1], fc.round(4).to_string())


if __name__ == "__main__":
    main()
