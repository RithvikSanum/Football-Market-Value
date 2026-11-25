"""Football transfer market index (brief section 15.1).

price = real value x market index;   target = log(fee) - log(index at transfer date)

Methods (all real-time: the level used at date t only uses information before t)
-------------------------------------------------------------------------------
Estimation sample: buyers in the 14 leagues covered since 2012 ("core"). The dataset records far more
small deals in later years (165 kept transfers in 2012 vs 2,121 in 2025), so an all-transfers index is
dominated by changing coverage; levels estimated on the core are applied to every transfer.
An all-sample variant is kept for comparison.

hedonic      Expanding OLS of log fee on player features + window fixed effects (2 windows / year).
             The level for window w is the FE of window w-1 from a fit on windows < w (carried
             forward). The first `min_history_windows` windows are a burn-in that uses one in-sample
             fit on those windows only (all inside the training period; asserted).
hedonic_now  Nowcast = carried level + beta * (TM-level movement since the window started) +
             shrunk mean residual of this window's deals dated strictly before t
             (weight n / (n + nowcast.shrinkage_pseudo_count)). beta: slope of past window-to-window
             level changes on TM-level changes (windows < w only).
hedonic_seg  hedonic_now + segment deviations: buyer (PL vs other) and tier (top-end vs rest).
             Deviation = mean residual of the segment in the most recent completed window, shrunk
             toward 0 (= the overall index) by n / (n + shrinkage_pseudo_count). Top-end is defined by
             the hedonic *predicted* price (top 10% vs a threshold from earlier windows), not by the fee:
             segmenting on the outcome would bias the segment index and cannot be applied to a player
             who has not been sold.
repeat_sales Players sold 2+ times: d log fee = window effects (+1 later sale, -1 earlier sale) +
             controls (change in (age-26)^2, change in minutes share, change in data coverage).
             Random-walk smoothing between consecutive window effects (ridge on differences) so sparse
             windows borrow from neighbours. Expanding fit on pairs completed before window w.
tm_level     Median Transfermarkt value of top-5-league players at the last month-end before t.

No CPI or general inflation is used anywhere.
"""
from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd

from fv.config import load_config, path, setup_logging
from fv.features.windows import idx_label, window_idx, window_start

log = logging.getLogger(__name__)

TOP5 = ("GB1", "ES1", "IT1", "L1", "FR1")


# --------------------------------------------------------------------------- design matrix
def hedonic_X(f: pd.DataFrame) -> pd.DataFrame:
    """Player features available for every transfer (no Transfermarkt values)."""
    X = pd.DataFrame(index=f.index)
    a = f.age.clip(15, 38).fillna(25)          # 2 rows lack a date of birth
    X["age"] = a - 25
    X["age2"] = (a - 25) ** 2
    for p in ("Defender", "Midfield", "Attack"):
        X[f"pos_{p}"] = (f.position == p).astype(float)
    lg = lambda s: np.select([s.isin(TOP5), s.str.endswith("_lower"), s == "OTHER"],
                             ["top5", "lower", "other"], "covered")
    for side in ("from", "to"):
        g = pd.Series(lg(f[f"{side}_league"]), index=f.index)
        for k in ("top5", "lower", "other"):
            X[f"{side}_{k}"] = (g == k).astype(float)
        e = f[f"{side}_elo"]
        X[f"{side}_elo"] = ((e - 1500) / 200).fillna(0)
        X[f"{side}_elo_na"] = e.isna().astype(float)
    X["pl_buyer"] = f.pl_buyer.astype(float)
    # Only fixed-length (365-day) performance measures: appearance data start in 2012-07, so
    # cumulative career minutes (and 730-day windows early on) grow mechanically with calendar time
    # and would absorb the market trend.
    X["has_apps"] = (f.mins_365 > 0).astype(float)
    X["log_mins_365"] = np.log1p(f.mins_365.fillna(0))
    X["share_365"] = f.min_share_365.fillna(0)
    X["share_na"] = f.min_share_365.isna().astype(float)
    X["ga_p90"] = f.ga_p90_365.fillna(0).clip(0, 1.5)
    return X


def _ols(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray, float]:
    xtx = X.T @ X + ridge * np.eye(X.shape[1])
    inv = np.linalg.inv(xtx)
    b = inv @ X.T @ y
    r = y - X @ b
    s2 = float(r @ r) / max(len(y) - X.shape[1], 1)
    return b, inv * s2, s2


REF_WINDOW = 2   # 2013-S: first window with a full 365-day appearance look-back (data start 2012-07)


def _fit_hedonic(X: pd.DataFrame, y: pd.Series, w: pd.Series, n_windows: int):
    """OLS with window FE relative to REF_WINDOW. Returns beta (incl. intercept), FE and FE se
    (length n_windows; NaN for windows with no rows)."""
    present = sorted(set(w.unique()) - {REF_WINDOW})
    D = np.zeros((len(w), len(present)))
    for c, j in enumerate(present):
        D[:, c] = (w.values == j)
    Z = np.column_stack([np.ones(len(w)), X.values, D])
    b, cov, _ = _ols(Z, y.values)
    k = 1 + X.shape[1]
    fe = np.full(n_windows, np.nan)
    se = np.full(n_windows, np.nan)
    fe[REF_WINDOW], se[REF_WINDOW] = 0.0, 0.0
    for c, j in enumerate(present):
        fe[j], se[j] = b[k + c], np.sqrt(cov[k + c, k + c])
    return b[:k], fe, se


def _is_winter(j: int) -> int:
    return int(j % 2 == 1)          # idx 0 = 2012-S, 1 = 2013-W, ...


def _winter_effect(fe: np.ndarray) -> float:
    """Mean gap between each winter window FE and the average of its neighbouring summer FEs."""
    gaps = [fe[i] - (fe[i - 1] + fe[i + 1]) / 2 for i in range(REF_WINDOW + 1, len(fe) - 1)
            if _is_winter(i)]
    gaps = [g for g in gaps if np.isfinite(g)]
    return float(np.mean(gaps)) if gaps else 0.0


# --------------------------------------------------------------------------- TM level
def tm_level_monthly(con, leagues=TOP5, min_minutes: int = 900) -> pd.DataFrame:
    """Median latest TM value of established top-5-league players at each month-end.

    Population = players with >= `min_minutes` domestic-league minutes in those leagues in the
    12 months before the month-end (fixed-quality composition; the raw 'everyone with a valuation'
    median drifts with dataset coverage). Value = latest valuation on/before the month-end.
    """
    lg = "(" + ",".join(f"'{x}'" for x in leagues) + ")"
    return con.execute(f"""
      WITH m AS (SELECT unnest(generate_series(DATE '2013-06-30', DATE '2026-07-31', INTERVAL 1 MONTH)) AS d0),
      months AS (SELECT (date_trunc('month', d0) + INTERVAL 1 MONTH - INTERVAL 1 DAY)::DATE AS m FROM m),
      pop AS (
        SELECT months.m, a.player_id
        FROM months JOIN appearances a ON a.date <= months.m AND a.date > months.m - INTERVAL 365 DAY
        WHERE a.competition_id IN {lg}
        GROUP BY 1, 2 HAVING sum(a.minutes_played) >= {min_minutes}),
      val AS (
        SELECT pop.m, pop.player_id, arg_max(v.market_value_in_eur, v.date) AS mv
        FROM pop JOIN player_valuations v ON v.player_id = pop.player_id AND v.date <= pop.m
          AND v.date > pop.m - INTERVAL 365 DAY
        GROUP BY 1, 2)
      SELECT m AS month_end, median(mv) AS tm_median, count(*) AS n_players
      FROM val WHERE mv > 0 GROUP BY 1 ORDER BY 1""").df()


def tm_level_on(dates: pd.Series, tm: pd.DataFrame) -> pd.Series:
    """log TM level at the last month-end strictly before each date (published in real time)."""
    s = tm.assign(month_end=pd.to_datetime(tm.month_end)).sort_values("month_end")
    s = s.assign(avail=s.month_end + pd.Timedelta(days=1))
    d = pd.DataFrame({"date": pd.to_datetime(dates).values, "_i": np.arange(len(dates))}).sort_values("date")
    m = pd.merge_asof(d, s[["avail", "tm_median"]], left_on="date", right_on="avail", direction="backward",
                      allow_exact_matches=True)
    return pd.Series(np.log(m.sort_values("_i").tm_median.values), index=dates.index)


# --------------------------------------------------------------------------- hedonic rolling
def rolling_hedonic(f: pd.DataFrame, tm: pd.DataFrame, cfg: dict,
                    est: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns per-transfer log levels and a per-window table.

    est: boolean mask of the estimation sample (fits, segment deviations and nowcast residuals use
    only these rows); levels are applied to every row.
    """
    est = pd.Series(True, index=f.index) if est is None else est
    mi = cfg["market_index"]
    k_seg = mi["shrinkage_pseudo_count"]
    k_now = mi["nowcast"]["shrinkage_pseudo_count"]
    burn = mi["min_history_windows"]
    q_top = mi["segments"]["tier"]["top_end_quantile"]
    train_end_idx = int(window_idx(pd.Series([pd.Timestamp(cfg["splits"]["train_end"])])).iloc[0])
    assert burn <= train_end_idx, "burn-in windows must lie inside the training period"

    X = hedonic_X(f)
    y = np.log(f.fee_eur)
    w = f.window_idx
    tml = tm_level_on(f.transfer_date, tm)
    n_w = int(w.max()) + 1

    out = pd.DataFrame(index=f.index, columns=["lvl_hedonic", "lvl_hedonic_now", "lvl_hedonic_seg",
                                               "se_hedonic", "seg_top", "pred_hedonic"], dtype=float)
    win_rows = []
    level_hist: dict[int, float] = {}        # carried level for each window (real-time)
    tm_start = {j: float(tm_level_on(pd.Series([window_start(j)]), tm).iloc[0]) for j in range(n_w + 1)}

    # burn-in: one in-sample fit on the first `burn` windows
    est = est & (w >= REF_WINDOW)
    m0 = (w < burn) & est
    b0, fe0, se0 = _fit_hedonic(X[m0], y[m0], w[m0], burn)
    fe0[:REF_WINDOW] = 0.0          # windows before the reference take the reference level
    se0[:REF_WINDOW] = np.nan
    for j in range(burn):
        level_hist[j] = fe0[j]

    for j in range(n_w):
        rows = w == j
        if j < burn:
            beta, fe, se, lvl, lvl_se = b0, fe0, se0, fe0[j], se0[j]
            season = _winter_effect(fe0)
            hist = m0
        else:
            hist = (w < j) & est
            beta, fe, se = _fit_hedonic(X[hist], y[hist], w[hist], j)
            season = _winter_effect(fe)
            # carry forward the last level, adjusted for the winter/summer seasonal gap
            lvl = fe[j - 1] + season * (_is_winter(j) - _is_winter(j - 1))
            lvl_se = se[j - 1]
            level_hist[j] = lvl
        pred_hist = beta[0] + X[hist].values @ beta[1:]
        thr = np.quantile(pred_hist, q_top)
        resid_hist = y[hist].values - pred_hist - np.nan_to_num(fe[w[hist].values])

        # TM beta from past consecutive windows (needs >= 4 changes)
        past = [i for i in range(1, j) if i in level_hist and i - 1 in level_hist]
        if mi["nowcast"]["use_tm_level_movement"] and len(past) >= 4 and j >= burn:
            dl = np.array([fe[i] - fe[i - 1] for i in range(1, j)])
            dt = np.array([tm_start[i + 1] - tm_start[i] for i in range(1, j)])
            ok = np.isfinite(dt)
            beta_tm = float(np.polyfit(dt[ok], dl[ok], 1)[0]) if ok.sum() >= 4 else 0.0
        else:
            beta_tm = 0.0

        # segment deviations from the most recent completed window in history
        last = j - 1 if j >= burn else j
        lastm = (w[hist] == last).values
        Xh = X[hist]
        dev = {}
        for name, mask in {"pl": Xh.pl_buyer.values == 1, "nonpl": Xh.pl_buyer.values == 0,
                           "top": pred_hist > thr, "rest": pred_hist <= thr}.items():
            r = resid_hist[lastm & mask]
            dev[name] = (len(r) / (len(r) + k_seg)) * (r.mean() if len(r) else 0.0)

        # rows of window j
        idx = f.index[rows]
        pred_j = beta[0] + X.loc[idx].values @ beta[1:]
        tm_move = (tml.loc[idx].values - tm_start[j]) if j >= burn else np.zeros(len(idx))
        tm_move = np.nan_to_num(tm_move)
        base_now = lvl + beta_tm * tm_move
        if j >= burn:
            # shrunk mean residual of this window's deals strictly before each date
            d = f.loc[idx, "transfer_date"].values
            r_now = y.loc[idx].values - pred_j - base_now
            core = est.loc[idx].values
            order = np.argsort(d[core], kind="stable")
            ds, rs = d[core][order], r_now[core][order]
            csum = np.concatenate([[0.0], np.cumsum(rs)])
            n_before = np.searchsorted(ds, d, side="left")     # core deals strictly before each date
            adj = csum[n_before] / (n_before + k_now)
        else:
            adj = np.zeros(len(idx))
        now = base_now + adj
        top = pred_j > thr
        pl = X.loc[idx, "pl_buyer"].values == 1
        seg = now + np.where(pl, dev["pl"], dev["nonpl"]) + np.where(top, dev["top"], dev["rest"])
        out.loc[idx, "lvl_hedonic"] = lvl
        out.loc[idx, "lvl_hedonic_now"] = now
        out.loc[idx, "lvl_hedonic_seg"] = seg
        out.loc[idx, "se_hedonic"] = lvl_se
        out.loc[idx, "seg_top"] = top.astype(float)
        out.loc[idx, "pred_hedonic"] = pred_j

        n_last = {k: int((lastm & m).sum()) for k, m in {
            "pl": Xh.pl_buyer.values == 1, "nonpl": Xh.pl_buyer.values == 0,
            "top": pred_hist > thr, "rest": pred_hist <= thr}.items()}
        win_rows.append({"window_idx": j, "window": idx_label(j), "burn_in": j < burn,
                         "level": lvl, "se": lvl_se, "winter_effect": season,
                         "level_deseason": lvl - season * _is_winter(j), "beta_tm": beta_tm, "n_deals": int(rows.sum()),
                         "n_hist": int(hist.sum()), **{f"dev_{k}": v for k, v in dev.items()},
                         **{f"n_{k}": v for k, v in n_last.items()},
                         "in_sample_fe": fe[j] if j < len(fe) else np.nan})
    return out, pd.DataFrame(win_rows)


def final_in_sample(f: pd.DataFrame, est: pd.Series | None = None) -> pd.DataFrame:
    """Full-sample hedonic FE (uses all data; for charts / 'what we know now' only, never a feature)."""
    n_w = int(f.window_idx.max()) + 1
    f = f if est is None else f[est]
    f = f[f.window_idx >= REF_WINDOW]
    X, y, w = hedonic_X(f), np.log(f.fee_eur), f.window_idx
    _, fe, se = _fit_hedonic(X, y, w, n_w)
    return pd.DataFrame({"window_idx": range(n_w), "fe_full": fe, "se_full": se})


# --------------------------------------------------------------------------- repeat sales
def repeat_sales(f: pd.DataFrame, cfg: dict, smooth: float = 4.0) -> tuple[pd.Series, pd.DataFrame]:
    rs = cfg["market_index"]["repeat_sales"]
    g = f.sort_values(["player_id", "transfer_date"])
    nxt = g.groupby("player_id").shift(-1)
    pairs = pd.DataFrame({
        "w1": g.window_idx, "w2": nxt.window_idx,
        "dy": np.log(nxt.fee_eur) - np.log(g.fee_eur),
        "d_age2": (nxt.age.clip(15, 38) - 26) ** 2 - (g.age.clip(15, 38) - 26) ** 2,
        "d_share": nxt.min_share_365.fillna(0) - g.min_share_365.fillna(0),
        "d_has": (nxt.mins_730 > 0).astype(float) - (g.mins_730 > 0).astype(float),
        "gap": (nxt.transfer_date - g.transfer_date).dt.days / 365.25,
    }).dropna(subset=["w2", "dy", "d_age2"])
    pairs = pairs[(pairs.gap <= rs["max_gap_years"]) & (pairs.w2 > pairs.w1)]
    pairs["w1"], pairs["w2"] = pairs.w1.astype(int), pairs.w2.astype(int)
    n_w = int(f.window_idx.max()) + 1
    burn = cfg["market_index"]["min_history_windows"]
    levels = {}
    rows = []
    for j in range(burn, n_w):
        p = pairs[pairs.w2 < j]
        if len(p) < 30:
            continue
        D = np.zeros((len(p), j - 1))
        for c, s in (("w2", 1.0), ("w1", -1.0)):
            ww = p[c].values
            m = ww >= 1
            D[np.where(m)[0], ww[m] - 1] += s
        C = p[["d_age2", "d_share", "d_has"]].values
        Z = np.column_stack([D, C])
        # random-walk smoothing: penalise differences of consecutive window effects
        P = np.zeros((j - 1, Z.shape[1]))
        for i in range(j - 1):
            P[i, i] = 1.0
            if i > 0:
                P[i, i - 1] = -1.0
        A = Z.T @ Z + smooth * P.T @ P + 1e-6 * np.eye(Z.shape[1])
        b = np.linalg.solve(A, Z.T @ p.dy.values)
        fe = np.concatenate([[0.0], b[: j - 1]])
        season = _winter_effect(fe)
        levels[j] = fe[j - 1] + season * (_is_winter(j) - _is_winter(j - 1))
        rows.append({"window_idx": j, "level_rs": levels[j], "n_pairs": len(p),
                     "level_rs_deseason": levels[j] - season * _is_winter(j)})
    lv = f.window_idx.map(levels)
    return lv, pd.DataFrame(rows)


# --------------------------------------------------------------------------- crowd ratio
def crowd_ratio(f: pd.DataFrame, days: int = 365, min_n: int = 30) -> pd.Series:
    """Real-time log median of fee / TM value (TM value >= lag days before) over deals in the
    `days` before each transfer (strictly earlier dates). Transfermarkt values ran ~20-40% below fees
    in 2012-20 and level with fees from 2021, so a fixed train-period scale factor drifts."""
    d = f[["transfer_date", "fee_eur", "log_tm_lag"]].copy()
    d["r"] = np.log(d.fee_eur) - d.log_tm_lag
    ok = d[d.r.notna()].sort_values("transfer_date")
    dates, vals = ok.transfer_date.values, ok.r.values
    out = pd.Series(np.nan, index=f.index)
    t = f.transfer_date.values
    lo = np.searchsorted(dates, t - np.timedelta64(days, "D"), side="left")
    hi = np.searchsorted(dates, t, side="left")            # strictly before t
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= min_n:
            out.iloc[i] = float(np.median(vals[a:b]))
    return out


# --------------------------------------------------------------------------- build
def build() -> None:
    cfg = load_config()
    f = pd.read_parquet(path("processed") / "features.parquet")
    f["transfer_date"] = pd.to_datetime(f.transfer_date)
    con = duckdb.connect()
    for t in ("player_valuations", "appearances"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet("
                    f"'{path('interim') / 'transfermarkt' / (t + '.parquet')}')")
    tm = tm_level_monthly(con, tuple(cfg["market_index"]["tm_level"]["leagues"]))
    tm.to_parquet(path("processed") / "tm_level_monthly.parquet", index=False)

    # core estimation sample: buyers in leagues covered since 2012 (stable composition)
    from fv.features.base import league_coverage
    cov = league_coverage()
    long_cov = {k for k, v in cov.items() if v <= pd.Timestamp("2012-12-31")}
    core = f.to_league.isin(long_cov)
    hed, win = rolling_hedonic(f, tm, cfg, est=core)
    hed_all, win_all = rolling_hedonic(f, tm, cfg)
    hed_all = hed_all[["lvl_hedonic", "lvl_hedonic_now"]].add_suffix("_allsample")
    win = win.merge(win_all[["window_idx", "level"]].rename(columns={"level": "level_allsample"}),
                    on="window_idx")
    hed = pd.concat([hed, hed_all], axis=1)
    rs_lvl, rs_win = repeat_sales(f, cfg)
    full = final_in_sample(f, core)
    win = win.merge(rs_win, on="window_idx", how="left").merge(full, on="window_idx", how="left")
    tm_start = tm_level_on(pd.Series([window_start(j) for j in win.window_idx]), tm)
    win["tm_level_start"] = tm_start.values
    win.to_parquet(path("processed") / "market_index_windows.parquet", index=False)

    idx = pd.DataFrame({"transfer_id": f.transfer_id})
    idx = pd.concat([idx, hed], axis=1)
    idx["lvl_repeat_sales"] = rs_lvl.values
    idx["lvl_tm"] = tm_level_on(f.transfer_date, tm).values
    idx["crowd_ratio"] = crowd_ratio(f).values
    idx.to_parquet(path("processed") / "market_index_transfers.parquet", index=False)
    log.info("market index: %d windows, %d transfers", len(win), len(idx))


def main() -> None:
    setup_logging()
    build()


if __name__ == "__main__":
    main()
