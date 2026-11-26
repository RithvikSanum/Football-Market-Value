"""Phase 3 final evaluation: the ONE scoring of the untouched test split (brief section 11).

Refuses to run twice (marker data/processed/final_test_done.json) unless --force; a forced re-run is
recorded in the report. Steps:
1. Refit the pricing models exactly as fv.models.pricing (train; early stopping on validation; CQR on
   calibration) and check they reproduce the saved validation predictions.
2. Test-set pricing metrics vs the Transfermarkt baselines, breakdowns, interval coverage, errors by
   window incl. test windows (drift).
3. Evaluators + consensus (weights from validation) on test.
4. 12-month forecast metrics on test snapshots vs baselines.
5. Undervalued-signal backtest on test snapshots.
6. Contract adjustment, forward-chained by season inside the test period (uses test labels AFTER
   step 2 is frozen; reported separately).
7. Market-forecast backtest summary (one-year-ahead vs no change).
Writes reports/final_report.md and figures.

--dry-run: exercises every code path WITHOUT touching test data: test rows are dropped, calibration
rows stand in for "test" (and calibration snapshots for test snapshots), no marker is written and the
report goes to the scratch file reports/_dry_run_final_report.md (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from fv.config import load_config, path, setup_logging
from fv.evaluation.metrics import breakdowns, coverage, scores, table
from fv.evaluation.report import GRID, INK2, MODELS, S1, S2, S3, S4, _md, _pct, _save, plt
from fv.features.build import STATS_ONLY, TM_CORRECTED, features_for
from fv.models import evaluators as ev
from fv.models.pricing import cqr_qhat, deflator, fit_lgb, fit_ridge, lgb_frame, load_dataset

log = logging.getLogger(__name__)
MARKER = "final_test_done.json"


# --------------------------------------------------------------------------- 1. refit pricing
def refit_pricing(df: pd.DataFrame, meta: dict, cfg: dict, extra: pd.DataFrame | None = None):
    """Returns predictions for df (and `extra` rows, e.g. snapshots) from models fit exactly as in
    pricing.main. Columns: <name> (EUR), <name>_real, <name>_q10/_q90 (real, CQR)."""
    alpha = 1 - cfg["intervals"]["coverage"]
    tr, va = df[df.split == "train"], df[df.split == "val"]
    out = pd.DataFrame(index=df.index)
    out_x = pd.DataFrame(index=extra.index) if extra is not None else None
    c_tm = float(np.nanmedian(tr.log_fee - tr.log_tm_lag))
    out["tm_baseline"] = np.exp(df.log_tm_lag + c_tm)
    out["tm_realtime_ratio"] = np.exp(df.log_tm_adj)
    for name, groups in (("ridge_stats", STATS_ONLY), ("ridge_tm", TM_CORRECTED)):
        feats = features_for(groups)
        r = fit_ridge(tr, tr.y, feats)
        out[name] = np.exp(r.predict(df[feats]) + df.log_index)
    for name, groups in (("lgb_stats", STATS_ONLY), ("lgb_tm", TM_CORRECTED)):
        feats = meta[name]["features"]
        best = meta[name]["params"]
        params = {k: v for k, v in best.items() if k not in ("objective", "cv_mdape")} | {"n_estimators": 3000}
        Xtr, cats = lgb_frame(tr, feats)
        Xva, _ = lgb_frame(va, feats, cats)
        Xall, _ = lgb_frame(df, feats, cats)
        m = fit_lgb(Xtr, tr.y, Xva, va.y, params, best["objective"], seed=cfg["seed"])
        out[f"{name}_real"] = m.predict(Xall)
        out[name] = np.exp(out[f"{name}_real"] + df.log_index)
        q = {}
        Xx = lgb_frame(extra, feats, cats)[0] if extra is not None else None
        qx = {}
        for qq in cfg["intervals"]["quantiles"]:
            mq = fit_lgb(Xtr, tr.y, Xva, va.y, params, "quantile", alpha=qq, seed=cfg["seed"])
            q[qq] = mq.predict(Xall)
            if Xx is not None:
                qx[qq] = mq.predict(Xx)
        lo, hi = np.minimum(q[0.1], q[0.9]), np.maximum(q[0.1], q[0.9])
        cal = (df.split == "calib").values
        qhat = cqr_qhat(df.y.values[cal], lo[cal], hi[cal], alpha)
        out[f"{name}_q10"], out[f"{name}_q90"] = lo - qhat, hi + qhat
        if Xx is not None:
            out_x[f"{name}_real"] = m.predict(Xx)
            lo2, hi2 = np.minimum(qx[0.1], qx[0.9]), np.maximum(qx[0.1], qx[0.9])
            out_x[f"{name}_q10"], out_x[f"{name}_q90"] = lo2 - qhat, hi2 + qhat
    return out, out_x


# --------------------------------------------------------------------------- figures
def fig_pred_actual(d: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, (lab, col), c in zip(axes, (("TM baseline (≥30d)", "tm_baseline"),
                                        ("LightGBM TM-corrected", "lgb_tm")), (S2, S1)):
        x, y = d.fee_eur / 1e6, d[col] / 1e6
        ax.scatter(x, y, s=6, alpha=0.35, color=c, edgecolors="none")
        lim = [0.2, 300]
        ax.plot(lim, lim, color=INK2, lw=1)
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_title(lab, loc="left"), ax.set_xlabel("Actual fee (€m)")
    axes[0].set_ylabel("Predicted (€m)")
    fig.suptitle("Test set (2022/23 onward): predicted vs actual fee", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "final_pred_vs_actual.png")


def fig_errors_by_window(err: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for (m, g), c in zip(err.groupby("model", sort=False), (S1, S2, S3)):
        ax.plot(g.window_idx, 100 * g.median_err, color=c, marker="o", ms=4, label=m)
    ax.axhline(0, color=INK2, lw=1)
    test0 = err[err.split == "test"].window_idx.min()
    ax.axvspan(test0 - 0.5, err.window_idx.max() + 0.5, color=GRID, alpha=0.5, lw=0)
    ax.text(test0, ax.get_ylim()[1] * 0.9, " test", color=INK2, fontsize=9)
    lab = dict(zip(err.window_idx, err.window))
    ticks = sorted(lab)[::2]
    ax.set_xticks(ticks, [lab[k] for k in ticks], rotation=45, fontsize=8)
    ax.set_ylabel("Median log error (pred − fee), %")
    ax.set_title("Pricing errors by transfer window (out-of-sample)", loc="left")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return _save(fig, "final_errors_by_window.png")


def fig_market_fan() -> str:
    from fv.models.market_forecast import series
    s = series()
    fc = pd.read_csv(path("processed") / "market_forecast.csv")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(s))
    ax.plot(x, np.exp(s.L - s.L.iloc[-1]), color=S1, label="Real-time index (deseasonalised)")
    xf = x[-1] + 2 * fc.horizon_years.values
    xs = np.r_[x[-1], xf]
    ax.fill_between(xs, np.exp(np.r_[0, fc.p10]), np.exp(np.r_[0, fc.bull]), color=S1, alpha=0.15,
                    lw=0, label="Monte Carlo 10–90%")
    ax.plot(xs, np.exp(np.r_[0, fc.base]), color=S1, ls="--", label="Base")
    ax.plot(xs, np.exp(np.r_[0, fc.bear]), color=S2, ls="--", label="Bear (worst historical fall)")
    lab = list(s.window) + [f"+{h}y" for h in fc.horizon_years]
    ax.set_xticks(list(x[::2]) + list(xf), lab[::2][: len(x[::2])] + [f"+{h}y" for h in fc.horizon_years],
                  rotation=45, fontsize=8)
    ax.set_ylabel("Index (latest window = 1)")
    ax.set_title("Transfer price index: history and 1–3 year forecast", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, "market_forecast_fan.png")


# --------------------------------------------------------------------------- main
def main(force: bool = False, dry: bool = False) -> None:
    setup_logging()
    cfg = load_config()
    marker = path("processed") / MARKER
    if marker.exists() and not force and not dry:
        raise SystemExit(f"final test already run ({marker.read_text()}); use --force to re-run")
    meta = json.loads((path("processed") / "pricing_meta.json").read_text())
    df = load_dataset()
    df["log_index"] = deflator(df, meta["deflator"])
    df["y"] = df.log_fee - df.log_index
    if dry:
        df = df[df.split != "test"].copy()

    # snapshot rows (for the signal backtest), priced with the same models
    from fv.features.snapshots import index_at  # noqa: F401  (index already in the panel)
    snap = pd.read_parquet(path("processed") / "snapshot_panel.parquet")
    snap["t"] = pd.to_datetime(snap.t)
    if dry:
        snap = snap[snap.t < pd.Timestamp(cfg["forecast"]["test_first"])].copy()
    preds, sp = refit_pricing(df, meta, cfg, extra=snap)

    # 1. reproduction check on validation
    old = pd.read_parquet(path("processed") / "predictions_phase1.parquet").set_index("transfer_id")
    v = df.split == "val"
    a = np.log(preds.loc[v, "lgb_tm"].values)
    b = np.log(old.loc[df.loc[v, "transfer_id"], "lgb_tm"].values)
    repro = float(np.max(np.abs(a - b)))
    log.info("reproduction check (val, max |log diff|): %.2e", repro)

    res = pd.concat([df, preds], axis=1)
    if dry:   # calibration rows play the part of the test set
        res["split"] = res.split.replace({"calib": "test"})
    te = res[res.split == "test"]
    common = te[te.tm_baseline.notna()]
    names = MODELS | {"TM transfers-table value (leaky, reference)": "tm_transfers_table"}
    res["tm_transfers_table"] = res.tm_value_transfers_table * meta["tm_scale_transfers_table"]
    te, common = res[res.split == "test"], res[(res.split == "test") & res.tm_baseline.notna()]
    t_common = _pct(table(common, names), ["MdAPE", "within_25", "within_50"])
    t_all = _pct(table(te, {k: v for k, v in MODELS.items() if not k.startswith("TM")}),
                 ["MdAPE", "within_25", "within_50"])
    bd = breakdowns(common, MODELS)
    cov_rows = []
    for name, lab in (("lgb_stats", "Stats-only"), ("lgb_tm", "TM-corrected")):
        c = coverage(te.y.values, te[f"{name}_q10"].values, te[f"{name}_q90"].values)
        cov_rows.append({"model": lab, "coverage": 100 * c["coverage"],
                         "below_q10": 100 * np.mean(te.y < te[f"{name}_q10"]),
                         "above_q90": 100 * np.mean(te.y > te[f"{name}_q90"]),
                         "median_width_ratio": c["median_width_ratio"]})
    cov_t = pd.DataFrame(cov_rows)

    # errors by window (out-of-sample: phase-1 OOF for train windows, held-out val/calib, test)
    err_rows = []
    for col, lab in (("lgb_tm", "LightGBM TM-corrected"), ("lgb_stats", "LightGBM stats-only"),
                     ("tm_realtime_ratio", "TM × real-time ratio")):
        e = res.assign(err=np.log(res[col]) - res.log_fee)
        if col.startswith("lgb"):
            oof = old[f"{col}_oof_real"].reindex(res.transfer_id).values
            tr_err = oof + res.log_index.values - res.log_fee.values
            e["err"] = np.where(res.split == "train", tr_err, e.err)
        g = e.dropna(subset=["err"]).groupby("window_idx").agg(
            median_err=("err", "median"), n=("err", "size"), split=("split", "last"))
        g = g[g.n >= 30].reset_index().assign(model=lab)
        err_rows.append(g)
    from fv.features.windows import idx_label
    err = pd.concat(err_rows).assign(window=lambda d: d.window_idx.map(idx_label))
    drift = []
    for lab, g in err.groupby("model", sort=False):
        x = g.window_idx.values / 2.0
        X = np.c_[np.ones(len(x)), x]
        beta, res_ss, *_ = np.linalg.lstsq(X, g.median_err.values, rcond=None)
        sig2 = float(np.sum((g.median_err.values - X @ beta) ** 2) / max(len(x) - 2, 1))
        se = np.sqrt(sig2 * np.linalg.inv(X.T @ X)[1, 1])
        test_bias = float(g[g.split == "test"].median_err.mean())
        drift.append({"model": lab, "slope_per_year": beta[1], "t": beta[1] / se, "windows": len(g),
                      "flag": abs(beta[1] / se) > 2, "mean_test_window_bias": test_bias})
    drift = pd.DataFrame(drift)

    # 3. evaluators
    pool = ev.transfer_pool()
    held = res[res.split.isin(["val", "calib", "test"])].copy()
    held["t"] = held.transfer_date
    comps = ev.comparables(held, pool)
    held = held.join(comps)
    held["log_model"] = np.log(held.lgb_tm)
    held["log_crowd"] = held.log_tm_adj
    held["book_value"] = ev.book_value(held)
    w = ev.fit_weights(held[held.split == "val"])
    held = held.join(ev.consensus(held, w["weights"]))
    held["consensus"] = np.exp(held.log_consensus)
    held["comps"] = np.exp(held.log_comps)
    held["crowd"] = np.exp(held.log_crowd)
    ht = held[held.split == "test"].dropna(subset=["log_model", "log_comps", "log_crowd"])
    ev_t = _pct(table(ht, {"Model (LightGBM TM-corrected)": "lgb_tm", "Comparables (K=10)": "comps",
                           "Crowd (TM × real-time ratio)": "crowd", "Consensus": "consensus"}),
                ["MdAPE", "within_25", "within_50"])
    ht = ht.assign(spread_band=pd.qcut(ht.evaluator_spread, 4, labels=["low", "mid-low", "mid-high", "high"]))
    spread_t = ht.groupby("spread_band", observed=True).apply(
        lambda g: pd.Series({"n": len(g), "consensus_MdAPE": scores(g.fee_eur, g.consensus)["MdAPE"],
                             "median_spread_log": g.evaluator_spread.median()})).reset_index()
    spread_t["consensus_MdAPE"] = (100 * spread_t.consensus_MdAPE).map("{:.1f}%".format)
    book_cov = float(held.book_value.notna().mean())
    book_ratio = float(np.nanmedian(held.fee_eur / held.book_value.replace(0, np.nan)))

    # 4. forecast metrics on test
    from fv.models.forecast import metrics as fmetrics
    fp = pd.read_parquet(path("processed") / "forecast_predictions.parquet")
    ft = fp[(fp.split == ("calib" if dry else "test")) & fp.y_real.notna()]
    f_rows = [{"model": lab, **fmetrics(ft.y_real.values, ft[c].values)}
              for lab, c in (("no change", "pred_no_change"), ("age curve", "pred_age_curve"),
                             ("LightGBM", "pred_lgb"))]
    f_rows.append({"model": "80% interval coverage", "n": len(ft),
                   "MAE": float(((ft.y_real >= ft.pred_q10) & (ft.y_real <= ft.pred_q90)).mean())})
    f_t = pd.DataFrame(f_rows)

    # 5. undervalued signal on test snapshots
    s = snap.join(sp)
    s = s.merge(fp[["transfer_id", "y_real", "pred_age_curve"]], on="transfer_id", how="left")
    s["model_value"] = np.exp(s.lgb_tm_real + s.log_index)
    s["ratio"] = np.log(s.model_value / s.tm_now)
    t0 = pd.Timestamp(cfg["forecast"]["calib_dates"][0] if dry else cfg["forecast"]["test_first"])
    stest = s[(s.t >= t0) & s.ratio.notna()].copy()
    q = cfg["forecast"]["signal_top_quantile"]
    stest["flag"] = stest.groupby("t").ratio.transform(lambda r: r >= r.quantile(q))
    sold = stest.sale_fee_12m.notna()
    sig = stest.groupby("flag").apply(lambda g: pd.Series({
        "players": len(g), "with_12m_TM_value": int(g.y_fwd12.notna().sum()),
        "median_12m_TM_change": float(np.nanmedian(np.exp(g.y_fwd12) - 1)),
        "median_12m_change_vs_age_curve": float(np.nanmedian(g.y_real - g.pred_age_curve)),
        "share_sold_12m": float(g.sale_fee_12m.notna().mean()),
        "median_sale_fee_over_TM": float(np.nanmedian(g.sale_fee_12m / g.tm_now)),
        "median_age": float(g.age.median())})).reset_index()
    sig["flag"] = sig.flag.map({True: f"flagged (top {100 * (1 - q):.0f}% model/TM)", False: "others"})
    by_date = stest.groupby(["t", "flag"]).y_fwd12.median().unstack()
    hit = float(np.mean(by_date[True] > by_date[False])) if True in by_date else np.nan
    # bootstrap CI for the difference in median 12m change (flagged - others)
    rng = np.random.default_rng(cfg["seed"])
    fl, ot = stest[stest.flag].y_fwd12.dropna().values, stest[~stest.flag].y_fwd12.dropna().values
    diffs = [np.median(rng.choice(fl, len(fl))) - np.median(rng.choice(ot, len(ot))) for _ in range(1000)]
    sig_ci = np.quantile(diffs, [0.05, 0.95])
    # age-adjusted: the player part of the change minus the age-curve expectation
    fa = stest[stest.flag].eval("y_real - pred_age_curve").dropna().values
    oa = stest[~stest.flag].eval("y_real - pred_age_curve").dropna().values
    adj_diffs = [np.median(rng.choice(fa, len(fa))) - np.median(rng.choice(oa, len(oa))) for _ in range(1000)]
    adj_d, adj_ci = float(np.median(fa) - np.median(oa)), np.quantile(adj_diffs, [0.05, 0.95])

    # 6. contract adjustment (forward-chained inside test; after the frozen test scoring above)
    from fv.features.contract import contract_at_transfer
    c = te.merge(contract_at_transfer(te), on="transfer_id", how="left")
    c = c.dropna(subset=["contract_years_left"]).copy()
    c["resid"] = c.log_fee - np.log(c.lgb_tm)
    c["season"] = c.transfer_season
    c_rows = []
    seasons = sorted(c.season.unique())
    for sn in seasons[1:]:
        fit, sc = c[c.season < sn], c[c.season == sn]
        if len(fit) < 200 or len(sc) < 50:
            continue
        X = lambda d: np.c_[np.ones(len(d)), np.minimum(d.contract_years_left, 4),
                            (d.contract_years_left <= 1).astype(float), d.age - 25]
        beta, *_ = np.linalg.lstsq(X(fit), fit.resid.values, rcond=None)
        adj = np.exp(np.log(sc.lgb_tm) + X(sc) @ beta)
        c_rows.append({"season": sn, "fit_on": f"{seasons[0]}–{seasons[seasons.index(sn) - 1]}",
                       "n": len(sc), "MdAPE_main": scores(sc.fee_eur, sc.lgb_tm)["MdAPE"],
                       "MdAPE_with_contract": scores(sc.fee_eur, adj)["MdAPE"],
                       "beta_years_left": beta[1], "beta_le1y": beta[2]})
    contract_t = pd.DataFrame(c_rows, columns=["season", "fit_on", "n", "MdAPE_main", "MdAPE_with_contract",
                                               "beta_years_left", "beta_le1y"])

    # 7. market backtest
    mb = pd.read_csv(path("processed") / "market_backtest_summary.csv")
    mf = pd.read_csv(path("processed") / "market_forecast.csv")

    figs = {"pva": fig_pred_actual(common), "err": fig_errors_by_window(err), "fan": fig_market_fan()}
    run = {"run_at": datetime.now().isoformat(timespec="seconds"), "forced": bool(force and marker.exists()),
           "reproduction_max_abs_log_diff": repro}
    md = f"""# Final report: test set (checkpoint 3)

**This is the single scoring of the untouched test split** (transfers from {cfg['splits']['calib_end']};
snapshots from {cfg['forecast']['test_first']}). Run at {run['run_at']}{' (FORCED RE-RUN)' if run['forced'] else ''}.
Models were refit exactly as in phase 1/2 (train → early stopping on validation → CQR on calibration);
reproduction check on validation: max |log difference| = {repro:.1e}.

## 1. Pricing: test set

Common subset (deals with a TM value ≥ 30 days before):

{_md(t_common)}

All test deals (statistical models):

{_md(t_all)}

![Predicted vs actual]({figs['pva']})

### Breakdowns (MdAPE, common subset)

""" + "\n\n".join(f"**By {k.replace('_', ' ')}**\n\n{_md(_pct(v, [c for c in v.columns if c not in (k, 'n')]))}"
                  for k, v in bd.items()) + f"""

### Price ranges (80%, CQR)

{_md(cov_t, {'coverage': '{:.1f}', 'below_q10': '{:.1f}', 'above_q90': '{:.1f}', 'median_width_ratio': '{:.2f}'})}

### Drift (errors by window, including test windows)

![Errors by window]({figs['err']})

{_md(drift, {'slope_per_year': '{:+.3f}', 't': '{:+.2f}', 'mean_test_window_bias': '{:+.3f}'})}

## 2. Evaluators and consensus (test, deals with all three evaluators)

Consensus weights (stacking: convex blend with the lowest validation MAE in logs):
{', '.join(f"{k} {v:.2f}" for k, v in w['weights'].items())} (validation n = {w['n_common']}; inverse-MAE
weights would have been {', '.join(f"{k} {v:.2f}" for k, v in w['inverse_mae_weights'].items())}).

{_md(ev_t)}

Does disagreement between evaluators signal uncertainty? Consensus error by evaluator-spread quartile:

{_md(spread_t, {'median_spread_log': '{:.2f}'})}

Book value (estimate; last fee into the selling club written off over 5 years) is available for
{book_cov:.0%} of held-out deals; median fee / book value = {book_ratio:.2f}. It is a floor indicator only.

## 3. 12-month value forecast (test snapshots, player part = TM change minus TM market move)

{_md(f_t, {'MAE': '{:.3f}', 'RMSE': '{:.3f}', 'R2_vs_no_change': '{:+.3f}', 'spearman': '{:.3f}', 'direction_hit_|y|>0.1': '{:.1%}'})}

(For the interval row, `MAE` holds the empirical coverage.)

## 4. Undervalued-signal backtest (test snapshots)

Flag = model median ÷ current TM value in the top decile on each snapshot date.

{_md(sig, {'median_12m_TM_change': '{:+.1%}', 'median_12m_change_vs_age_curve': '{:+.3f}', 'share_sold_12m': '{:.1%}', 'median_sale_fee_over_TM': '{:.2f}', 'median_age': '{:.1f}'})}

Difference in median 12-month TM change (flagged − others): {np.median(fl) - np.median(ot):+.3f} log
points (bootstrap 90% CI {sig_ci[0]:+.3f} to {sig_ci[1]:+.3f}); flagged group ahead on {hit:.0%} of
test snapshot dates.

**Age-adjusted** (player part of the change minus the age-curve expectation, so the flag is not simply
"young players"): flagged − others = {adj_d:+.3f} log points (90% CI {adj_ci[0]:+.3f} to {adj_ci[1]:+.3f}).

## 5. Contract adjustment (forward-chained inside the test period)

Fitted on the main model's log residual (years left capped at 4, ≤ 1 year flag, age), on earlier test
seasons only, scored on the next. **Uses test-period labels after the frozen test scoring above.**

{_md(_pct(contract_t, ['MdAPE_main', 'MdAPE_with_contract']), {'beta_years_left': '{:+.3f}', 'beta_le1y': '{:+.3f}'})}

## 6. Market forecast

One-year-ahead backtest of the real-time index (log points; `MAE_common` = same origins for all):

{_md(mb, {'MAE': '{:.3f}', 'RMSE': '{:.3f}', 'bias': '{:+.3f}', 'MAE_common': '{:.3f}'})}

![Market fan]({figs['fan']})

{_md(mf[['horizon_years', 'origin', 'p10', 'base', 'bull', 'bear', 'bear_from', 'bear_to', 'drift_mode']], {'p10': '{:+.3f}', 'base': '{:+.3f}', 'bull': '{:+.3f}', 'bear': '{:+.3f}'})}
"""
    if dry:
        (path("reports") / "_dry_run_final_report.md").write_text(md)
        log.info("dry run OK (no test data used, nothing saved)")
        return
    (path("reports") / "final_report.md").write_text(md)
    res[res.split == "test"][["transfer_id", "fee_eur", *preds.columns]].to_parquet(
        path("processed") / "predictions_test.parquet", index=False)
    held.to_parquet(path("processed") / "evaluators_heldout.parquet", index=False)
    s.to_parquet(path("processed") / "snapshot_priced.parquet", index=False)
    marker.write_text(json.dumps(run))
    log.info("wrote final report")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.force, a.dry_run)
