"""Phase 1 model report: reports/model_report.md + reports/figures/*.png (validation only)."""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fv.config import load_config, path, setup_logging  # noqa: E402
from fv.evaluation.metrics import breakdowns, coverage, table  # noqa: E402
from fv.features.build import GROUPS  # noqa: E402
from fv.features.windows import idx_label  # noqa: E402

log = logging.getLogger(__name__)

# Reference palette (dataviz skill, light mode): categorical slots in fixed order + text/surface roles.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODELS = {"TM baseline (≥30d)": "tm_baseline", "TM × real-time ratio": "tm_realtime_ratio",
          "Ridge, stats only": "ridge_stats",
          "Ridge, with TM": "ridge_tm", "LightGBM stats-only": "lgb_stats",
          "LightGBM TM-corrected": "lgb_tm"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, name: str) -> str:
    f = path("figures") / name
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    d = df.copy()
    for c, f in (fmt or {}).items():
        if c in d:
            d[c] = d[c].map(lambda v: f.format(v) if pd.notna(v) else "")
    return d.to_markdown(index=False)


def _pct(df: pd.DataFrame, cols) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        d[c] = (100 * d[c]).map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
    return d


# --------------------------------------------------------------------------- index chart
def index_frame(cfg) -> pd.DataFrame:
    w = pd.read_parquet(path("processed") / "market_index_windows.parquet")
    base = w.iloc[-1]                                     # base date 2026-06-30 lies in the last window
    w["date"] = pd.to_datetime([idx_label(i).replace("-S", "-07-01").replace("-W", "-01-15")
                                for i in w.window_idx])
    z = 1.645
    w["overall"] = np.exp(w.level_deseason - base.level_deseason)
    w["lo"] = np.exp(w.level_deseason - z * w.se - base.level_deseason)
    w["hi"] = np.exp(w.level_deseason + z * w.se - base.level_deseason)
    for s in ("pl", "nonpl", "top", "rest"):
        # anchored to the overall index (not to each segment's own, noisy, last window)
        w[s] = np.exp(w.level_deseason + w[f"dev_{s}"] - base.level_deseason)
    w["repeat"] = np.exp(w.level_rs_deseason - w.level_rs_deseason.iloc[-1])
    w["tm"] = np.exp(w.tm_level_start - w.tm_level_start.iloc[-1])
    return w[w.window_idx >= 2]


def fig_index(w: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    ax = axes[0]
    ax.fill_between(w.date, w.lo, w.hi, color=S1, alpha=0.15, lw=0, label="90% band (hedonic)")
    ax.plot(w.date, w.overall, color=S1, label="Hedonic (real-time, main)")
    ax.plot(w.date, w.repeat, color=S2, label="Repeat sales")
    ax.plot(w.date, w.tm, color=S3, label="TM top-5 median value")
    ax.set_title("Market index: methods")
    ax.legend(loc="upper left", fontsize=8)
    for ax, (a, b, la, lb, t) in zip(axes[1:], [("pl", "nonpl", "PL buyers", "Other buyers", "By buyer"),
                                                ("top", "rest", "Top-end (top 10% predicted)", "Rest",
                                                 "By tier")]):
        ax.fill_between(w.date, w.lo, w.hi, color=S1, alpha=0.10, lw=0)
        ax.plot(w.date, w[a], color=S1, label=la)
        ax.plot(w.date, w[b], color=S2, label=lb)
        ax.set_title(f"Segment indices: {t.lower()} (partial pooling)")
        ax.text(0.99, 0.02, "band: overall index 90%", transform=ax.transAxes, ha="right", fontsize=8,
                color=INK2)
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes:
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("")
    axes[0].set_ylabel("Index (base = 1.0 at 30 Jun 2026)")
    fig.suptitle("Football transfer market index, real-time levels (de-seasonalised)", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "market_index.png")


def fig_deflators(d: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = d[d.split == "val"].deflator.tolist()
    ypos = np.arange(len(names))
    for k, (split, col, off) in enumerate((("val", S1, -0.15), ("calib", S2, 0.15))):
        s = d[d.split == split].set_index("deflator").loc[names]
        ax.errorbar(100 * s.diff_vs_none, ypos + off,
                    xerr=[100 * (s.diff_vs_none - s.diff_90lo), 100 * (s.diff_90hi - s.diff_vs_none)],
                    fmt="o", color=col, ms=6, capsize=0, lw=1.5,
                    label={"val": "Validation 2020/21", "calib": "Calibration 2021/22"}[split])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlabel("MdAPE difference vs no deflator (percentage points; < 0 = better)")
    ax.set_title("Deflator comparison (LightGBM stats-only, paired bootstrap 90% CI)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    fig.tight_layout()
    return _save(fig, "deflator_comparison.png")


# --------------------------------------------------------------------------- errors by window
def errors_by_window(r: pd.DataFrame, model: str) -> pd.DataFrame:
    """Signed log error (pred - actual). Train windows: expanding OOF; val/calib: held-out preds."""
    e = pd.Series(np.nan, index=r.index)
    tr = r.split == "train"
    e[tr] = r.loc[tr, f"{model}_oof_real"] - r.loc[tr, "y"]
    ho = r.split.isin(["val", "calib"])
    e[ho] = np.log(r.loc[ho, model]) - r.loc[ho, "log_fee"]
    return r.assign(err=e).dropna(subset=["err"])


MIN_WINDOW_N = 30


def drift_test(e: pd.DataFrame) -> dict:
    g = e.groupby("window_idx").err
    g = g.median()[g.size() >= MIN_WINDOW_N]
    x, y = g.index.values.astype(float), g.values
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return {"slope_per_year": 2 * b, "t": b / se, "windows": len(g), "flag": bool(abs(b / se) > 2)}


def fig_errors(r: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    tests = []
    for model, col, lab in (("lgb_tm", S1, "LightGBM TM-corrected"), ("lgb_stats", S2, "LightGBM stats-only")):
        e = errors_by_window(r, model)
        g = e.groupby("window_idx").err
        ok = g.size() >= MIN_WINDOW_N
        med, q1, q3 = g.median()[ok], g.quantile(.25)[ok], g.quantile(.75)[ok]
        x = [pd.Timestamp(idx_label(i).replace("-S", "-07-01").replace("-W", "-01-15")) for i in med.index]
        axes[0].fill_between(x, q1, q3, color=col, alpha=0.12, lw=0)
        axes[0].plot(x, med, color=col, marker="o", ms=4, label=f"{lab} (median, IQR band)")
        tests.append({"model": lab, "segment": "all", **drift_test(e)})
        for ax, (seg, a, b, la, lb) in zip(axes[1:], [("pl_buyer", 1, 0, "PL buyers", "Other buyers"),
                                                       ("seg_top", 1, 0, "Top-end", "Rest")]):
            if model != "lgb_tm":
                continue
            for val, c2, l2 in ((a, S1, la), (b, S2, lb)):
                es = e[e[seg] == val]
                gs = es.groupby("window_idx").err
                m = gs.median()[gs.size() >= MIN_WINDOW_N]
                xs = [pd.Timestamp(idx_label(i).replace("-S", "-07-01").replace("-W", "-01-15")) for i in m.index]
                ax.plot(xs, m, color=c2, marker="o", ms=4, label=l2)
                tests.append({"model": lab, "segment": l2, **drift_test(es)})
            ax.set_title(f"TM-corrected: by {'buyer' if seg == 'pl_buyer' else 'tier'}")
            ax.legend(fontsize=8)
    axes[0].set_title("Median signed log error by window")
    axes[0].legend(fontsize=8, loc="lower left")
    va0 = pd.Timestamp(load_config()["splits"]["train_end"])
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(va0, color=INK2, lw=0.8, ls="--")
        ax.text(va0, ax.get_ylim()[1], " held-out →", color=INK2, fontsize=8, va="top")
    axes[0].set_ylabel("log(pred / actual); > 0 = over-predicts")
    fig.tight_layout()
    return _save(fig, "errors_by_window.png"), pd.DataFrame(tests)


# --------------------------------------------------------------------------- other figures
def fig_pred_actual(va: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(va.fee_eur / 1e6, va.lgb_tm / 1e6, s=10, color=S1, alpha=0.45, lw=0)
    lim = [0.2, 300]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("Actual fee (€m, log)"), ax.set_ylabel("Predicted median (€m, log)")
    ax.set_title("LightGBM TM-corrected, validation 2020/21", loc="left")
    fig.tight_layout()
    return _save(fig, "pred_vs_actual.png")


def fig_calibration(va: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for m, lab in (("lgb_stats", "Stats-only"), ("lgb_tm", "TM-corrected")):
        for kind, lo, hi in (("raw quantiles", f"{m}_q10_raw", f"{m}_q90_raw"), ("CQR", f"{m}_q10", f"{m}_q90")):
            c = coverage(va.y.values, va[lo].values, va[hi].values)
            rows.append({"model": lab, "interval": kind, "below_q10": float(np.mean(va.y < va[lo])),
                         "above_q90": float(np.mean(va.y > va[hi])), **c})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lab = [f"{a}\n{b}" for a, b in zip(d.model, d.interval)]
    ax.axvspan(75, 85, color=S3, alpha=0.12, lw=0)
    ax.text(80, -0.62, "target 75–85%", ha="center", va="center", color=INK2, fontsize=8)
    ax.scatter(100 * d.coverage, range(len(d)), color=S1, s=60, zorder=3)
    for i, v in enumerate(d.coverage):
        ax.text(100 * v + 0.6, i, f"{100 * v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(d)), lab), ax.set_ylim(len(d) - 0.5, -0.9), ax.set_xlim(60, 95)
    ax.set_xlabel("Empirical coverage of 80% interval, validation (%)")
    ax.set_title("Interval calibration", loc="left")
    fig.tight_layout()
    return _save(fig, "calibration.png"), d


def shap_tables(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_parquet(path("processed") / f"shap_{name}.parquet")
    feats = [c for c in s.columns if c not in ("transfer_id", "base_value")]
    imp = s[feats].abs().mean().sort_values(ascending=False)
    grp = {f: g for g, fs in GROUPS.items() for f in fs}
    by_g = s[feats].T.groupby(lambda f: grp[f]).sum().T.abs().mean().sort_values(ascending=False)
    imp = imp.rename_axis("feature").rename("mean_abs_shap").reset_index()
    by_g = by_g.rename_axis("group").rename("mean_abs_shap").reset_index()
    return imp, by_g


def fig_shap(imp: pd.DataFrame, by_g: pd.DataFrame, title: str, fname: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    top = imp.head(15).iloc[::-1]
    axes[0].barh(top.feature, top.mean_abs_shap, color=S1, height=0.65)
    axes[0].set_title("Top 15 features (mean |SHAP|, log points)", loc="left")
    g = by_g.iloc[::-1]
    axes[1].barh(g.group, g.mean_abs_shap, color=S1, height=0.55)
    axes[1].set_title("By factor group (|sum of group SHAP|)", loc="left")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, fname)


def fig_aging() -> str:
    c = pd.read_csv(path("processed") / "aging_curves_position.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (pos, g), col in zip(c.groupby("position"), (S1, S2, S3, S4)):
        g = g[(g.age >= 17) & (g.age <= 36)]
        v = np.exp(g.log_value_curve - g.log_value_curve.max())
        peak = int(g.peak_age.iloc[0])
        ax.plot(g.age, v, color=col, label=f"{pos} (peak {peak})")
    ax.set_xlabel("Age"), ax.set_ylabel("Relative value (peak = 1)")
    ax.set_title("Value aging curves by position (delta method, TM values, train period)", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "aging_curves_position.png")


# --------------------------------------------------------------------------- report
def main() -> None:
    setup_logging()
    cfg = load_config()
    pr = path("processed")
    r = pd.read_parquet(pr / "predictions_phase1.parquet")
    assert not (r.split == "test").any(), "test rows must not be present in phase 1"
    meta = json.loads((pr / "pricing_meta.json").read_text())
    funnel = pd.read_csv(pr / "filter_funnel.csv")
    defl = pd.read_csv(pr / "deflator_comparison.csv")
    elo = json.loads((pr / "elo_params.json").read_text())
    va = r[r.split == "val"]
    common = va[va.tm_baseline.notna()]

    f_index = fig_index(index_frame(cfg))
    f_defl = fig_deflators(defl)
    f_err, drift = fig_errors(r)
    f_pa = fig_pred_actual(va)
    f_cal, cal = fig_calibration(va)
    imp_tm, grp_tm = shap_tables("lgb_tm")
    imp_st, grp_st = shap_tables("lgb_stats")
    f_shap_tm = fig_shap(imp_tm, grp_tm, "SHAP: LightGBM TM-corrected (val + calib rows)", "shap_lgb_tm.png")
    f_shap_st = fig_shap(imp_st, grp_st, "SHAP: LightGBM stats-only (val + calib rows)", "shap_lgb_stats.png")
    f_age = fig_aging()

    pct = ["MdAPE", "within_25", "within_50"]
    m_common = _pct(table(common, MODELS | {"TM transfers-table value (leaky, reference)": "tm_transfers_table"}), pct)
    m_all = _pct(table(va, {k: v for k, v in MODELS.items() if not k.startswith("TM")}), pct)
    bd = breakdowns(common, MODELS)
    defl_t = defl.copy()
    for c in ("MdAPE", "diff_vs_none", "diff_90lo", "diff_90hi", "within_25", "within_50"):
        defl_t[c] = (100 * defl_t[c]).round(1)
    cal_t = cal.copy()
    for c in ("coverage", "below_q10", "above_q90"):
        cal_t[c] = (100 * cal_t[c]).round(1)

    parts = [f"""# Model report — Phase 1 (validation only)

*Generated by `make evaluate`. The test split (transfers from {cfg['splits']['calib_end']}) has not been
scored. All numbers below are validation (2020/21) unless stated.*

## Setup

* Universe: permanent transfers with known fee ≥ €{cfg['training']['min_fee_eur']:,}, July 2012 → July 2026.
* Splits: train < {cfg['splits']['train_end']} · validation → {cfg['splits']['val_end']} ·
  calibration → {cfg['splits']['calib_end']} · test after (untouched).
* Target: real value = log(fee) − log(market index). Deflator used: **{meta['deflator']}**
  (see the market-index section for why this overrides the pure validation rule).
* Predictions are medians (exp of predicted log). MdAPE = median |pred / fee − 1|.
* Fees are in EUR as recorded; GBP uses the ECB rate on the transfer date. Currency check: see
  `reports/data_audit.md` section 7 (fees agreed in £ were converted at the rate of the day; no
  restatement at a single recent rate).

### Filter funnel

{_md(funnel)}

## 1. Headline metrics (validation 2020/21)

Common subset where a Transfermarkt value ≥30 days before the deal exists (n = {len(common)}):

{_md(m_common, {'MAE_log': '{:.3f}'})}

All validation rows (n = {len(va)}; the TM baseline cannot price rows without a TM value):

{_md(m_all, {'MAE_log': '{:.3f}'})}

TM baseline scale factor (fit on train, as the brief specifies): ×{meta['tm_scale']:.3f}.
"TM × real-time ratio" instead multiplies the TM value by the median fee/TM ratio of deals in the prior
365 days (known in real time). The ratio was ~1.2–1.4 in 2012–20 and ~1.0 from 2021: Transfermarkt
values caught up with fees after COVID, so a fixed train-period factor over-prices later deals. The
TM-corrected model uses the same real-time-adjusted TM value (`log_tm_adj`).

![Predicted vs actual]({f_pa})

### Breakdowns (MdAPE, common subset)

"""]
    for k, v in bd.items():
        parts.append(f"**By {k.replace('_', ' ')}**\n\n" + _md(_pct(v, list(MODELS))) + "\n")
    parts.append(f"""
## 2. Market index

![Market index]({f_index})

The main index is a rolling hedonic regression (two windows per year) estimated on buyers in the 14
leagues covered since 2012, with a real-time nowcast (carried level + TM-level movement + shrunk
mean residual of the current window's earlier deals). Segment indices are partial-pooling deviations
(pseudo-count {cfg['market_index']['shrinkage_pseudo_count']}). Bands: ±1.645 standard errors of the window effect.

**Method comparison.** Each deflator was used to build the target for LightGBM stats-only (default
params), and the predictions converted back to money:

![Deflator comparison]({f_defl})

{_md(defl_t[['split', 'deflator', 'n', 'MdAPE', 'diff_vs_none', 'diff_90lo', 'diff_90hi']])}

*(MdAPE and differences in percentage points.)* On validation the lowest MdAPE is **no deflator**, but
its lead over the hedonic nowcast is within noise (90% CI includes 0), and on calibration all methods
are tied. Prices in 2020–22 sat close to the training-period average, so an index cannot show its value
there; it matters when the model has to price at levels outside its training range (test period and
current valuations). The TM market level is clearly worst on validation. **Decision flagged for the
user:** keep the hedonic nowcast (current setting) or follow the validation rule (no deflator).

## 3. Price ranges (conformalized quantile regression)

![Calibration]({f_cal})

{_md(cal_t, {'median_width_ratio': '{:.2f}'})}

`median_width_ratio` = median of (upper / lower) bound. CQR offsets were fit on the calibration split
(stats-only q̂ = {meta['lgb_stats']['cqr_qhat']:.3f}, TM-corrected q̂ = {meta['lgb_tm']['cqr_qhat']:.3f} log points)
and are evaluated here on validation, which is held out from both training and calibration.

## 4. Diagnostics: errors by window and segment

![Errors by window]({f_err})

Training windows use expanding-window out-of-fold predictions (fit on earlier windows only); validation
and calibration windows use the held-out predictions. Drift test = OLS slope of window median errors on
time (flag if |t| > 2):

{_md(drift.assign(slope_per_year=drift.slope_per_year.map('{:+.3f}'.format), t=drift.t.map('{:+.2f}'.format)))}

Windows with fewer than {MIN_WINDOW_N} deals are excluded. **Drift found and partly fixed:** a first
run flagged upward drift overall (the model over-priced post-2020 deals by ~10%). Cause: the median
fee / TM-value ratio was ~1.2–1.4 in 2012–20 and ~1.0 from 2021, so TM values learned at the training
ratio over-price later deals. Using the real-time ratio (`log_tm_adj`) removed the overall drift.
**Still flagged:** PL buyers (only {int(drift.loc[drift.segment == 'PL buyers', 'windows'].iloc[0])} windows with ≥{MIN_WINDOW_N} deals):
under-priced in 2015–17 (the Premier League TV-deal surge), close to zero since. The segmented
deflator is the candidate fix; it was the best deflator on calibration but within noise.

## 5. Explanations (SHAP)

![SHAP TM-corrected]({f_shap_tm})

![SHAP stats-only]({f_shap_st})

Group importance, TM-corrected:

{_md(grp_tm, {'mean_abs_shap': '{:.3f}'})}

## 6. Features and components

* **Club strength:** own two-level Elo from {88216:,} club games (ClubElo API unreachable). Tuned on
  pre-{cfg['splits']['train_end']} games: K_club = {elo['k_club']:.0f}, K_league = {elo['k_league']:.0f},
  home advantage = {elo['hfa']:.0f}. Match-outcome MSE {elo['mse']:.4f} vs {elo['mse_naive_home_mean']:.4f} naive.
* **Age index:** delta-method curves by position (train-period TM values, demeaned by year).

![Aging curves]({f_age})

* Feature groups in this phase: {', '.join(GROUPS)}. Hype (E), availability (F) and pace/profile (I)
  arrive in phase 2.

## 7. Caveats

* Early stopping used the validation split (as the brief specifies), so validation metrics are slightly
  optimistic for the LightGBM models; hyper-parameters were tuned with CV inside training only.
* Validation is the COVID-affected 2020/21 season.
* Appearance data begin in 2012-07: career features are left-truncated (experience is normalised by
  observable years) and about half of all transfers have no appearance data in the prior year
  (players from leagues outside the dataset).
* Club Elo is our own; leagues with few cross-league games (e.g. Brazil, Saudi Arabia) have no
  absolute rating (within-league standing only).
""")
    out = path("reports") / "model_report.md"
    out.write_text("\n".join(parts))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
