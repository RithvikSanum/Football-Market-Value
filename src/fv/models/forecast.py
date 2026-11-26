"""12-month value forecast (brief section 7), on the snapshot panel (fv.features.snapshots).

Target (player part only): y_real = log(TM value at t+12m / TM value at t) - market move, where the
market move is the change in the TM market level (median TM value of established top-5-league
players, the same series as the TM-level index cross-check) over the same 12 months. The market part
comes from the market forecast (section 15.3); the two are combined in fv.models.combined. Assumption
(documented): the forecast fair value = current fair value x predicted ratio.

Models
- baseline "no change": 0
- baseline "age curve": change on the player's sub-position value curve from age to age + 1, plus the
  train-period mean of (y_real - curve change) (curves are demeaned by year)
- LightGBM (point = median quantile), quantiles 0.1 / 0.5 / 0.9 + CQR on the calibration snapshots
  -> bear / base / bull. Features: all pricing groups A-G plus the TM value at t.
Splits by snapshot date (config forecast.*); targets of training dates are realised before
pricing train_end. Test rows are predicted but never scored here (fv.evaluation.final_test does that).
Output: data/processed/forecast_predictions.parquet, forecast_meta.json, forecast_val_metrics.csv
"""
from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from fv.config import load_config, path, setup_logging
from fv.features.build import TM_CORRECTED, features_for
from fv.features.market_index import tm_level_on
from fv.models.pricing import cqr_qhat, lgb_frame

log = logging.getLogger(__name__)
PARAMS = dict(n_estimators=4000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)


def forecast_features() -> list[str]:
    f = [c for c in features_for(TM_CORRECTED) if c != "log_tm_adj"]
    return f + ["log_tm_now"]


def assign_split(t: pd.Series, cfg: dict) -> pd.Series:
    fc = cfg["forecast"]
    t = pd.to_datetime(t)
    val, cal = pd.to_datetime(fc["val_dates"]), pd.to_datetime(fc["calib_dates"])
    return pd.Series(np.select([t <= pd.Timestamp(fc["train_last"]), t.isin(val), t.isin(cal),
                                t >= pd.Timestamp(fc["test_first"])],
                               ["train", "val", "calib", "test"], "gap"), index=t.index)


def load_panel() -> pd.DataFrame:
    cfg = load_config()
    p = pd.read_parquet(path("processed") / "snapshot_panel.parquet")
    p["t"] = pd.to_datetime(p.t)
    tm = pd.read_parquet(path("processed") / "tm_level_monthly.parquet")
    p["market_move"] = (tm_level_on(p.t + pd.Timedelta(days=365), tm) - tm_level_on(p.t, tm)).values
    p["y_real"] = p.y_fwd12 - p.market_move
    p["split"] = assign_split(p.t, cfg)
    return p


def age_curve_change(p: pd.DataFrame) -> pd.Series:
    c = pd.read_csv(path("processed") / "aging_curves.csv")
    sub = c[c.level == "sub_position"]
    pos = c[c.level == "position"]
    out = np.full(len(p), np.nan)
    for (lvl, key), g in pd.concat([sub, pos]).groupby(["level", "profile"]):
        col = p.sub_position if lvl == "sub_position" else p.position
        rows = (col == key).values & np.isnan(out)
        if rows.any():
            a, v = g.age.values.astype(float), g.log_value_curve.values
            x = p.age.values[rows].astype(float)
            out[rows] = np.interp(x + 1, a, v) - np.interp(x, a, v)
    return pd.Series(out, index=p.index)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    big = np.abs(y) > 0.1
    from scipy.stats import spearmanr
    return {"n": int(ok.sum()), "MAE": float(np.mean(np.abs(pred - y))),
            "RMSE": float(np.sqrt(np.mean((pred - y) ** 2))),
            "R2_vs_no_change": float(1 - np.sum((pred - y) ** 2) / np.sum(y ** 2)),
            "spearman": float(spearmanr(pred, y).statistic) if pred.std() > 0 else np.nan,
            "direction_hit_|y|>0.1": float(np.mean(np.sign(pred[big]) == np.sign(y[big])))}


def main() -> None:
    setup_logging()
    cfg = load_config()
    alpha = 1 - cfg["intervals"]["coverage"]
    p = load_panel()
    feats = forecast_features()
    has = p.y_real.notna()
    tr, va, ca = (p[has & (p.split == s)] for s in ("train", "val", "calib"))
    log.info("forecast rows with target: %s", p[has].split.value_counts().to_dict())

    curve = age_curve_change(p)
    k = float(np.nanmean(tr.y_real - curve.loc[tr.index]))
    p["pred_no_change"] = 0.0
    p["pred_age_curve"] = curve + k

    Xtr, cats = lgb_frame(tr, feats)
    Xva, _ = lgb_frame(va, feats, cats)
    Xall, _ = lgb_frame(p, feats, cats)
    preds, iters = {}, {}
    for q in cfg["intervals"]["quantiles"]:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=cfg["seed"], verbose=-1, **PARAMS)
        m.fit(Xtr, tr.y_real, eval_X=(Xva,), eval_y=(va.y_real,), eval_metric="quantile",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        preds[q], iters[q] = m.predict(Xall), int(m.best_iteration_)
        if q == 0.5:
            m.booster_.save_model(str(path("processed") / "model_forecast_q50.txt"))
            imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    lo, hi = np.minimum(preds[0.1], preds[0.9]), np.maximum(preds[0.1], preds[0.9])
    cal = (p.split == "calib").values & has.values
    qhat = cqr_qhat(p.y_real.values[cal], lo[cal], hi[cal], alpha)
    p["pred_lgb"] = preds[0.5]
    p["pred_q10"], p["pred_q90"] = lo - qhat, hi + qhat
    p["pred_q10_raw"], p["pred_q90_raw"] = lo, hi

    rows = []
    for split, d in (("val", p[has & (p.split == "val")]), ("calib", p[has & (p.split == "calib")])):
        for name, col in (("no change", "pred_no_change"), ("age curve", "pred_age_curve"),
                          ("LightGBM", "pred_lgb")):
            rows.append({"split": split, "model": name, **metrics(d.y_real.values, d[col].values)})
        inside = (d.y_real >= d.pred_q10) & (d.y_real <= d.pred_q90)
        rows.append({"split": split, "model": "80% interval coverage", "n": len(d),
                     "MAE": float(inside.mean())})
    mt = pd.DataFrame(rows)
    mt.to_csv(path("processed") / "forecast_val_metrics.csv", index=False)
    log.info("forecast metrics (y_real, log points):\n%s", mt.round(4).to_string())
    keep = ["transfer_id", "player_id", "t", "split", "age", "position", "sub_position", "to_league",
            "tm_now", "y_fwd12", "market_move", "y_real", "pred_no_change", "pred_age_curve", "pred_lgb",
            "pred_q10", "pred_q90", "pred_q10_raw", "pred_q90_raw"]
    p[keep].to_parquet(path("processed") / "forecast_predictions.parquet", index=False)
    meta = {"features": feats, "params": PARAMS, "best_iterations": iters, "cqr_qhat": qhat,
            "age_curve_intercept": k, "top_features_gain": imp.head(15).round(1).to_dict()}
    (path("processed") / "forecast_meta.json").write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
