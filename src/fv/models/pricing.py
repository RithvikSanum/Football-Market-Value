"""Phase 1 pricing models (brief section 7) trained on real value = log(fee) - log(market index).

Stages
1. Choose the market-index deflator on validation (LightGBM stats-only, fixed default params).
2. Baselines: Transfermarkt value (>=30 days before) x one scale factor fit on train; ridge.
3. LightGBM stats-only (groups A-D) and TM-corrected (A-D + G, H): Optuna tuning with expanding-window
   CV inside the training period, final fit with early stopping on validation. Fixed seeds.
4. Quantile models (0.1/0.5/0.9) + conformalized quantile regression on the calibration split.
5. Out-of-fold predictions over training windows (expanding) for drift diagnostics.
6. SHAP values on validation.

The test split (>= splits.calib_end) is never scored here: rows are excluded before any prediction.
"""
from __future__ import annotations

import json
import logging
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fv.config import load_config, path, setup_logging
from fv.evaluation.metrics import scores
from fv.features.build import CATEGORICAL, STATS_ONLY, TM_CORRECTED, features_for

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFLATORS = {
    "none": None,
    "hedonic (carry-forward)": "lvl_hedonic",
    "hedonic nowcast": "lvl_hedonic_now",
    "hedonic nowcast + segments": "lvl_hedonic_seg",
    "hedonic nowcast, all-transfer sample": "lvl_hedonic_now_allsample",
    "repeat sales": "lvl_repeat_sales",
    "TM market level": "lvl_tm",
}
DEFAULT_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)


# --------------------------------------------------------------------------- data
def assign_split(d: pd.Series, cfg: dict) -> pd.Series:
    s = cfg["splits"]
    d = pd.to_datetime(d)
    return pd.Series(np.select(
        [d < pd.Timestamp(s["train_end"]), d < pd.Timestamp(s["val_end"]),
         d < pd.Timestamp(s["calib_end"])], ["train", "val", "calib"], "test"), index=d.index)


def load_dataset() -> pd.DataFrame:
    cfg = load_config()
    f = pd.read_parquet(path("processed") / "features.parquet")
    idx = pd.read_parquet(path("processed") / "market_index_transfers.parquet")
    df = f.merge(idx, on="transfer_id", how="left")
    df["transfer_date"] = pd.to_datetime(df.transfer_date)
    df["split"] = assign_split(df.transfer_date, cfg)
    df["log_fee"] = np.log(df.fee_eur)
    df = df.sort_values("transfer_date").reset_index(drop=True)
    # crowd ratio is undefined for the first months of data (needs 30 prior deals): training rows only
    first = df.crowd_ratio.dropna().iloc[0]
    assert not (df.crowd_ratio.isna() & (df.split != "train")).any()
    df["crowd_ratio"] = df.crowd_ratio.fillna(first)
    df["log_tm_adj"] = df.log_tm_lag + df.crowd_ratio
    return df


def deflator(df: pd.DataFrame, name: str) -> pd.Series:
    col = DEFLATORS[name]
    if col is None:
        return pd.Series(0.0, index=df.index)
    s = df[col].astype(float)
    # Repeat-sales / TM level start later than the training window: early training rows take the first
    # available level (training rows only; affects no validation / calibration / test row).
    first = s[s.notna()].iloc[0] if s.notna().any() else 0.0
    early = s.isna() & (df.split == "train")
    assert not (s.isna() & (df.split != "train")).any(), f"{name}: missing level outside training"
    return s.where(~early, first)


def lgb_frame(df: pd.DataFrame, feats: list[str], cats: dict[str, list] | None = None):
    X = df[feats].copy()
    cats = cats or {}
    for c in [c for c in CATEGORICAL if c in feats]:
        if c not in cats:
            cats[c] = sorted(X[c].dropna().unique().tolist())
        X[c] = pd.Categorical(X[c].where(X[c].isin(cats[c])), categories=cats[c])  # unseen -> NaN
    return X, cats


# --------------------------------------------------------------------------- LightGBM
def fit_lgb(Xtr, ytr, Xva=None, yva=None, params=None, objective="regression", alpha=None, seed=42):
    p = {**DEFAULT_PARAMS, **(params or {})}
    p.pop("objective", None)
    m = lgb.LGBMRegressor(objective=objective, alpha=alpha if alpha else 0.9, random_state=seed,
                          verbose=-1, **p)
    if Xva is not None:
        metric = "quantile" if objective == "quantile" else "l1"
        m.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), eval_metric=metric,
              callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(Xtr, ytr)
    return m


def mdape_real(y, pred) -> float:
    return float(np.median(np.abs(np.exp(pred - y) - 1)))


def cv_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds inside training: validate on each of the last 3 training seasons."""
    tr = df[df.split == "train"]
    folds = []
    for start in (10, 12, 14):          # windows 2017-S/2018-W, 2018-S/2019-W, 2019-S/2020-W
        a = tr.index[tr.window_idx < start].to_numpy()
        b = tr.index[tr.window_idx.isin([start, start + 1])].to_numpy()
        folds.append((a, b))
    return folds


def tune(df: pd.DataFrame, y: pd.Series, feats: list[str], n_trials: int, seed: int) -> dict:
    folds = cv_folds(df)
    X, cats = lgb_frame(df, feats)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 63, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            n_estimators=2000,
        )
        obj = trial.suggest_categorical("objective", ["regression", "huber", "regression_l1"])
        errs = []
        for a, b in folds:
            m = fit_lgb(X.loc[a], y.loc[a], X.loc[b], y.loc[b], params, obj, seed=seed)
            errs.append(mdape_real(y.loc[b].values, m.predict(X.loc[b])))
        trial.set_user_attr("n_iter", m.best_iteration_)
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    log.info("best CV MdAPE %.4f params %s", study.best_value, study.best_params)
    return study.best_params | {"cv_mdape": study.best_value}


# --------------------------------------------------------------------------- ridge
def fit_ridge(df_tr: pd.DataFrame, y_tr: pd.Series, feats: list[str]):
    cats = [c for c in CATEGORICAL if c in feats]
    nums = [c for c in feats if c not in cats]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nums),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cats),
    ])
    model = make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 30), cv=TimeSeriesSplit(5)))
    order = df_tr.transfer_date.argsort().values
    model.fit(df_tr[feats].iloc[order], y_tr.iloc[order])
    return model


# --------------------------------------------------------------------------- CQR
def cqr_qhat(y_cal, lo_cal, hi_cal, alpha: float) -> float:
    e = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(e)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(e)[k - 1])


# --------------------------------------------------------------------------- pipeline
def choose_deflator(df: pd.DataFrame, seed: int, n_boot: int = 1000) -> pd.DataFrame:
    """Compare deflators with LightGBM stats-only (default params) on validation (train -> val) and,
    as a second look, on calibration (train+val -> calib). Paired bootstrap CI of the MdAPE difference
    vs no deflator."""
    feats = features_for(STATS_ONLY)
    rng = np.random.default_rng(seed)
    rows = []
    for split, fit_on in (("val", ["train"]), ("calib", ["train", "val"])):
        tr, te = df[df.split.isin(fit_on)], df[df.split == split]
        ape = {}
        for name in DEFLATORS:
            lvl = deflator(df, name)
            y = df.log_fee - lvl
            Xtr, cats = lgb_frame(tr, feats)
            Xte, _ = lgb_frame(te, feats, cats)
            m = fit_lgb(Xtr, y.loc[tr.index], params=DEFAULT_PARAMS, seed=seed)
            pred_fee = np.exp(m.predict(Xte) + lvl.loc[te.index])
            ape[name] = np.abs(pred_fee / te.fee_eur.values - 1)
            rows.append({"split": split, "deflator": name, **scores(te.fee_eur, pred_fee)})
        A = pd.DataFrame(ape)
        boot = pd.DataFrame([A.iloc[rng.integers(0, len(A), len(A))].median() for _ in range(n_boot)])
        diff = boot.sub(boot["none"], axis=0)
        for name in DEFLATORS:
            r = next(r for r in rows if r["split"] == split and r["deflator"] == name)
            r["diff_vs_none"] = float(A[name].median() - A["none"].median())
            r["diff_90lo"], r["diff_90hi"] = float(diff[name].quantile(.05)), float(diff[name].quantile(.95))
        log.info("%s: %s", split, {n: round(A[n].median(), 4) for n in DEFLATORS})
    return pd.DataFrame(rows)


def oof_train(df, y, feats, params, n_iter, seed) -> pd.Series:
    """Expanding-window out-of-fold predictions for training windows from 2015-S (diagnostics)."""
    tr = df[df.split == "train"]
    out = pd.Series(np.nan, index=df.index)
    X, cats = lgb_frame(df, feats)
    p = {k: v for k, v in params.items() if k not in ("objective", "cv_mdape")} | {"n_estimators": n_iter}
    for j in range(6, int(tr.window_idx.max()) + 1):
        a = tr.index[tr.window_idx < j]
        b = tr.index[tr.window_idx == j]
        if len(b) == 0:
            continue
        m = fit_lgb(X.loc[a], y.loc[a], params=p, objective=params.get("objective", "regression"), seed=seed)
        out.loc[b] = m.predict(X.loc[b])
    return out


def main() -> None:
    setup_logging()
    cfg = load_config()
    seed = cfg["seed"]
    n_trials = cfg.get("tuning", {}).get("n_trials", 40)
    alpha = 1 - cfg["intervals"]["coverage"]
    out_dir = path("processed")

    df = load_dataset()
    df = df[df.split != "test"].copy()          # test is never touched in phase 1
    log.info("rows: %s", df.split.value_counts().to_dict())

    # 1. deflator choice
    defl = choose_deflator(df, seed)
    defl.to_csv(out_dir / "deflator_comparison.csv", index=False)
    method = cfg["market_index"]["method"]
    v = defl[defl.split == "val"]
    chosen = v.sort_values("MdAPE").iloc[0].deflator if method == "auto" else method
    log.info("chosen deflator: %s", chosen)
    df["log_index"] = deflator(df, chosen)
    df["y"] = df.log_fee - df.log_index
    tr, va, ca = (df[df.split == s] for s in ("train", "val", "calib"))
    held = df[df.split.isin(["val", "calib"])]

    preds = pd.DataFrame(index=df.index)
    meta: dict = {"deflator": chosen}

    # 2. baselines
    c_tm = float(np.nanmedian(tr.log_fee - tr.log_tm_lag))
    preds["tm_baseline"] = np.exp(df.log_tm_lag + c_tm)
    preds["tm_realtime_ratio"] = np.exp(df.log_tm_adj)
    c_tm2 = float(np.nanmedian(tr.log_fee - np.log(tr.tm_value_transfers_table)))
    preds["tm_transfers_table"] = np.exp(np.log(df.tm_value_transfers_table) + c_tm2)
    meta["tm_scale"] = float(np.exp(c_tm))
    meta["tm_scale_transfers_table"] = float(np.exp(c_tm2))
    for name, groups in (("ridge_stats", STATS_ONLY), ("ridge_tm", TM_CORRECTED)):
        feats = features_for(groups)
        r = fit_ridge(tr, tr.y, feats)
        preds[name] = np.exp(r.predict(df[feats]) + df.log_index)
        meta[f"{name}_alpha"] = float(r[-1].alpha_)

    # 3-4. LightGBM point + quantiles
    shap_out = {}
    for name, groups in (("lgb_stats", STATS_ONLY), ("lgb_tm", TM_CORRECTED)):
        feats = features_for(groups)
        best = tune(df, df.y, feats, n_trials, seed)
        params = {k: v for k, v in best.items() if k not in ("objective", "cv_mdape")} | {"n_estimators": 3000}
        Xtr, cats = lgb_frame(tr, feats)
        Xva, _ = lgb_frame(va, feats, cats)
        Xall, _ = lgb_frame(df, feats, cats)
        m = fit_lgb(Xtr, tr.y, Xva, va.y, params, best["objective"], seed=seed)
        preds[name] = np.exp(m.predict(Xall) + df.log_index)
        preds[f"{name}_real"] = m.predict(Xall)
        meta[name] = {"params": best, "best_iteration": int(m.best_iteration_), "features": feats}
        m.booster_.save_model(str(out_dir / f"model_{name}.txt"))

        q = {}
        for qq in cfg["intervals"]["quantiles"]:
            mq = fit_lgb(Xtr, tr.y, Xva, va.y, params, "quantile", alpha=qq, seed=seed)
            q[qq] = mq.predict(Xall)
        lo, hi = np.minimum(q[0.1], q[0.9]), np.maximum(q[0.1], q[0.9])
        cal = (df.split == "calib").values
        qhat = cqr_qhat(df.y.values[cal], lo[cal], hi[cal], alpha)
        preds[f"{name}_q10_raw"], preds[f"{name}_q90_raw"] = lo, hi
        preds[f"{name}_q50"] = q[0.5]
        preds[f"{name}_q10"], preds[f"{name}_q90"] = lo - qhat, hi + qhat
        meta[name]["cqr_qhat"] = qhat

        preds[f"{name}_oof_real"] = oof_train(df, df.y, feats, best, int(m.best_iteration_), seed)

        import shap
        Xh, _ = lgb_frame(held, feats, cats)
        sv = shap.TreeExplainer(m.booster_).shap_values(Xh)
        shap_out[name] = pd.DataFrame(sv, columns=feats, index=held.index).assign(
            transfer_id=held.transfer_id.values, base_value=float(np.mean(m.predict(Xtr))))

    keep = ["transfer_id", "player_name", "transfer_date", "split", "window_idx", "fee_eur", "fee_gbp",
            "log_fee", "log_index", "y", "to_league", "from_league", "position", "age", "pl_buyer",
            "seg_top", "log_tm_lag", "crowd_ratio", "from_club_name", "to_club_name"]
    res = pd.concat([df[keep], preds], axis=1)
    res.to_parquet(out_dir / "predictions_phase1.parquet", index=False)
    for k, v in shap_out.items():
        v.to_parquet(out_dir / f"shap_{k}.parquet", index=False)
    (out_dir / "pricing_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("done")


if __name__ == "__main__":
    main()
