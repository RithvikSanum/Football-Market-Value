"""Ablation study (brief section 6): does each factor group help?

For each model (stats-only, TM-corrected): the full feature set, the full set minus each group, and the
phase-1 feature set. LightGBM with the tuned parameters of the full model (pricing_meta.json), averaged
over 3 seeds; target = real value (same deflator as pricing).
- validation: fit on train, early stopping on validation (as the main models), score validation
- calibration: fit on train + validation with the iteration count from the validation fit, score calib
Paired bootstrap 90% CI of the MdAPE change vs the full set. Test is never touched.
Output: data/processed/ablation.csv
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from fv.config import load_config, path, setup_logging
from fv.evaluation.metrics import scores
from fv.features.build import PHASE1_STATS, STATS_ONLY, TM_CORRECTED, features_for
from fv.models.pricing import deflator, fit_lgb, lgb_frame, load_dataset

log = logging.getLogger(__name__)
SEEDS = (42, 43, 44)


def _predict(df, feats, params, objective):
    tr, va, ca = (df[df.split == s] for s in ("train", "val", "calib"))
    trva = df[df.split.isin(["train", "val"])]
    pv, pc = np.zeros(len(va)), np.zeros(len(ca))
    for s in SEEDS:
        Xtr, cats = lgb_frame(tr, feats)
        Xva, _ = lgb_frame(va, feats, cats)
        m = fit_lgb(Xtr, tr.y, Xva, va.y, params | {"n_estimators": 3000}, objective, seed=s)
        pv += m.predict(Xva) / len(SEEDS)
        X2, cats2 = lgb_frame(trva, feats)
        Xc, _ = lgb_frame(ca, feats, cats2)
        m2 = fit_lgb(X2, trva.y, params=params | {"n_estimators": int(m.best_iteration_)},
                     objective=objective, seed=s)
        pc += m2.predict(Xc) / len(SEEDS)
    return (np.exp(pv + va.log_index.values), np.exp(pc + ca.log_index.values))


def main() -> None:
    setup_logging()
    cfg = load_config()
    meta = json.loads((path("processed") / "pricing_meta.json").read_text())
    df = load_dataset()
    df = df[df.split != "test"].copy()
    df["log_index"] = deflator(df, meta["deflator"])
    df["y"] = df.log_fee - df.log_index
    va, ca = df[df.split == "val"], df[df.split == "calib"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for model, groups, key in (("LightGBM stats-only", STATS_ONLY, "lgb_stats"),
                               ("LightGBM TM-corrected", TM_CORRECTED, "lgb_tm")):
        best = meta[key]["params"]
        params = {k: v for k, v in best.items() if k not in ("objective", "cv_mdape")}
        variants = {"full": groups} | {f"− {g}": [x for x in groups if x != g] for g in groups}
        extra = [g for g in groups if g not in PHASE1_STATS + ["G_momentum", "H_crowd"]]
        variants["phase-1 set (no " + ", ".join(extra) + ")"] = [g for g in groups if g not in extra]
        ape = {}
        for name, gs in variants.items():
            pv, pc = _predict(df, features_for(gs), params, best["objective"])
            ape[name] = (np.abs(pv / va.fee_eur.values - 1), np.abs(pc / ca.fee_eur.values - 1))
            sv, sc = scores(va.fee_eur, pv), scores(ca.fee_eur, pc)
            rows.append({"model": model, "variant": name, "val_MdAPE": sv["MdAPE"],
                         "val_within_50": sv["within_50"], "calib_MdAPE": sc["MdAPE"]})
            log.info("%s | %s: val %.4f calib %.4f", model, name, sv["MdAPE"], sc["MdAPE"])
        for split, j in (("val", 0), ("calib", 1)):
            A = pd.DataFrame({k: v[j] for k, v in ape.items()})
            idx = rng.integers(0, len(A), (1000, len(A)))
            boot = np.stack([np.median(A.values[i], axis=0) for i in idx])
            d = boot - boot[:, [0]]
            for k, name in enumerate(A.columns):
                r = next(r for r in rows if r["model"] == model and r["variant"] == name)
                r[f"{split}_diff_vs_full"] = float(np.median(A[name]) - np.median(A["full"]))
                r[f"{split}_diff_90lo"], r[f"{split}_diff_90hi"] = np.quantile(d[:, k], [0.05, 0.95])
    out = pd.DataFrame(rows)
    out.to_csv(path("processed") / "ablation.csv", index=False)
    log.info("wrote ablation.csv\n%s", out.round(4).to_string())


if __name__ == "__main__":
    main()
