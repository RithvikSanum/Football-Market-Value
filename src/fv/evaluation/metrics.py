"""Pricing metrics on money values (predictions are medians: exp of predicted log)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEE_BANDS = [0, 5e6, 20e6, 50e6, np.inf]
FEE_LABELS = ["<€5m", "€5–20m", "€20–50m", "€50m+"]
AGE_BANDS = [0, 21, 24, 27, 30, 99]
AGE_LABELS = ["≤20", "21–23", "24–26", "27–29", "30+"]


def scores(fee: np.ndarray, pred: np.ndarray) -> dict:
    fee, pred = np.asarray(fee, float), np.asarray(pred, float)
    ok = np.isfinite(pred) & (pred > 0)
    fee, pred = fee[ok], pred[ok]
    ape = np.abs(pred / fee - 1)
    return {"n": int(ok.sum()), "MdAPE": float(np.median(ape)),
            "MAE_log": float(np.mean(np.abs(np.log(pred) - np.log(fee)))),
            "within_25": float(np.mean(ape <= 0.25)), "within_50": float(np.mean(ape <= 0.5))}


def table(df: pd.DataFrame, preds: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"model": m, **scores(df.fee_eur, df[c])} for m, c in preds.items()])


def league_group(s: pd.Series) -> pd.Series:
    names = {"GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A", "L1": "Bundesliga",
             "FR1": "Ligue 1"}
    return s.map(names).fillna("Other")


def breakdowns(df: pd.DataFrame, preds: dict[str, str]) -> dict[str, pd.DataFrame]:
    d = df.assign(
        fee_band=pd.cut(df.fee_eur, FEE_BANDS, labels=FEE_LABELS, right=False),
        buying_league=league_group(df.to_league),
        position_group=df.position.fillna("Unknown"),
        age_band=pd.cut(df.age, AGE_BANDS, labels=AGE_LABELS, right=False),
    )
    out = {}
    for dim in ("fee_band", "buying_league", "position_group", "age_band"):
        rows = []
        for k, g in d.groupby(dim, observed=True):
            r = {dim: k, "n": len(g)}
            for m, c in preds.items():
                r[m] = scores(g.fee_eur, g[c])["MdAPE"]
            rows.append(r)
        out[dim] = pd.DataFrame(rows)
    return out


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    inside = (y >= lo) & (y <= hi)
    return {"coverage": float(np.mean(inside)),
            "median_width_ratio": float(np.median(np.exp(hi - lo)))}
