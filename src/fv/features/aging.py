"""Group D: value aging curves by position group (delta method), fit on the training period only.

For each player, pair each valuation with the valuation closest to 12 months later (300-430 days).
The annualised change in log TM value is demeaned by calendar year (removes market-wide inflation),
averaged by integer age, and summed cumulatively into a curve. Comparing the same player across
years avoids survivor bias. Peak age = curve maximum.

Phase 2 (profile-aware): curves also by sub-position (e.g. Centre-Back vs Full-Back), with each age's
mean change shrunk toward the position-group mean: (n * sub + K * pos) / (n + K), K = SHRINK_K.
age_index = age - peak_age(sub-position, falling back to position); age_curve_fwd2 = expected change in
log value from aging over the next 2 years on the same curve. (Archetype curves need SoFIFA / Understat
and are pending.)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fv.config import path

log = logging.getLogger(__name__)

AGES = range(16, 39)
SHRINK_K = 200


def fit_curves(con, train_end: str, min_pairs: int = 30) -> pd.DataFrame:
    v = con.execute(f"""
      SELECT v.player_id, v.date, v.market_value_in_eur AS mv, p.position, p.sub_position,
             p.date_of_birth
      FROM player_valuations v JOIN players p USING (player_id)
      WHERE v.market_value_in_eur > 0 AND v.date < DATE '{train_end}' AND p.position <> 'Missing'
    """).df()
    v["date"] = pd.to_datetime(v.date)
    v = v.sort_values(["player_id", "date"])
    nxt = v.rename(columns={"date": "date2", "mv": "mv2"})[["player_id", "date2", "mv2"]]
    v["target"] = v.date + pd.Timedelta(days=365)
    m = pd.merge_asof(v.sort_values("target"), nxt.sort_values("date2"), left_on="target",
                      right_on="date2", by="player_id", direction="nearest",
                      tolerance=pd.Timedelta(days=65))
    m = m.dropna(subset=["mv2"])
    m = m[m.date2 < pd.Timestamp(train_end)]
    gap = (m.date2 - m.date).dt.days / 365.25
    m["d"] = np.log(m.mv2 / m.mv) / gap
    m["age"] = np.floor((m.date - pd.to_datetime(m.date_of_birth)).dt.days / 365.25)
    m["d"] = m.d - m.groupby(m.date.dt.year).d.transform("mean")
    def curve(mean: np.ndarray) -> np.ndarray:
        # value at age a+1 = value at age a + mean change during age a
        return np.concatenate([[0.0], np.cumsum(mean[:-1])])

    rows, pos_mean = [], {}
    for pos, g in m.groupby("position"):
        agg = g.groupby("age").d.agg(["mean", "count"]).reindex(AGES)
        agg.loc[agg["count"] < min_pairs, "mean"] = np.nan
        agg["mean"] = agg["mean"].interpolate(limit_direction="both")
        pos_mean[pos] = agg["mean"].values
        for a, c, n in zip(AGES, curve(agg["mean"].values), agg["count"].fillna(0)):
            rows.append(("position", pos, pos, a, c, int(n)))
    for (pos, sp), g in m.dropna(subset=["sub_position"]).groupby(["position", "sub_position"]):
        if pos not in pos_mean or sp == pos:
            continue
        agg = g.groupby("age").d.agg(["mean", "count"]).reindex(AGES)
        n = agg["count"].fillna(0).values
        mu = np.where(n > 0, agg["mean"].fillna(0).values, 0.0)
        shrunk = (n * mu + SHRINK_K * pos_mean[pos]) / (n + SHRINK_K)
        for a, c, k in zip(AGES, curve(shrunk), n):
            rows.append(("sub_position", sp, pos, a, c, int(k)))
    curves = pd.DataFrame(rows, columns=["level", "profile", "position", "age", "log_value_curve",
                                         "n_pairs"])
    peaks = curves.loc[curves.groupby(["level", "profile"]).log_value_curve.idxmax(),
                       ["level", "profile", "age"]].rename(columns={"age": "peak_age"})
    curves = curves.merge(peaks, on=["level", "profile"])
    curves.to_csv(path("processed") / "aging_curves.csv", index=False)
    curves[curves.level == "position"].to_csv(path("processed") / "aging_curves_position.csv", index=False)
    log.info("peak ages: %s", dict(zip(peaks.profile, peaks.peak_age)))
    return curves


def add_age_index(df: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    if "level" not in curves:                       # phase-1 file: position level only
        curves = curves.assign(level="position", profile=curves.position)
    pos = curves[curves.level == "position"]
    sub = curves[curves.level == "sub_position"]
    key = np.where(df.sub_position.isin(sub.profile.unique()), df.sub_position, df.position)
    lvl = np.where(df.sub_position.isin(sub.profile.unique()), "sub_position", "position")
    prof = pd.concat([sub, pos])
    peak = prof.drop_duplicates(["level", "profile"]).set_index(["level", "profile"]).peak_age
    idx = pd.MultiIndex.from_arrays([lvl, key])
    df["peak_age"] = peak.reindex(idx).values
    df["age_index"] = df.age - df.peak_age
    # expected 2-year aging change in log value on the profile curve (linear interpolation in age)
    fwd = np.full(len(df), np.nan)
    for (lv, pr), g in prof.groupby(["level", "profile"]):
        rows = (lvl == lv) & (key == pr)
        if not rows.any():
            continue
        a, c = g.age.values.astype(float), g.log_value_curve.values
        x = df.age.values[rows].astype(float)
        fwd[rows] = np.interp(x + 2, a, c) - np.interp(x, a, c)
    df["age_curve_fwd2"] = fwd
    return df
