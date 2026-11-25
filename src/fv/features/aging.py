"""Group D: value aging curves by position group (delta method), fit on the training period only.

For each player, pair each valuation with the valuation closest to 12 months later (300-430 days).
The annualised change in log TM value is demeaned by calendar year (removes market-wide inflation),
averaged by integer age, and summed cumulatively into a curve. Comparing the same player across
years avoids survivor bias. Peak age = curve maximum; age_index = age - peak_age(position).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fv.config import path

log = logging.getLogger(__name__)

AGES = range(16, 39)


def fit_curves(con, train_end: str, min_pairs: int = 30) -> pd.DataFrame:
    v = con.execute(f"""
      SELECT v.player_id, v.date, v.market_value_in_eur AS mv, p.position, p.date_of_birth
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
    rows = []
    for pos, g in m.groupby("position"):
        agg = g.groupby("age").d.agg(["mean", "count"]).reindex(AGES)
        agg.loc[agg["count"] < min_pairs, "mean"] = np.nan
        agg["mean"] = agg["mean"].interpolate(limit_direction="both")
        # value at age a+1 = value at age a + mean change during age a
        curve = np.concatenate([[0.0], np.cumsum(agg["mean"].values[:-1])])
        for a, c, n in zip(AGES, curve, agg["count"].fillna(0)):
            rows.append((pos, a, c, int(n)))
    curves = pd.DataFrame(rows, columns=["position", "age", "log_value_curve", "n_pairs"])
    peaks = curves.loc[curves.groupby("position").log_value_curve.idxmax(), ["position", "age"]]
    curves = curves.merge(peaks.rename(columns={"age": "peak_age"}), on="position")
    curves.to_csv(path("processed") / "aging_curves_position.csv", index=False)
    log.info("peak ages: %s", dict(zip(peaks.position, peaks.age)))
    return curves


def add_age_index(df: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    peak = curves.drop_duplicates("position").set_index("position").peak_age
    df["peak_age"] = df.position.map(peak)
    df["age_index"] = df.age - df.peak_age
    return df
