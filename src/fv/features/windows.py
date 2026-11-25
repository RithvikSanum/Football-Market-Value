"""Transfer windows.

Each calendar year has two windows: summer "YYYY-S" (covers 1 Jun - 31 Dec) and winter "YYYY-W"
(covers 1 Jan - 31 May). Deals outside the official windows (free agents, off-window leagues) belong
to the most recent window. `window_idx` is a running integer (2012-S = 0, 2013-W = 1, ...).
"""
from __future__ import annotations

import pandas as pd

BASE_YEAR = 2012


def window_label(d: pd.Series) -> pd.Series:
    d = pd.to_datetime(d)
    return d.dt.year.astype(str) + "-" + (d.dt.month >= 6).map({True: "S", False: "W"})


def window_idx(d: pd.Series) -> pd.Series:
    d = pd.to_datetime(d)
    return (d.dt.year - BASE_YEAR) * 2 + (d.dt.month >= 6).astype(int) - 1


def window_start(idx: int) -> pd.Timestamp:
    year = BASE_YEAR + (idx + 1) // 2
    return pd.Timestamp(year=year, month=6 if (idx + 1) % 2 else 1, day=1)


def idx_label(idx: int) -> str:
    year = BASE_YEAR + (idx + 1) // 2
    return f"{year}-{'S' if (idx + 1) % 2 else 'W'}"


def late_window(d: pd.Series) -> pd.Series:
    """Last 3 days of a window, approximated by deadline days 31 Aug / 31 Jan (29 Aug - 2 Sep,
    29 Jan - 2 Feb). League deadlines differ slightly; documented as an approximation."""
    d = pd.to_datetime(d)
    md = d.dt.month * 100 + d.dt.day
    return md.between(829, 902) | md.between(129, 202)
