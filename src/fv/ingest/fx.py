"""ECB euro reference rates (free, no key). Cached to data/raw/fx/ and data/interim/fx.parquet.

Rates are "units of currency per 1 EUR" on each TARGET business day since 1999-01-04.
`rate_on(date)` uses the last published rate on or before the date (weekends / holidays).
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from fv.config import load_config, path, setup_logging

log = logging.getLogger(__name__)


def _fetch_ecb(cur: str) -> pd.DataFrame:
    cfg = load_config()
    url = cfg["fx"]["ecb_url"].format(cur=cur)
    r = requests.get(url, headers={"User-Agent": cfg["http"]["user_agent"]},
                     timeout=cfg["http"]["timeout_seconds"])
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    return df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": f"{cur.lower()}_per_eur"})


def _fetch_frankfurter(cur: str) -> pd.DataFrame:
    cfg = load_config()
    url = f"{cfg['fx']['frankfurter_url']}/1999-01-04..?from=EUR&to={cur}"
    r = requests.get(url, headers={"User-Agent": cfg["http"]["user_agent"]},
                     timeout=cfg["http"]["timeout_seconds"])
    r.raise_for_status()
    rates = r.json()["rates"]
    return pd.DataFrame({"date": list(rates), f"{cur.lower()}_per_eur": [v[cur] for v in rates.values()]})


def download(force: bool = False) -> Path:
    cfg = load_config()
    raw = path("raw") / "fx"
    raw.mkdir(parents=True, exist_ok=True)
    out = path("interim") / "fx.parquet"
    if out.exists() and not force:
        log.info("cached: %s", out)
        return out
    frames = []
    for cur in cfg["fx"]["currencies"]:
        cache = raw / f"eur_{cur.lower()}.csv"
        if cache.exists() and not force:
            df = pd.read_csv(cache)
        else:
            try:
                df = _fetch_ecb(cur) if cfg["fx"]["source"] == "ecb" else _fetch_frankfurter(cur)
                log.info("fetched %s from %s", cur, cfg["fx"]["source"])
            except Exception as e:  # noqa: BLE001
                log.warning("primary FX source failed (%s); trying frankfurter", e)
                df = _fetch_frankfurter(cur)
            df.to_csv(cache, index=False)
            time.sleep(cfg["http"]["min_delay_seconds"])
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df.set_index("date"))
    fx = pd.concat(frames, axis=1).sort_index().dropna(how="all").reset_index()
    fx.to_parquet(out, index=False)
    return out


def load() -> pd.DataFrame:
    return pd.read_parquet(path("interim") / "fx.parquet")


def rate_on(dates: pd.Series, cur: str = "GBP") -> pd.Series:
    """Units of `cur` per EUR on each date (last ECB fixing on or before the date)."""
    fx = load()[["date", f"{cur.lower()}_per_eur"]].sort_values("date")
    d = pd.DataFrame({"date": pd.to_datetime(dates).values, "_i": range(len(dates))}).sort_values("date")
    m = pd.merge_asof(d, fx, on="date", direction="backward").sort_values("_i")
    return pd.Series(m[f"{cur.lower()}_per_eur"].values, index=dates.index)


def main() -> None:
    setup_logging()
    download()


if __name__ == "__main__":
    main()
