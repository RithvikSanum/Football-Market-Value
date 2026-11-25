"""Contract history from old Kaggle versions of transfermarkt-datasets (davidcariboo/player-scores, CC0).

Only players.csv is read from each version's zip, using HTTP range requests (the zip's central directory
plus that one member), so we never download the full archives. Snapshot date = the zip entry's
modification time. Cached per version in data/raw/contracts/ and never re-downloaded.
Output: data/interim/contract_snapshots.parquet (version, snapshot_date, player_id, current_club_id,
contract_expiration_date).
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime

import pandas as pd
import requests

from fv.config import load_config, path, setup_logging

log = logging.getLogger(__name__)
URL = "https://www.kaggle.com/api/v1/datasets/download/davidcariboo/player-scores?datasetVersionNumber={}"
COLS = ["player_id", "current_club_id", "contract_expiration_date"]


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, sess: requests.Session):
        self.sess = sess
        r = sess.get(url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("server does not support range requests")
        self.url = r.url                                  # signed storage URL after redirect
        self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self.bytes_read = 0
        self.buf, self.buf_start = b"", 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        n = min(n, self.size - self.pos)
        b0 = self.buf_start
        if not (b0 <= self.pos and self.pos + n <= b0 + len(self.buf)):   # read ahead >= 1 MB
            end = min(self.pos + max(n, 1 << 20), self.size) - 1
            r = self.sess.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
            r.raise_for_status()
            self.buf, self.buf_start = r.content, self.pos
            self.bytes_read += len(r.content)
        i = self.pos - self.buf_start
        out = self.buf[i:i + n]
        self.pos += len(out)
        return out


def fetch_version(v: int, sess: requests.Session) -> pd.DataFrame | None:
    f = RangeFile(URL.format(v), sess)
    zf = zipfile.ZipFile(f)
    names = [n for n in zf.namelist() if n.rsplit("/", 1)[-1] == "players.csv"]
    if not names:
        return None
    info = zf.getinfo(names[0])
    snap = datetime(*info.date_time).date()
    with zf.open(info) as fh:
        df = pd.read_csv(fh, usecols=lambda c: c in COLS)
    if "contract_expiration_date" not in df:
        return None
    log.info("v%d: snapshot %s, %d players, %.1f MB read", v, snap, len(df), f.bytes_read / 1e6)
    return df.assign(version=v, snapshot_date=pd.Timestamp(snap))


# every 4th version, plus all versions inside the gaps (> 40 days) that sampling leaves
GAP_FILL = [226, 227, 228, 246, 247, 248, 602, 603, 604, 650, 651, 652]


def download(first: int = 193, last: int = 679, step: int = 4) -> pd.DataFrame:
    cfg = load_config()
    raw = path("raw") / "contracts"
    raw.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = cfg["http"]["user_agent"]
    versions = sorted(set(range(first, last + 1, step)) | {last} | set(GAP_FILL))
    frames = []
    for v in versions:
        f = raw / f"players_v{v}.parquet"
        if not f.exists():
            df = fetch_version(v, sess)
            (df if df is not None else pd.DataFrame(columns=COLS)).to_parquet(f, index=False)
            time.sleep(cfg["http"]["min_delay_seconds"])
        d = pd.read_parquet(f)
        if len(d):
            frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    # format changed over time ("2023-06-30" vs "2023-06-30 00:00:00"): parse the date part only
    out["contract_expiration_date"] = pd.to_datetime(
        out.contract_expiration_date.astype("string").str[:10], format="%Y-%m-%d", errors="coerce")
    out.to_parquet(path("interim") / "contract_snapshots.parquet", index=False)
    log.info("contract snapshots: %d versions, %s .. %s", out.version.nunique(),
             out.snapshot_date.min().date(), out.snapshot_date.max().date())
    return out


if __name__ == "__main__":
    setup_logging()
    download()
