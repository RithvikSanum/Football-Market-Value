"""Ingest transfermarkt-datasets (dcaribou) into Parquet.

Steps (all idempotent):
1. Download the zip of gzipped CSVs once (skipped if cached).
2. Convert each table to Parquet in data/interim/transfermarkt/ (skipped if newer than the zip).
3. Archive a dated copy of the `players` table in data/snapshots/players/ so contract-expiry
   history accumulates from now on (one copy per dataset version, keyed by the zip's
   Last-Modified date and the run date).
"""
from __future__ import annotations

import datetime as dt
import logging
import shutil
import zipfile
from pathlib import Path

import duckdb
import requests

from fv.config import load_config, path, setup_logging

log = logging.getLogger(__name__)


def download(force: bool = False) -> Path:
    cfg = load_config()
    src = cfg["sources"]["transfermarkt"]
    dest_dir = path("raw") / "transfermarkt"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src["archive"]
    if dest.exists() and not force:
        log.info("cached: %s", dest)
        return dest
    url = f"{src['base_url']}/{src['archive']}"
    log.info("downloading %s", url)
    headers = {"User-Agent": cfg["http"]["user_agent"]}
    with requests.get(url, headers=headers, stream=True, timeout=cfg["http"]["timeout_seconds"]) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        (dest_dir / "last_modified.txt").write_text(r.headers.get("Last-Modified", ""))
    return dest


def to_parquet(zip_path: Path) -> Path:
    out_dir = path("interim") / "transfermarkt"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = zip_path.parent / "csv"
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            table = member.removesuffix(".csv.gz")
            pq = out_dir / f"{table}.parquet"
            if pq.exists() and pq.stat().st_mtime > zip_path.stat().st_mtime:
                log.info("cached: %s", pq.name)
                continue
            zf.extract(member, extract_dir)
            log.info("converting %s", member)
            con = duckdb.connect()
            con.execute(
                f"COPY (SELECT * FROM read_csv_auto('{extract_dir / member}', sample_size=-1)) "
                f"TO '{pq}' (FORMAT PARQUET)"
            )
            con.close()
    shutil.rmtree(extract_dir, ignore_errors=True)
    return out_dir


def snapshot_players(pq_dir: Path) -> Path:
    """Archive a dated copy of players.parquet (contract history builds up over time)."""
    snap_dir = path("snapshots") / "players"
    con = duckdb.connect()
    max_val = con.execute(
        f"SELECT max(date) FROM read_parquet('{pq_dir / 'player_valuations.parquet'}')"
    ).fetchone()[0]
    con.close()
    tag = f"data-{max_val:%Y%m%d}_run-{dt.date.today():%Y%m%d}"
    dest = snap_dir / f"players_{tag}.parquet"
    if not dest.exists():
        shutil.copy2(pq_dir / "players.parquet", dest)
        log.info("archived players snapshot: %s", dest.name)
    return dest


def main() -> None:
    setup_logging()
    pq_dir = to_parquet(download())
    snapshot_players(pq_dir)


if __name__ == "__main__":
    main()
