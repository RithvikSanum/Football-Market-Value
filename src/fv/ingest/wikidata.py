"""Transfermarkt player id -> English Wikipedia article, via Wikidata property P2446 (CC0).

Queried in batches of TM ids (a full P2446 scan times out). Raw JSON per batch is cached in
data/raw/wikidata/ and never re-downloaded. Output: data/interim/wikidata_tm_enwiki.parquet
(player_id, qid, article, wd_dob) — one row per TM id found; ids with no item are simply absent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from urllib.parse import unquote

import duckdb
import pandas as pd
import requests

from fv.config import load_config, path, setup_logging

log = logging.getLogger(__name__)

QUERY = """SELECT ?tm ?item ?article ?dob WHERE {
  VALUES ?tm { %s }
  ?item wdt:P2446 ?tm .
  OPTIONAL { ?item wdt:P569 ?dob . }
  OPTIONAL { ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }
}"""


def player_ids() -> list[int]:
    """Players we need hype for: every player in the transfer universe plus all current players."""
    con = duckdb.connect()
    tm = path("interim") / "transfermarkt"
    ids = set(con.sql(f"select distinct player_id from '{tm}/transfers.parquet'").df().player_id)
    ids |= set(con.sql(f"select player_id from '{tm}/players.parquet' where last_season >= 2025")
               .df().player_id)
    feats = path("processed") / "features.parquet"
    if feats.exists():
        ids |= set(pd.read_parquet(feats, columns=["player_id"]).player_id)
    return sorted(int(i) for i in ids)


def _fetch(ids: list[int], cfg: dict) -> dict:
    q = QUERY % " ".join(f'"{i}"' for i in ids)
    for attempt in range(4):
        r = requests.post(cfg["sources"]["wikidata"]["sparql_url"], data={"query": q},
                          headers={"User-Agent": cfg["http"]["user_agent"],
                                   "Accept": "application/sparql-results+json"},
                          timeout=cfg["http"]["timeout_seconds"] + 30)
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("Retry-After", 10 * (attempt + 1)))
            log.warning("wikidata %s; retry in %ss", r.status_code, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Wikidata SPARQL failed after retries")


def download() -> pd.DataFrame:
    cfg = load_config()
    raw = path("raw") / "wikidata"
    raw.mkdir(parents=True, exist_ok=True)
    ids = player_ids()
    bs = cfg["sources"]["wikidata"]["batch_size"]
    rows = []
    for k in range(0, len(ids), bs):
        batch = ids[k:k + bs]
        key = hashlib.sha1(",".join(map(str, batch)).encode()).hexdigest()[:16]
        f = raw / f"batch_{key}.json"
        if f.exists():
            data = json.loads(f.read_text())
        else:
            data = _fetch(batch, cfg)
            f.write_text(json.dumps(data))
            time.sleep(cfg["http"]["min_delay_seconds"])
        for b in data["results"]["bindings"]:
            art = b.get("article", {}).get("value")
            rows.append({"player_id": int(b["tm"]["value"]),
                         "qid": b["item"]["value"].rsplit("/", 1)[-1],
                         "article": unquote(art.rsplit("/wiki/", 1)[-1]) if art else None,
                         "wd_dob": b.get("dob", {}).get("value", "")[:10] or None})
        if (k // bs) % 10 == 0:
            log.info("wikidata %d/%d ids", min(k + bs, len(ids)), len(ids))
    df = pd.DataFrame(rows).drop_duplicates(["player_id", "qid", "article"])
    # one item per TM id; if several (rare duplicates), keep the one with an enwiki article
    df = (df.sort_values(["player_id", "article"], na_position="last")
            .drop_duplicates("player_id"))
    out = path("interim") / "wikidata_tm_enwiki.parquet"
    df.to_parquet(out, index=False)
    log.info("ids queried %d; with Wikidata item %d; with enwiki article %d",
             len(ids), len(df), df.article.notna().sum())
    return df


if __name__ == "__main__":
    setup_logging()
    download()
