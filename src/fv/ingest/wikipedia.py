"""English Wikipedia daily page views (Wikimedia Analytics API, CC0) for mapped players.

Access policy (doc.wikimedia.org/generated-data-platform/aqs/analytics-api/documentation/access-policy):
descriptive User-Agent; wait for each request to finish before sending another. We do exactly that
(sequential, plus http.wiki_delay_seconds between requests; 429s honour Retry-After). A faster pace
hit 429s within seconds, so we use 1 request/sec.

Cached in shards under data/raw/wikipedia/ (article, date, views); articles already fetched (including
ones with no data) are listed in done.txt and never re-downloaded. Combined output:
data/interim/wiki_pageviews.parquet.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import quote

import pandas as pd
import requests

from fv.config import load_config, path, setup_logging

log = logging.getLogger(__name__)
SHARD = 50


def needed_player_ids() -> set[int]:
    """Players in the transfer universe plus current players in the current-valuation leagues."""
    cfg = load_config()
    ids = set(pd.read_parquet(path("processed") / "features.parquet", columns=["player_id"]).player_id)
    p = pd.read_parquet(path("interim") / "transfermarkt" / "players.parquet",
                        columns=["player_id", "last_season", "current_club_domestic_competition_id"])
    cur = p[(p.last_season >= 2025)
            & p.current_club_domestic_competition_id.isin(cfg["current_valuation"]["leagues"])]
    return ids | set(cur.player_id)


def user_agent(cfg: dict) -> str:
    """Wikimedia's 2026 rate limits give 200 req/min only to a *compliant* User-Agent, i.e. one with
    contact details (email or URL); without it a client counts as unidentified (10 req/min)."""
    contact = (cfg["http"].get("contact") or "").strip()
    ua = cfg["http"]["user_agent"]
    return f"{ua.split(' (')[0]} ({contact}; non-commercial research) python-requests" if contact else ua


def _fetch(sess: requests.Session, article: str, cfg: dict) -> pd.DataFrame:
    w = cfg["sources"]["wikipedia"]
    url = w["pageviews_url"].format(article=quote(article.replace(" ", "_"), safe=""),
                                    start=w["start"], end=w["end"])
    for attempt in range(5):
        try:
            r = sess.get(url, timeout=cfg["http"]["timeout_seconds"])
        except requests.exceptions.RequestException as e:   # dropped connection / timeout
            log.warning("%s for %s; retry in %ss", type(e).__name__, article, 30 * (attempt + 1))
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code == 404:            # no views recorded for this title in the range
            return pd.DataFrame(columns=["article", "date", "views"])
        if r.status_code == 429 or r.status_code >= 500:
            wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            log.warning("%s for %s; retry in %ss", r.status_code, article, wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        items = r.json().get("items", [])
        return pd.DataFrame({"article": article,
                             "date": pd.to_datetime([i["timestamp"][:8] for i in items]),
                             "views": [int(i["views"]) for i in items]})
    raise RuntimeError(f"pageviews failed for {article}")


def download() -> None:
    cfg = load_config()
    raw = path("raw") / "wikipedia"
    raw.mkdir(parents=True, exist_ok=True)
    done_f = raw / "done.txt"
    done = set(done_f.read_text().splitlines()) if done_f.exists() else set()
    m = pd.read_parquet(path("interim") / "wikidata_tm_enwiki.parquet")
    m = m[m.player_id.isin(needed_player_ids())]
    todo = sorted(set(m.article.dropna()) - done)
    log.info("articles: %d total, %d cached, %d to fetch", m.article.notna().sum(), len(done), len(todo))
    sess = requests.Session()
    sess.headers["User-Agent"] = user_agent(cfg)
    delay = cfg["http"].get("wiki_delay_seconds", 0.1)
    buf, names = [], []
    n_shard = len(list(raw.glob("shard_*.parquet")))

    def flush():
        nonlocal buf, names, n_shard
        if not names:
            return
        df = pd.concat(buf, ignore_index=True) if buf else pd.DataFrame(columns=["article", "date", "views"])
        df.astype({"views": "int32"}).to_parquet(raw / f"shard_{n_shard:04d}.parquet", index=False)
        with done_f.open("a") as fh:
            fh.write("\n".join(names) + "\n")
        n_shard += 1
        buf, names = [], []

    failed = []
    for i, art in enumerate(todo, 1):
        try:
            d = _fetch(sess, art, cfg)
        except RuntimeError as e:          # not marked done: retried on the next run
            log.warning("%s; skipping for now", e)
            failed.append(art)
            continue
        if len(d):
            buf.append(d)
        names.append(art)
        time.sleep(delay)
        if len(names) >= SHARD:
            flush()
            log.info("pageviews %d/%d", i, len(todo))
    flush()
    if failed:
        log.warning("%d articles failed this run (re-run to retry): %s", len(failed), failed[:10])
    combine()


def combine() -> None:
    raw = path("raw") / "wikipedia"
    df = pd.concat([pd.read_parquet(f) for f in sorted(raw.glob("shard_*.parquet"))], ignore_index=True)
    df = df.drop_duplicates(["article", "date"])
    out = path("interim") / "wiki_pageviews.parquet"
    df.to_parquet(out, index=False)
    log.info("wrote %s: %d rows, %d articles", out, len(df), df.article.nunique())


if __name__ == "__main__":
    setup_logging()
    download()
