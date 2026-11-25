"""Club strength: our own two-level Elo built from transfermarkt-datasets games.

ClubElo's API (api.clubelo.com) was unreachable in Nov 2026, so we compute ratings ourselves.

Model
-----
rating(club) = league_rating(league) + offset(club)           for clubs playing in a covered league
rating(club) = standalone rating                               for clubs seen only in cups / Europe

* Every game (domestic league, domestic cup, European / Club World Cup; national teams excluded)
  updates the two clubs' offsets: delta = K_club * G * (S - E), with G the goal-difference multiplier
  (World Football Elo convention) and E = 1 / (1 + 10^(-(R_home + HFA - R_away) / 400)).
* Games between different leagues additionally move the league ratings by K_league * G * (S - E),
  so European results propagate to every club in the league.
* K is boosted for new clubs / leagues (burn-in): K * (1 + 2 * exp(-n / 15)).
* A club entering a league (promotion, or a league that enters the dataset in 2024) starts at the
  20th-percentile offset of that league, or at its existing standalone rating if it has one.
* Absolute ratings are only meaningful for leagues linked to others by enough cross-league games;
  `elo_abs_ok` is False (and features use NaN) when a league has < MIN_CROSS cross-league games.
* Strictly sequential: the rating at date t uses only games before t. Hyper-parameters (K, HFA) are
  tuned on games before the training cut-off only.
"""
from __future__ import annotations

import itertools
import logging
import math
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

from fv.config import load_config, path, setup_logging

log = logging.getLogger(__name__)

MIN_CROSS = 30
INIT = 1500.0
UEFA = {"CL", "CLQ", "EL", "ELQ", "UCOL", "ECLQ", "USC", "KLUB"}


def load_games() -> pd.DataFrame:
    d = path("interim") / "transfermarkt"
    con = duckdb.connect()
    g = con.execute(f"""
      SELECT g.game_id, g.date, g.competition_id, coalesce(g.competition_type, 'other') AS ctype,
             g.home_club_id, g.away_club_id, g.home_club_goals hg, g.away_club_goals ag, c.country_name
      FROM read_parquet('{d / 'games.parquet'}') g
      LEFT JOIN read_parquet('{d / 'competitions.parquet'}') c USING (competition_id)
      WHERE coalesce(g.competition_type, '') <> 'national_team_competition'
        AND g.home_club_id IS NOT NULL AND g.away_club_id IS NOT NULL
      ORDER BY g.date, g.game_id""").df()
    # country -> top-flight league id (for placing lower-division cup opponents)
    top = con.execute(f"""SELECT country_name, competition_id FROM read_parquet('{d / 'competitions.parquet'}')
                          WHERE type = 'domestic_league'""").df()
    g.attrs["country_league"] = dict(zip(top.country_name, top.competition_id))
    return g


def _g_mult(gd: int) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


def run(games: pd.DataFrame, k_club: float, k_league: float, hfa: float,
        record: bool = True) -> tuple[pd.DataFrame | None, dict]:
    """Process games in order. Returns (per-game pre-ratings, scores on 2014-07..train_end games)."""
    country_league = games.attrs["country_league"]
    league_r: dict[str, float] = {}
    league_n: dict[str, int] = defaultdict(int)          # cross-league games played by the league
    offset: dict[int, float] = {}
    alone: dict[int, float] = {}                         # standalone ratings (no league)
    club_league: dict[int, str] = {}
    n_games: dict[int, int] = defaultdict(int)
    members: dict[str, set] = defaultdict(set)

    score_from = pd.Timestamp("2014-07-01")
    score_to = pd.Timestamp(load_config()["splits"]["train_end"])
    se, ns, sx, nx = 0.0, 0, 0.0, 0
    out = []

    def rating(c: int) -> float:
        if c in club_league:
            return league_r[club_league[c]] + offset[c]
        return alone.get(c, np.nan)

    def place(c: int, league: str) -> None:
        """Put club c into league (promotion / first sighting / league joins dataset)."""
        prev = rating(c) if (c in club_league or c in alone) else None
        if league not in league_r:
            # league rating: mean of standalone ratings of its founding members if any, else INIT
            league_r[league] = INIT
        if c in club_league:
            members[club_league[c]].discard(c)
        offs = [offset[m] for m in members[league]]
        if prev is not None and len(members[league]) == 0:
            league_r[league] = prev
            offset[c] = 0.0
        elif prev is not None:
            offset[c] = prev - league_r[league]
        else:
            offset[c] = float(np.percentile(offs, 20)) if len(offs) >= 5 else 0.0
        club_league[c] = league
        members[league].add(c)
        alone.pop(c, None)

    def first_seen(c: int, row) -> None:
        # standalone: European newcomer -> 25th pct of current ratings; domestic cup -> top flight - 150
        if row.competition_id in UEFA:
            cur = [rating(x) for x in list(club_league)[:2000]]
            alone[c] = float(np.nanpercentile(cur, 25)) if cur else INIT - 100
        else:
            lg = country_league.get(row.country_name)
            alone[c] = (league_r.get(lg, INIT)) - 150

    for row in games.itertuples(index=False):
        h, a = row.home_club_id, row.away_club_id
        if row.ctype == "domestic_league":
            for c in (h, a):
                if club_league.get(c) != row.competition_id:
                    place(c, row.competition_id)
        for c in (h, a):
            if c not in club_league and c not in alone:
                first_seen(c, row)
        rh, ra = rating(h), rating(a)
        lh, la = club_league.get(h), club_league.get(a)
        e = 1.0 / (1.0 + 10 ** (-(rh + hfa - ra) / 400))
        s = 1.0 if row.hg > row.ag else 0.5 if row.hg == row.ag else 0.0
        if record:
            out.append((row.game_id, row.date, h, a, rh, ra,
                        offset.get(h, np.nan), offset.get(a, np.nan), lh, la,
                        league_n[lh] if lh else 0, league_n[la] if la else 0))
        if score_from <= row.date < score_to:
            se += (s - e) ** 2
            ns += 1
            if lh != la and lh and la:  # cross-league among covered leagues
                sx += (s - e) ** 2
                nx += 1
        gm = _g_mult(row.hg - row.ag)
        # A league club's rating (and its league's) only moves on games against other league clubs:
        # standalone opponents (lower divisions in cups, uncovered European leagues) are uncalibrated
        # and would otherwise inflate leagues with many such games. Standalone clubs always update.
        both_league = lh is not None and la is not None
        for c, sign in ((h, 1), (a, -1)):
            k = k_club * (1 + 2 * math.exp(-n_games[c] / 15))
            d = sign * k * gm * (s - e)
            if c in club_league:
                if both_league:
                    offset[c] += d
            else:
                alone[c] += d
            n_games[c] += 1
        if both_league and lh != la:
            for lg, sign in ((lh, 1), (la, -1)):
                kl = k_league * (1 + 2 * math.exp(-league_n[lg] / 30))
                league_r[lg] += sign * kl * gm * (s - e)
                league_n[lg] += 1
        if not record and row.date >= score_to:
            break

    pre = None
    if record:
        pre = pd.DataFrame(out, columns=["game_id", "date", "home_club_id", "away_club_id",
                                         "home_elo", "away_elo", "home_elo_rel", "away_elo_rel",
                                         "home_league", "away_league", "home_league_cross_n",
                                         "away_league_cross_n"])
    final = {lg: league_r[lg] for lg in league_r}
    return pre, {"mse": se / max(ns, 1), "n": ns, "mse_cross": sx / max(nx, 1), "n_cross": nx,
                 "league_ratings": final, "league_cross_n": dict(league_n)}


def tune(games: pd.DataFrame) -> dict:
    """Club K and home advantage on all games; league K on cross-league games (train era only)."""
    best = None
    for kc, hfa in itertools.product([6, 8, 10, 12, 15], [40, 50, 60, 70]):
        _, sc = run(games, kc, 2, hfa, record=False)
        log.info("K_club=%s HFA=%s  MSE=%.5f (n=%d)", kc, hfa, sc["mse"], sc["n"])
        if best is None or sc["mse"] < best[0]:
            best = (sc["mse"], kc, hfa)
    _, kc, hfa = best
    bestl = None
    for kl in [1, 2, 4, 8, 12, 16]:
        _, sc = run(games, kc, kl, hfa, record=False)
        log.info("K_league=%s  cross-league MSE=%.5f (n=%d)", kl, sc["mse_cross"], sc["n_cross"])
        if bestl is None or sc["mse_cross"] < bestl[0]:
            bestl = (sc["mse_cross"], kl, sc["mse"])
    params = dict(k_club=kc, k_league=bestl[1], hfa=hfa, mse=bestl[2], mse_cross=bestl[0])
    log.info("best %s", params)
    return params


def build() -> None:
    games = load_games()
    params = tune(games)
    pre, sc = run(games, params["k_club"], params["k_league"], params["hfa"])
    lr = pd.DataFrame({"league": list(sc["league_ratings"]),
                       "rating_end": list(sc["league_ratings"].values())})
    lr["cross_games"] = lr.league.map(sc["league_cross_n"]).fillna(0).astype(int)
    lr.sort_values("rating_end", ascending=False).to_csv(path("processed") / "elo_league_ratings.csv",
                                                          index=False)
    # naive reference: always predict the train-era mean result for the home side
    sc_ref = games[(games.date >= "2014-07-01") & (games.date < load_config()["splits"]["train_end"])]
    s_ref = np.where(sc_ref.hg > sc_ref.ag, 1.0, np.where(sc_ref.hg == sc_ref.ag, 0.5, 0.0))
    params["mse_naive_home_mean"] = float(((s_ref - s_ref.mean()) ** 2).mean())
    for side in ("home", "away"):
        ok = pre[f"{side}_league_cross_n"].ge(MIN_CROSS) | pre[f"{side}_league"].isna()
        pre[f"{side}_elo_abs_ok"] = ok
    pre.to_parquet(path("processed") / "elo_games.parquet", index=False)
    # long per-club table of pre-game ratings (rating known going INTO that game)
    long = pd.concat([
        pre.rename(columns={f"{s}_club_id": "club_id", f"{s}_elo": "elo", f"{s}_elo_rel": "elo_rel",
                            f"{s}_league": "league", f"{s}_elo_abs_ok": "elo_abs_ok"})
           [["game_id", "date", "club_id", "elo", "elo_rel", "league", "elo_abs_ok"]]
        for s in ("home", "away")]).sort_values(["club_id", "date"])
    long.to_parquet(path("processed") / "elo_club_history.parquet", index=False)
    pd.Series(params).to_json(path("processed") / "elo_params.json")
    log.info("wrote elo tables: %d games, %d clubs", len(pre), long.club_id.nunique())


def main() -> None:
    setup_logging()
    build()


if __name__ == "__main__":
    main()
