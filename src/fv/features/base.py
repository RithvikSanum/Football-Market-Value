"""Group A: base features (bio, leagues, club strength, window, tenure).

Leakage notes
* Selling league = league of the selling club's last domestic league game before the transfer.
* Buying league = league of the buying club's next domestic league game within 200 days (the league a
  club will play in is known when the deal is done: promotion/relegation is settled in May), falling
  back to its last league game before the transfer. This is the one feature allowed to read a date
  after the transfer; the leakage test exempts it explicitly.
* Club Elo = rating going into the club's last game before the transfer (own Elo, fv.features.elo).
* Clubs with no league games before t: if the club's (clubs-table) league is covered only from a later
  date (e.g. Brazil from 2024), use that league; if the league was already covered, the club must have
  been in a lower tier -> "<league>_lower" (e.g. GB1_lower ~ Championship). League coverage start dates
  are dataset metadata (data/processed/league_coverage.csv), not player information.
* Position, foot and height are static profile fields (current profile); treated as time-invariant.
"""
from __future__ import annotations

import pandas as pd

from fv.config import path
from fv.features import sources

from fv.features.windows import late_window, window_idx, window_label

LEAGUE_GAMES = """
  CREATE OR REPLACE TEMP VIEW club_league_games AS
  SELECT game_id, date, competition_id, home_club_id AS club_id FROM games WHERE competition_type = 'domestic_league'
  UNION ALL
  SELECT game_id, date, competition_id, away_club_id FROM games WHERE competition_type = 'domestic_league'
"""


def league_coverage() -> dict[str, pd.Timestamp]:
    """First game date per domestic league in the full dataset (static metadata, built once)."""
    f = path("processed") / "league_coverage.csv"
    if not f.exists():
        c = sources.connect().execute("""SELECT competition_id, min(date) AS start FROM games
            WHERE competition_type = 'domestic_league' GROUP BY 1""").df()
        c.to_csv(f, index=False)
    c = pd.read_csv(f, parse_dates=["start"])
    return dict(zip(c.competition_id, c.start))


def _fill_league(league: pd.Series, club_league: pd.Series, t: pd.Series, cov: dict) -> pd.Series:
    start = club_league.map(cov)
    fill = club_league.where(t < start, club_league + "_lower").where(club_league.notna())
    return league.fillna(fill).fillna("OTHER")


def base_features(con) -> pd.DataFrame:
    con.execute(LEAGUE_GAMES)
    df = con.execute("""
    WITH tg AS (SELECT * FROM targets),
    fl AS (SELECT tg.transfer_id, arg_max(g.competition_id, g.date) AS from_league
           FROM tg JOIN club_league_games g ON g.club_id = tg.from_club_id AND g.date < tg.t GROUP BY 1),
    tl_next AS (SELECT tg.transfer_id, arg_min(g.competition_id, g.date) AS to_league_next
           FROM tg JOIN club_league_games g ON g.club_id = tg.to_club_id
             AND g.date >= tg.t AND g.date < tg.t + INTERVAL 200 DAY GROUP BY 1),
    tl_prev AS (SELECT tg.transfer_id, arg_max(g.competition_id, g.date) AS to_league_prev
           FROM tg JOIN club_league_games g ON g.club_id = tg.to_club_id AND g.date < tg.t GROUP BY 1),
    fe AS (SELECT tg.transfer_id, arg_max(e.elo, e.date) AS from_elo_raw, arg_max(e.elo_rel, e.date) AS from_elo_rel,
                  arg_max(e.elo_abs_ok, e.date) AS from_ok, max(e.date) AS from_elo_date
           FROM tg JOIN elo_club_history e ON e.club_id = tg.from_club_id AND e.date < tg.t GROUP BY 1),
    te AS (SELECT tg.transfer_id, arg_max(e.elo, e.date) AS to_elo_raw, arg_max(e.elo_rel, e.date) AS to_elo_rel,
                  arg_max(e.elo_abs_ok, e.date) AS to_ok, max(e.date) AS to_elo_date
           FROM tg JOIN elo_club_history e ON e.club_id = tg.to_club_id AND e.date < tg.t GROUP BY 1),
    ten AS (SELECT tg.transfer_id, max(x.transfer_date) AS joined_date
           FROM tg JOIN transfers x ON x.player_id = tg.player_id AND x.to_club_id = tg.from_club_id
             AND x.transfer_date < tg.t GROUP BY 1),
    seen AS (SELECT tg.transfer_id, min(l.date) AS first_seen
           FROM tg JOIN game_lineups l ON l.player_id = tg.player_id AND l.club_id = tg.from_club_id
             AND l.date < tg.t GROUP BY 1)
    SELECT tg.transfer_id, tg.t,
           p.date_of_birth, p.position, p.sub_position, p.foot, p.height_in_cm,
           fl.from_league, coalesce(tl_next.to_league_next, tl_prev.to_league_prev) AS to_league,
           CASE WHEN fe.from_ok THEN fe.from_elo_raw END AS from_elo, fe.from_elo_rel, fe.from_elo_date,
           CASE WHEN te.to_ok THEN te.to_elo_raw END AS to_elo, te.to_elo_rel, te.to_elo_date,
           coalesce(ten.joined_date, seen.first_seen) AS joined_from_club,
           cf.domestic_competition_id AS from_club_table_league,
           ct.domestic_competition_id AS to_club_table_league
    FROM tg LEFT JOIN players p USING (player_id)
    LEFT JOIN clubs cf ON cf.club_id = tg.from_club_id LEFT JOIN clubs ct ON ct.club_id = tg.to_club_id
    LEFT JOIN fl USING (transfer_id) LEFT JOIN tl_next USING (transfer_id) LEFT JOIN tl_prev USING (transfer_id)
    LEFT JOIN fe USING (transfer_id) LEFT JOIN te USING (transfer_id)
    LEFT JOIN ten USING (transfer_id) LEFT JOIN seen USING (transfer_id)
    """).df()
    country = dict(con.execute("SELECT competition_id, country_name FROM competitions").fetchall())
    df["age"] = (df.t - pd.to_datetime(df.date_of_birth)).dt.days / 365.25
    cov = league_coverage()
    df["from_league"] = _fill_league(df.from_league, df.from_club_table_league, df.t, cov)
    df["to_league"] = _fill_league(df.to_league, df.to_club_table_league, df.t, cov)
    df = df.drop(columns=["from_club_table_league", "to_club_table_league"])
    fc = df.from_league.str.replace("_lower", "").map(country)
    tc = df.to_league.str.replace("_lower", "").map(country)
    df["cross_border"] = (fc != tc).where(fc.notna() & tc.notna()).astype("float")
    df["pl_buyer"] = (df.to_league == "GB1").astype(int)
    df["window"] = window_label(df.t).str[-1]
    df["window_idx"] = window_idx(df.t)
    df["late_window"] = late_window(df.t).astype(int)
    df["tenure_years"] = (df.t - pd.to_datetime(df.joined_from_club)).dt.days / 365.25
    df["elo_diff"] = df.to_elo - df.from_elo
    df["position"] = df.position.replace({"Missing": None})
    return df.drop(columns=["date_of_birth", "joined_from_club"])
