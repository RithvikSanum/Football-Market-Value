"""Group C: experience (career to date, data from 2012-07), strictly before t.

* League-adjusted minutes: each match's minutes x w(opponent Elo), w = clip((Elo - 1200) / 400,
  0.25, 2.0) so a 1600 opponent counts 1.0 and a 2000 opponent 2.0. Opponents without an
  identifiable absolute Elo (leagues with few cross-league games, e.g. Brazil) get w = 0.5.
* Big-game minutes: opponent Elo >= the 75th percentile of pre-game Elo across domestic-league games
  in the 12 months before that month.
* Experience per observable year: data start in 2012-07, so careers are left-truncated; we divide
  league-adjusted minutes by the years observable since max(age 17, 2012-07-01).
* International caps: only current totals exist -> not usable historically (checkpoint 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EURO = ("CL", "EL", "UCOL", "USC")


def _big_game_thresholds(con) -> pd.DataFrame:
    g = con.execute("""
      SELECT date, home_elo AS elo FROM elo_games WHERE home_elo_abs_ok AND home_league IS NOT NULL
      UNION ALL SELECT date, away_elo FROM elo_games WHERE away_elo_abs_ok AND away_league IS NOT NULL
    """).df()
    if g.empty:
        return pd.DataFrame(columns=["month", "big_thr"])
    g["date"] = pd.to_datetime(g.date)
    months = pd.date_range(g.date.min().to_period("M").to_timestamp(), g.date.max(), freq="MS")
    rows = []
    for m in months:
        w = g.elo[(g.date < m) & (g.date >= m - pd.DateOffset(years=1))]
        rows.append((m, float(np.percentile(w, 75)) if len(w) > 500 else np.nan))
    return pd.DataFrame(rows, columns=["month", "big_thr"])


def experience_features(con, start: str) -> pd.DataFrame:
    con.register("big_thr", _big_game_thresholds(con))
    df = con.execute(f"""
    WITH tg AS (SELECT * FROM targets),
    ap AS (
      SELECT a.player_id, a.date, a.minutes_played AS m, a.competition_id, g.competition_type,
             CASE WHEN a.player_club_id = e.home_club_id
                  THEN CASE WHEN e.away_elo_abs_ok THEN e.away_elo END
                  ELSE CASE WHEN e.home_elo_abs_ok THEN e.home_elo END END AS opp_elo
      FROM appearances a JOIN games g USING (game_id) LEFT JOIN elo_games e USING (game_id)),
    ap2 AS (
      SELECT ap.*, coalesce(greatest(least((opp_elo - 1200) / 400.0, 2.0), 0.25), 0.5) AS w,
             (opp_elo >= b.big_thr) AS big
      FROM ap LEFT JOIN big_thr b ON b.month = date_trunc('month', ap.date))
    SELECT tg.transfer_id,
      sum(CASE WHEN ap2.competition_type = 'domestic_league' THEN ap2.m ELSE 0 END) AS career_league_mins,
      sum(ap2.m * ap2.w) AS adj_mins,
      sum(CASE WHEN ap2.competition_id IN {EURO} THEN ap2.m ELSE 0 END) AS euro_mins,
      sum(CASE WHEN ap2.big THEN ap2.m ELSE 0 END) AS big_game_mins,
      sum(CASE WHEN ap2.competition_id = tg.to_league THEN ap2.m ELSE 0 END) AS buying_league_mins,
      max(ap2.date) AS last_app_any
    FROM tg LEFT JOIN ap2 ON ap2.player_id = tg.player_id AND ap2.date < tg.t
    GROUP BY 1
    """).df()
    return df


def finish_experience(df: pd.DataFrame, start: str) -> pd.DataFrame:
    """Needs age and t (from base features)."""
    for c in ("career_league_mins", "adj_mins", "euro_mins", "big_game_mins", "buying_league_mins"):
        df[c] = df[c].fillna(0)
    born17 = df.t - pd.to_timedelta((df.age - 17) * 365.25, unit="D")
    obs_from = born17.where(born17 > pd.Timestamp(start), pd.Timestamp(start))
    years = ((df.t - obs_from).dt.days / 365.25).clip(lower=0.5)
    df["exp_per_year"] = df.adj_mins / years
    df["obs_years"] = years
    return df
