"""Group B: performance over the last 365 / 730 days (appearances + lineups), all strictly before t.

Minutes share = domestic-league minutes / (90 x the club's league games since the player was first
seen in that club's matchday squad within the window). Covers only competitions in the dataset
(top flights of covered leagues, cups, Europe); players from uncovered leagues get NaN, which
LightGBM handles natively.
"""
from __future__ import annotations

import pandas as pd


def _share_sql(lo: int, hi: int, name: str) -> str:
    """Minutes share over [t - lo days, t - hi days)."""
    return f"""
    {name}_seen AS (
      SELECT tg.transfer_id, l.club_id, min(l.date) AS first_seen
      FROM tg JOIN game_lineups l ON l.player_id = tg.player_id
        AND l.date >= tg.t - INTERVAL {lo} DAY AND l.date < tg.t - INTERVAL {hi} DAY
      GROUP BY 1, 2),
    {name}_team AS (
      SELECT s.transfer_id, count(*) AS team_games
      FROM {name}_seen s JOIN tg USING (transfer_id)
      JOIN club_league_games g ON g.club_id = s.club_id AND g.date >= s.first_seen
        AND g.date < tg.t - INTERVAL {hi} DAY
      GROUP BY 1),
    {name}_mins AS (
      SELECT tg.transfer_id, sum(a.minutes_played) AS league_mins
      FROM tg JOIN appearances a ON a.player_id = tg.player_id
        AND a.date >= tg.t - INTERVAL {lo} DAY AND a.date < tg.t - INTERVAL {hi} DAY
      JOIN games g ON g.game_id = a.game_id AND g.competition_type = 'domestic_league'
      GROUP BY 1),
    {name} AS (
      SELECT tm.transfer_id, coalesce(m.league_mins, 0) / (90.0 * tm.team_games) AS {name}
      FROM {name}_team tm LEFT JOIN {name}_mins m USING (transfer_id))"""


def _agg_sql(days: int) -> str:
    s = f"_{days}"
    return f"""
    agg{s} AS (
      SELECT tg.transfer_id,
        sum(a.minutes_played) AS mins{s}, count(*) AS apps{s},
        sum(a.goals) AS goals{s}, sum(a.assists) AS assists{s},
        sum(a.yellow_cards) AS yc{s}, sum(a.red_cards) AS rc{s}, max(a.date) AS last_app{s}
      FROM tg JOIN appearances a ON a.player_id = tg.player_id
        AND a.date >= tg.t - INTERVAL {days} DAY AND a.date < tg.t
      GROUP BY 1),
    st{s} AS (
      SELECT tg.transfer_id, count(*) AS starts{s}
      FROM tg JOIN game_lineups l ON l.player_id = tg.player_id AND l.type = 'starting_lineup'
        AND l.date >= tg.t - INTERVAL {days} DAY AND l.date < tg.t
      GROUP BY 1)"""


def performance_features(con) -> pd.DataFrame:
    sql = "WITH tg AS (SELECT * FROM targets)," + ",".join([
        _agg_sql(365), _agg_sql(730),
        _share_sql(365, 0, "min_share_365"), _share_sql(730, 0, "min_share_730"),
        _share_sql(182, 0, "min_share_recent"), _share_sql(365, 182, "min_share_prior"),
    ]) + """
    SELECT tg.transfer_id, agg_365.*, agg_730.*, st_365.starts_365, st_730.starts_730,
           min_share_365.min_share_365, min_share_730.min_share_730,
           min_share_recent.min_share_recent, min_share_prior.min_share_prior
    FROM tg LEFT JOIN agg_365 USING (transfer_id) LEFT JOIN agg_730 USING (transfer_id)
    LEFT JOIN st_365 USING (transfer_id) LEFT JOIN st_730 USING (transfer_id)
    LEFT JOIN min_share_365 USING (transfer_id) LEFT JOIN min_share_730 USING (transfer_id)
    LEFT JOIN min_share_recent USING (transfer_id) LEFT JOIN min_share_prior USING (transfer_id)
    """
    df = con.execute(sql).df()
    df = df.loc[:, ~df.columns.duplicated()]
    for d in (365, 730):
        m = df[f"mins_{d}"]
        ok = m >= 90
        for c in ("goals", "assists", "yc", "rc"):
            df[f"{c}_p90_{d}"] = (df[f"{c}_{d}"] * 90 / m).where(ok)
        df[f"ga_p90_{d}"] = df[f"goals_p90_{d}"] + df[f"assists_p90_{d}"]
        df[f"mins_{d}"] = m.fillna(0)
        df[f"starts_{d}"] = df[f"starts_{d}"].fillna(0)
        df = df.drop(columns=[f"{c}_{d}" for c in ("goals", "assists", "yc", "rc")] + [f"apps_{d}"])
    for c in ("min_share_365", "min_share_730", "min_share_recent", "min_share_prior"):
        df[c] = df[c].clip(upper=1.0)
    df["min_share_trend"] = df.min_share_recent - df.min_share_prior
    df = df.drop(columns=["min_share_recent", "min_share_prior", "last_app_730"])
    return df
