"""Build the transfer feature table: data/processed/features.parquet."""
from __future__ import annotations

import logging

import pandas as pd

from fv.config import load_config, path, setup_logging
from fv.features import aging, sources
from fv.features.availability import AVAIL_COLS, availability_features
from fv.features.base import base_features
from fv.features.hype import HYPE_COLS, add_abnormal_hype, hype_features
from fv.features.experience import experience_features, finish_experience
from fv.features.momentum import momentum_features
from fv.features.performance import performance_features
from fv.features.universe import build_universe, data_cutoff

log = logging.getLogger(__name__)

# Factor groups (brief section 6). Phase 2 adds E (hype), F (availability), I (pace/profile).
GROUPS: dict[str, list[str]] = {
    "A_base": ["age", "position", "sub_position", "foot", "height_in_cm", "from_league", "to_league",
               "cross_border", "pl_buyer", "from_elo", "from_elo_rel", "to_elo", "to_elo_rel",
               "elo_diff", "window", "late_window", "tenure_years"],
    "B_performance": ["mins_365", "starts_365", "goals_p90_365", "assists_p90_365", "ga_p90_365",
                      "yc_p90_365", "rc_p90_365", "mins_730", "starts_730", "goals_p90_730",
                      "assists_p90_730", "ga_p90_730", "min_share_365", "min_share_730",
                      "min_share_trend"],
    "C_experience": ["career_league_mins", "adj_mins", "euro_mins", "big_game_mins",
                     "buying_league_mins", "exp_per_year"],
    "D_age_index": ["age_index", "age_curve_fwd2"],
    # English Wikipedia views (abnormal_hype is fitted on training-period transfers in main())
    "E_hype": [*HYPE_COLS, "abnormal_hype"],
    # inferred unavailability from lineups (not confirmed injuries)
    "F_availability": AVAIL_COLS,
    "G_momentum": ["tm_dlog_6m", "tm_dlog_12m", "tm_dist_high"],
    # TM value >= 30 days before, times the real-time median fee/TM ratio of the prior 365 days
    # (log_tm_adj, built in fv.models.pricing.load_dataset from market_index.crowd_ratio)
    "H_crowd": ["log_tm_adj"],
}
CATEGORICAL = ["position", "sub_position", "foot", "from_league", "to_league", "window"]
PHASE1_STATS = ["A_base", "B_performance", "C_experience", "D_age_index"]
# Phase 2: hype (E) and inferred availability (F) are not TM-derived, so both models get them.
# Pace / archetypes (I) are pending a SoFIFA alternative (SoFIFA blocks automated access).
STATS_ONLY = PHASE1_STATS + ["E_hype", "F_availability"]
TM_CORRECTED = STATS_ONLY + ["G_momentum", "H_crowd"]


def features_for(groups: list[str]) -> list[str]:
    return [f for g in groups for f in GROUPS[g]]


def build_features(con, targets: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """targets: transfer_id, player_id, t, from_club_id, to_club_id. Reads only `con` views."""
    cfg = load_config()
    con.register("targets_df", targets[["transfer_id", "player_id", "t", "from_club_id", "to_club_id"]])
    con.execute("CREATE OR REPLACE TEMP TABLE targets AS SELECT * FROM targets_df")
    b = base_features(con)
    con.register("tl", b[["transfer_id", "to_league"]])
    con.execute("CREATE OR REPLACE TEMP TABLE targets AS SELECT t.*, tl.to_league FROM targets_df t "
                "JOIN tl USING (transfer_id)")
    p = performance_features(con)
    e = experience_features(con, cfg["splits"]["train_start"])
    m = momentum_features(con, cfg["training"]["tm_value_lag_days"])
    av = availability_features(con)
    hy = hype_features(con)
    df = b.merge(p, on="transfer_id", how="left").merge(e, on="transfer_id", how="left") \
          .merge(m, on="transfer_id", how="left").merge(av, on="transfer_id", how="left") \
          .merge(hy, on="transfer_id", how="left")
    df = finish_experience(df, cfg["splits"]["train_start"])
    df = aging.add_age_index(df, curves)
    return df


def main() -> None:
    setup_logging()
    cfg = load_config()
    con = sources.connect()
    uni, funnel = build_universe(con)
    funnel.to_csv(path("processed") / "filter_funnel.csv", index=False)
    log.info("universe: %d transfers", len(uni))
    curves = aging.fit_curves(con, cfg["splits"]["train_end"])
    targets = uni.rename(columns={"transfer_date": "t"})
    feats = build_features(con, targets, curves)
    keep = ["transfer_id", "player_id", "player_name", "transfer_date", "transfer_season",
            "from_club_id", "to_club_id", "from_club_name", "to_club_name", "fee_eur", "fee_gbp",
            "gbp_per_eur", "tm_value_transfers_table"]
    out = uni[keep].merge(feats.drop(columns=["t"]), on="transfer_id", how="left")
    td = pd.to_datetime(out.transfer_date)
    train = (td >= cfg["splits"]["train_start"]) & (td < cfg["splits"]["train_end"])
    out, info = add_abnormal_hype(out, train)
    log.info("abnormal hype regression (train rows): %s", info)
    out.to_parquet(path("processed") / "features.parquet", index=False)
    log.info("wrote features: %s", out.shape)


if __name__ == "__main__":
    main()
