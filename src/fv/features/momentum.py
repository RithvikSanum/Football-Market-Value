"""Groups G (market momentum) and H (crowd value): Transfermarkt valuations dated >= lag days before t.

Excluded from the "stats only" model. `tm_value_transfers_table` (transfers.market_value_in_eur) is
kept for the report comparison only and is never a model feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_features(con, lag_days: int) -> pd.DataFrame:
    df = con.execute(f"""
    WITH tg AS (SELECT *, t - INTERVAL {lag_days} DAY AS t0 FROM targets),
    v AS (SELECT * FROM player_valuations WHERE market_value_in_eur > 0)
    SELECT tg.transfer_id,
      (SELECT arg_max(v.market_value_in_eur, v.date) FROM v WHERE v.player_id = tg.player_id AND v.date <= tg.t0) AS tm_lag,
      (SELECT max(v.date) FROM v WHERE v.player_id = tg.player_id AND v.date <= tg.t0) AS tm_lag_date,
      (SELECT arg_max(v.market_value_in_eur, v.date) FROM v WHERE v.player_id = tg.player_id
         AND v.date <= tg.t0 - INTERVAL 182 DAY) AS tm_6m,
      (SELECT arg_max(v.market_value_in_eur, v.date) FROM v WHERE v.player_id = tg.player_id
         AND v.date <= tg.t0 - INTERVAL 365 DAY) AS tm_12m,
      (SELECT max(v.market_value_in_eur) FROM v WHERE v.player_id = tg.player_id AND v.date <= tg.t0) AS tm_high
    FROM tg""").df()
    df["log_tm_lag"] = np.log(df.tm_lag)
    df["tm_dlog_6m"] = np.log(df.tm_lag / df.tm_6m)
    df["tm_dlog_12m"] = np.log(df.tm_lag / df.tm_12m)
    df["tm_dist_high"] = np.log(df.tm_lag / df.tm_high)
    return df.drop(columns=["tm_6m", "tm_12m", "tm_high"])
