"""Contract length at transfer, reconstructed from dated Kaggle snapshots of players.csv.

For a transfer on date t: take the latest snapshot dated strictly before t in which the player was
registered at the selling club (current_club_id == from_club_id). contract_years_left =
(contract_expiration_date - t) / 365.25, clipped at 0 (an expired date in a stale snapshot = out of
contract / renewal not yet recorded). snapshot_age_days records how stale the snapshot is.
Only transfers from Dec 2022 onward can be covered (first snapshot 13 Nov 2022), and the source has no
snapshot between 11 Apr 2025 and 14 Feb 2026. Used only by the contract-adjustment model (checkpoint 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fv.config import path

MAX_SNAPSHOT_AGE_DAYS = 365


def contract_at_transfer(tr: pd.DataFrame) -> pd.DataFrame:
    """tr: transfer_id, player_id, transfer_date, from_club_id. Returns transfer_id, contract_years_left,
    snapshot_date, snapshot_age_days."""
    s = pd.read_parquet(path("interim") / "contract_snapshots.parquet")
    s = s.dropna(subset=["contract_expiration_date"]).sort_values("snapshot_date")
    s["snapshot_date"] = s.snapshot_date.astype("datetime64[us]")
    s["contract_expiration_date"] = s.contract_expiration_date.astype("datetime64[us]")
    s["current_club_id"] = s.current_club_id.astype("int64")
    t = tr[["transfer_id", "player_id", "transfer_date", "from_club_id"]].copy()
    t["transfer_date"] = pd.to_datetime(t.transfer_date).astype("datetime64[us]")
    t["from_club_id"] = t.from_club_id.astype("int64")
    t["lookup"] = t.transfer_date - pd.Timedelta(days=1)       # strictly before t
    m = pd.merge_asof(t.sort_values("lookup"),
                      s.rename(columns={"current_club_id": "from_club_id"})[
                          ["player_id", "from_club_id", "snapshot_date", "contract_expiration_date"]],
                      left_on="lookup", right_on="snapshot_date", by=["player_id", "from_club_id"],
                      direction="backward")
    m["snapshot_age_days"] = (m.transfer_date - m.snapshot_date).dt.days
    stale = m.snapshot_age_days > MAX_SNAPSHOT_AGE_DAYS
    m.loc[stale, ["snapshot_date", "contract_expiration_date"]] = pd.NaT
    m["contract_years_left"] = ((m.contract_expiration_date - m.transfer_date).dt.days / 365.25).clip(lower=0)
    m.loc[stale, ["snapshot_age_days", "contract_years_left"]] = np.nan
    return m[["transfer_id", "contract_years_left", "snapshot_date", "snapshot_age_days"]]
