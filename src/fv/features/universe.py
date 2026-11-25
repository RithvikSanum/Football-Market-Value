"""The modelling universe: permanent transfers with a known fee >= threshold, with a filter funnel."""
from __future__ import annotations

import logging

import pandas as pd

from fv.config import load_config, path

log = logging.getLogger(__name__)


def data_cutoff(con) -> str:
    return str(con.execute("SELECT max(date) FROM games").fetchone()[0])


def build_universe(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    min_fee = cfg["training"]["min_fee_eur"]
    start = cfg["splits"]["train_start"]
    cutoff = data_cutoff(con)
    t = con.execute("SELECT * FROM transfers").df()
    steps = [("all transfer rows", len(t))]

    def step(name: str, mask: pd.Series) -> None:
        nonlocal t
        t = t[mask]
        steps.append((name, len(t)))

    step(f"dated on/before data cutoff {cutoff}", t.transfer_date <= pd.Timestamp(cutoff))
    step(f"dated on/after training-window start {start}", t.transfer_date >= pd.Timestamp(start))
    step("fee known (not NaN)", t.transfer_fee.notna())
    step("fee > 0 (drops loans, end-of-loan, free)", t.transfer_fee > 0)
    step(f"fee >= €{min_fee:,.0f}", t.transfer_fee >= min_fee)
    step("selling club != buying club", t.from_club_id != t.to_club_id)
    t = t.drop_duplicates(["player_id", "transfer_date", "from_club_id", "to_club_id"])
    steps.append(("de-duplicated", len(t)))

    t = t.reset_index(drop=True)
    t["transfer_id"] = (t.player_id.astype(str) + "_" + t.transfer_date.dt.strftime("%Y%m%d")
                        + "_" + t.from_club_id.astype(str) + "_" + t.to_club_id.astype(str))
    funnel = pd.DataFrame(steps, columns=["filter", "rows remaining"])
    funnel["removed"] = (-funnel["rows remaining"].diff()).fillna(0).astype(int)
    fx = pd.read_parquet(path("processed") / "transfers_fx.parquet",
                         columns=["player_id", "transfer_date", "from_club_id", "to_club_id",
                                  "gbp_per_eur", "fee_gbp"]).drop_duplicates(
        ["player_id", "transfer_date", "from_club_id", "to_club_id"])
    t = t.merge(fx, on=["player_id", "transfer_date", "from_club_id", "to_club_id"], how="left")
    t = t.rename(columns={"transfer_fee": "fee_eur", "market_value_in_eur": "tm_value_transfers_table"})
    return t, funnel
