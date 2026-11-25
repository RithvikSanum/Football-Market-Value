"""Attach fees in EUR and GBP at the ECB rate on the transfer date.

Finding (brief 15.2): transfermarkt-datasets stores fees natively in EUR. Fees agreed in GBP were
converted by Transfermarkt at (approximately) the rate around the deal date, not restated at one
recent rate, so EUR is treated as the fee currency and GBP is derived with the transfer-date rate.
"""
from __future__ import annotations

import logging

import duckdb

from fv.config import path, setup_logging
from fv.ingest import fx

log = logging.getLogger(__name__)


def build() -> None:
    src = path("interim") / "transfermarkt" / "transfers.parquet"
    out = path("processed") / "transfers_fx.parquet"
    t = duckdb.connect().execute(f"SELECT * FROM read_parquet('{src}')").df()
    t["gbp_per_eur"] = fx.rate_on(t["transfer_date"], "GBP")
    t["fee_eur"] = t["transfer_fee"]
    t["fee_gbp"] = t["fee_eur"] * t["gbp_per_eur"]
    t.to_parquet(out, index=False)
    log.info("wrote %s (%d rows)", out, len(t))


def main() -> None:
    setup_logging()
    fx.download()
    build()


if __name__ == "__main__":
    main()
