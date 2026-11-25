import pandas as pd
import pytest

from fv.config import path
from fv.ingest import fx

pytestmark = pytest.mark.skipif(not (path("interim") / "fx.parquet").exists(), reason="run make ingest")


def test_rate_uses_last_fixing_on_or_before_date():
    # 2019-08-03 is a Saturday -> Friday 2019-08-02 fixing; must never look forward.
    rates = fx.load().set_index("date")["gbp_per_eur"]
    got = fx.rate_on(pd.Series(pd.to_datetime(["2019-08-03"])))
    assert got.iloc[0] == rates.loc["2019-08-02"]


def test_known_gbp_deal_matches_transfer_date_rate():
    # Maguire: £80m headline, dataset €87m, Aug 2019. Implied rate within 1% of ECB on the date.
    gbp = 87e6 * fx.rate_on(pd.Series(pd.to_datetime(["2019-08-05"]))).iloc[0]
    assert abs(gbp / 80e6 - 1) < 0.01


def test_rate_preserves_input_order():
    d = pd.Series(pd.to_datetime(["2024-01-02", "2010-01-04", "2018-06-01"]))
    r = fx.rate_on(d)
    assert r.iloc[1] != r.iloc[0] and len(r) == 3
