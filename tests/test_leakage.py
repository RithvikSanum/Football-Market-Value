"""Leakage tests (brief section 4): every feature uses only information available before the transfer."""
import numpy as np
import pandas as pd
import pytest

from fv.config import load_config, path

FEATS = path("processed") / "features.parquet"
pytestmark = pytest.mark.skipif(not FEATS.exists(), reason="run make features first")

# to_league is the buying club's league for the coming season (read from its next fixture; known when
# the deal is done). Features derived from it are compared only when both builds agree on to_league.
TO_LEAGUE_DERIVED = {"to_league", "pl_buyer", "cross_border", "buying_league_mins"}
DATE_COLS = {"from_elo_date", "to_elo_date", "last_app_365", "last_app_any", "tm_lag_date"}


@pytest.fixture(scope="module")
def feats() -> pd.DataFrame:
    f = pd.read_parquet(FEATS)
    f["transfer_date"] = pd.to_datetime(f.transfer_date)
    return f


@pytest.fixture(scope="module")
def sample(feats) -> pd.DataFrame:
    # stratified over years so early, validation and test-period rows are all covered
    return feats.groupby(feats.transfer_date.dt.year, group_keys=False).apply(
        lambda g: g.sample(1, random_state=7)).head(14)


def test_feature_dates_precede_transfer(feats):
    lag = load_config()["training"]["tm_value_lag_days"]
    for c in DATE_COLS - {"tm_lag_date"}:
        d = pd.to_datetime(feats[c])
        assert (d.isna() | (d < feats.transfer_date)).all(), c
    tm = pd.to_datetime(feats.tm_lag_date)
    assert (tm.isna() | (tm <= feats.transfer_date - pd.Timedelta(days=lag))).all()


def test_features_identical_when_future_data_removed(feats, sample):
    from fv.features import aging, sources
    from fv.features.build import build_features

    curves = pd.read_csv(path("processed") / "aging_curves.csv")
    for _, row in sample.iterrows():
        t = row.transfer_date
        con = sources.connect(cutoff=str(t.date()))        # every dated table truncated at < t
        tg = pd.DataFrame([{"transfer_id": row.transfer_id, "player_id": row.player_id, "t": t,
                            "from_club_id": row.from_club_id, "to_club_id": row.to_club_id}])
        got = build_features(con, tg, curves).iloc[0]
        same_to = got["to_league"] == row["to_league"]
        for c in got.index:
            if c in ("t", "transfer_id") or (c in TO_LEAGUE_DERIVED and not same_to) or c == "to_league":
                continue
            a, b = got[c], row[c]
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (float, np.floating, int, np.integer)) and not isinstance(a, bool):
                assert np.isclose(float(a), float(b), rtol=1e-9, atol=1e-9), (row.transfer_id, c, a, b)
            else:
                assert str(a) == str(b), (row.transfer_id, c, a, b)


def test_elo_is_sequential():
    """Ratings going into games before T must not change when games from T onwards are removed."""
    from fv.features.elo import load_games, run
    import json

    p = json.loads((path("processed") / "elo_params.json").read_text())
    games = load_games()
    T = pd.Timestamp("2019-01-01")
    full, _ = run(games, p["k_club"], p["k_league"], p["hfa"])
    cut = games[games.date < T].copy()
    cut.attrs = games.attrs
    part, _ = run(cut, p["k_club"], p["k_league"], p["hfa"])
    a = full[full.date < T].set_index("game_id")[["home_elo", "away_elo"]]
    b = part.set_index("game_id")[["home_elo", "away_elo"]]
    assert np.allclose(a.loc[b.index].values, b.values, equal_nan=True)


@pytest.mark.parametrize("cut", ["2019-08-15", "2021-01-20", "2023-07-31"])
def test_market_index_uses_only_earlier_deals(feats, cut):
    """Index level for deals on date T is unchanged when all deals after T are removed."""
    from fv.features.market_index import rolling_hedonic
    from fv.features.base import league_coverage

    cfg = load_config()
    tm = pd.read_parquet(path("processed") / "tm_level_monthly.parquet")
    cov = league_coverage()
    long_cov = {k for k, v in cov.items() if v <= pd.Timestamp("2012-12-31")}
    T = pd.Timestamp(cut)
    on_t = feats.transfer_date.between(T - pd.Timedelta(days=6), T)
    T = feats.loc[on_t, "transfer_date"].max()
    full, _ = rolling_hedonic(feats, tm, cfg, est=feats.to_league.isin(long_cov))
    f2 = feats[feats.transfer_date <= T]
    part, _ = rolling_hedonic(f2, tm, cfg, est=f2.to_league.isin(long_cov))
    rows = f2.index[f2.transfer_date == T]
    for c in ("lvl_hedonic", "lvl_hedonic_now", "lvl_hedonic_seg"):
        assert np.allclose(full.loc[rows, c].astype(float), part.loc[rows, c].astype(float)), c


def test_crowd_ratio_uses_only_earlier_deals(feats):
    from fv.features.market_index import crowd_ratio

    full = crowd_ratio(feats)
    T = pd.Timestamp("2021-08-31")
    f2 = feats[feats.transfer_date <= T]
    part = crowd_ratio(f2)
    rows = f2.index[f2.transfer_date == f2.transfer_date.max()]
    assert np.allclose(full.loc[rows], part.loc[rows], equal_nan=True)


def test_availability_uses_only_earlier_matches(feats):
    """Inferred-unavailability features (incl. ongoing spells) are unchanged when data from t on is removed."""
    from fv.features import sources
    from fv.features.availability import AVAIL_COLS, availability_features

    pick = feats[(feats.long_spells_24m > 0) | (feats.unavail_now == 1)]
    pick = pd.concat([pick[pick.unavail_now == 1].head(4), pick.sample(6, random_state=3)])
    assert len(pick) >= 6
    for _, row in pick.iterrows():
        con = sources.connect(cutoff=str(row.transfer_date.date()))
        con.register("tg", pd.DataFrame([{"transfer_id": row.transfer_id, "player_id": row.player_id,
                                          "t": row.transfer_date, "from_club_id": row.from_club_id}]))
        con.execute("CREATE TEMP TABLE targets AS SELECT * FROM tg")
        got = availability_features(con).iloc[0]
        for c in AVAIL_COLS:
            assert np.isclose(got[c], row[c], equal_nan=True), (row.transfer_id, c, got[c], row[c])
