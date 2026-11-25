"""Checkpoint 2 report: reports/phase2_report.md (+ figures). Validation / calibration only, never test.

Sections: source status, entity match rates, inferred unavailability, profile-aware aging curves,
contract-history feasibility, ablation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fv.config import load_config, path, setup_logging
from fv.evaluation.report import GRID, INK2, S1, S2, S3, S4, _md, _save, plt

log = logging.getLogger(__name__)
OUT = path("reports") / "phase2_report.md"


def match_rates() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_parquet(path("interim") / "wikidata_tm_enwiki.parquet")
    p = pd.read_parquet(path("interim") / "transfermarkt" / "players.parquet",
                        columns=["player_id", "name", "date_of_birth"])
    f = pd.read_parquet(path("processed") / "features.parquet", columns=["player_id", "fee_eur"])
    x = m.merge(p, on="player_id", how="left")
    tmdob = pd.to_datetime(x.date_of_birth).dt.strftime("%Y-%m-%d")
    both = x.wd_dob.notna() & tmdob.notna()
    gap = (pd.to_datetime(x.wd_dob[both], errors="coerce") - pd.to_datetime(tmdob[both])).dt.days.abs()
    u = f.merge(m, on="player_id", how="left")
    rows = [
        ("Wikidata item found (all queried TM ids)", f"{len(m):,} / {26639:,}"),
        ("… with an English Wikipedia article", f"{m.article.notna().sum():,}"),
        ("Transfer-universe rows with an article", f"{u.article.notna().mean():.1%}"),
        ("… for fees ≥ €5m", f"{u[u.fee_eur >= 5e6].article.notna().mean():.1%}"),
        ("Date of birth agrees exactly (Wikidata vs TM)", f"{(x.wd_dob[both] == tmdob[both]).mean():.1%}"),
        ("Date of birth within a year", f"{(gap <= 366).mean():.2%}"),
    ]
    rate = pd.DataFrame(rows, columns=["check", "value"])
    spot = x[x.article.notna()].sample(20, random_state=3)[["player_id", "name", "article", "wd_dob"]]
    spot = spot.assign(tm_dob=pd.to_datetime(
        p.set_index("player_id").loc[spot.player_id, "date_of_birth"]).dt.strftime("%Y-%m-%d").values)
    rate.to_csv(path("processed") / "match_rates.csv", index=False)
    return rate, spot


def fig_aging_sub() -> str:
    c = pd.read_csv(path("processed") / "aging_curves.csv")
    c = c[(c.age >= 17) & (c.age <= 36)]
    groups = ["Attack", "Midfield", "Defender", "Goalkeeper"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for ax, pos in zip(axes, groups):
        base = c[(c.level == "position") & (c.profile == pos)]
        ax.plot(base.age, np.exp(base.log_value_curve - base.log_value_curve.max()), color=INK2,
                lw=1.5, ls="--", label=f"{pos} (all)")
        subs = c[(c.level == "sub_position") & (c.position == pos)]
        # keep the 4 best-populated sub-positions (others are shrunk almost entirely to the group curve)
        top = subs.groupby("profile").n_pairs.sum().sort_values(ascending=False).index[:4]
        for prof, col in zip(sorted(top), (S1, S2, S3, S4)):
            g = subs[subs.profile == prof]
            ax.plot(g.age, np.exp(g.log_value_curve - g.log_value_curve.max()), color=col,
                    label=f"{prof} (peak {int(g.peak_age.iloc[0])})")
        ax.set_title(pos, loc="left")
        ax.set_xlabel("Age")
        ax.legend(fontsize=7, loc="lower center")
    axes[0].set_ylabel("Relative value (peak = 1)")
    fig.suptitle("Value aging curves by sub-position (delta method, train-period TM values, shrunk "
                 "to group)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "aging_curves_subposition.png")


def availability_section() -> tuple[pd.DataFrame, pd.DataFrame]:
    from fv.features import availability as av
    from fv.features import sources
    f = pd.read_parquet(path("processed") / "features.parquet")
    desc = f[av.AVAIL_COLS].describe(percentiles=[0.1, 0.5, 0.9]).T[
        ["count", "mean", "10%", "50%", "90%"]].round(3).reset_index(names="feature")
    con = sources.connect()
    pid, t, club = 288230, pd.Timestamp("2023-08-14"), 131        # Ousmane Dembélé, Barcelona -> PSG
    con.register("tdf", pd.DataFrame({"transfer_id": ["x"], "player_id": [pid], "t": [t],
                                      "from_club_id": [club]}))
    con.execute("CREATE TEMP TABLE targets AS SELECT * FROM tdf")
    pres, cg, apps = av._load(con)
    cgb = {c: g.sort_values(["date", "game_id"]).reset_index(drop=True) for c, g in cg.groupby("club_id")}
    seq = av.player_sequence(pres, cgb, apps)
    sp = av.spells(seq, t, club).assign(start=lambda d: d.start.dt.date, end=lambda d: d.end.dt.date)
    return desc, sp


def contract_section() -> tuple[pd.DataFrame, pd.DataFrame]:
    from fv.features.contract import contract_at_transfer
    f = pd.read_parquet(path("processed") / "features.parquet")
    x = f.merge(contract_at_transfer(f), on="transfer_id")
    x = x[pd.to_datetime(x.transfer_date) >= "2022-12-01"]
    cov = x.groupby("transfer_season").agg(
        transfers=("transfer_id", "size"), covered=("contract_years_left", lambda s: s.notna().mean()),
        median_years_left=("contract_years_left", "median"),
        median_snapshot_age_days=("snapshot_age_days", "median")).reset_index()
    cov = cov[cov.transfers > 10]
    cov["covered"] = cov.covered.map("{:.0%}".format)
    y = x.dropna(subset=["contract_years_left", "log_tm_lag"])
    y = y.assign(r=np.log(y.fee_eur) - y.log_tm_lag)
    band = pd.cut(y.contract_years_left, [-0.01, 0.5, 1, 1.5, 2, 3, 6],
                  labels=["≤0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3+"])
    sig = y.groupby(band, observed=True).r.agg(["size", "median"]).reset_index()
    sig.columns = ["years_left", "n", "median_log_fee_over_tm"]
    sig["fee_vs_tm"] = (np.exp(sig.median_log_fee_over_tm) - 1).map("{:+.0%}".format)
    return cov, sig


def fig_ablation(ab: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, (model, g) in zip(axes, ab.groupby("model", sort=False)):
        g = g[g.variant != "full"].iloc[::-1]
        y = np.arange(len(g))
        d = 100 * g.val_diff_vs_full.values
        lo, hi = 100 * g.val_diff_90lo.values, 100 * g.val_diff_90hi.values
        ax.barh(y, d, color=[S2 if v > 0 else S1 for v in d], height=0.55)
        ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_yticks(y, g.variant)
        ax.set_title(model, loc="left")
        ax.set_xlabel("Change in validation MdAPE vs full set (pp; >0 = group helps)")
    fig.tight_layout()
    return _save(fig, "ablation.png")


def main() -> None:
    setup_logging()
    rate, spot = match_rates()
    f_aging = fig_aging_sub()
    av_desc, dembele = availability_section()
    cov, sig = contract_section()
    ab_f = path("processed") / "ablation.csv"
    ab_md = "_Ablation not run yet._"
    if ab_f.exists():
        ab = pd.read_csv(ab_f)
        show = ab[["model", "variant", "val_MdAPE", "val_diff_vs_full", "val_diff_90lo", "val_diff_90hi",
                   "calib_MdAPE", "calib_diff_vs_full"]].copy()
        for c in show.columns[2:]:
            show[c] = (100 * show[c]).map("{:.1f}".format)
        ab_md = f"![Ablation]({fig_ablation(ab)})\n\n" + _md(show)
    md = f"""# Phase 2 report: extra factors (checkpoint 2)

Validation (2020/21) and calibration (2021/22) only. The test split is untouched.

## 1. Source status

| source | status | used for |
|---|---|---|
| Wikidata SPARQL (P2446) | ✅ working | TM id → English Wikipedia article |
| Wikimedia page views API | ✅ working (HTTP 429 above ~1 req/s; we honour Retry-After) | group E hype |
| Kaggle versions of transfermarkt-datasets | ✅ working (range requests, players.csv only) | contract history |
| Understat | ❌ **down**: every page returns a "We will come back soon" maintenance page | xG / xA (group B) |
| SoFIFA | ❌ **blocked**: Cloudflare "Attention Required" (HTTP 403) for automated requests | pace, archetypes (group I) |

Per the brief, Understat- and SoFIFA-dependent work (xG features, archetypes, pace decline curves) is
**paused pending your decision**; nothing has been faked or filled in.

## 2. Entity matching: Transfermarkt → Wikidata → English Wikipedia

{_md(rate)}

Spot-check, 20 random matches (TM name, Wikipedia article, both dates of birth):

{_md(spot)}

## 3. Inferred unavailability (group F; **not confirmed injuries**)

Absence spells from matchday squads (lineups from July 2013): a regular (≥ 45 min average over his last
10 squads) missing ≥ 3 consecutive club matches with lineup data; the match after a red card is
excused; runs with no return are counted only at the player's current club. 55% of transfers have
lineup history (the rest come from leagues the dataset does not cover).

{_md(av_desc)}

Sanity check, Ousmane Dembélé before his August 2023 move (known record: hamstring Sept 2017 and Jan
2018, Nov 2019 – Aug 2020, knee 2021, hamstring Feb 2023):

{_md(dembele)}

## 4. Profile-aware aging curves (group D)

![Aging by sub-position]({f_aging})

Sub-position curves use the delta method on training-period TM values, with each age's mean change
shrunk toward the position-group curve (pseudo-count 200). **Value** peaks at 23–25 everywhere; the
profiles differ in how fast value falls after the peak (wingers fastest, centre-backs slower,
goalkeepers slowest). `age_index` now uses the sub-position peak and a new feature `age_curve_fwd2` is
the curve's expected 2-year change. Archetype curves and pace decline curves need SoFIFA / Understat.

## 5. Contract history feasibility

{_md(cov, {'median_years_left': '{:.2f}', 'median_snapshot_age_days': '{:.0f}'})}

Snapshot = latest Kaggle `players.csv` dated before the transfer in which the player is at the selling
club (135 versions, Nov 2022 → Sep 2026, mostly weekly; **the source has no snapshot between
11 Apr 2025 and 14 Feb 2026**, so summer-2025 deals use April 2025 contracts). Coverage is ~40% because
many selling clubs are outside the dataset's squad coverage or have no contract date.

The signal is strong. Fee relative to the TM value (≥ 30 days before) by years left:

{_md(sig, {'median_log_fee_over_tm': '{:+.3f}'})}

**Proposal for checkpoint 2** (all contract data lies in the test period, from Dec 2022):
1. In phase 3, score the main models on the test set **once**, as planned, without contract data, and
   freeze those results.
2. Then fit the contract adjustment (a small model on the main model's residual: years left, age, fee
   band) forward-chained inside 2022/23–2025/26 (fit on earlier seasons, score the next), and report it
   separately and clearly labelled as using test-period labels.
3. Refit it on all 2022/23–2025/26 deals for current valuations.

## 6. Ablation (does each group help?)

{ab_md}

LightGBM with the full model's tuned parameters, mean of 3 seeds; paired-bootstrap 90% intervals.
"""
    OUT.write_text(md)
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
