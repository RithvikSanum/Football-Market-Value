"""Group F: INFERRED UNAVAILABILITY (absence spells from lineups), not confirmed injuries.

Pluggable: `availability_features(con)` returns one row per target with the AVAIL_COLS columns;
a real injury dataset can replace it by producing the same columns.

Definitions (brief section 6F), all from matches strictly before t:
- Club stints come from the player's matchday-squad history (game_lineups): consecutive squad
  appearances for the same club form a block; the block's club matches are that club's games with
  lineup data from the block's first squad date until the next block starts.
- A match is "present" if the player is in the matchday squad (starting XI or bench).
- Regular = averaged >= 45 minutes (50%) over his previous 10 matchday squads at any club (at least 5),
  so a long or repeated injury does not by itself remove regular status, and new signings carry it over.
- The match right after a red card is excused (dropped from the sequence; approximate suspension).
- Spell = run of >= 3 consecutive absences that starts while the player is a regular.
  Long spell = >= 5 matches. Runs are counted only up to t (no future knowledge of length).
- A run with no return to the squad before t is kept only if it is at the player's current club
  (the selling club for transfers) in his latest stint; otherwise it is treated as a departure, and
  those trailing matches are dropped from the denominator.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

AVAIL_COLS = ["avail_365", "avail_730", "avail_1095", "spell_mins_missed_365", "long_spells_24m",
              "months_since_long_spell", "spell_recurrence", "unavail_now"]
MIN_SPELL, LONG_SPELL, REG_WINDOW, REG_SHARE, MIN_PRIOR = 3, 5, 10, 0.5, 5
MONTHS_CAP = 60.0


def _load(con) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pres = con.sql("""
        SELECT l.player_id, l.club_id, l.game_id, l.date
        FROM game_lineups l WHERE l.player_id IN (SELECT DISTINCT player_id FROM targets)""").df()
    clubs = pres.club_id.unique().tolist() or [-1]
    con.register("av_clubs", pd.DataFrame({"club_id": clubs}))
    cg = con.sql("""
        SELECT DISTINCT l.club_id, l.game_id, l.date FROM game_lineups l
        WHERE l.club_id IN (SELECT club_id FROM av_clubs)""").df()
    apps = con.sql("""
        SELECT a.player_id, a.game_id, a.minutes_played AS mins, a.red_cards AS rc
        FROM appearances a WHERE a.player_id IN (SELECT DISTINCT player_id FROM targets)""").df()
    for d in (pres, cg):
        d["date"] = pd.to_datetime(d.date)
    return pres, cg, apps


def player_sequence(pres_p: pd.DataFrame, cg_by_club: dict, apps_p: pd.DataFrame) -> pd.DataFrame:
    """Match-level rows for one player: block, club, date, present, regular, run_id (absence runs)."""
    pres_p = pres_p.sort_values(["date", "game_id"])
    blk = (pres_p.club_id != pres_p.club_id.shift()).cumsum().values
    starts = pres_p.groupby(blk).agg(club_id=("club_id", "first"), start=("date", "first"))
    starts["next_start"] = starts.start.shift(-1).fillna(pd.Timestamp("2200-01-01"))
    present_games = set(zip(pres_p.club_id, pres_p.game_id))
    mins = dict(zip(apps_p.game_id, apps_p.mins.fillna(0)))
    red = set(apps_p.game_id[apps_p.rc.fillna(0) > 0])
    # regular status from the player's own last REG_WINDOW squad appearances (any club), before each date
    pm = np.array([mins.get(gid, 0.0) for gid in pres_p.game_id], float)
    pdates = pres_p.date.values
    pcs = np.r_[0.0, np.cumsum(pm)]
    out = []
    for b, s in starts.iterrows():
        g = cg_by_club.get(s.club_id)
        if g is None:
            continue
        g = g[(g.date >= s.start) & (g.date < s.next_start)]
        if g.empty:
            continue
        pres = np.array([(s.club_id, gid) in present_games for gid in g.game_id])
        m = np.array([mins.get(gid, 0.0) if p else 0.0 for gid, p in zip(g.game_id, pres)], float)
        rc = np.array([gid in red for gid in g.game_id])
        excused = np.r_[False, rc[:-1]] & ~pres          # absence right after a red card
        keep = ~excused
        g, pres, m = g[keep], pres[keep], m[keep]
        k = np.searchsorted(pdates, g.date.values, side="left")   # squad apps strictly before date
        lo = np.maximum(k - REG_WINDOW, 0)
        prior = k - lo
        share = np.where(prior > 0, (pcs[k] - pcs[lo]) / (90.0 * np.maximum(prior, 1)), 0.0)
        regular = (prior >= MIN_PRIOR) & (share >= REG_SHARE)
        out.append(pd.DataFrame({"block": b, "club_id": s.club_id, "date": g.date.values,
                                 "present": pres, "regular": regular}))
    if not out:
        return pd.DataFrame(columns=["block", "club_id", "date", "present", "regular", "run_id"])
    seq = pd.concat(out, ignore_index=True)
    # absence runs within a block; run_id = -1 for present rows
    newrun = (~seq.present) & ((seq.present.shift(fill_value=True)) | (seq.block != seq.block.shift()))
    rid = newrun.cumsum()
    seq["run_id"] = np.where(seq.present, -1, rid)
    first = seq[~seq.present].groupby("run_id").regular.first()
    seq["run_regular"] = seq.run_id.map(first.astype(float)).fillna(0).astype(bool)
    return seq


def features_at(seq: pd.DataFrame, t: pd.Timestamp, current_club) -> dict:
    nan = {c: np.nan for c in AVAIL_COLS}
    s = seq[seq.date < t]
    if s.empty:
        return nan
    last_block = s.block.max()
    is_current = (s.block == last_block) & (s.club_id == current_club)
    # trailing rows after the last present match of each block (unless current stint): departure
    last_pres = s[s.present].groupby("block").date.max()
    lp = s.block.map(last_pres)
    valid = is_current | (s.date <= lp)
    s = s[valid]
    ab = s[(s.run_id >= 0) & s.run_regular]
    runs = ab.groupby("run_id").agg(n=("date", "size"), start=("date", "min"), end=("date", "max"),
                                    block=("block", "first"))
    runs = runs[runs.n >= MIN_SPELL]
    spell_rows = s.run_id.isin(runs.index)
    out = {}
    for days, name in ((365, "avail_365"), (730, "avail_730"), (1095, "avail_1095")):
        w = s.date >= t - pd.Timedelta(days=days)
        den = int(w.sum())
        out[name] = 1 - (spell_rows & w).sum() / den if den >= 5 else np.nan
    w365 = s.date >= t - pd.Timedelta(days=365)
    out["spell_mins_missed_365"] = float(90 * (spell_rows & w365).sum()) if w365.sum() >= 5 else np.nan
    if s.empty or np.isnan(out["avail_1095"]) and np.isnan(out["avail_365"]):
        return {**nan, **{k: v for k, v in out.items() if k in nan}}
    long = runs[runs.n >= LONG_SPELL]
    recent = long[long.end >= t - pd.Timedelta(days=730)]
    out["long_spells_24m"] = float(len(recent))
    closed_end = s[s.present].date.max()
    ongoing = not runs.empty and runs.end.max() > (closed_end if pd.notna(closed_end) else pd.Timestamp.min)
    out["unavail_now"] = float(ongoing)
    if long.empty:
        out["months_since_long_spell"] = MONTHS_CAP
    else:
        last_end = long.end.max()
        months = 0.0 if (ongoing and long.end.max() == runs.end.max()) else (t - last_end).days / 30.44
        out["months_since_long_spell"] = min(months, MONTHS_CAP)
    st = np.sort(recent.start.values)
    out["spell_recurrence"] = float(len(st) >= 2 and (np.diff(st) <= np.timedelta64(365, "D")).any())
    return out


def availability_features(con) -> pd.DataFrame:
    """One row per target (transfer_id) with AVAIL_COLS. `targets` view: transfer_id, player_id, t,
    from_club_id. Uses only rows of `con` views (truncated by date in leakage tests)."""
    tg = con.sql("SELECT transfer_id, player_id, t, from_club_id FROM targets").df()
    tg["t"] = pd.to_datetime(tg.t)
    pres, cg, apps = _load(con)
    cg_by_club = {c: g.sort_values(["date", "game_id"]).reset_index(drop=True)
                  for c, g in cg.groupby("club_id")}
    pres_by, apps_by = dict(tuple(pres.groupby("player_id"))), dict(tuple(apps.groupby("player_id")))
    empty_apps = apps.iloc[:0]
    rows = []
    for pid, g in tg.groupby("player_id"):
        if pid not in pres_by:
            rows += [{"transfer_id": r.transfer_id} for r in g.itertuples()]
            continue
        seq = player_sequence(pres_by[pid], cg_by_club, apps_by.get(pid, empty_apps))
        for r in g.itertuples():
            rows.append({"transfer_id": r.transfer_id, **features_at(seq, r.t, r.from_club_id)})
    df = pd.DataFrame(rows)
    for c in AVAIL_COLS:
        if c not in df:
            df[c] = np.nan
    log.info("availability: %d targets, %.0f%% with history", len(df), 100 * df.avail_365.notna().mean())
    return df[["transfer_id", *AVAIL_COLS]]


def spells(seq: pd.DataFrame, t: pd.Timestamp, current_club) -> pd.DataFrame:
    """Spell list (start, end, matches) before t, for the player-page timeline (phase 4)."""
    s = seq[seq.date < t]
    if s.empty:
        return pd.DataFrame(columns=["start", "end", "matches"])
    last_pres = s[s.present].groupby("block").date.max()
    is_current = (s.block == s.block.max()) & (s.club_id == current_club)
    s = s[is_current | (s.date <= s.block.map(last_pres))]
    ab = s[(s.run_id >= 0) & s.run_regular]
    r = ab.groupby("run_id").agg(start=("date", "min"), end=("date", "max"), matches=("date", "size"))
    return r[r.matches >= MIN_SPELL].reset_index(drop=True)
