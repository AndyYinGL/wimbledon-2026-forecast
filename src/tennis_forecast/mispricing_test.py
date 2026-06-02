"""Mispricing study, step 2: does the market over-price recent form relative to
the i.i.d. model, and is that premium real information or overreaction?

Builds recent-form features for each GS match's p1 (pre-match), then:
  Test 1  diff ~ recent_form           (does market pay a premium for form?)
  Test 2  p1_won ~ model_p1 + diff      (does the premium predict outcomes?)

diff = market_p1 - model_p1. If Test 1 finds a positive premium and Test 2
finds diff does NOT predict the outcome (coef ~0), the premium is overreaction
-- a mispricing -- consistent with the harness result that form doesn't predict
true play. Uses the match table from mispricing.build_match_table plus a
full-ATP recent-form lookup from the odds files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
ODDS = ROOT / "data" / "raw" / "odds"


def _load_all_results(years=range(2010, 2026)) -> pd.DataFrame:
    """All ATP matches (winner/loser/date) from the odds files, for form calc."""
    frames = []
    for y in years:
        f = ODDS / (f"{y}.xlsx" if y >= 2013 else f"{y}.xls")
        if not f.exists():
            continue
        df = pd.read_excel(f)
        frames.append(df[["Date", "Winner", "Loser"]])
    allm = pd.concat(frames, ignore_index=True)
    allm["Date"] = pd.to_datetime(allm["Date"], errors="coerce")
    return allm.dropna(subset=["Date"]).sort_values("Date")


def _form_lookup(allm: pd.DataFrame):
    """Build, per player, a time-sorted list of (date, won) for streak/winrate."""
    rec = {}
    for _, r in allm.iterrows():
        w, l, d = str(r["Winner"]), str(r["Loser"]), r["Date"]
        rec.setdefault(w, []).append((d, 1))
        rec.setdefault(l, []).append((d, 0))
    for k in rec:
        rec[k].sort(key=lambda x: x[0])
    return rec


def recent_form(rec, name, as_of, n=20):
    """Win rate over last n matches and current streak, strictly before as_of."""
    hist = [w for (d, w) in rec.get(name, []) if d < as_of]
    if not hist:
        return np.nan, np.nan
    last = hist[-n:]
    winrate = np.mean(last)
    streak = 0
    for w in reversed(hist):
        if w == 1:
            streak += 1
        else:
            break
    return winrate, streak


def main():
    mt = pd.read_parquet(PROCESSED / "mispricing_matches.parquet")
    # we need each match's date and p1's odds-name to look up form; rebuild from
    # odds is heavy, so approximate as_of by slam start (form pre-tournament).
    SLAM_START = {"Australian Open": "01-15", "French Open": "05-22",
                  "Wimbledon": "07-03", "US Open": "08-28"}

    # map p1_id back to an odds name for form lookup
    id2name = (pd.read_csv(PROCESSED / "odds_name_to_id.csv")
               .dropna(subset=["player_id"])
               .drop_duplicates("player_id")
               .set_index("player_id")["odds_name"].to_dict())

    allm = _load_all_results()
    rec = _form_lookup(allm)

    winrates, streaks = [], []
    for _, r in mt.iterrows():
        as_of = pd.Timestamp(f"{int(r['year'])}-{SLAM_START[r['slam']]}")
        nm = id2name.get(r["p1_id"])
        wr, st = recent_form(rec, nm, as_of) if nm else (np.nan, np.nan)
        winrates.append(wr); streaks.append(st)
    mt["recent_winrate"] = winrates
    mt["recent_streak"] = streaks
    mt["diff"] = mt["market_p1"] - mt["model_p1"]

    d = mt.dropna(subset=["recent_winrate", "diff", "model_p1", "p1_won"]).copy()
    print(f"matches with form: {len(d)}")
    print()

    # Test 1: does market pay a premium for recent form? diff ~ recent_winrate
    X1 = sm.add_constant(d[["recent_winrate"]])
    m1 = sm.OLS(d["diff"], X1).fit()
    print("=== Test 1: diff (market - model) ~ recent_winrate ===")
    print(f"  recent_winrate coef: {m1.params['recent_winrate']:+.4f}  "
          f"(t={m1.tvalues['recent_winrate']:.2f}, p={m1.pvalues['recent_winrate']:.4f})")
    print()

    # Test 2: does diff predict the outcome, controlling for model? (logit)
    X2 = sm.add_constant(d[["model_p1", "diff"]])
    m2 = sm.Logit(d["p1_won"], X2).fit(disp=0)
    print("=== Test 2: p1_won ~ model_p1 + diff (logit) ===")
    for k in ["model_p1", "diff"]:
        print(f"  {k:12s} coef {m2.params[k]:+.4f}  "
              f"(z={m2.tvalues[k]:.2f}, p={m2.pvalues[k]:.4f})")
    print()
    print("Reading: Test1 +coef = market pays for form. Test2 diff coef ~0 or")
    print("negative = that premium does NOT predict outcomes = overreaction.")


if __name__ == "__main__":
    main()