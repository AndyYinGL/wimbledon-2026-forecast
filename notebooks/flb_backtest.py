"""Favorite-longshot bias backtest: flat-stake ROI from betting favorites vs
longshots at actual Pinnacle closing odds.

Uses RAW odds (Winner/Loser + PSW/PSL) so settlement is correct -- NOT the
orientation-randomised match table. Bet 1 unit per match.
  - bet favorite : back the lower-odds side; win -> (odds-1), lose -> -1
  - bet longshot : back the higher-odds side
ROI = mean profit per unit staked. vig is included (raw odds), so ROI is
negative by construction; FLB shows as favorites losing LESS than longshots
(or even positive on extreme favorites). Also bucket ROI by implied prob.

Scope toggle: Grand Slam men's (Best of==5), and all ATP for a larger sample.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ODDS = Path.home() / "wimbledon-2026-forecast" / "data" / "raw" / "odds"


def load_odds(years=range(2011, 2026), gs_only=True):
    frames = []
    for y in years:
        f = ODDS / (f"{y}.xlsx" if y >= 2013 else f"{y}.xls")
        if not f.exists():
            continue
        df = pd.read_excel(f)
        if gs_only:
            df = df[(df["Series"] == "Grand Slam") & (df["Best of"] == 5)]
        frames.append(df[["PSW", "PSL", "Comment"]])
    d = pd.concat(frames, ignore_index=True)
    # keep completed matches with valid odds
    d = d[(d.get("Comment", "Completed") == "Completed")]
    d = d.dropna(subset=["PSW", "PSL"])
    d = d[(d["PSW"] > 1) & (d["PSL"] > 1)]
    return d


def backtest(d, label):
    psw, psl = d["PSW"].values, d["PSL"].values
    # favorite = lower odds side. Did favorite win? Winner's odds = PSW.
    fav_is_winner = psw < psl              # favorite (lower odds) is the Winner
    fav_odds = np.minimum(psw, psl)
    dog_odds = np.maximum(psw, psl)

    # bet favorite: profit = (fav_odds-1) if favorite won else -1
    fav_profit = np.where(fav_is_winner, fav_odds - 1, -1.0)
    # bet longshot: profit = (dog_odds-1) if longshot won (i.e. favorite did NOT) else -1
    dog_profit = np.where(~fav_is_winner, dog_odds - 1, -1.0)

    n = len(d)
    print(f"=== {label}  (n={n}) ===")
    print(f"  bet favorite : ROI {fav_profit.mean():+.4f}  ({fav_profit.mean()*100:+.2f}%)")
    print(f"  bet longshot : ROI {dog_profit.mean():+.4f}  ({dog_profit.mean()*100:+.2f}%)")
    print(f"  favorite win rate: {fav_is_winner.mean():.4f}")

    # bucket favorite ROI by implied prob of the favorite (de-vig)
    rw, rl = 1 / psw, 1 / psl
    fav_implied = np.maximum(rw, rl) / (rw + rl)
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cats = pd.cut(fav_implied, bins)
    bt = pd.DataFrame({"cat": cats, "profit": fav_profit})
    roi_by = bt.groupby("cat", observed=True)["profit"].agg(["mean", "size"])
    print("  favorite ROI by implied-prob bucket:")
    print(roi_by.to_string())
    print()


def main():
    print("Backtest: flat 1-unit, actual Pinnacle closing odds (vig included)\n")
    gs = load_odds(gs_only=True)
    backtest(gs, "Grand Slam men's 2011-2025")
    allm = load_odds(gs_only=False)
    backtest(allm, "All ATP 2011-2025")


if __name__ == "__main__":
    main()