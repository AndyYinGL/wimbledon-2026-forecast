"""FLB stability across years: flat-stake ROI of betting favorites, per year,
to test whether the Grand Slam 'favorites are profitable' edge is persistent or
driven by a few seasons (literature notes 2025 reversed the usual FLB).

Raw Pinnacle odds, 1-unit flat, vig included. Favorite = lower-odds side.
Also reports the 0.7-0.9 implied bucket (where GS favorites were +EV overall).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ODDS = Path.home() / "wimbledon-2026-forecast" / "data" / "raw" / "odds"


def load_year(y, gs_only=True):
    f = ODDS / (f"{y}.xlsx" if y >= 2013 else f"{y}.xls")
    if not f.exists():
        return None
    df = pd.read_excel(f)
    if gs_only:
        df = df[(df["Series"] == "Grand Slam") & (df["Best of"] == 5)]
    df = df[df.get("Comment", "Completed") == "Completed"]
    df = df.dropna(subset=["PSW", "PSL"])
    df = df[(df["PSW"] > 1) & (df["PSL"] > 1)]
    return df


def fav_roi(df):
    psw, psl = df["PSW"].values, df["PSL"].values
    fav_is_winner = psw < psl
    fav_odds = np.minimum(psw, psl)
    profit = np.where(fav_is_winner, fav_odds - 1, -1.0)
    # 0.7-0.9 implied bucket
    rw, rl = 1 / psw, 1 / psl
    fav_implied = np.maximum(rw, rl) / (rw + rl)
    mask = (fav_implied > 0.7) & (fav_implied <= 0.9)
    bucket_roi = profit[mask].mean() if mask.sum() else np.nan
    return profit.mean(), len(df), bucket_roi, mask.sum()


def run(gs_only, label):
    print(f"=== {label} ===")
    print(f"  {'year':6s} {'n':>5s} {'fav ROI':>10s} {'0.7-0.9 ROI':>14s} {'n_bucket':>9s}")
    all_profit = []
    for y in range(2011, 2026):
        df = load_year(y, gs_only)
        if df is None or len(df) == 0:
            continue
        roi, n, broi, nb = fav_roi(df)
        print(f"  {y:<6d} {n:5d} {roi:+9.2%} {broi:+13.2%} {nb:9d}")
    print()


def main():
    run(gs_only=True, label="Grand Slam men's: favorite ROI by year")
    run(gs_only=False, label="All ATP: favorite ROI by year")


if __name__ == "__main__":
    main()