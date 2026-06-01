"""Explore tennis-data.co.uk odds: implied probabilities and join feasibility
with our Sackmann-based model. Goal (design B, Moskowitz-style): does the
market's implied win prob deviate from a skill-based model in a way correlated
with recent winning streaks? (Market paying for a hot hand that, per our
harness, players don't actually have.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ODDS = Path.home() / "wimbledon-2026-forecast" / "data" / "raw" / "odds"


def load_year(y):
    f = ODDS / (f"{y}.xlsx" if y >= 2013 else f"{y}.xls")
    return pd.read_excel(f)


def main():
    df = pd.concat([load_year(y) for y in range(2011, 2025)], ignore_index=True)
    print("total ATP matches 2011-2024:", len(df))

    # Grand Slam, our two slams
    gs = df[df["Series"] == "Grand Slam"].copy()
    print("Grand Slam matches:", len(gs))
    print("tournaments:", gs["Tournament"].unique())
    print()

    # implied prob from Pinnacle (sharp book), de-vigged
    d = gs.dropna(subset=["PSW", "PSL"]).copy()
    raw_w = 1 / d["PSW"]
    raw_l = 1 / d["PSL"]
    d["mkt_p_winner"] = raw_w / (raw_w + raw_l)   # de-vig, winner's implied prob
    print(f"Pinnacle odds available: {len(d)} of {len(gs)} GS matches")
    print(f"mean market implied prob of WINNER: {d['mkt_p_winner'].mean():.4f}")
    print("  (should be >0.5: favourites win more often)")
    print()

    # favourite accuracy sanity check: when market favours winner (>0.5), how often right?
    # here winner already won by construction, so check calibration differently:
    # bucket by mkt_p_winner and the win rate is trivially 1. Instead check the
    # distribution and that favourites (high implied) dominate.
    print("distribution of winner's market implied prob:")
    print(d["mkt_p_winner"].describe())
    print()

    # Wimbledon / US Open only, men
    for slam in ["Wimbledon", "US Open"]:
        s = d[d["Tournament"] == slam]
        print(f"{slam}: {len(s)} matches, years {sorted(s['Date'].dt.year.unique())}")


if __name__ == "__main__":
    main()