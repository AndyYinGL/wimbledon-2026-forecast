"""Model-edge backtest, LEAK-FREE. The earlier version let bet direction depend
on the winner's-perspective edge, so it backed known winners. Here everything is
in a fixed, outcome-independent player1 frame (orientation randomised at build):
bet p1 iff model_p1 - market_p1 > thresh; settle p1_won at p1's real odds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "wimbledon-2026-forecast"
PROCESSED = ROOT / "data" / "processed"


def backtest(d, thresh):
    edge = d["model_p1"] - d["market_p1"]      # pre-match, outcome-independent
    bet = edge > thresh                        # back p1 when model likes p1 more
    sub = d[bet]
    if len(sub) == 0:
        return np.nan, 0
    # p1 won -> collect p1's odds; else -1
    profit = np.where(sub["p1_won"] == 1, sub["odds_p1"] - 1, -1.0)
    return profit.mean(), len(sub)


def main():
    d = pd.read_parquet(PROCESSED / "mispricing_matches.parquet")
    d = d.dropna(subset=["model_p1", "market_p1", "p1_won", "odds_p1"]).copy()
    print(f"matches: {len(d)}")
    print(f"sanity: p1_won base rate {d['p1_won'].mean():.4f} (should be ~0.5)")
    print(f"sanity: mean(model_p1 - market_p1) {(d['model_p1']-d['market_p1']).mean():+.5f} "
          f"(should be ~0 in neutral frame)\n")

    print("=== leak-free model-edge backtest (vig included) ===")
    for th in [0.0, 0.05, 0.10, 0.15]:
        roi, nb = backtest(d, th)
        print(f"  threshold {th:.2f}: ROI {roi:+.2%}  bets={nb}")
    print()
    print("=== by year, threshold 0.05 ===")
    for y in sorted(d["year"].unique()):
        roi, nb = backtest(d[d["year"] == y], 0.05)
        print(f"  {y}: ROI {roi:+.2%}  bets={nb}")


if __name__ == "__main__":
    main()