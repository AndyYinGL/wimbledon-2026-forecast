"""Mispricing robustness + favorite-longshot bias (FLB) test.

(A) Does the model add anything beyond the market?  p1_won ~ model + market.
    If market dominates and model coef ~0, the market subsumes the i.i.d. model.

(B) Favorite-longshot bias: bucket by market implied prob, compare to actual
    win rate. FLB (documented in ATP tennis) = favorites win MORE than implied,
    longshots LESS. This is a within-market pricing bias and the real candidate
    for exploitable mispricing, independent of whether our model beats the market.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path.home() / "wimbledon-2026-forecast"
PROCESSED = ROOT / "data" / "processed"


def main():
    d = pd.read_parquet(PROCESSED / "mispricing_matches.parquet")
    d = d.dropna(subset=["market_p1", "model_p1", "p1_won"]).copy()
    n = len(d)
    print(f"matches: {n}\n")

    # (A) does model add anything beyond market?
    X = sm.add_constant(d[["market_p1", "model_p1"]])
    m = sm.Logit(d["p1_won"], X).fit(disp=0)
    print("=== (A) p1_won ~ market + model (logit) ===")
    for k in ["market_p1", "model_p1"]:
        print(f"  {k:11s} coef {m.params[k]:+.3f}  z={m.tvalues[k]:.2f}  p={m.pvalues[k]:.4f}")
    print("  (market dominant + model~0  =>  market subsumes the model)\n")

    # (B) favorite-longshot bias: bucket by market implied prob
    print("=== (B) Favorite-longshot bias: market implied vs actual win rate ===")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    d["bucket"] = pd.cut(d["market_p1"], bins)
    g = d.groupby("bucket", observed=True).agg(
        mean_implied=("market_p1", "mean"),
        actual_winrate=("p1_won", "mean"),
        n=("p1_won", "size"),
    )
    g["gap_actual_minus_implied"] = g["actual_winrate"] - g["mean_implied"]
    print(g.to_string())
    print()
    print("  FLB signature: favorites (high implied) actual > implied (gap>0),")
    print("  longshots (low implied) actual < implied (gap<0).")

    # quantify FLB slope: regress (actual - implied) on implied
    d["resid"] = d["p1_won"] - d["market_p1"]
    Xf = sm.add_constant(d[["market_p1"]])
    mf = sm.OLS(d["resid"], Xf).fit()
    print(f"\n  FLB slope (resid ~ implied): {mf.params['market_p1']:+.4f} "
          f"(t={mf.tvalues['market_p1']:.2f}, p={mf.pvalues['market_p1']:.4f})")
    print("  positive slope => favorites underpriced (classic FLB)")


if __name__ == "__main__":
    main()