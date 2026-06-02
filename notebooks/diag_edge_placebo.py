import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "wimbledon-2026-forecast"
d = pd.read_parquet(ROOT / "data/processed/mispricing_matches.parquet")
d = d.dropna(subset=["model_p1", "market_p1", "p1_won", "odds_p1"]).copy()
rng = np.random.default_rng(42)


def bt(edge, th=0.05):
    sub = d[edge > th]
    if len(sub) == 0:
        return float("nan"), 0
    profit = np.where(sub["p1_won"] == 1, sub["odds_p1"] - 1, -1.0)
    return profit.mean(), len(sub)


# 1. real model edge
roi_real, n_real = bt(d["model_p1"] - d["market_p1"])
print(f"REAL  model edge        : ROI {roi_real:+.2%}  bets={n_real}")

# 2. placebo: random model_p
fake = pd.Series(rng.random(len(d)), index=d.index)
roi_f, n_f = bt(fake - d["market_p1"])
print(f"PLACEBO random model_p  : ROI {roi_f:+.2%}  bets={n_f}")

# 3. placebo: market + noise as model
noisy = d["market_p1"] + pd.Series(rng.normal(0, 0.1, len(d)), index=d.index)
roi_m, n_m = bt(noisy - d["market_p1"])
print(f"PLACEBO market+noise    : ROI {roi_m:+.2%}  bets={n_m}")

# 4. reference: bet p1 in every match
ref = np.where(d["p1_won"] == 1, d["odds_p1"] - 1, -1.0).mean()
print(f"(ref) bet p1 all matches: ROI {ref:+.2%}  bets={len(d)}")