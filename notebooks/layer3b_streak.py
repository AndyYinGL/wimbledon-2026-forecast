"""Layer 3b: does a winning STREAK on serve predict the next serve point,
after controlling for pre-match skill?

serve_streak = the server's run of consecutive serve points won in this match,
as of the START of the current point (past results only, reset to 0 on a loss),
on the server's own serve-point stream. Same diagnostic as 3a: real momentum or
strength leakage? Watch the streak coef, whether logit(p_serve) is pushed down,
and ECE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "wimbledon-2026-forecast" / "notebooks"))
import harness_common as hc


def compute_streak(won: pd.Series) -> np.ndarray:
    """Consecutive prior wins entering each point (past only, reset on loss)."""
    out = np.zeros(len(won), dtype=int)
    run = 0
    for i, v in enumerate(won.values):
        out[i] = run
        run = run + 1 if v == 1 else 0
    return out


def main():
    df = hc.load_data()
    df = df.sort_values(["match_id", "PointServer", "PointNumber"]).copy()
    df["serve_streak"] = (
        df.groupby(["match_id", "PointServer"], sort=False)["server_won"]
          .transform(lambda s: compute_streak(s))
    )

    print("serve_streak distribution (entering point):",
          df["serve_streak"].value_counts().sort_index().head(8).to_dict())
    print()

    hc.evaluate(df, ["f_logit_pserve", "serve_streak"], "L3b-streak")

    print()
    print("  raw P(win next serve pt | entering streak):")
    for k in range(5):
        sub = df[df["serve_streak"] == k]["server_won"]
        if len(sub):
            print(f"    streak={k}: {sub.mean():.4f}  (n={len(sub)})")


if __name__ == "__main__":
    main()