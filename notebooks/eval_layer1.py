"""Harness anchor: evaluate the layer-1 constant p_serve baseline on the
per-point dataset (Plan A, predict next point).

Layer-1 has no fitted parameters -- p_serve comes straight from walk-forward
belief snapshots -- so its log-loss / Brier / calibration on the full 883k
points is already an honest out-of-sample number. This is the reference every
later model (layer-2, layer-3, black-box) must beat.

Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "wimbledon-2026-forecast"
sys.path.insert(0, str(ROOT / "src"))

from tennis_forecast import pricing

DATA = ROOT / "data" / "processed" / "pbp_points_dataset.parquet"


def main():
    df = pd.read_parquet(DATA)
    print(f"points: {len(df)}")

    probs = df["p_serve"].tolist()
    outcomes = df["server_won"].tolist()

    ll = pricing.log_loss(probs, outcomes)
    bs = pricing.brier_score(probs, outcomes)

    print()
    print("=== Layer-1 baseline (constant p_serve, i.i.d.) ===")
    print(f"  log-loss : {ll:.5f}")
    print(f"  Brier    : {bs:.5f}")
    print(f"  base rate (server_won): {sum(outcomes)/len(outcomes):.5f}")
    print(f"  mean p_serve          : {sum(probs)/len(probs):.5f}")

    # Reference: log-loss of always predicting the base rate (a naive constant).
    base = sum(outcomes) / len(outcomes)
    ll_const = pricing.log_loss([base] * len(outcomes), outcomes)
    print(f"  (naive constant base-rate log-loss: {ll_const:.5f})")

    print()
    print("=== Reliability curve ===")
    rc = pricing.reliability_curve(probs, outcomes, n_bins=10)
    print("raw return:", rc)


if __name__ == "__main__":
    main()