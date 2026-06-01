"""Harness Layer 3a: does the previous serve point predict the next one,
after controlling for pre-match skill? (The literal i.i.d. test.)

Momentum feature: prev_serve_won -- for each serve point, whether THIS server
won his PREVIOUS serve point in the same match. Defined on the server's own
serve-point sequence (not the raw previous point, which may be the opponent's
serve), so it tests "serving form persistence" rather than generic scoreboard
flow.

Causal boundary: strictly past information. Grouped by (match_id, server),
ordered by PointNumber, take shift(1); each server's first serve point in a
match has no predecessor and is dropped. No cross-match / cross-server borrow.

Protocol: train 2011-2023, test 2024, compare to Layer 1 on the same test set.
This is the bare single-feature run -- a diagnostic to see if prev_serve_won is
real signal or, like game_lead in Layer 2, a strength-leakage artifact.
Computation only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path.home() / "wimbledon-2026-forecast"
sys.path.insert(0, str(ROOT / "src"))
from tennis_forecast import pricing

DATA = ROOT / "data" / "processed" / "pbp_points_dataset.parquet"


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(probs, outcomes, n_bins=10):
    probs = np.asarray(probs); outcomes = np.asarray(outcomes)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    tot = len(probs); e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += abs(probs[m].mean() - outcomes[m].mean()) * m.sum() / tot
    return e


def main():
    df = pd.read_parquet(DATA)

    # identify the server's player id per point (1 -> p1 side, 2 -> p2 side).
    # We don't have player ids per row here, so use (match_id, PointServer) as
    # the server's own serve-point stream within a match.
    df = df.sort_values(["match_id", "PointServer", "PointNumber"]).copy()

    # prev_serve_won: did this server win his PREVIOUS serve point in this match?
    grp = df.groupby(["match_id", "PointServer"], sort=False)
    df["prev_serve_won"] = grp["server_won"].shift(1)

    # drop each server's first serve point (no predecessor)
    before = len(df)
    df = df.dropna(subset=["prev_serve_won"]).copy()
    df["prev_serve_won"] = df["prev_serve_won"].astype(int)
    df["f_logit_pserve"] = logit(df["p_serve"])
    print(f"dropped {before - len(df)} first-serve-points; {len(df)} remain")

    feat_cols = ["f_logit_pserve", "prev_serve_won"]
    train = df[df["year"] <= 2023]
    test  = df[df["year"] == 2024]
    yte = test["server_won"].values.tolist()
    print(f"train {len(train)}  test(2024) {len(test)}  matches {test['match_id'].nunique()}")
    print()

    # Layer 1 on same test set
    p1 = test["p_serve"].values
    ll1 = pricing.log_loss(p1.tolist(), yte)

    # Layer 3a
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train[feat_cols].values, train["server_won"].values)
    p3 = clf.predict_proba(test[feat_cols].values)[:, 1]
    ll3 = pricing.log_loss(p3.tolist(), yte)
    coef = dict(zip(feat_cols, clf.coef_[0]))

    print("=== 2024 test set: Layer 1 vs Layer 3a (prev serve point) ===")
    print(f"  Layer 1   log-loss {ll1:.5f}  ECE {ece(p1, yte):.5f}")
    print(f"  Layer 3a  log-loss {ll3:.5f}  ECE {ece(p3, yte):.5f}")
    print(f"  improvement (L1 - L3a): {ll1 - ll3:+.6f}")
    print()
    print("  coefficients (logit scale):")
    print(f"    f_logit_pserve   {coef['f_logit_pserve']:+.4f}")
    print(f"    prev_serve_won   {coef['prev_serve_won']:+.4f}")
    print(f"    intercept        {clf.intercept_[0]:+.4f}")
    print()
    # descriptive (raw, uncontrolled) for contrast -- subject to selection +
    # Miller-Sanjurjo bias; shown only to compare with the controlled coef.
    won_prev = df[df["prev_serve_won"] == 1]["server_won"].mean()
    lost_prev = df[df["prev_serve_won"] == 0]["server_won"].mean()
    print(f"  raw P(win | won prev serve pt):  {won_prev:.4f}")
    print(f"  raw P(win | lost prev serve pt): {lost_prev:.4f}")
    print(f"  raw gap (uncontrolled): {won_prev - lost_prev:+.4f}")


if __name__ == "__main__":
    main()