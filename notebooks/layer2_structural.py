"""Harness Layer 2: does non-momentum STRUCTURAL score state beat Layer 1?

Layer 1 = constant per-match p_serve (pure i.i.d.). Layer 2 adds where we are
in the match (set number, who serves, server's game-score lead in the set) via
logistic regression with logit(p_serve) as an offset-like feature -- so it is a
structural correction on top of Layer 1, not a fresh model.

All score features are start-of-point state (pbp convention), so no
within-point leakage. Protocol: train 2011-2023, test 2024, compare to Layer 1
on the SAME 2024 test set.

Computation only -- no plots (those go in a notebook later).
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

    # --- align score to SERVER perspective (label is server_won) ---
    srv_is_p1 = df["PointServer"] == 1
    df["server_games"]   = np.where(srv_is_p1, df["P1GamesWon"], df["P2GamesWon"])
    df["returner_games"] = np.where(srv_is_p1, df["P2GamesWon"], df["P1GamesWon"])
    df["server_game_lead"] = df["server_games"] - df["returner_games"]
    df["total_games"]      = df["server_games"] + df["returner_games"]
    df["f_logit_pserve"]   = logit(df["p_serve"])

    feat_cols = ["f_logit_pserve", "SetNo", "server_game_lead", "total_games"]

    train = df[df["year"] <= 2023]
    test  = df[df["year"] == 2024]
    print(f"train points: {len(train)}   test points (2024): {len(test)}")
    print(f"test slams: {list(test['slam'].unique())}   "
          f"test matches: {test['match_id'].nunique()}")
    print()

    Xtr, ytr = train[feat_cols].values, train["server_won"].values
    Xte, yte = test[feat_cols].values,  test["server_won"].values

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    p2 = clf.predict_proba(Xte)[:, 1]
    p1 = test["p_serve"].values  # Layer 1 on same test set, no fitting

    yl = yte.tolist()
    ll1, ll2 = pricing.log_loss(p1.tolist(), yl), pricing.log_loss(p2.tolist(), yl)
    b1, b2 = pricing.brier_score(p1.tolist(), yl), pricing.brier_score(p2.tolist(), yl)

    print("=== 2024 test set: Layer 1 vs Layer 2 ===")
    print(f"  Layer 1  log-loss {ll1:.5f}  Brier {b1:.5f}  ECE {ece(p1, yte):.5f}")
    print(f"  Layer 2  log-loss {ll2:.5f}  Brier {b2:.5f}  ECE {ece(p2, yte):.5f}")
    print(f"  log-loss improvement (L1 - L2): {ll1 - ll2:+.6f}")
    print()
    print("Layer-2 coefficients (logit scale):")
    for name, coef in zip(feat_cols, clf.coef_[0]):
        print(f"  {name:18s} {coef:+.4f}")
    print(f"  intercept          {clf.intercept_[0]:+.4f}")


if __name__ == "__main__":
    main()