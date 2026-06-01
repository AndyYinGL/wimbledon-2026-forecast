"""Harness Layer 2: does non-momentum STRUCTURAL score state beat Layer 1?

Runs three configurations on the same 2011-2023 train / 2024 test split and
prints a comparison table:
  - Layer 1            : constant p_serve (no fitting)
  - Layer 2 exogenous  : logit(p_serve) + SetNo + total_games
  - Layer 2 + game_lead: also server's game lead in the current set

The third is a deliberate leakage demo: being ahead proxies for in-match
strength, so it "improves" log-loss while pushing the p_serve coefficient down
and worsening calibration. Computation only -- no plots.
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


def fit_eval(train, test, feat_cols):
    Xtr, ytr = train[feat_cols].values, train["server_won"].values
    Xte, yte = test[feat_cols].values, test["server_won"].values
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return p, dict(zip(feat_cols, clf.coef_[0]))


def main():
    df = pd.read_parquet(DATA)

    srv_is_p1 = df["PointServer"] == 1
    df["server_games"]   = np.where(srv_is_p1, df["P1GamesWon"], df["P2GamesWon"])
    df["returner_games"] = np.where(srv_is_p1, df["P2GamesWon"], df["P1GamesWon"])
    df["server_game_lead"] = df["server_games"] - df["returner_games"]
    df["total_games"]      = df["server_games"] + df["returner_games"]
    df["f_logit_pserve"]   = logit(df["p_serve"])

    base_feats = ["f_logit_pserve", "SetNo", "total_games"]
    df = df.dropna(subset=base_feats + ["server_game_lead", "server_won"]).copy()

    train = df[df["year"] <= 2023]
    test  = df[df["year"] == 2024]
    yte = test["server_won"].values.tolist()
    print(f"train {len(train)}  test(2024) {len(test)}  matches {test['match_id'].nunique()}")
    print()

    # Layer 1: constant p_serve, no fitting
    p1 = test["p_serve"].values
    ll1 = pricing.log_loss(p1.tolist(), yte)

    # Layer 2 exogenous
    p2a, coef_a = fit_eval(train, test, base_feats)
    ll2a = pricing.log_loss(p2a.tolist(), yte)

    # Layer 2 + game_lead (leakage demo)
    p2b, coef_b = fit_eval(train, test, base_feats + ["server_game_lead"])
    ll2b = pricing.log_loss(p2b.tolist(), yte)

    print(f"{'model':32s} {'log-loss':>9} {'Δ vs L1':>9} "
          f"{'logit(ps)':>10} {'ECE':>8}")
    print(f"{'L1 constant p_serve':32s} {ll1:9.5f} {'--':>9} "
          f"{'--':>10} {ece(p1, yte):8.5f}")
    print(f"{'L2 exogenous (SetNo+depth)':32s} {ll2a:9.5f} {ll1-ll2a:+9.5f} "
          f"{coef_a['f_logit_pserve']:10.4f} {ece(p2a, yte):8.5f}")
    print(f"{'L2 + game_lead (leakage)':32s} {ll2b:9.5f} {ll1-ll2b:+9.5f} "
          f"{coef_b['f_logit_pserve']:10.4f} {ece(p2b, yte):8.5f}")


if __name__ == "__main__":
    main()