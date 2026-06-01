"""Shared utilities for the momentum harness layers.

Every layer file imports from here so logit / ECE / the train-test protocol /
the report format stay identical across layers and can't silently drift.

Protocol fixed here: train 2011-2023, test 2024; Layer 1 = constant p_serve on
the same test set (no fitting); a challenger model adds feature(s) via logistic
regression with logit(p_serve) kept in, so we can watch whether the p_serve
coefficient gets pushed down (a leakage tell).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path.home() / "wimbledon-2026-forecast"
sys.path.insert(0, str(ROOT / "src"))
from tennis_forecast import pricing  # noqa: E402

DATA = ROOT / "data" / "processed" / "pbp_points_dataset.parquet"


def load_data() -> pd.DataFrame:
    """Load the per-point dataset and add logit(p_serve) once."""
    df = pd.read_parquet(DATA)
    df["f_logit_pserve"] = logit(df["p_serve"])
    return df


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


def split(df: pd.DataFrame):
    """Walk-forward split: train <=2023, test ==2024."""
    return df[df["year"] <= 2023], df[df["year"] == 2024]


def evaluate(df: pd.DataFrame, feat_cols, label: str):
    """Fit a logistic challenger on feat_cols, compare to Layer 1 on 2024.

    Prints log-loss / ECE for L1 and the challenger, the improvement, and the
    fitted coefficients (so the logit(p_serve) coefficient is always visible).
    feat_cols MUST include 'f_logit_pserve' to keep p_serve in the model.
    """
    train, test = split(df)
    yte = test["server_won"].values.tolist()

    p1 = test["p_serve"].values
    ll1 = pricing.log_loss(p1.tolist(), yte)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train[feat_cols].values, train["server_won"].values)
    pc = clf.predict_proba(test[feat_cols].values)[:, 1]
    llc = pricing.log_loss(pc.tolist(), yte)
    coef = dict(zip(feat_cols, clf.coef_[0]))

    print(f"train {len(train)}  test(2024) {len(test)}  "
          f"matches {test['match_id'].nunique()}")
    print(f"=== 2024 test: Layer 1 vs {label} ===")
    print(f"  Layer 1   log-loss {ll1:.5f}  ECE {ece(p1, yte):.5f}")
    print(f"  {label:9s} log-loss {llc:.5f}  ECE {ece(pc, yte):.5f}")
    print(f"  improvement (L1 - {label}): {ll1 - llc:+.6f}")
    print("  coefficients (logit scale):")
    for name in feat_cols:
        print(f"    {name:18s} {coef[name]:+.4f}")
    print(f"    {'intercept':18s} {clf.intercept_[0]:+.4f}")
    return {"ll1": ll1, "ll_challenger": llc, "coef": coef}