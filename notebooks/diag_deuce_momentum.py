"""Gauriot-Page style check: is the prev-serve-point effect stronger on
'tight' high-leverage points (30-30 and deuce) than overall?

Gauriot & Page (2019) find the hot-hand effect jumps on near-even points
(to ~15%), explained by strategic effort allocation. We test whether our
prev_serve_won effect is larger on the tight subset (s_before == r_before and
>= 2, i.e. 30-30 / deuce) than on all points.

Pre-point within-game score (s_before/r_before) is rebuilt from past results
only (leak-free). prev_serve_won is the server's previous serve-point result
in this match. Train 2011-2023 / test 2024 for the controlled comparison;
raw gaps reported on all years (descriptive).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path.home() / "wimbledon-2026-forecast" / "notebooks"))
import harness_common as hc
from tennis_forecast import pricing


def main():
    df = hc.load_data()

    # prev serve point won (server's own serve-point stream)
    df = df.sort_values(["match_id", "PointServer", "PointNumber"]).copy()
    g = df.groupby(["match_id", "PointServer"], sort=False)
    df["prev_serve_won"] = g["server_won"].shift(1)

    # pre-point within-game score, leak-free
    df = df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).copy()
    gk = ["match_id", "SetNo", "GameNo"]
    gg = df.groupby(gk, sort=False)
    df["_s"] = df["server_won"].astype(int)
    df["_r"] = 1 - df["_s"]
    df["s_before"] = gg["_s"].cumsum() - df["_s"]
    df["r_before"] = gg["_r"].cumsum() - df["_r"]

    # 'tight' near-even points: 30-30 and deuce (equal and >=2)
    df["tight"] = ((df["s_before"] == df["r_before"]) & (df["s_before"] >= 2))

    df = df.dropna(subset=["prev_serve_won"]).copy()
    df["prev_serve_won"] = df["prev_serve_won"].astype(int)

    def raw_gap(sub, name):
        won = sub[sub["prev_serve_won"] == 1]["server_won"]
        lost = sub[sub["prev_serve_won"] == 0]["server_won"]
        print(f"  {name}: n={len(sub)}")
        print(f"    P(win | won prev) : {won.mean():.4f}  (n={len(won)})")
        print(f"    P(win | lost prev): {lost.mean():.4f}  (n={len(lost)})")
        print(f"    raw gap           : {won.mean() - lost.mean():+.4f}")

    print("=== RAW prev-serve gap: all points vs tight (30-30/deuce) ===")
    raw_gap(df, "ALL points")
    raw_gap(df[df["tight"]], "TIGHT points (30-30/deuce)")
    print()

    # controlled: logistic regression on the tight subset, train/test
    tight = df[df["tight"]].copy()
    tr = tight[tight["year"] <= 2023]
    te = tight[tight["year"] == 2024]
    feats = ["f_logit_pserve", "prev_serve_won"]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(tr[feats].values, tr["server_won"].values)
    coef = dict(zip(feats, clf.coef_[0]))
    print("=== CONTROLLED on tight subset (train<=2023, test 2024) ===")
    print(f"  tight train {len(tr)}  tight test {len(te)}")
    print(f"  prev_serve_won coef : {coef['prev_serve_won']:+.4f}")
    print(f"  f_logit_pserve coef : {coef['f_logit_pserve']:+.4f}")
    print(f"  (compare: full-sample prev_serve_won coef was +0.0483)")


if __name__ == "__main__":
    main()