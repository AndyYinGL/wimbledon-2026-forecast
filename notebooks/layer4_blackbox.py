"""Layer 4: black-box falsification check. Can a gradient-boosted tree, free to
find any nonlinear interaction among ALL start-of-point features, beat the
Layer-1 constant p_serve out of sample?

If even XGBoost -- unconstrained by our hand-built linear features -- cannot
beat Layer 1, then "momentum doesn't help" is not an artifact of poor feature
design: the sequential structure simply isn't there to exploit.

All features are start-of-point (no current-point result leaks in). We keep
logit(p_serve) in so the tree must improve ON TOP of skill, and we run two
configs -- with and without server_game_lead (the known strength-leakage
feature) -- to see how much of any 'gain' is just that leakage again.

Walk-forward: train 2011-2023, test 2024. Computation only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path.home() / "wimbledon-2026-forecast" / "notebooks"))
import harness_common as hc
from tennis_forecast import pricing


def compute_streak(won: pd.Series) -> np.ndarray:
    out = np.zeros(len(won), dtype=int); run = 0
    for i, v in enumerate(won.values):
        out[i] = run; run = run + 1 if v == 1 else 0
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["match_id", "PointNumber"]).copy()
    srv1 = df["PointServer"] == 1

    # structural (server perspective) -- GamesWon is game-level (pre-point safe)
    df["server_games"]   = np.where(srv1, df["P1GamesWon"], df["P2GamesWon"])
    df["returner_games"] = np.where(srv1, df["P2GamesWon"], df["P1GamesWon"])
    df["server_game_lead"] = df["server_games"] - df["returner_games"]
    df["total_games"]      = df["server_games"] + df["returner_games"]

    # momentum (history, per server stream)
    df = df.sort_values(["match_id", "PointServer", "PointNumber"]).copy()
    g = df.groupby(["match_id", "PointServer"], sort=False)
    df["prev_serve_won"] = g["server_won"].shift(1)
    df["serve_streak"] = g["server_won"].transform(lambda s: compute_streak(s))

    # previous service game held (game scale)
    gkey = ["match_id", "SetNo", "GameNo"]
    games = (df.groupby(gkey, sort=False)
               .agg(server=("PointServer", "last"),
                    held=("server_won", "last"),
                    pnum=("PointNumber", "first"))
               .reset_index().sort_values(["match_id", "pnum"]))
    games["prev_game_held"] = games.groupby(["match_id", "server"],
                                            sort=False)["held"].shift(1)
    df = df.merge(games[gkey + ["prev_game_held"]], on=gkey, how="left")

    # break point faced -- LEAK-FREE rebuild from pre-point within-game score
    df = df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).copy()
    gg = df.groupby(gkey, sort=False)
    df["_srv_pt"] = df["server_won"].astype(int)
    df["_ret_pt"] = 1 - df["_srv_pt"]
    df["s_before"] = gg["_srv_pt"].cumsum() - df["_srv_pt"]
    df["r_before"] = gg["_ret_pt"].cumsum() - df["_ret_pt"]
    df["server_faces_bp"] = (
        (df["r_before"] >= 3) & (df["r_before"] - df["s_before"] >= 1)
    ).astype(int)

    df["f_logit_pserve"] = hc.logit(df["p_serve"])
    return df

    # momentum (history, per server stream)
    df = df.sort_values(["match_id", "PointServer", "PointNumber"]).copy()
    g = df.groupby(["match_id", "PointServer"], sort=False)
    df["prev_serve_won"] = g["server_won"].shift(1)
    df["serve_streak"] = g["server_won"].transform(lambda s: compute_streak(s))

    # previous service game held (game scale)
    gkey = ["match_id", "SetNo", "GameNo"]
    games = (df.groupby(gkey, sort=False)
               .agg(server=("PointServer", "last"),
                    held=("server_won", "last"),
                    pnum=("PointNumber", "first"))
               .reset_index().sort_values(["match_id", "pnum"]))
    games["prev_game_held"] = games.groupby(["match_id", "server"],
                                            sort=False)["held"].shift(1)
    df = df.merge(games[gkey + ["prev_game_held"]], on=gkey, how="left")

    df["f_logit_pserve"] = hc.logit(df["p_serve"])
    return df


def run_xgb(train, test, feats):
    Xtr, ytr = train[feats].values, train["server_won"].values
    Xte, yte = test[feats].values, test["server_won"].values
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, eval_metric="logloss", n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return p, clf


def main():
    df = build_features(hc.load_data())
    # need momentum history present; drop rows lacking prev features
    df = df.dropna(subset=["prev_serve_won", "prev_game_held"]).copy()

    train, test = hc.split(df)
    yte = test["server_won"].values.tolist()
    p1 = test["p_serve"].values
    ll1 = pricing.log_loss(p1.tolist(), yte)
    print(f"train {len(train)}  test(2024) {len(test)}  "
          f"matches {test['match_id'].nunique()}")
    print(f"Layer 1  log-loss {ll1:.5f}  ECE {hc.ece(p1, yte):.5f}")
    print()

    base = ["f_logit_pserve", "SetNo", "total_games", "server_faces_bp",
            "prev_serve_won", "serve_streak", "prev_game_held"]

    for label, feats in [
        ("XGB no game_lead", base),
        ("XGB + game_lead", base + ["server_game_lead"]),
    ]:
        p, clf = run_xgb(train, test, feats)
        ll = pricing.log_loss(p.tolist(), yte)
        ptr = clf.predict_proba(train[feats].values)[:, 1]
        ll_tr = pricing.log_loss(ptr.tolist(), train["server_won"].values.tolist())
        print(f"=== {label} ({len(feats)} features) ===")
        print(f"  TRAIN log-loss {ll_tr:.5f}   TEST log-loss {ll:.5f}   "
              f"gap {ll - ll_tr:+.5f}")
        print(f"  ECE {hc.ece(p, yte):.5f}  improvement (L1-XGB) {ll1 - ll:+.6f}")
        imp = sorted(zip(feats, clf.feature_importances_),
                     key=lambda x: -x[1])
        print("  feature importance:")
        for name, val in imp:
            print(f"    {name:18s} {val:.4f}")
        print()


if __name__ == "__main__":
    main()