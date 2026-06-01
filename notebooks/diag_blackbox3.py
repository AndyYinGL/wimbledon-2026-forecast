"""Diagnostic 3: robustness of the no-game_lead XGBoost improvement (~+0.0048).
Vary max_depth and random seed; if the improvement is stable and positive it is
real signal (to be attributed later); if it flips sign across seeds it is noise.
game_lead excluded throughout (known leak).
"""
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


def build(df):
    df = df.sort_values(["match_id", "PointNumber"]).copy()
    srv1 = df["PointServer"] == 1
    df["total_games"] = (np.where(srv1, df["P1GamesWon"], df["P2GamesWon"])
                         + np.where(srv1, df["P2GamesWon"], df["P1GamesWon"]))
    df = df.sort_values(["match_id", "PointServer", "PointNumber"]).copy()
    g = df.groupby(["match_id", "PointServer"], sort=False)
    df["prev_serve_won"] = g["server_won"].shift(1)
    df["serve_streak"] = g["server_won"].transform(lambda s: compute_streak(s))
    gkey = ["match_id", "SetNo", "GameNo"]
    games = (df.groupby(gkey, sort=False)
               .agg(server=("PointServer", "last"), held=("server_won", "last"),
                    pnum=("PointNumber", "first"))
               .reset_index().sort_values(["match_id", "pnum"]))
    games["prev_game_held"] = games.groupby(["match_id", "server"],
                                            sort=False)["held"].shift(1)
    df = df.merge(games[gkey + ["prev_game_held"]], on=gkey, how="left")
    df = df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).copy()
    gg = df.groupby(gkey, sort=False)
    df["_s"] = df["server_won"].astype(int); df["_r"] = 1 - df["_s"]
    df["sb"] = gg["_s"].cumsum() - df["_s"]; df["rb"] = gg["_r"].cumsum() - df["_r"]
    df["server_faces_bp"] = ((df["rb"] >= 3) & (df["rb"] - df["sb"] >= 1)).astype(int)
    df["f_logit_pserve"] = hc.logit(df["p_serve"])
    return df.dropna(subset=["prev_serve_won", "prev_game_held"]).copy()


def main():
    df = build(hc.load_data())
    tr = df[df["year"] <= 2023]; te = df[df["year"] == 2024]
    yte = te["server_won"].values.tolist()
    ll1 = pricing.log_loss(te["p_serve"].tolist(), yte)
    feats = ["f_logit_pserve", "SetNo", "total_games", "prev_game_held",
             "prev_serve_won", "serve_streak", "server_faces_bp"]
    print(f"L1 log-loss {ll1:.5f}   (no game_lead throughout)\n")

    print("vary max_depth (seed=0):")
    for depth in [2, 3, 4, 6]:
        clf = XGBClassifier(n_estimators=300, max_depth=depth, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            eval_metric="logloss", n_jobs=-1, random_state=0)
        clf.fit(tr[feats].values, tr["server_won"].values)
        p = clf.predict_proba(te[feats].values)[:, 1]
        ll = pricing.log_loss(p.tolist(), yte)
        print(f"  depth={depth}: test {ll:.5f}  impr {ll1-ll:+.6f}")

    print("\nvary seed (depth=4):")
    imps = []
    for seed in range(5):
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            eval_metric="logloss", n_jobs=-1, random_state=seed)
        clf.fit(tr[feats].values, tr["server_won"].values)
        p = clf.predict_proba(te[feats].values)[:, 1]
        ll = pricing.log_loss(p.tolist(), yte)
        imps.append(ll1 - ll)
        print(f"  seed={seed}: impr {ll1-ll:+.6f}")
    print(f"  mean {np.mean(imps):+.6f}  std {np.std(imps):.6f}")


if __name__ == "__main__":
    main()