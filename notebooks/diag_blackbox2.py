"""Diagnostic 2: add black-box features one at a time (on top of p_serve) to
locate which feature(s) produce the XGBoost out-of-sample improvement.
All features are leak-free start-of-point. game_lead tested last (known leak).
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
    df["server_games"]   = np.where(srv1, df["P1GamesWon"], df["P2GamesWon"])
    df["returner_games"] = np.where(srv1, df["P2GamesWon"], df["P1GamesWon"])
    df["server_game_lead"] = df["server_games"] - df["returner_games"]
    df["total_games"]      = df["server_games"] + df["returner_games"]
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
    print("L1 log-loss", round(ll1, 5))

    steps = [
        ["f_logit_pserve"],
        ["f_logit_pserve", "SetNo"],
        ["f_logit_pserve", "SetNo", "total_games"],
        ["f_logit_pserve", "SetNo", "total_games", "prev_game_held"],
        ["f_logit_pserve", "SetNo", "total_games", "prev_game_held", "prev_serve_won"],
        ["f_logit_pserve", "SetNo", "total_games", "prev_game_held", "prev_serve_won", "serve_streak"],
        ["f_logit_pserve", "SetNo", "total_games", "prev_game_held", "prev_serve_won", "serve_streak", "server_faces_bp"],
        ["f_logit_pserve", "SetNo", "total_games", "prev_game_held", "prev_serve_won", "serve_streak", "server_faces_bp", "server_game_lead"],
    ]
    for feats in steps:
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            eval_metric="logloss", n_jobs=-1)
        clf.fit(tr[feats].values, tr["server_won"].values)
        p = clf.predict_proba(te[feats].values)[:, 1]
        ll = pricing.log_loss(p.tolist(), yte)
        print(f"  +{feats[-1]:18s} ({len(feats)}) test {ll:.5f}  impr {ll1-ll:+.6f}")


if __name__ == "__main__":
    main()