"""Within-match check: is the +0.0049 black-box gain just cross-match strength
(selection), or genuine within-match sequential structure?

Idea: add an explicit IN-MATCH strength proxy -- the server's running serve-win
rate so far in THIS match (cumulative, shifted, pre-point). If the gain comes
from the tree inferring 'who is strong this match' through score features, this
direct proxy should absorb it and the other score features should add little
on top. If a gain survives beyond the in-match strength proxy, that is real
within-match structure.

All features leak-free start-of-point. game_lead excluded. Train 2011-2023,
test 2024.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path.home() / "wimbledon-2026-forecast" / "notebooks"))
import harness_common as hc
from tennis_forecast import pricing


def compute_streak(won):
    out = np.zeros(len(won), dtype=int); run = 0
    for i, v in enumerate(won.values):
        out[i] = run; run = run + 1 if v == 1 else 0
    return out


def build(df):
    df = df.sort_values(["match_id", "PointServer", "PointNumber"]).copy()
    g = df.groupby(["match_id", "PointServer"], sort=False)
    df["prev_serve_won"] = g["server_won"].shift(1)
    df["serve_streak"] = g["server_won"].transform(lambda s: compute_streak(s))

    # in-match cumulative serve-win rate for this server, pre-point (shifted)
    csum = g["server_won"].cumsum() - df["server_won"]
    cnt = g.cumcount()
    df["inmatch_serve_rate"] = (csum / cnt.replace(0, np.nan)).fillna(df["p_serve"])

    srv1 = df["PointServer"] == 1
    df = df.sort_values(["match_id", "PointNumber"]).copy()
    srv1 = df["PointServer"] == 1
    df["total_games"] = (np.where(srv1, df["P1GamesWon"], df["P2GamesWon"])
                         + np.where(srv1, df["P2GamesWon"], df["P1GamesWon"]))

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


def run(tr, te, feats, yte, ll1, label):
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        eval_metric="logloss", n_jobs=-1, random_state=0)
    clf.fit(tr[feats].values, tr["server_won"].values)
    p = clf.predict_proba(te[feats].values)[:, 1]
    ll = pricing.log_loss(p.tolist(), yte)
    print(f"  {label:42s} test {ll:.5f}  impr {ll1-ll:+.6f}")


def main():
    df = build(hc.load_data())
    tr = df[df["year"] <= 2023]; te = df[df["year"] == 2024]
    yte = te["server_won"].values.tolist()
    ll1 = pricing.log_loss(te["p_serve"].tolist(), yte)
    print(f"L1 log-loss {ll1:.5f}\n")

    score = ["f_logit_pserve", "total_games", "prev_game_held",
             "prev_serve_won", "serve_streak", "server_faces_bp"]
    run(tr, te, score, yte, ll1, "score features (the +0.0049 set)")
    run(tr, te, ["f_logit_pserve", "inmatch_serve_rate"], yte, ll1,
        "p_serve + in-match strength proxy only")
    run(tr, te, score + ["inmatch_serve_rate"], yte, ll1,
        "score features + in-match strength proxy")


if __name__ == "__main__":
    main()