"""Diagnostic: is the XGBoost 'improvement' from faces_bp, or from XGB
re-fitting p_serve nonlinearly? Compare XGB with [p_serve] vs [p_serve+faces_bp].
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path.home() / "wimbledon-2026-forecast" / "notebooks"))
import harness_common as hc
from tennis_forecast import pricing


def main():
    df = hc.load_data()
    df = df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).copy()
    gk = ["match_id", "SetNo", "GameNo"]
    gg = df.groupby(gk, sort=False)
    df["_s"] = df["server_won"].astype(int)
    df["_r"] = 1 - df["_s"]
    df["sb"] = gg["_s"].cumsum() - df["_s"]
    df["rb"] = gg["_r"].cumsum() - df["_r"]
    df["faces_bp"] = ((df["rb"] >= 3) & (df["rb"] - df["sb"] >= 1)).astype(int)
    df["f_logit_pserve"] = hc.logit(df["p_serve"])

    tr = df[df["year"] <= 2023]
    te = df[df["year"] == 2024]
    yte = te["server_won"].values.tolist()
    ll1 = pricing.log_loss(te["p_serve"].tolist(), yte)
    print("L1 log-loss", round(ll1, 5))

    for feats in [["f_logit_pserve"], ["f_logit_pserve", "faces_bp"]]:
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            eval_metric="logloss", n_jobs=-1)
        clf.fit(tr[feats].values, tr["server_won"].values)
        p = clf.predict_proba(te[feats].values)[:, 1]
        ll = pricing.log_loss(p.tolist(), yte)
        print(feats, "test log-loss", round(ll, 5), "impr", round(ll1 - ll, 6))


if __name__ == "__main__":
    main()