"""Layer 3d: are serve points HARDER to win when the server faces a break point?
(Klaassen-Magnus: the server does worse on 'important' points.)

server_faces_bp = 1 if the returner holds a break point on THIS point. It is a
property of the current point (not past history), so there is no previous-point
leakage. BUT it is a *selected* situation: a server reaches break point partly
because the current game has gone badly, which the season-constant p_serve
cannot see. So a negative effect mixes a genuine big-point effect with
situational selection -- flagged in interpretation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "wimbledon-2026-forecast" / "notebooks"))
import harness_common as hc


def main():
    df = hc.load_data()

    # server faces a break point = the RETURNER holds the break point.
    # P1BreakPoint==1 means P1 (returner on that point) holds BP; verified those
    # rows all have PointServer==2, and symmetrically for P2BreakPoint.
    srv1 = df["PointServer"] == 1
    df["server_faces_bp"] = np.where(srv1, df["P2BreakPoint"], df["P1BreakPoint"])
    df["server_faces_bp"] = df["server_faces_bp"].fillna(0).astype(int)

    rate = df["server_faces_bp"].mean()
    print(f"server_faces_bp rate: {rate:.4f}")
    print()

    # raw, uncontrolled comparison (mixes selection effect)
    bp = df[df["server_faces_bp"] == 1]["server_won"]
    nobp = df[df["server_faces_bp"] == 0]["server_won"]
    print("  raw P(server wins):")
    print(f"    facing break point : {bp.mean():.4f}  (n={len(bp)})")
    print(f"    not                : {nobp.mean():.4f}  (n={len(nobp)})")
    print(f"    raw gap            : {bp.mean() - nobp.mean():+.4f}")
    print()

    hc.evaluate(df, ["f_logit_pserve", "server_faces_bp"], "L3d-breakpt")


if __name__ == "__main__":
    main()