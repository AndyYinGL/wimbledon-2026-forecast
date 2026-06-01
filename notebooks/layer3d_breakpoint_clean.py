"""Layer 3d (clean): does facing a break point lower the server's win prob,
after controlling for skill? -- using a LEAK-FREE break-point flag.

The dataset's P1BreakPoint / P2BreakPoint columns are post-point state (they
move with the current point's result), so using them leaks the outcome. Here we
rebuild the break-point situation from scratch using only PAST within-game
scoring: count each side's points won earlier in the current game (shift, so
strictly pre-point), then apply the tennis rule

    returner faces/holds break point  <=>  r >= 3 and (r - s) >= 1

where s = points the server has won so far in this game, r = points the
returner has won so far. This covers 0-40/15-40/30-40 and AD-out after deuce.

Train 2011-2023, test 2024. Computation only.
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
    df = df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).copy()

    # who won each point: server or returner (using server_won, the label's
    # source -- but we will SHIFT so only past points count).
    # server_won == 1 -> server won that point; else returner won it.
    gkey = ["match_id", "SetNo", "GameNo"]
    grp = df.groupby(gkey, sort=False)

    # cumulative points won by server/returner BEFORE the current point
    # (cumulative count of past points in the game, shifted by 1).
    df["_srv_pt"] = df["server_won"].astype(int)
    df["_ret_pt"] = 1 - df["_srv_pt"]
    df["s_before"] = grp["_srv_pt"].cumsum() - df["_srv_pt"]   # server pts before
    df["r_before"] = grp["_ret_pt"].cumsum() - df["_ret_pt"]   # returner pts before

    # break point faced by the server, computed from PRE-point score only
    df["server_faces_bp"] = (
        (df["r_before"] >= 3) & (df["r_before"] - df["s_before"] >= 1)
    ).astype(int)

    rate = df["server_faces_bp"].mean()
    print(f"server_faces_bp rate (leak-free): {rate:.4f}")
    print()

    # raw, uncontrolled
    bp = df[df["server_faces_bp"] == 1]["server_won"]
    nobp = df[df["server_faces_bp"] == 0]["server_won"]
    print("  raw P(server wins):")
    print(f"    facing break point : {bp.mean():.4f}  (n={len(bp)})")
    print(f"    not                : {nobp.mean():.4f}  (n={len(nobp)})")
    print(f"    raw gap            : {bp.mean() - nobp.mean():+.4f}")
    print()

    hc.evaluate(df, ["f_logit_pserve", "server_faces_bp"], "L3d-bp-clean")


if __name__ == "__main__":
    main()