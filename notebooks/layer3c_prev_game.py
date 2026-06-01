"""Layer 3c: does holding/losing the PREVIOUS service game predict the next
serve point, after controlling for pre-match skill? (Game-scale momentum.)

prev_game_held = 1 if the server held his PREVIOUS service game in this match,
0 if he was broken. Defined on the server's own sequence of service games;
the first service game per server (no predecessor) is dropped. Strictly past
information. Same diagnostic: real momentum or strength leakage?
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
    df = df.sort_values(["match_id", "PointNumber"]).copy()

    # 1) per service game: did the server hold? (last point of the game won)
    game_key = ["match_id", "SetNo", "GameNo"]
    games = (
        df.groupby(game_key, sort=False)
          .agg(server=("PointServer", "last"),
               held=("server_won", "last"),
               pnum=("PointNumber", "first"))
          .reset_index()
    )
    games = games.sort_values(["match_id", "pnum"])

    # 2) per (match, server): previous service game's hold result
    games["prev_game_held"] = (
        games.groupby(["match_id", "server"], sort=False)["held"].shift(1)
    )

    # 3) map back to points via (match, SetNo, GameNo)
    df = df.merge(
        games[game_key + ["prev_game_held"]], on=game_key, how="left"
    )

    before = len(df)
    df = df.dropna(subset=["prev_game_held"]).copy()
    df["prev_game_held"] = df["prev_game_held"].astype(int)
    print(f"dropped {before - len(df)} points in each server's first service "
          f"game; {len(df)} remain")
    print()

    # raw uncontrolled
    held = df[df["prev_game_held"] == 1]["server_won"]
    broke = df[df["prev_game_held"] == 0]["server_won"]
    print("  raw P(win serve pt | previous service game):")
    print(f"    held previous  : {held.mean():.4f}  (n={len(held)})")
    print(f"    broken previous: {broke.mean():.4f}  (n={len(broke)})")
    print(f"    raw gap        : {held.mean() - broke.mean():+.4f}")
    print()

    hc.evaluate(df, ["f_logit_pserve", "prev_game_held"], "L3c-prevgame")


if __name__ == "__main__":
    main()