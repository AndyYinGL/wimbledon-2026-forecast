"""Single-match sanity check of the full chain:
  pbp match -> player_ids -> as-of date -> filter beliefs -> p_serve.

Pick one well-known match (2023 Wimbledon final, Alcaraz vs Djokovic) and
print everything by hand so we can eyeball whether p_serve is sane BEFORE
running the chain in bulk. Read-only.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "wimbledon-2026-forecast"
sys.path.insert(0, str(ROOT / "src"))

from tennis_forecast import data
from tennis_forecast.filter import run_filter
from tennis_forecast.pipeline import skills_to_pserve

PROCESSED = ROOT / "data" / "processed"
PBP = ROOT / "data" / "raw" / "pbp"

# Representative as-of dates per slam (tournaments start here each year).
SLAM_START = {"wimbledon": "07-03", "usopen": "08-28"}


def name_to_id_map() -> dict:
    df = pd.read_csv(PROCESSED / "pbp_name_to_id.csv")
    df = df.dropna(subset=["player_id"])
    return {row["pbp_name"]: int(row["player_id"]) for _, row in df.iterrows()}


def main():
    name2id = name_to_id_map()

    # --- locate the 2023 Wimbledon men's final ---
    matches = pd.read_csv(PBP / "2023-wimbledon-matches.csv")
    # men's final: match_num first digit '1' (men), second digit '7' (final)
    final = matches[matches["match_num"].astype(str).str.startswith("17")]
    print("Candidate men's final row(s):")
    print(final[["match_id", "match_num", "player1", "player2"]].to_string(index=False))
    print()

    row = final.iloc[0]
    p1_name, p2_name = row["player1"], row["player2"]
    p1_id, p2_id = name2id.get(p1_name), name2id.get(p2_name)
    print(f"player1: {p1_name!r} -> id {p1_id}")
    print(f"player2: {p2_name!r} -> id {p2_id}")
    print()

    # --- as-of date: use only data strictly BEFORE the tournament start ---
    year, slam = int(row["year"]), row["slam"]
    as_of = pd.Timestamp(f"{year}-{SLAM_START[slam]}")
    print(f"as-of date (train cutoff): {as_of.date()}")

    # --- run filter on observations strictly before as-of ---
    m = data.load_atp_matches(range(2010, year + 1))
    obs = data.serve_return_counts(m)
    obs = obs[obs["tourney_date"] < as_of]
    print(f"training observations before cutoff: {len(obs)}")
    beliefs = run_filter(obs)
    print(f"players with beliefs: {len(beliefs)}")
    print()

    # --- pull the two beliefs ---
    b1, b2 = beliefs.get(p1_id), beliefs.get(p2_id)
    for name, b in [(p1_name, b1), (p2_name, b2)]:
        if b is None:
            print(f"  !! no belief for {name}")
        else:
            print(f"  {name:20s} serve_mean={b.serve_mean:+.3f} "
                  f"return_mean={b.return_mean:+.3f} "
                  f"serve_grass={b.serve_grass_mean:+.3f} "
                  f"return_grass={b.return_grass_mean:+.3f}")
    print()

    # --- p_serve both directions (grass=True for Wimbledon) ---
    if b1 and b2:
        p1_serving = skills_to_pserve(b1, b2, grass=True)
        p2_serving = skills_to_pserve(b2, b1, grass=True)
        print(f"p_serve [{p1_name} serving vs {p2_name}] = {p1_serving:.4f}")
        print(f"p_serve [{p2_name} serving vs {p1_name}] = {p2_serving:.4f}")
        print()
        print("Sanity check: top grass servers usually land ~0.65-0.72.")


if __name__ == "__main__":
    main()