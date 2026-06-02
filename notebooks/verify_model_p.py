"""Verify the model_p chain on one Grand Slam match before batching:
odds row -> ids -> pre-match belief -> p_serve -> markov match win prob,
compared to the market's de-vigged implied prob. Read-only.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "wimbledon-2026-forecast"
sys.path.insert(0, str(ROOT / "src"))

from tennis_forecast import data
from tennis_forecast.filter import run_filter
from tennis_forecast.data import serve_return_counts
from tennis_forecast.pipeline import skills_to_pserve
from tennis_forecast.markov import match_win_prob

PROCESSED = ROOT / "data" / "processed"
ODDS = ROOT / "data" / "raw" / "odds"

SLAM_GRASS = {"Wimbledon": True}  # others -> False


def odds_id_map():
    df = pd.read_csv(PROCESSED / "odds_name_to_id.csv").dropna(subset=["player_id"])
    return {r["odds_name"]: int(r["player_id"]) for _, r in df.iterrows()}


def main():
    name2id = odds_id_map()

    # locate 2023 Wimbledon final
    df = pd.read_excel(ODDS / "2023.xlsx")
    gs = df[(df["Series"] == "Grand Slam") & (df["Best of"] == 5)
            & (df["Tournament"] == "Wimbledon") & (df["Round"] == "The Final")]
    row = gs.iloc[0]
    w, l = row["Winner"], row["Loser"]
    print(f"final: {w} def. {l}  ({row['Tournament']})")

    wid, lid = name2id.get(w), name2id.get(l)
    print(f"  ids: {w}={wid}  {l}={lid}")

    # market implied prob (de-vig Pinnacle), winner's perspective
    rw, rl = 1 / row["PSW"], 1 / row["PSL"]
    mkt_w = rw / (rw + rl)
    print(f"  market implied P(winner wins): {mkt_w:.4f}  (PSW={row['PSW']}, PSL={row['PSL']})")

    # pre-match belief snapshot: train filter on matches before the tournament
    as_of = pd.Timestamp("2023-07-03")
    m = data.load_atp_matches(range(2010, 2024))
    obs = serve_return_counts(m)
    obs = obs[obs["tourney_date"] < as_of]
    beliefs = run_filter(obs)

    bw, bl = beliefs.get(wid), beliefs.get(lid)
    grass = SLAM_GRASS.get(row["Tournament"], False)
    print(f"  grass={grass}")
    for nm, b in [(w, bw), (l, bl)]:
        if b is None:
            print(f"  !! no belief for {nm}")
        else:
            print(f"  {nm:20s} serve_mean={b.serve_mean:+.3f} var={b.serve_var:.4f} "
                  f"last={b.last_date.date()}")

    # p_serve each way, then markov match win prob (best of 5)
    p_w_serve = skills_to_pserve(bw, bl, grass=grass)
    p_l_serve = skills_to_pserve(bl, bw, grass=grass)
    print(f"  p_serve: {w}={p_w_serve:.4f}  {l}={p_l_serve:.4f}")

    model_w = match_win_prob(p_w_serve, p_l_serve, best_of=5)
    print(f"  model P(winner wins) = {model_w:.4f}")
    print()
    print(f"  COMPARE  market={mkt_w:.4f}  model={model_w:.4f}  diff={mkt_w-model_w:+.4f}")


if __name__ == "__main__":
    main()