"""Mispricing study, step 1: build a per-match table of model vs market win
probabilities for Grand Slam men's singles, then compare each to actual
outcomes. If the i.i.d. model is far worse-calibrated than the market, the
market-minus-model gap is mostly model error, not market overreaction -- so we
check this BEFORE any mispricing test.

Each row: a GS men's match with
  p1_id, p2_id (p1 = the odds 'Winner', p2 = the 'Loser' -- but we evaluate on a
  neutral 'player1 wins' basis using a fixed orientation, see below),
  market_p1  (de-vigged Pinnacle implied prob that p1 wins),
  model_p1   (i.i.d. filter+markov prob that p1 wins),
  p1_won     (1 always, since p1 is the winner -- so we RANDOMISE orientation).

Walk-forward: belief snapshot uses only matches before each tournament.
Surface: Wimbledon -> grass; others -> non-grass (filter only has a grass term).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .filter import run_filter
from .data import serve_return_counts
from .pipeline import skills_to_pserve
from .markov import match_win_prob

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ODDS = RAW / "odds"

SLAM_START = {"Australian Open": "01-15", "French Open": "05-22",
              "Wimbledon": "07-03", "US Open": "08-28"}
GRASS = {"Wimbledon": True}  # others -> False


def _odds_id_map() -> dict:
    df = pd.read_csv(PROCESSED / "odds_name_to_id.csv").dropna(subset=["player_id"])
    return {r["odds_name"]: int(r["player_id"]) for _, r in df.iterrows()}


def _load_odds_year(y: int) -> pd.DataFrame:
    f = ODDS / (f"{y}.xlsx" if y >= 2013 else f"{y}.xls")
    df = pd.read_excel(f)
    gs = df[(df["Series"] == "Grand Slam") & (df["Best of"] == 5)].copy()
    gs["year"] = y
    return gs


def build_match_table(years=range(2011, 2026), write: bool = True) -> pd.DataFrame:
    name2id = _odds_id_map()
    rng = np.random.default_rng(0)  # for orientation randomisation

    rows = []
    for y in years:
        gs = _load_odds_year(y)
        # belief snapshot per (year, slam): trained before that tournament
        # we approximate with a single pre-season cutoff per slam
        for slam, sub in gs.groupby("Tournament"):
            if slam not in SLAM_START:
                continue
            as_of = pd.Timestamp(f"{y}-{SLAM_START[slam]}")
            m = data.load_atp_matches(range(2010, y + 1))
            obs = serve_return_counts(m)
            obs = obs[obs["tourney_date"] < as_of]
            beliefs = run_filter(obs)
            grass = GRASS.get(slam, False)

            for _, r in sub.iterrows():
                wid = name2id.get(str(r["Winner"]))
                lid = name2id.get(str(r["Loser"]))
                if wid is None or lid is None:
                    continue
                bw, bl = beliefs.get(wid), beliefs.get(lid)
                if bw is None or bl is None:
                    continue
                if pd.isna(r["PSW"]) or pd.isna(r["PSL"]):
                    continue

                # market de-vig, winner's implied prob
                rw, rl = 1 / r["PSW"], 1 / r["PSL"]
                mkt_w = rw / (rw + rl)

                # model prob winner wins
                pw = skills_to_pserve(bw, bl, grass=grass)
                pl = skills_to_pserve(bl, bw, grass=grass)
                model_w = match_win_prob(pw, pl, best_of=5)

                # randomise orientation so label isn't always 1
                if rng.random() < 0.5:
                    # keep winner as p1
                    rows.append(dict(year=y, slam=slam, round=r["Round"],
                                     p1_id=wid, p2_id=lid,
                                     market_p1=mkt_w, model_p1=model_w, p1_won=1,
                                     odds_p1=r["PSW"], odds_p2=r["PSL"],
                                     wrank=r["WRank"], lrank=r["LRank"],
                                     sw=bw.serve_var, sl=bl.serve_var))
                else:
                    # flip: loser as p1
                    rows.append(dict(year=y, slam=slam, round=r["Round"],
                                     p1_id=lid, p2_id=wid,
                                     market_p1=1 - mkt_w, model_p1=1 - model_w, p1_won=0,
                                     odds_p1=r["PSL"], odds_p2=r["PSW"],
                                     wrank=r["LRank"], lrank=r["WRank"],
                                     sw=bl.serve_var, sl=bw.serve_var))

    df = pd.DataFrame(rows)
    if write:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED / "mispricing_matches.parquet", index=False)
    return df


def _logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


if __name__ == "__main__":
    df = build_match_table()
    n = len(df)
    print(f"GS men's matches with model+market+belief: {n}")
    print(f"  years: {sorted(df['year'].unique())}")
    print(f"  base rate p1_won: {df['p1_won'].mean():.4f}  (should be ~0.5, randomised)")
    print()
    print("=== accuracy vs actual outcome (log-loss, lower better) ===")
    print(f"  market : {_logloss(df['market_p1'], df['p1_won']):.5f}")
    print(f"  model  : {_logloss(df['model_p1'],  df['p1_won']):.5f}")
    print(f"  coin   : {_logloss([0.5]*n, df['p1_won']):.5f}  (0.5 baseline)")
    print()
    # correlation of the two probabilities
    print(f"  corr(market, model): {df['market_p1'].corr(df['model_p1']):.4f}")
    print(f"  mean |market - model|: {(df['market_p1'] - df['model_p1']).abs().mean():.4f}")