"""Build the per-point dataset for the falsification harness (Plan A).

One row = one point. Columns:
  identity : match_id, year, slam, PointNumber
  label    : server_won  (PointWinner == PointServer)
  layer-1  : p_serve      (constant per match: server's point-win prob vs
                           returner, from the season-snapshot filter beliefs)
  layer-2 raw (derived features added later): SetNo, P1GamesWon, P2GamesWon,
             GameNo, P1Score, P2Score, PointServer, PointWinner

Boundaries enforced here:
  * walk-forward (cross-year): for season Y, the filter trains only on match
    data strictly BEFORE that year's tournament start -> belief snapshot.
  * cleaning: keep PointServer in {1,2} (drops 0X/0Y placeholder rows);
    keep men's singles only (match_num first digit '1'); drop matches whose
    players have no belief (e.g. veterans outside the active pool).

Slam matching uses the exact "-{slam}-" pattern, NOT a substring test, because
"usopen" is a substring of "ausopen" -- a plain `in` check silently pulled
Australian Open files in and mislabelled them as US Open.

This version builds the clean base only. Derived layer-2/3 features (previous
point, streaks, break points, ...) are added in a separate step so the
within-match causal boundary can be handled explicitly there.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import data
from .filter import run_filter
from .pipeline import skills_to_pserve

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PBP = RAW / "pbp"
PROCESSED = ROOT / "data" / "processed"

SLAMS = ("wimbledon", "usopen")
SLAM_START = {"wimbledon": "07-03", "usopen": "08-28"}
GRASS = {"wimbledon": True, "usopen": False}

# Raw point columns carried through for later feature engineering.
POINT_COLS = [
    "match_id", "PointNumber", "SetNo", "GameNo",
    "P1GamesWon", "P2GamesWon", "P1Score", "P2Score",
    "PointServer", "PointWinner",
    "P1BreakPoint", "P2BreakPoint", "P1BreakPointWon", "P2BreakPointWon",
]


def _name_to_id() -> dict:
    df = pd.read_csv(PROCESSED / "pbp_name_to_id.csv").dropna(subset=["player_id"])
    return {r["pbp_name"]: int(r["player_id"]) for _, r in df.iterrows()}


def _belief_snapshot(year: int):
    """Filter beliefs trained only on matches strictly before this year's slams."""
    cutoff = pd.Timestamp(f"{year}-{SLAM_START['wimbledon']}")
    m = data.load_atp_matches(range(2010, year + 1))
    obs = data.serve_return_counts(m)
    obs = obs[obs["tourney_date"] < cutoff]
    return run_filter(obs)


def _slam_of(name: str) -> str | None:
    """Exact slam from a filename, avoiding the usopen/ausopen substring trap."""
    for s in SLAMS:
        if f"-{s}-" in name:
            return s
    return None


def _pbp_files() -> list[Path]:
    return sorted(
        p for p in PBP.glob("*-points.csv")
        if _slam_of(p.name) is not None
        and "doubles" not in p.name and "mixed" not in p.name
    )


def build_dataset(write: bool = True) -> pd.DataFrame:
    name2id = _name_to_id()
    files = _pbp_files()

    all_rows = []
    stats = {"matches_total": 0, "matches_dropped_nobelief": 0,
             "matches_dropped_noid": 0, "points_kept": 0}

    # Group files by year so we run the filter once per season.
    years = sorted({int(p.name[:4]) for p in files})
    for year in years:
        beliefs = _belief_snapshot(year)
        year_files = [p for p in files if p.name.startswith(str(year))]

        for pts_path in year_files:
            slam = _slam_of(pts_path.name)
            grass = GRASS[slam]
            matches_path = PBP / pts_path.name.replace("-points.csv", "-matches.csv")
            if not matches_path.exists():
                continue

            matches = pd.read_csv(matches_path, usecols=["match_id", "match_num",
                                                         "player1", "player2"])
            # men's singles only
            matches = matches[matches["match_num"].astype(str).str[0] == "1"]

            points = pd.read_csv(pts_path, low_memory=False)
            points = points[[c for c in POINT_COLS if c in points.columns]]
            # Drop non-point placeholder rows (PointNumber like 0X, 0Y, 106D):
            # real points have purely numeric PointNumber.
            pn = pd.to_numeric(points["PointNumber"], errors="coerce")
            points = points[pn.notna()].copy()
            points["PointNumber"] = pn[pn.notna()].astype(int)
            points = points[points["PointServer"].isin([1, 2])]

            for _, mrow in matches.iterrows():
                stats["matches_total"] += 1
                p1id = name2id.get(mrow["player1"])
                p2id = name2id.get(mrow["player2"])
                if p1id is None or p2id is None:
                    stats["matches_dropped_noid"] += 1
                    continue
                b1, b2 = beliefs.get(p1id), beliefs.get(p2id)
                if b1 is None or b2 is None:
                    stats["matches_dropped_nobelief"] += 1
                    continue

                # p_serve in each direction
                p1_serving = skills_to_pserve(b1, b2, grass=grass)  # player1 serves
                p2_serving = skills_to_pserve(b2, b1, grass=grass)  # player2 serves

                mp = points[points["match_id"] == mrow["match_id"]].copy()
                if mp.empty:
                    continue
                mp["year"] = year
                mp["slam"] = slam
                mp["server_won"] = (mp["PointWinner"] == mp["PointServer"]).astype(int)
                # pick the right p_serve per point based on who serves
                mp["p_serve"] = mp["PointServer"].map({1: p1_serving, 2: p2_serving})

                all_rows.append(mp)
                stats["points_kept"] += len(mp)

    dataset = pd.concat(all_rows, ignore_index=True)
    if write:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(PROCESSED / "pbp_points_dataset.parquet", index=False)

    # report
    print("=== build_dataset summary ===")
    print(f"  seasons processed       : {years}")
    print(f"  matches seen            : {stats['matches_total']}")
    print(f"  dropped (no id)         : {stats['matches_dropped_noid']}")
    print(f"  dropped (no belief)     : {stats['matches_dropped_nobelief']}")
    kept_matches = (stats['matches_total'] - stats['matches_dropped_noid']
                    - stats['matches_dropped_nobelief'])
    print(f"  matches kept            : {kept_matches}")
    print(f"  points kept             : {stats['points_kept']}")
    print(f"  overall server_won rate : {dataset['server_won'].mean():.4f}")
    print(f"  mean p_serve            : {dataset['p_serve'].mean():.4f}")
    if write:
        print("  written to: data/processed/pbp_points_dataset.parquet")
    return dataset


if __name__ == "__main__":
    build_dataset()