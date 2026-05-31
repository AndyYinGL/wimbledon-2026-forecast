"""Data layer: historical match data, live Wimbledon draw/results, market odds.

All functions return tidy pandas DataFrames with documented columns. Raw
downloads go in data/raw/ (gitignored); tidy parquet in data/processed/.

Implement one function at a time. Start here — everything downstream needs data.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

# Anchor for the global serve baseline (avg serve point-win sum, ATP men ~1.29).
ATP_SERVE_SUM = 1.29


def load_atp_matches(years: range, raw_dir: Path = RAW) -> pd.DataFrame:
    """Load + concat Sackmann atp_matches_{year}.csv over `years`.

    Returns one row per match with normalised columns:
      tourney_date (datetime), surface, tourney_level,
      winner_id, winner_name, loser_id, loser_name,
      and serve stats: w_svpt, w_1stWon, w_2ndWon, l_svpt, l_1stWon, l_2ndWon.
    """
    keep = [
        "tourney_date", "surface", "tourney_level",
        "winner_id", "winner_name", "loser_id", "loser_name",
        "w_svpt", "w_1stWon", "w_2ndWon",
        "l_svpt", "l_1stWon", "l_2ndWon",
    ]
    frames = []
    for year in years:
        path = raw_dir / f"atp_matches_{year}.csv"
        if not path.exists():
            print(f"  skip (not found): {path.name}")
            continue
        df = pd.read_csv(path, usecols=lambda c: c in keep)
        df["year"] = year
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No atp_matches_*.csv found in {raw_dir}")

    matches = pd.concat(frames, ignore_index=True)
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"], format="%Y%m%d"
    )
    matches = matches.sort_values("tourney_date").reset_index(drop=True)
    return matches


def serve_return_counts(matches: pd.DataFrame) -> pd.DataFrame:
    """Turn each match into per-match serve/return point COUNTS — the binomial
    observation that lets the filter separate serve skill from return skill.

    Each match yields TWO rows (one per server):
      tourney_date, surface, server_id, server_name, returner_id, returner_name,
      points_won (serve points won by server), points_played (serve points).

    This version keeps only matches with serve stats present and drops Carpet.
    Retirement handling is deferred; `points_played` is carried so short,
    unrepresentative matches can be filtered later.
    """
    df = matches[matches["surface"] != "Carpet"].copy()

    needed = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]
    df = df.dropna(subset=needed)
    df = df[(df["w_svpt"] > 0) & (df["l_svpt"] > 0)]

    base = ["tourney_date", "surface"]

    # Row 1: winner served.
    winner_rows = df[base].copy()
    winner_rows["server_id"] = df["winner_id"]
    winner_rows["server_name"] = df["winner_name"]
    winner_rows["returner_id"] = df["loser_id"]
    winner_rows["returner_name"] = df["loser_name"]
    winner_rows["points_won"] = df["w_1stWon"] + df["w_2ndWon"]
    winner_rows["points_played"] = df["w_svpt"]

    # Row 2: loser served.
    loser_rows = df[base].copy()
    loser_rows["server_id"] = df["loser_id"]
    loser_rows["server_name"] = df["loser_name"]
    loser_rows["returner_id"] = df["winner_id"]
    loser_rows["returner_name"] = df["winner_name"]
    loser_rows["points_won"] = df["l_1stWon"] + df["l_2ndWon"]
    loser_rows["points_played"] = df["l_svpt"]

    obs = pd.concat([winner_rows, loser_rows], ignore_index=True)
    obs = obs.sort_values("tourney_date").reset_index(drop=True)
    return obs


def load_wimbledon_draw() -> pd.DataFrame:
    """Current men's draw + results so far (live source / tennis API).

    TODO: columns slot, player_id, status ('alive'|'out'), round_reached.
    Re-fetch each round; this drives the live forecast.
    """
    raise NotImplementedError


def load_market_odds(odds_dir: Path = RAW / "odds") -> pd.DataFrame:
    """Load tennis-data.co.uk ATP closing odds for the model-vs-market study.

    Reads every yearly .xls/.xlsx in odds_dir and returns one row per match:
      date, surface, round, winner_name, loser_name, comment (e.g. 'Completed'),
      odds_w, odds_l   -- Pinnacle decimal odds, falling back to market average,
      p_market_w       -- de-vigged market-implied prob that the winner wins.

    Note: names follow tennis-data convention ('Federer R.'); reconciling them
    with Sackmann ids is a separate name-normalisation step (done later).
    """
    files = sorted(odds_dir.glob("*.xls*"))
    if not files:
        raise FileNotFoundError(f"No odds files in {odds_dir}")

    frames = []
    for path in files:
        df = pd.read_excel(path)
        out = pd.DataFrame({
            "date": pd.to_datetime(df["Date"], errors="coerce"),
            "surface": df.get("Surface"),
            "round": df.get("Round"),
            "winner_name": df["Winner"],
            "loser_name": df["Loser"],
            "comment": df.get("Comment"),
        })
        # Pinnacle first, fall back to market average where Pinnacle is missing.
        out["odds_w"] = df.get("PSW").fillna(df.get("AvgW")) if "PSW" in df else df.get("AvgW")
        out["odds_l"] = df.get("PSL").fillna(df.get("AvgL")) if "PSL" in df else df.get("AvgL")
        frames.append(out)

    odds = pd.concat(frames, ignore_index=True)
    odds = odds.dropna(subset=["odds_w", "odds_l"])

    # De-vig (proportional): strip the bookmaker overround to a true probability.
    inv_w = 1.0 / odds["odds_w"]
    inv_l = 1.0 / odds["odds_l"]
    odds["p_market_w"] = inv_w / (inv_w + inv_l)

    odds = odds.sort_values("date").reset_index(drop=True)
    return odds
