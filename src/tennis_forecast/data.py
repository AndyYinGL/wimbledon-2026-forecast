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


def load_atp_matches(years: range) -> pd.DataFrame:
    """Load + concat Sackmann atp_matches_{year}.csv over `years`.

    TODO: normalise player ids; keep date, surface, tourney_level, and serve
    stats (w_svpt, w_1stWon, w_2ndWon, l_svpt, l_1stWon, l_2ndWon, ...).
    Returns one row per match.
    """
    raise NotImplementedError


def serve_return_counts(matches: pd.DataFrame) -> pd.DataFrame:
    """Turn each match into the per-match serve/return point COUNTS the filter
    consumes (the binomial observation that identifies serve vs return skill).

    TODO: for each match emit, per player, serve_points_won / serve_points_played
    (and thus the opponent's return points). Flag matches missing serve stats
    (fall back to win/loss) and retirements/walkovers (exclude or down-weight).
    """
    raise NotImplementedError


def load_wimbledon_draw() -> pd.DataFrame:
    """Current men's draw + results so far (live source / tennis API).

    TODO: columns slot, player_id, status ('alive'|'out'), round_reached.
    Re-fetch each round; this drives the live forecast.
    """
    raise NotImplementedError


def load_market_odds() -> pd.DataFrame:
    """Closing odds for the model-vs-market calibration study.

    TODO: columns player_id / match_id, decimal_odds, ts. De-vig in pricing.py.
    """
    raise NotImplementedError
