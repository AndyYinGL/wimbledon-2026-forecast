"""Resolve tennis-data.co.uk odds player names to Sackmann player_ids.

Odds names are 'Surname X.' (e.g. 'Sinner J.', 'Van De Zandschulp B.'),
unlike the pbp feed's 'X. Surname'. Resolution order (highest priority last):
  1. unique 'surname + first-initial' match within the active player pool
  2. manual override table (hand-verified, authoritative)

Candidate pool = players appearing in ATP match data 2010-2026 (the players the
filter holds beliefs for), which removes false ambiguity from historical
same-surname players and guarantees a matched id is usable.

Writes an auditable mapping to data/processed/odds_name_to_id.csv with a
match_method column (initial / manual / unresolved). Scope: Grand Slam men's
singles (Best of == 5).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from . import data

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ODDS = RAW / "odds"
PLAYERS = RAW / "atp_players.csv"
OVERRIDES = RAW / "odds_name_overrides.csv"


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c))


def norm(s: str) -> str:
    return strip_accents(s).strip().lower()


def parse_odds_name(name: str):
    """'Sinner J.' -> ('j', 'sinner'); 'Van De Zandschulp B.' -> ('b', 'van de zandschulp').

    Returns (first_initial_lower, surname_norm) or None if the trailing token
    isn't a 1-2 letter initial.
    """
    raw = str(name).strip()
    toks = raw.split()
    if len(toks) < 2:
        return None
    initial_tok = toks[-1]
    letters = initial_tok.replace(".", "")
    if letters.isalpha() and 1 <= len(letters) <= 2:
        return letters[0].lower(), norm(" ".join(toks[:-1]))
    return None


def collect_odds_names() -> set[str]:
    names: set[str] = set()
    for y in range(2011, 2026):
        f = ODDS / (f"{y}.xlsx" if y >= 2013 else f"{y}.xls")
        if not f.exists():
            continue
        df = pd.read_excel(f)
        gs = df[(df["Series"] == "Grand Slam") & (df["Best of"] == 5)]
        names.update(gs["Winner"].dropna().astype(str))
        names.update(gs["Loser"].dropna().astype(str))
    return names


def _build_lookup(active_ids: set):
    players = pd.read_csv(PLAYERS, usecols=["player_id", "name_first", "name_last"])
    players = players.dropna(subset=["name_first", "name_last"])
    players = players[players["player_id"].isin(active_ids)]
    by_si = {}
    for pid, fn, ln in zip(players["player_id"], players["name_first"], players["name_last"]):
        by_si.setdefault((norm(ln), norm(fn)[:1]), []).append(pid)
    return by_si


def _load_overrides() -> dict:
    if not OVERRIDES.exists():
        return {}
    df = pd.read_csv(OVERRIDES)
    out = {}
    for _, row in df.iterrows():
        pid = row.get("player_id")
        if pd.notna(pid) and str(pid).strip() != "":
            out[str(row["odds_name"]).strip()] = int(pid)
    return out


def resolve_odds_names(active_ids: set, write: bool = True) -> pd.DataFrame:
    by_si = _build_lookup(active_ids)
    overrides = _load_overrides()
    names = collect_odds_names()

    rows = []
    for name in sorted(names):
        pid, method = None, "unresolved"
        parsed = parse_odds_name(name)
        if parsed:
            cands = by_si.get((parsed[1], parsed[0]), [])
            if len(cands) == 1:
                pid, method = cands[0], "initial"
        if name in overrides:
            pid, method = overrides[name], "manual"
        rows.append({"odds_name": name, "player_id": pid, "match_method": method})

    out = pd.DataFrame(rows)
    if write:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_csv(PROCESSED / "odds_name_to_id.csv", index=False)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from tennis_forecast import data as _data

    m = _data.load_atp_matches(range(2010, 2027))
    active = set(m["winner_id"]) | set(m["loser_id"])

    df = resolve_odds_names(active)
    total = len(df)
    by_method = df["match_method"].value_counts()
    resolved = df["player_id"].notna().sum()
    print(f"Total distinct GS men's odds names : {total}")
    for method in ("initial", "manual", "unresolved"):
        print(f"  {method:11s}: {by_method.get(method, 0)}")
    print(f"Resolved: {resolved}/{total} ({resolved/total:.1%})")
    print("Written to: data/processed/odds_name_to_id.csv")
    un = df[df["player_id"].isna()]
    if len(un):
        print("\n=== still unresolved (expected: Barrios M. edge case) ===")
        for n in un["odds_name"]:
            print(f"  {n}")