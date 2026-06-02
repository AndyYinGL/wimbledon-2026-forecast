"""Probe: can tennis-data odds player names map to Sackmann ids?
Odds format is 'Surname X.' (e.g. 'Sinner J.', 'Van De Zandschulp B.').
Read-only, print-only. Scope: Grand Slam men's (Best of == 5).
Candidate pool = active men 2010-2026 (same as pbp resolution).
"""
import sys
from pathlib import Path

import unicodedata
import pandas as pd

ROOT = Path.home() / "wimbledon-2026-forecast"
sys.path.insert(0, str(ROOT / "src"))
from tennis_forecast import data

ODDS = ROOT / "data" / "raw" / "odds"
PLAYERS = ROOT / "data" / "raw" / "atp_players.csv"


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c))


def norm(s):
    return strip_accents(s).strip().lower()


def collect_odds_names():
    names = set()
    for y in range(2011, 2025):
        f = ODDS / (f"{y}.xlsx" if y >= 2013 else f"{y}.xls")
        df = pd.read_excel(f)
        gs = df[(df["Series"] == "Grand Slam") & (df["Best of"] == 5)]
        names.update(gs["Winner"].dropna().astype(str))
        names.update(gs["Loser"].dropna().astype(str))
    return names


def parse_odds_name(name):
    """'Sinner J.' -> ('j', 'sinner'); 'Van De Zandschulp B.' -> ('b', 'van de zandschulp')."""
    raw = str(name).strip()
    toks = raw.split()
    if len(toks) < 2:
        return None
    # last token is the initial (e.g. 'J.'); rest is surname
    initial_tok = toks[-1]
    if initial_tok.replace(".", "").isalpha() and len(initial_tok.replace(".", "")) <= 2:
        initial = initial_tok.replace(".", "")[0].lower()
        surname = norm(" ".join(toks[:-1]))
        return initial, surname
    return None


def build_lookups(active_ids):
    players = pd.read_csv(PLAYERS, usecols=["player_id", "name_first", "name_last"])
    players = players.dropna(subset=["name_first", "name_last"])
    players = players[players["player_id"].isin(active_ids)]
    by_si = {}
    for pid, fn, ln in zip(players["player_id"], players["name_first"], players["name_last"]):
        key = (norm(ln), norm(fn)[:1])
        by_si.setdefault(key, []).append(pid)
    return by_si


def main():
    m = data.load_atp_matches(range(2010, 2027))
    active = set(m["winner_id"]) | set(m["loser_id"])
    by_si = build_lookups(active)
    names = collect_odds_names()

    matched, ambiguous, unmatched = 0, [], []
    for name in sorted(names):
        p = parse_odds_name(name)
        if p is None:
            unmatched.append(name); continue
        cands = by_si.get((p[1], p[0]), [])
        if len(cands) == 1:
            matched += 1
        elif len(cands) >= 2:
            ambiguous.append((name, len(cands)))
        else:
            unmatched.append(name)

    total = len(names)
    print(f"distinct GS men's odds names: {total}")
    print(f"  unique-initial matched : {matched} ({matched/total:.1%})")
    print(f"  ambiguous (2+)         : {len(ambiguous)}")
    print(f"  unmatched              : {len(unmatched)}")
    print()
    if ambiguous:
        print("=== ambiguous ===")
        for n, c in ambiguous: print(f"  {n}  ({c})")
        print()
    if unmatched:
        print("=== unmatched ===")
        for n in unmatched: print(f"  {n}")


if __name__ == "__main__":
    main()