"""One-off probe (v3): map pbp men's-singles names to atp_players.csv via
(1) exact full name, (2) UNIQUE initial form -- but restrict the candidate
pool to players who actually appear in the ATP match data (2010-2026), i.e.
the players the filter has beliefs for. This collapses false ambiguity from
decades of historical same-surname players in the full table.

Read-only, print-only. No joins, no files written.
Scope: wimbledon + usopen, men's singles only (match_num first digit '1').
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.home() / "wimbledon-2026-forecast" / "src"))
from tennis_forecast import data  # noqa: E402

PBP = Path.home() / "wimbledon-2026-forecast" / "data" / "raw" / "pbp"
PLAYERS = Path.home() / "wimbledon-2026-forecast" / "data" / "raw" / "atp_players.csv"

SLAMS = ("wimbledon", "usopen")


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def norm(s: str) -> str:
    return strip_accents(s).strip().lower()


INITIAL_RE = re.compile(r"^([A-Za-z])\.?(?:[A-Za-z]\.?)*\s+(.+)$")


def parse_initial(name: str):
    raw = str(name).strip()
    first_tok = raw.split()[0] if raw.split() else ""
    if (("." in first_tok and len(first_tok.replace(".", "")) <= 2)
            or (len(first_tok) == 1 and first_tok.isalpha())):
        m = INITIAL_RE.match(raw)
        if m:
            return m.group(1).lower(), norm(m.group(2))
    return None


def active_male_ids() -> set:
    m = data.load_atp_matches(range(2010, 2027))
    return set(m["winner_id"]) | set(m["loser_id"])


def collect_pbp_names() -> set[str]:
    names: set[str] = set()
    files = [
        p for p in PBP.glob("*-matches.csv")
        if any(slam in p.name for slam in SLAMS)
        and "doubles" not in p.name
        and "mixed" not in p.name
    ]
    print(f"Scanning {len(files)} matches files (men's singles only) ...")
    for path in files:
        df = pd.read_csv(path, usecols=["match_num", "player1", "player2"])
        df = df[df["match_num"].astype(str).str[0] == "1"]
        for col in ("player1", "player2"):
            names.update(df[col].dropna().astype(str))
    return names


def build_lookups(active_ids: set):
    """Lookups restricted to active players (those with filter beliefs)."""
    players = pd.read_csv(PLAYERS, usecols=["player_id", "name_first", "name_last"])
    players = players.dropna(subset=["name_first", "name_last"])
    players = players[players["player_id"].isin(active_ids)]

    exact = {}
    by_surname_initial = {}
    for pid, fn, ln in zip(players["player_id"], players["name_first"], players["name_last"]):
        exact.setdefault(norm(f"{fn} {ln}"), pid)
        key = (norm(ln), norm(fn)[:1])
        by_surname_initial.setdefault(key, []).append(pid)
    return exact, by_surname_initial


def main():
    active = active_male_ids()
    print(f"Active male player pool (2010-2026): {len(active)}")
    pbp_names = collect_pbp_names()
    exact, by_si = build_lookups(active)

    matched_exact = 0
    matched_initial = 0
    ambiguous = []
    manual = []

    for name in sorted(pbp_names):
        if norm(name) in exact:
            matched_exact += 1
            continue
        parsed = parse_initial(name)
        if parsed:
            initial, surname = parsed
            cands = by_si.get((surname, initial), [])
            if len(cands) == 1:
                matched_initial += 1
            elif len(cands) >= 2:
                ambiguous.append((name, len(cands)))
            else:
                manual.append(name)
        else:
            manual.append(name)

    total = len(pbp_names)
    auto = matched_exact + matched_initial
    print()
    print(f"Distinct men's-singles names in pbp : {total}")
    print(f"  matched exactly (full name)       : {matched_exact}")
    print(f"  matched via UNIQUE initial form   : {matched_initial}")
    print(f"  auto-matched total                : {auto} ({auto/total:.1%})")
    print(f"  ambiguous initial (2+ candidates) : {len(ambiguous)}")
    print(f"  unmatched (need manual)           : {len(manual)}")
    print()
    if ambiguous:
        print("=== Ambiguous initial forms (auto-rejected, need manual) ===")
        for name, n in ambiguous:
            print(f"  {name}  ({n} candidates)")
        print()
    if manual:
        print("=== Unmatched, need manual mapping ===")
        for name in manual:
            print(f"  {name}")


if __name__ == "__main__":
    main()