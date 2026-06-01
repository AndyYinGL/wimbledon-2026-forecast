"""Name normalisation: map point-by-point player names to Sackmann player_ids.

Resolution order (highest priority last so it wins):
  1. exact full name match (within the active player pool)
  2. unique initial-form match ("R. Federer" -> sole R-surnamed player)
  3. manual override table (hand-verified, authoritative)

The candidate pool is restricted to players appearing in ATP match data, i.e.
the players the filter holds beliefs for; this both removes false ambiguity
from historical same-surname players and guarantees a matched id is usable.

Writes an auditable mapping to data/processed/pbp_name_to_id.csv with a
match_method column (exact / initial / manual) so every link can be reviewed.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PLAYERS = RAW / "atp_players.csv"
OVERRIDES = RAW / "pbp_name_overrides.csv"

SLAMS = ("wimbledon", "usopen")


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def norm(s: str) -> str:
    return strip_accents(s).strip().lower()


INITIAL_RE = re.compile(r"^([A-Za-z])\.?(?:[A-Za-z]\.?)*\s+(.+)$")


def parse_initial(name: str):
    """('r', 'federer') for 'R. Federer'/'C Alcaraz'; None for full names."""
    raw = str(name).strip()
    first_tok = raw.split()[0] if raw.split() else ""
    if (("." in first_tok and len(first_tok.replace(".", "")) <= 2)
            or (len(first_tok) == 1 and first_tok.isalpha())):
        m = INITIAL_RE.match(raw)
        if m:
            return m.group(1).lower(), norm(m.group(2))
    return None


def _build_lookups(active_ids: set):
    players = pd.read_csv(PLAYERS, usecols=["player_id", "name_first", "name_last"])
    players = players.dropna(subset=["name_first", "name_last"])
    players = players[players["player_id"].isin(active_ids)]

    exact = {}                       # norm full name -> id
    by_surname_initial = {}          # (surname_norm, initial) -> [ids]
    for pid, fn, ln in zip(players["player_id"], players["name_first"], players["name_last"]):
        exact.setdefault(norm(f"{fn} {ln}"), pid)
        key = (norm(ln), norm(fn)[:1])
        by_surname_initial.setdefault(key, []).append(pid)
    return exact, by_surname_initial


def _load_overrides() -> dict:
    """pbp_name -> player_id from the hand-verified table (blank ids skipped)."""
    if not OVERRIDES.exists():
        return {}
    df = pd.read_csv(OVERRIDES)
    out = {}
    for _, row in df.iterrows():
        pid = row.get("player_id")
        if pd.notna(pid) and str(pid).strip() != "":
            out[str(row["pbp_name"]).strip()] = int(pid)
    return out


def collect_pbp_names() -> set[str]:
    """All men's-singles player names across wimbledon/usopen matches files."""
    names: set[str] = set()
    files = [
        p for p in (RAW / "pbp").glob("*-matches.csv")
        if any(slam in p.name for slam in SLAMS)
        and "doubles" not in p.name and "mixed" not in p.name
    ]
    for path in files:
        df = pd.read_csv(path, usecols=["match_num", "player1", "player2"])
        df = df[df["match_num"].astype(str).str[0] == "1"]
        for col in ("player1", "player2"):
            names.update(df[col].dropna().astype(str))
    return names


def resolve_names(active_ids: set, write: bool = True) -> pd.DataFrame:
    """Resolve every pbp men's-singles name to a player_id where possible.

    Returns a DataFrame: pbp_name, player_id (nullable), match_method.
    Also writes it to data/processed/pbp_name_to_id.csv when `write`.
    """
    exact, by_si = _build_lookups(active_ids)
    overrides = _load_overrides()
    pbp_names = collect_pbp_names()

    rows = []
    for name in sorted(pbp_names):
        pid, method = None, "unresolved"
        if norm(name) in exact:
            pid, method = exact[norm(name)], "exact"
        else:
            parsed = parse_initial(name)
            if parsed:
                cands = by_si.get((parsed[1], parsed[0]), [])
                if len(cands) == 1:
                    pid, method = cands[0], "initial"
        # manual override wins if present
        if name in overrides:
            pid, method = overrides[name], "manual"
        rows.append({"pbp_name": name, "player_id": pid, "match_method": method})

    out = pd.DataFrame(rows)
    if write:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_csv(PROCESSED / "pbp_name_to_id.csv", index=False)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from tennis_forecast import data

    m = data.load_atp_matches(range(2010, 2027))
    active = set(m["winner_id"]) | set(m["loser_id"])

    df = resolve_names(active)
    total = len(df)
    by_method = df["match_method"].value_counts()
    resolved = df["player_id"].notna().sum()

    print(f"Total distinct pbp men's-singles names : {total}")
    for method in ("exact", "initial", "manual", "unresolved"):
        print(f"  {method:11s}: {by_method.get(method, 0)}")
    print(f"Resolved: {resolved}/{total} ({resolved/total:.1%})")
    print("Written to: data/processed/pbp_name_to_id.csv")

    unresolved = df[df["player_id"].isna()]
    if len(unresolved):
        print("\n=== Still unresolved ===")
        for n in unresolved["pbp_name"]:
            print(f"  {n}")