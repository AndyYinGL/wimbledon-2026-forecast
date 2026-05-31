"""Monte Carlo simulation of matches and the tournament draw.

The match simulator plays out points under the exact scoring rules, so it is
the ground-truth that validates ``markov.py``. The tournament simulator runs
the remaining draw many times to produce the *distribution of game events*:
each player's probability of reaching each round and of winning the title.
"""

from __future__ import annotations

import random
from collections import defaultdict


def simulate_match(
    p_a: float,
    p_b: float,
    best_of: int = 5,
    rng: random.Random | None = None,
) -> bool:
    """Simulate one match point-by-point. Returns True iff A wins.

    Mirrors the rules in ``markov.py`` (tiebreak at 6-6, 10-point deciding-set
    tiebreak, serve alternation within and across sets).
    """
    rng = rng or random
    sets_to_win = best_of // 2 + 1
    sets_a = sets_b = 0
    set_index = 0

    while sets_a < sets_to_win and sets_b < sets_to_win:
        a_serves_first = set_index % 2 == 0
        final_set = sets_a == sets_to_win - 1 and sets_b == sets_to_win - 1
        a_won_set = _simulate_set(p_a, p_b, a_serves_first, final_set, rng)
        if a_won_set:
            sets_a += 1
        else:
            sets_b += 1
        set_index += 1

    return sets_a == sets_to_win


def _simulate_set(p_a, p_b, a_serves_first, final_set, rng) -> bool:
    games_a = games_b = 0
    while True:
        if games_a >= 6 and games_a - games_b >= 2:
            return True
        if games_b >= 6 and games_b - games_a >= 2:
            return False
        if games_a == 6 and games_b == 6:
            return _simulate_tiebreak(p_a, p_b, 10 if final_set else 7, rng)
        total = games_a + games_b
        a_serves = a_serves_first if total % 2 == 0 else not a_serves_first
        p_serve = p_a if a_serves else p_b
        server_wins_game = _simulate_game(p_serve, rng)
        a_won_game = server_wins_game if a_serves else not server_wins_game
        if a_won_game:
            games_a += 1
        else:
            games_b += 1


def _simulate_game(p: float, rng) -> bool:
    """Return True iff the server wins the game."""
    s = r = 0
    while True:
        if s >= 4 and s - r >= 2:
            return True
        if r >= 4 and r - s >= 2:
            return False
        if rng.random() < p:
            s += 1
        else:
            r += 1


def _simulate_tiebreak(p_a, p_b, target, rng) -> bool:
    sa = sb = 0
    while True:
        if sa >= target and sa - sb >= 2:
            return True
        if sb >= target and sb - sa >= 2:
            return False
        m = sa + sb
        a_serves = ((m + 1) // 2) % 2 == 0
        p_point = p_a if a_serves else (1.0 - p_b)
        if rng.random() < p_point:
            sa += 1
        else:
            sb += 1


def simulate_match_winprob(p_a, p_b, best_of=5, n_sims=20000, seed=0) -> float:
    """Monte Carlo estimate of P(A wins). Used to validate markov.match_win_prob."""
    rng = random.Random(seed)
    wins = sum(simulate_match(p_a, p_b, best_of, rng) for _ in range(n_sims))
    return wins / n_sims


# --- Tournament draw --------------------------------------------------------

def simulate_tournament(
    bracket: list[str],
    p_serve: dict[str, float],
    best_of: int = 5,
    n_sims: int = 10000,
    seed: int = 0,
) -> dict:
    """Simulate a single-elimination draw ``n_sims`` times.

    Args:
        bracket: player ids in draw order; length must be a power of two.
                 Use a sentinel id with a bye-tier p_serve to represent gaps.
        p_serve: map of player id -> serve point-win probability (clay-fitted).
        best_of: 5 for men's Grand Slam.

    Returns dict with, per player:
        'champion'  : P(win title)
        'reach'     : {round_size: P(reach that round)}  e.g. 16 -> P(reach R16)
    """
    n = len(bracket)
    assert n & (n - 1) == 0, "bracket size must be a power of two"
    rng = random.Random(seed)

    champion = defaultdict(int)
    reach = defaultdict(lambda: defaultdict(int))

    for _ in range(n_sims):
        alive = list(bracket)
        round_size = n
        while round_size > 1:
            for pid in alive:
                reach[pid][round_size] += 1
            nxt = []
            for i in range(0, round_size, 2):
                a, b = alive[i], alive[i + 1]
                pa = p_serve[a]
                pb = p_serve[b]
                from .markov import match_win_prob  # local import avoids cycle
                # Use analytical prob as the per-match coin; cheaper & exact-enough.
                a_wins = rng.random() < match_win_prob(pa, pb, best_of)
                nxt.append(a if a_wins else b)
            alive = nxt
            round_size //= 2
        champion[alive[0]] += 1
        reach[alive[0]][1] += 1

    return {
        "champion": {pid: champion[pid] / n_sims for pid in bracket},
        "reach": {
            pid: {rs: reach[pid][rs] / n_sims for rs in sorted(reach[pid])}
            for pid in bracket
        },
    }
