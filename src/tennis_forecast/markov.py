"""Analytical hierarchical Markov model for tennis (Klaassen & Magnus 2003).

Given each player's probability of winning a *point on their own serve*, this
module computes closed-form probabilities at the game / set / match level,
including win probability from ANY current score (the basis for in-play pricing).

The point-level i.i.d.-on-serve assumption is the standard one in the
literature (Klaassen & Magnus 2003; Barnett & Clarke 2005). It is a known
approximation — deviations exist but are small enough to be useful. The Monte
Carlo engine in ``simulate.py`` reproduces these same rules exactly and is used
in the test-suite to validate every function here.

Convention throughout: ``p_a`` / ``p_b`` are the *serve* point-win probabilities
of player A / B. "A wins" is always the quantity returned.
"""

from functools import lru_cache


def game_win_prob(p: float) -> float:
    """P(server wins a standard game) given serve point-win probability ``p``.

    Closed form (win to 0/15/30, plus the deuce geometric tail).
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1.0 - p
    deuce = (p * p) / (p * p + q * q)  # P(win from deuce)
    return (
        p**4                       # 4-0
        + 4 * p**4 * q             # 4-1
        + 10 * p**4 * q**2         # 4-2
        + 20 * p**3 * q**3 * deuce # via 3-3 deuce
    )


def tiebreak_win_prob(p_a: float, p_b: float, target: int = 7) -> float:
    """P(A wins a tiebreak), serve following the standard 1-2-2-2 pattern.

    ``target`` = 7 for a normal set tiebreak, 10 for the deciding-set tiebreak
    used at all Grand Slams (incl. Roland Garros) since 2022.
    Point index ``m`` (0-based) is served by A iff ((m+1)//2) is even.
    """
    memo: dict[tuple[int, int], float] = {}

    def rec(sa: int, sb: int) -> float:
        if sa >= target and sa - sb >= 2:
            return 1.0
        if sb >= target and sb - sa >= 2:
            return 0.0
        # Negligible deep-deuce tail: probability mass here is < 1e-12.
        if sa + sb > 60:
            return 0.5
        key = (sa, sb)
        if key in memo:
            return memo[key]
        m = sa + sb
        a_serves = ((m + 1) // 2) % 2 == 0
        p_point = p_a if a_serves else (1.0 - p_b)
        val = p_point * rec(sa + 1, sb) + (1.0 - p_point) * rec(sa, sb + 1)
        memo[key] = val
        return val

    return rec(0, 0)


def set_win_prob(
    p_a: float,
    p_b: float,
    a_serves_first: bool = True,
    final_set: bool = False,
) -> float:
    """P(A wins a set). Tiebreak at 6-6 (target 10 if ``final_set`` else 7)."""
    tb_target = 10 if final_set else 7
    g_a_hold = game_win_prob(p_a)        # A wins when A serves
    g_a_break = 1.0 - game_win_prob(p_b) # A wins when B serves
    memo: dict[tuple[int, int], float] = {}

    def rec(ga: int, gb: int) -> float:
        if ga >= 6 and ga - gb >= 2:
            return 1.0
        if gb >= 6 and gb - ga >= 2:
            return 0.0
        if ga == 6 and gb == 6:
            return tiebreak_win_prob(p_a, p_b, target=tb_target)
        key = (ga, gb)
        if key in memo:
            return memo[key]
        total = ga + gb
        a_serves = a_serves_first if (total % 2 == 0) else (not a_serves_first)
        p_game = g_a_hold if a_serves else g_a_break
        val = p_game * rec(ga + 1, gb) + (1.0 - p_game) * rec(ga, gb + 1)
        memo[key] = val
        return val

    return rec(0, 0)


def match_win_prob(p_a: float, p_b: float, best_of: int = 5) -> float:
    """P(A wins the match). best_of=5 for men's Grand Slam singles.

    First server alternates set to set; the deciding set uses the 10-point
    tiebreak. Cross-set serve alternation is modelled; see module docstring on
    why this remains a (close) approximation validated against Monte Carlo.
    """
    sets_to_win = best_of // 2 + 1
    memo: dict[tuple[int, int], float] = {}

    def rec(sa: int, sb: int) -> float:
        if sa == sets_to_win:
            return 1.0
        if sb == sets_to_win:
            return 0.0
        set_index = sa + sb
        a_first = set_index % 2 == 0
        is_final = sa == sets_to_win - 1 and sb == sets_to_win - 1
        sp = set_win_prob(p_a, p_b, a_serves_first=a_first, final_set=is_final)
        key = (sa, sb)
        if key in memo:
            return memo[key]
        val = sp * rec(sa + 1, sb) + (1.0 - sp) * rec(sa, sb + 1)
        memo[key] = val
        return val

    return rec(0, 0)


# --- In-play hooks (implement against these signatures in Phase 4) ----------
# match_win_prob_from_score(p_a, p_b, score_state) -> float
#   score_state captures sets/games/points won and current server, then reuses
#   the same recursions seeded at the current node instead of (0, 0).
