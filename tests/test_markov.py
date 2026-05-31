"""Validation suite. The key invariant: the analytical Markov engine and the
independent Monte Carlo simulator must agree. If they ever diverge, one of them
is wrong — this is the regression test that protects the mathematical core.

Run:  python -m pytest tests/ -q     (or)     python tests/test_markov.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_forecast.markov import game_win_prob, set_win_prob, match_win_prob
from tennis_forecast.simulate import simulate_match_winprob


def test_game_symmetry():
    assert abs(game_win_prob(0.5) - 0.5) < 1e-12
    assert game_win_prob(0.6) > 0.6          # serving advantage amplifies edge
    assert abs(game_win_prob(0.0)) < 1e-12
    assert abs(game_win_prob(1.0) - 1.0) < 1e-12


def test_match_symmetry():
    # Equal servers -> 50/50 match regardless of format.
    assert abs(match_win_prob(0.65, 0.65, best_of=5) - 0.5) < 1e-9
    assert abs(set_win_prob(0.65, 0.65) - 0.5) < 1e-9


def test_monotonicity():
    base = match_win_prob(0.66, 0.64, best_of=5)
    better = match_win_prob(0.68, 0.64, best_of=5)
    assert better > base > 0.5


def test_format_amplifies_favourite():
    # Best-of-5 should favour the stronger player more than best-of-3.
    bo5 = match_win_prob(0.67, 0.63, best_of=5)
    bo3 = match_win_prob(0.67, 0.63, best_of=3)
    assert bo5 > bo3 > 0.5


def test_analytical_matches_monte_carlo():
    """The headline invariant: closed form == simulation, within MC error."""
    cases = [(0.65, 0.62), (0.70, 0.60), (0.62, 0.62), (0.68, 0.55)]
    for p_a, p_b in cases:
        analytic = match_win_prob(p_a, p_b, best_of=5)
        mc = simulate_match_winprob(p_a, p_b, best_of=5, n_sims=40000, seed=1)
        assert abs(analytic - mc) < 0.01, (p_a, p_b, analytic, mc)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll invariants hold: analytical engine == Monte Carlo.")
