"""Synthetic-recovery test for the Stage-1 filter.

We invent players with KNOWN true serve/return skills, simulate many matches
from those skills, run the filter, and check it recovers the truth. If the
filter's math is wrong this test fails — that's its whole job.
"""

import sys
import math
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tennis_forecast.filter import run_filter, MU, _logistic


def _make_synthetic(n_players=30, n_matches=4000, seed=0):
    """Generate matches from known true skills.

    Returns (observations DataFrame, true_serve dict, true_return dict).
    """
    rng = random.Random(seed)
    # True skills drawn around 0 (the same scale the filter assumes).
    true_serve = {p: rng.gauss(0, 0.5) for p in range(n_players)}
    true_return = {p: rng.gauss(0, 0.5) for p in range(n_players)}

    rows = []
    for _ in range(n_matches):
        a, b = rng.sample(range(n_players), 2)
        # Each "match" = ~10 service games each, ~6 points per game ≈ 60 serve pts.
        for server, returner in [(a, b), (b, a)]:
            n = 60
            p = _logistic(MU + true_serve[server] - true_return[returner])
            won = sum(1 for _ in range(n) if rng.random() < p)
            rows.append({
                "server_id": server, "returner_id": returner,
                "points_won": won, "points_played": n,
            })
    obs = pd.DataFrame(rows)
    return obs, true_serve, true_return
def test_synthetic_recovery():
    obs, true_serve, true_return = _make_synthetic()
    beliefs = run_filter(obs)

    # Skills are only identified up to a shared shift between serve & return,
    # so compare each estimate to truth after removing the mean offset.
    def _centered_corr(est: dict, truth: dict):
        ids = list(truth)
        e = [est[i] for i in ids]
        t = [truth[i] for i in ids]
        e_off = sum(e) / len(e)
        t_off = sum(t) / len(t)
        ec = [x - e_off for x in e]
        tc = [x - t_off for x in t]
        num = sum(a * b for a, b in zip(ec, tc))
        den = math.sqrt(sum(a * a for a in ec) * sum(b * b for b in tc))
        return num / den

    est_serve = {i: b.serve_mean for i, b in beliefs.items()}
    est_return = {i: b.return_mean for i, b in beliefs.items()}

    corr_serve = _centered_corr(est_serve, true_serve)
    corr_return = _centered_corr(est_return, true_return)

    print(f"serve-skill recovery corr:  {corr_serve:.3f}")
    print(f"return-skill recovery corr: {corr_return:.3f}")

    # With 4000 matches the filter should track the truth strongly.
    assert corr_serve > 0.9, corr_serve
    assert corr_return > 0.9, corr_return


if __name__ == "__main__":
    test_synthetic_recovery()
    print("\nPASS: filter recovers known serve & return skills.")
