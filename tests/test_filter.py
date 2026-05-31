"""Synthetic-recovery + drift + grass-shrinkage tests for the filter.

We invent players with KNOWN true skills, simulate matches, run the filter, and
check it recovers the truth. If the filter's math is wrong, these fail.
"""

import sys
import math
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tennis_forecast.filter import run_filter, MU, _logistic


def _make_synthetic(n_players=30, n_matches=4000, seed=0):
    """Generate matches from known true serve/return skills."""
    rng = random.Random(seed)
    true_serve = {p: rng.gauss(0, 0.5) for p in range(n_players)}
    true_return = {p: rng.gauss(0, 0.5) for p in range(n_players)}

    rows = []
    for _ in range(n_matches):
        a, b = rng.sample(range(n_players), 2)
        for server, returner in [(a, b), (b, a)]:
            n = 60
            p = _logistic(MU + true_serve[server] - true_return[returner])
            won = sum(1 for _ in range(n) if rng.random() < p)
            rows.append({
                "tourney_date": pd.Timestamp("2015-01-01"),
                "surface": "Hard",
                "server_id": server, "returner_id": returner,
                "points_won": won, "points_played": n,
            })
    obs = pd.DataFrame(rows)
    return obs, true_serve, true_return


def test_synthetic_recovery():
    obs, true_serve, true_return = _make_synthetic()
    beliefs = run_filter(obs)

    def _centered_corr(est, truth):
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

    assert corr_serve > 0.9, corr_serve
    assert corr_return > 0.9, corr_return


def test_drift_tracks_changing_skill():
    """A player's true serve skill RISES over time; with drift the filter should
    estimate them stronger late than early."""
    rng = random.Random(7)

    star = 0
    n_others = 20
    rows = []
    start = pd.Timestamp("2015-01-01")

    for week in range(150):
        date = start + pd.Timedelta(weeks=week)
        true_serve_star = -0.4 + 0.8 * (week / 150)
        opp = rng.randint(1, n_others)
        for server, returner, s_skill in [
            (star, opp, true_serve_star),
            (opp, star, 0.0),
        ]:
            n = 60
            p = _logistic(MU + s_skill - 0.0)
            won = sum(1 for _ in range(n) if rng.random() < p)
            rows.append({
                "tourney_date": date,
                "surface": "Hard",
                "server_id": server, "returner_id": returner,
                "points_won": won, "points_played": n,
            })

    obs = pd.DataFrame(rows)

    early = run_filter(obs[obs["tourney_date"] < start + pd.Timedelta(weeks=50)])
    late = run_filter(obs)

    early_skill = early[star].serve_mean
    late_skill = late[star].serve_mean
    print(f"star serve skill  early: {early_skill:.3f}   late: {late_skill:.3f}")
    assert late_skill > early_skill + 0.1, (early_skill, late_skill)


def test_grass_shrinkage():
    """A player with a real grass bonus but FEW grass matches: the grass offset
    should move in the right direction but stay shrunk (well below the raw +0.6)."""
    rng = random.Random(3)

    star = 0
    rows = []
    d = pd.Timestamp("2018-06-01")
    for _ in range(200):
        opp = rng.randint(1, 15)
        for srv, ret in [(star, opp), (opp, star)]:
            n = 60
            p = _logistic(MU + 0.0)
            won = sum(1 for _ in range(n) if rng.random() < p)
            rows.append({"tourney_date": d, "surface": "Hard",
                         "server_id": srv, "returner_id": ret,
                         "points_won": won, "points_played": n})
    for _ in range(8):
        opp = rng.randint(1, 15)
        n = 60
        p = _logistic(MU + 0.6)
        won = sum(1 for _ in range(n) if rng.random() < p)
        rows.append({"tourney_date": d, "surface": "Grass",
                     "server_id": star, "returner_id": opp,
                     "points_won": won, "points_played": n})

    obs = pd.DataFrame(rows)
    b = run_filter(obs)[star]
    print(f"grass serve offset (true +0.6, only 8 matches): {b.serve_grass_mean:.3f}")
    assert b.serve_grass_mean > 0.05, b.serve_grass_mean
    assert b.serve_grass_mean < 0.6, b.serve_grass_mean


if __name__ == "__main__":
    test_synthetic_recovery()
    test_drift_tracks_changing_skill()
    test_grass_shrinkage()
    print("\nPASS: recovery, drift, and grass shrinkage all work.")