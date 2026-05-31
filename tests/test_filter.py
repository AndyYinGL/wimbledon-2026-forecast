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

def test_drift_tracks_changing_skill():
    """A player's true serve skill RISES over time; with drift the filter should
    estimate them stronger in the late period than the early period."""
    import pandas as pd
    rng = random.Random(7)
    from tennis_forecast.filter import run_filter

    star = 0          # the improving player
    n_others = 20
    rows = []
    start = pd.Timestamp("2015-01-01")

    for week in range(150):                       # ~3 years of weekly matches
        date = start + pd.Timedelta(weeks=week)
        true_serve_star = -0.4 + 0.8 * (week / 150)   # rises from -0.4 to +0.4
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

    # Filter on the first third vs the full history; the star's serve estimate
    # should be clearly higher at the end than early on.
    early = run_filter(obs[obs["tourney_date"] < start + pd.Timedelta(weeks=50)])
    late = run_filter(obs)

    early_skill = early[star].serve_mean
    late_skill = late[star].serve_mean
    print(f"star serve skill  early: {early_skill:.3f}   late: {late_skill:.3f}")
    assert late_skill > early_skill + 0.2, (early_skill, late_skill)

def test_grass_shrinkage():
    """A player with a genuine grass bonus but FEW grass matches: the grass
    offset should move in the right direction but stay small (shrunk), not blow
    up. A player with NO grass data should keep a ~0 grass offset."""
    import pandas as pd
    rng = random.Random(3)
    from tennis_forecast.filter import run_filter

    star = 0
    rows = []
    d = pd.Timestamp("2018-06-01")
    # 200 hard-court matches (lots of overall data), grass offset should stay ~0
    for _ in range(200):
        opp = rng.randint(1, 15)
        for srv, ret in [(star, opp), (opp, star)]:
            n = 60
            p = _logistic(MU + 0.0)
            won = sum(1 for _ in range(n) if rng.random() < p)
            rows.append({"tourney_date": d, "surface": "Hard",
                         "server_id": srv, "returner_id": ret,
                         "points_won": won, "points_played": n})
    # only 8 grass matches, but star serves MUCH better on grass (true +0.6)
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
    # Shrinkage: offset is positive (caught the grass bonus) but well below the
    # raw +0.6 (pulled toward 0 because grass data is thin).
    assert b.serve_grass_mean > 0.05, b.serve_grass_mean
    assert b.serve_grass_mean < 0.6, b.serve_grass_mean

if __name__ == "__main__":
    test_synthetic_recovery()
    test_drift_tracks_changing_skill()
    test_grass_shrinkage()
    print("\nPASS: recovery, drift, and grass shrinkage all work.")