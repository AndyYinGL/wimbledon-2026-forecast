"""Tune gamma with walk-forward CV across multiple test years.

For each gamma and each test year Y: train on 2010..Y-1, evaluate RAW
calibration on Y. Average over years => a robust choice of gamma, not one
overfit to a single season.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_forecast.data import load_atp_matches, serve_return_counts
from tennis_forecast.filter import run_filter, MU, _logistic
from tennis_forecast.markov import match_win_prob
from tennis_forecast.pricing import log_loss, brier_score


def make_pred_set(test_df, beliefs):
    def p_serve(s, r):
        bs, br = beliefs.get(s), beliefs.get(r)
        if bs is None or br is None:
            return None
        return _logistic(MU + bs.serve_mean - br.return_mean)
    probs, outs = [], []
    for r in test_df.itertuples(index=False):
        if r.winner_id < r.loser_id:
            a, b, w = r.winner_id, r.loser_id, 1
        else:
            a, b, w = r.loser_id, r.winner_id, 0
        pa, pb = p_serve(a, b), p_serve(b, a)
        if pa is None or pb is None:
            continue
        best_of = 5 if r.tourney_level == "G" else 3
        probs.append(match_win_prob(pa, pb, best_of))
        outs.append(w)
    return probs, outs


TEST_YEARS = [2022, 2023, 2024, 2025]
GAMMAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

# Pre-load each test year once.
test_sets = {}
for y in TEST_YEARS:
    df = load_atp_matches(range(y, y + 1))
    test_sets[y] = df[df["surface"] != "Carpet"]

print(f"{'gamma':>6} | " + " ".join(f"{y:>7}" for y in TEST_YEARS) + f" | {'mean_ll':>8} {'mean_acc':>8}")
print("-" * 72)

for gamma in GAMMAS:
    year_ll, year_acc = [], []
    for y in TEST_YEARS:
        train = serve_return_counts(load_atp_matches(range(2010, y)))
        beliefs = run_filter(train, gamma=gamma)
        probs, outs = make_pred_set(test_sets[y], beliefs)
        n = len(probs)
        acc = sum(1 for p, o in zip(probs, outs) if (p > 0.5) == (o == 1)) / n
        ll = log_loss(probs, outs)
        year_ll.append(ll)
        year_acc.append(acc)
    mean_ll = sum(year_ll) / len(year_ll)
    mean_acc = sum(year_acc) / len(year_acc)
    print(f"{gamma:>6.2f} | " + " ".join(f"{ll:>7.3f}" for ll in year_ll)
          + f" | {mean_ll:>8.4f} {mean_acc:>8.3f}")