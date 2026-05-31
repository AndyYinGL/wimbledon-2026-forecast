"""Scan TAU2 (grass shrinkage) via run_filter's tau2 parameter. Evaluate ON
GRASS only, compared to the no-offset baseline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_forecast.data import load_atp_matches, serve_return_counts
from tennis_forecast.filter import run_filter, MU, _logistic
from tennis_forecast.markov import match_win_prob
from tennis_forecast.pricing import log_loss, brier_score


def predict_grass(test_df, beliefs, use_grass):
    def p_serve(s, r):
        bs, br = beliefs.get(s), beliefs.get(r)
        if bs is None or br is None:
            return None
        sv, rt = bs.serve_mean, br.return_mean
        if use_grass:
            sv += bs.serve_grass_mean
            rt += br.return_grass_mean
        return _logistic(MU + sv - rt)
    probs, outs = [], []
    for row in test_df.itertuples(index=False):
        if row.winner_id < row.loser_id:
            a, b, w = row.winner_id, row.loser_id, 1
        else:
            a, b, w = row.loser_id, row.winner_id, 0
        pa, pb = p_serve(a, b), p_serve(b, a)
        if pa is None or pb is None:
            continue
        probs.append(match_win_prob(pa, pb, 5 if row.tourney_level == "G" else 3))
        outs.append(w)
    return probs, outs


train_obs = serve_return_counts(load_atp_matches(range(2010, 2024)))
test = load_atp_matches(range(2024, 2026))
grass = test[test["surface"] == "Grass"]

b0 = run_filter(train_obs)
p, o = predict_grass(grass, b0, use_grass=False)
n = len(p)
acc = sum(1 for x, y in zip(p, o) if (x > 0.5) == (y == 1)) / n
print("baseline NO-offset:  acc=%.3f log-loss=%.4f brier=%.4f" % (acc, log_loss(p, o), brier_score(p, o)))
print("%8s %7s %9s %7s" % ("tau2", "acc", "log-loss", "brier"))
for tau2 in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    beliefs = run_filter(train_obs, tau2=tau2)
    p, o = predict_grass(grass, beliefs, use_grass=True)
    n = len(p)
    acc = sum(1 for x, y in zip(p, o) if (x > 0.5) == (y == 1)) / n
    print("%8.3f %7.3f %9.4f %7.4f" % (tau2, acc, log_loss(p, o), brier_score(p, o)))