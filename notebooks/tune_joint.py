"""Joint scan of gamma (drift) x tau2 (grass shrinkage), evaluated on 2024+2025
overall (calibration-free RAW log-loss) to confirm the chosen combo is robust."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tennis_forecast.data import load_atp_matches, serve_return_counts
from tennis_forecast.filter import run_filter, MU, _logistic
from tennis_forecast.markov import match_win_prob
from tennis_forecast.pricing import log_loss


def make_pred_set(test_df, beliefs):
    def p_serve(s, r, grass):
        bs, br = beliefs.get(s), beliefs.get(r)
        if bs is None or br is None:
            return None
        sv, rt = bs.serve_mean, br.return_mean
        if grass:
            sv += bs.serve_grass_mean
            rt += br.return_grass_mean
        return _logistic(MU + sv - rt)
    probs, outs = [], []
    for row in test_df.itertuples(index=False):
        if row.winner_id < row.loser_id:
            a, b, w = row.winner_id, row.loser_id, 1
        else:
            a, b, w = row.loser_id, row.winner_id, 0
        grass = (row.surface == "Grass")
        pa, pb = p_serve(a, b, grass), p_serve(b, a, grass)
        if pa is None or pb is None:
            continue
        probs.append(match_win_prob(pa, pb, 5 if row.tourney_level == "G" else 3))
        outs.append(w)
    return probs, outs


# Train through 2024, evaluate on 2025 (the freshest fully-out-of-sample year).
train = serve_return_counts(load_atp_matches(range(2010, 2025)))
test = load_atp_matches(range(2025, 2026))
test = test[test["surface"] != "Carpet"]

print(f"{'gamma':>6} {'tau2':>7} {'acc':>7} {'log-loss':>9}")
for gamma in [0.08, 0.10, 0.12]:
    for tau2 in [0.005, 0.02, 0.05]:
        beliefs = run_filter(train, gamma=gamma, tau2=tau2)
        probs, outs = make_pred_set(test, beliefs)
        n = len(probs)
        acc = sum(1 for p, o in zip(probs, outs) if (p > 0.5) == (o == 1)) / n
        ll = log_loss(probs, outs)
        print(f"{gamma:>6.2f} {tau2:>7.3f} {acc:>7.3f} {ll:>9.4f}")