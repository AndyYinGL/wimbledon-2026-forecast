"""Does the grass offset help ON GRASS? Compare predictions with vs without
the grass offset, evaluated only on grass matches (Wimbledon-relevant slice).
"""

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


train = serve_return_counts(load_atp_matches(range(2010, 2024)))
beliefs = run_filter(train)

test = load_atp_matches(range(2024, 2026))
grass = test[test["surface"] == "Grass"]
print(f"grass matches in 2024-2025: {len(grass)}")

for use_grass in [False, True]:
    probs, outs = predict_grass(grass, beliefs, use_grass)
    n = len(probs)
    acc = sum(1 for p, o in zip(probs, outs) if (p > 0.5) == (o == 1)) / n
    label = "WITH grass offset" if use_grass else "WITHOUT (overall only)"
    print(f"  [{label:24s}] n={n}  acc={acc:.3f}  "
          f"log-loss={log_loss(probs, outs):.4f}  brier={brier_score(probs, outs):.4f}")