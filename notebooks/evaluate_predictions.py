"""Walk-forward out-of-sample evaluation with temperature-scaling calibration.

For each test year Y: train filter on all data BEFORE Y-1, fit temperature on
year Y-1 (holdout), evaluate on year Y. No look-ahead: training always precedes
the holdout, which precedes the test year.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from tennis_forecast.data import load_atp_matches, serve_return_counts
from tennis_forecast.filter import run_filter, MU, _logistic
from tennis_forecast.markov import match_win_prob
from tennis_forecast.pricing import (
    log_loss, brier_score, fit_temperature, apply_temperature,
)


def make_pred_set(test_df, beliefs):
    def p_serve(server, returner):
        bs, br = beliefs.get(server), beliefs.get(returner)
        if bs is None or br is None:
            return None
        return _logistic(MU + bs.serve_mean - br.return_mean)

    probs, outs = [], []
    for r in test_df.itertuples(index=False):
        if r.winner_id < r.loser_id:
            a, b, a_won = r.winner_id, r.loser_id, 1
        else:
            a, b, a_won = r.loser_id, r.winner_id, 0
        pa, pb = p_serve(a, b), p_serve(b, a)
        if pa is None or pb is None:
            continue
        best_of = 5 if r.tourney_level == "G" else 3
        probs.append(match_win_prob(pa, pb, best_of))
        outs.append(a_won)
    return probs, outs


def report(probs, outcomes, label):
    n = len(probs)
    acc = sum(1 for p, o in zip(probs, outcomes) if (p > 0.5) == (o == 1)) / n
    print(f"  [{label:28s}] n={n:4d}  acc={acc:.3f}  "
          f"log-loss={log_loss(probs, outcomes):.4f}  "
          f"brier={brier_score(probs, outcomes):.4f}")


def evaluate_year(test_year):
    """Train on everything up to test_year-1 (inclusive), fit T on that same
    last year as a holdout proxy, evaluate on test_year. Uses all available
    history (no year wasted), still strictly no future leakage into training."""
    print(f"\n=== test year {test_year} (train <= {test_year-1}) ===")
    train = serve_return_counts(load_atp_matches(range(2010, test_year)))
    beliefs = run_filter(train)

    # fit temperature on the most recent training year as a holdout proxy
    hold = load_atp_matches(range(test_year - 1, test_year))
    hold = hold[hold["surface"] != "Carpet"]
    hp, ho = make_pred_set(hold, beliefs)
    T, _ = fit_temperature(hp, ho)
    print(f"  fitted temperature: T = {T:.2f}")

    test = load_atp_matches(range(test_year, test_year + 1))
    test = test[test["surface"] != "Carpet"]
    tp, to = make_pred_set(test, beliefs)

    report(tp, to, "RAW")
    tp_cal = [apply_temperature(p, T) for p in tp]
    report(tp_cal, to, "CALIBRATED")
    return tp_cal, to


# Walk-forward over the two most recent complete-ish years.
for year in [2024, 2025]:
    evaluate_year(year)