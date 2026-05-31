"""Out-of-sample evaluation with temperature-scaling calibration.

Train filter on 2010-2023. Fit temperature T on 2024 (holdout). Evaluate on
2025, before and after calibration. T is fit on data the test set never sees,
so there is no leakage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from tennis_forecast.data import load_atp_matches, serve_return_counts
from tennis_forecast.filter import run_filter, MU, _logistic
from tennis_forecast.markov import match_win_prob
from tennis_forecast.pricing import (
    log_loss, brier_score, fit_temperature, apply_temperature,
)


def make_pred_set(test_df, beliefs):
    """Return (probs, outcomes) for matches in test_df, A/B assigned by id."""
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
    print(f"\n[{label}]  n={n}  acc={acc:.3f}  "
          f"log-loss={log_loss(probs, outcomes):.4f}  "
          f"brier={brier_score(probs, outcomes):.4f}")
    p, o = np.array(probs), np.array(outcomes)
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        hi = lo + 0.2 if lo < 0.8 else 1.01
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"    pred {lo:.1f}-{hi:.1f}: {m.sum():4d}  "
                  f"mean_pred {p[m].mean():.3f}  actual {o[m].mean():.3f}")


# 1. Train on 2010-2023.
train = serve_return_counts(load_atp_matches(range(2010, 2024)))
beliefs = run_filter(train)
print(f"trained on {len(train)} obs, {len(beliefs)} players")

# 2. Fit temperature on 2024 holdout.
hold = load_atp_matches(range(2024, 2025))
hold = hold[hold["surface"] != "Carpet"]
hp, ho = make_pred_set(hold, beliefs)
T, _ = fit_temperature(hp, ho)
print(f"fitted temperature on 2024: T = {T:.2f}")

# 3. Evaluate on 2025, before vs after calibration.
test = load_atp_matches(range(2025, 2026))
test = test[test["surface"] != "Carpet"]
tp, to = make_pred_set(test, beliefs)

report(tp, to, "2025 RAW (uncalibrated)")
tp_cal = [apply_temperature(p, T) for p in tp]
report(tp_cal, to, "2025 CALIBRATED (temperature scaled)")