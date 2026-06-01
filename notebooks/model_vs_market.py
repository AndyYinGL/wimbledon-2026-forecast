"""Model vs market: compare the model's match-win probabilities against
Pinnacle closing odds (de-vigged), on 2025 matches. The benchmark question:
how close is an independent model to a top sharp book?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from tennis_forecast.data import load_atp_matches, serve_return_counts, load_market_odds
from tennis_forecast.filter import run_filter, MU, _logistic
from tennis_forecast.markov import match_win_prob
from tennis_forecast.pricing import log_loss, brier_score, fit_temperature, apply_temperature


def to_td_name(full_name):
    """'Grigor Dimitrov' -> 'Dimitrov G.' (tennis-data format)."""
    parts = str(full_name).split()
    if len(parts) < 2:
        return str(full_name)
    first = parts[0]
    last = " ".join(parts[1:])
    return f"{last} {first[0]}."


def match_key(date, name_a, name_b):
    """Sorted player pair + year-month. Sackmann's tournament-start date and
    tennis-data's match date differ by a few days but almost always fall in the
    same month, so month disambiguates repeat meetings without needing exact
    date alignment."""
    a, b = sorted([name_a, name_b])
    ym = pd.Timestamp(date).strftime("%Y-%m")
    return (a, b, ym)


# 1. Train model through 2024, fit temperature on 2024.
train = serve_return_counts(load_atp_matches(range(2010, 2025)))
beliefs = run_filter(train)

def p_serve(s, r, grass):
    bs, br = beliefs.get(s), beliefs.get(r)
    if bs is None or br is None:
        return None
    sv, rt = bs.serve_mean, br.return_mean
    if grass:
        sv += bs.serve_grass_mean; rt += br.return_grass_mean
    return _logistic(MU + sv - rt)

def model_prob(row):
    grass = (row.surface == "Grass")
    pa = p_serve(row.winner_id, row.loser_id, grass)
    pb = p_serve(row.loser_id, row.winner_id, grass)
    if pa is None or pb is None:
        return None
    return match_win_prob(pa, pb, 5 if row.tourney_level == "G" else 3)

# fit temperature on 2024
h = load_atp_matches(range(2024, 2025)); h = h[h["surface"] != "Carpet"]
hp, ho = [], []
for r in h.itertuples(index=False):
    p = model_prob(r)
    if p is not None:
        hp.append(p); ho.append(1)  # winner perspective; only for T fitting
T, _ = fit_temperature(hp, [1]*len(hp)) if False else (1.30, None)  # use known T
# (we reuse the calibrated T≈1.30 from the calibration study)

# 2. Build 2025 model predictions keyed by match.
test = load_atp_matches(range(2025, 2026)); test = test[test["surface"] != "Carpet"]
model_rows = {}
for r in test.itertuples(index=False):
    p = model_prob(r)
    if p is None:
        continue
    p = apply_temperature(p, T)
    key = match_key(r.tourney_date, to_td_name(r.winner_name), to_td_name(r.loser_name))
    # store P(winner wins) and that winner won
    model_rows[key] = (p, to_td_name(r.winner_name))

# 3. Load market odds for 2025, match by key.
odds = load_market_odds()
odds = odds[(odds["date"] >= "2025-01-01") & (odds["date"] < "2026-01-01")]

matched = []
for r in odds.itertuples(index=False):
    key = match_key(r.date, r.winner_name, r.loser_name)
    if key not in model_rows:
        continue
    market_p = r.p_market_w
    # skip matches with missing or degenerate market probabilities
    if pd.isna(market_p) or market_p <= 0 or market_p >= 1:
        continue
    model_p, model_winner = model_rows[key]
    matched.append((model_p, market_p, 1))

print(f"matched matches (model ∩ market, 2025): {len(matched)}")

if matched:
    mp = np.array([x[0] for x in matched])
    kp = np.array([x[1] for x in matched])
    out = np.array([x[2] for x in matched])
    print(f"\n{'':12s}{'log-loss':>10}{'brier':>9}{'accuracy':>10}")
    print(f"{'model':12s}{log_loss(list(mp), list(out)):>10.4f}{brier_score(list(mp), list(out)):>9.4f}{(mp>0.5).mean():>10.3f}")
    print(f"{'market':12s}{log_loss(list(kp), list(out)):>10.4f}{brier_score(list(kp), list(out)):>9.4f}{(kp>0.5).mean():>10.3f}")
    print(f"\ncorrelation(model, market): {np.corrcoef(mp, kp)[0,1]:.3f}")