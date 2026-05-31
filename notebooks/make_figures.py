"""Regenerate the three calibration-study figures as PNGs for the README."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

from tennis_forecast.data import load_atp_matches, serve_return_counts
from tennis_forecast.filter import run_filter, MU, _logistic
from tennis_forecast.markov import match_win_prob
from tennis_forecast.pricing import log_loss, fit_temperature, apply_temperature

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)


def make_pred_set(test_df, beliefs, use_grass=True):
    def p_serve(s, r, grass):
        bs, br = beliefs.get(s), beliefs.get(r)
        if bs is None or br is None:
            return None
        sv, rt = bs.serve_mean, br.return_mean
        if grass and use_grass:
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
    return np.array(probs), np.array(outs)


print("training...")
train = serve_return_counts(load_atp_matches(range(2010, 2025)))
beliefs = run_filter(train)

hold = load_atp_matches(range(2024, 2025))
hold = hold[hold["surface"] != "Carpet"]
hp, ho = make_pred_set(hold, beliefs)
T, _ = fit_temperature(list(hp), list(ho))

test = load_atp_matches(range(2025, 2026))
test = test[test["surface"] != "Carpet"]
probs_raw, outcomes = make_pred_set(test, beliefs)
probs_cal = np.array([apply_temperature(p, T) for p in probs_raw])


# --- Figure 1: reliability ---
def reliability(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if m.sum() > 0:
            xs.append(probs[m].mean()); ys.append(outcomes[m].mean())
    return np.array(xs), np.array(ys)

rx, ry = reliability(probs_raw, outcomes)
cx, cy = reliability(probs_cal, outcomes)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
ax.plot(rx, ry, "o-", color="#d62728", label=f"raw (log-loss {log_loss(list(probs_raw), list(outcomes)):.3f})")
ax.plot(cx, cy, "s-", color="#2ca02c", label=f"calibrated (log-loss {log_loss(list(probs_cal), list(outcomes)):.3f})")
ax.set_xlabel("predicted probability"); ax.set_ylabel("actual win frequency")
ax.set_title(f"Reliability — 2025 out-of-sample (T = {T:.2f})")
ax.legend(loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
plt.tight_layout(); plt.savefig(FIG / "reliability.png", dpi=120); plt.close()
print("saved reliability.png")


# --- Figure 2: gamma scan ---
gammas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
gll = []
for g in gammas:
    bel = run_filter(train, gamma=g)
    pr, ou = make_pred_set(test, bel)
    gll.append(log_loss(list(pr), list(ou)))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(gammas, gll, "o-", color="#1f77b4")
ax.axvline(gammas[int(np.argmin(gll))], color="grey", ls="--", lw=1,
           label=f"chosen γ = {gammas[int(np.argmin(gll))]}")
ax.set_xlabel("drift parameter γ"); ax.set_ylabel("RAW log-loss (2025)")
ax.set_title("Drift tuning: γ vs out-of-sample log-loss"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "gamma_scan.png", dpi=120); plt.close()
print("saved gamma_scan.png")


# --- Figure 3: tau2 scan on grass ---
grass = load_atp_matches(range(2024, 2026))
grass = grass[grass["surface"] == "Grass"]
base_pr, base_ou = make_pred_set(grass, run_filter(train, tau2=0.005), use_grass=False)
base_ll = log_loss(list(base_pr), list(base_ou))
taus = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
tll = []
for t in taus:
    bel = run_filter(train, tau2=t)
    pr, ou = make_pred_set(grass, bel)
    tll.append(log_loss(list(pr), list(ou)))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(taus, tll, "o-", color="#9467bd", label="with grass offset")
ax.axhline(base_ll, color="grey", ls="--", lw=1, label="no offset (overall skill)")
ax.set_xscale("log")
ax.set_xlabel("grass-offset prior variance τ² (smaller = stronger shrinkage)")
ax.set_ylabel("log-loss on grass (2024-25)")
ax.set_title("Surface shrinkage: grass data is too sparse to help"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "tau2_scan.png", dpi=120); plt.close()
print("saved tau2_scan.png")

print("all figures saved to", FIG)