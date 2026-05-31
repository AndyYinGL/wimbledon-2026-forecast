"""Pricing + calibration layer — the market-maker view and the headline metrics.

These are pure functions (no external data), so they are implemented for real
here. They turn model probabilities into two-sided prices and, crucially, score
the model against reality and against the market as RG-2026 unfolds.
"""

from __future__ import annotations

import math


# --- Odds <-> probability ---------------------------------------------------

def prob_to_decimal_odds(p: float) -> float:
    """Fair decimal odds (no margin)."""
    if p <= 0:
        return float("inf")
    return 1.0 / p


def quote_two_sided(p: float, margin: float = 0.05) -> tuple[float, float]:
    """Quote two-sided decimal odds with a total ``margin`` (overround).

    A liquidity provider prices BOTH sides so implied probs sum to 1 + margin,
    split proportionally. This is the correct bilateral-overround model (the
    bug fixed in v1) — both sides carry the margin, not one.
    """
    pa, pb = p, 1.0 - p
    over = 1.0 + margin
    imp_a, imp_b = pa * over, pb * over
    return 1.0 / imp_a, 1.0 / imp_b


def devig_proportional(odds: dict[str, float]) -> dict[str, float]:
    """Strip the overround from a book of decimal odds -> true probabilities.

    Proportional (a.k.a. multiplicative) method: divide each implied prob by
    the booksum. Adequate for outright markets; note favourite-longshot bias
    means more refined methods (Shin, power) can matter — worth a comparison.
    """
    implied = {k: 1.0 / v for k, v in odds.items()}
    booksum = sum(implied.values())
    return {k: v / booksum for k, v in implied.items()}


# --- Scoring: how good are the probabilities? -------------------------------

def log_loss(prob_pred: list[float], outcome: list[int]) -> float:
    """Mean log-loss. Lower is better; the standard probabilistic score."""
    eps = 1e-15
    n = len(outcome)
    return -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in zip(prob_pred, outcome)
    ) / n


def brier_score(prob_pred: list[float], outcome: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(prob_pred, outcome)) / len(outcome)


def reliability_curve(prob_pred, outcome, n_bins: int = 10):
    """Calibration data for a reliability diagram.

    Returns list of (bin_center, mean_predicted, empirical_rate, count).
    A well-calibrated model lies on the diagonal — what a market maker needs
    to avoid being picked off. This is the project's headline deliverable.
    """
    bins: list[list[float]] = [[] for _ in range(n_bins)]
    outs: list[list[int]] = [[] for _ in range(n_bins)]
    for p, y in zip(prob_pred, outcome):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append(p)
        outs[idx].append(y)
    rows = []
    for i in range(n_bins):
        if not bins[i]:
            continue
        rows.append(
            (
                (i + 0.5) / n_bins,
                sum(bins[i]) / len(bins[i]),
                sum(outs[i]) / len(outs[i]),
                len(bins[i]),
            )
        )
    return rows
# --- Temperature scaling (calibration) --------------------------------------

def _logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def apply_temperature(prob, T):
    """Soften (T>1) or sharpen (T<1) a probability via temperature scaling."""
    z = _logit(prob) / T
    return 1.0 / (1.0 + math.exp(-z))


def fit_temperature(probs, outcomes, grid=None):
    """Find the temperature T that minimises log-loss on (probs, outcomes).

    T > 1 pulls over-confident probabilities toward 0.5 (fixes over-confidence).
    Fit this on a holdout (e.g. 2024) and apply to the test set (2025) to avoid
    leakage.
    """
    if grid is None:
        grid = [0.5 + 0.05 * i for i in range(150)]  # 0.5 .. 7.95
    best_T, best_ll = 1.0, float("inf")
    for T in grid:
        scaled = [apply_temperature(p, T) for p in probs]
        ll = log_loss(scaled, outcomes)
        if ll < best_ll:
            best_ll, best_T = ll, T
    return best_T, best_ll
