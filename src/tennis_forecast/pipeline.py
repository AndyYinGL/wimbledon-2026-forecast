"""Live forecast pipeline: glue the layers into a round-by-round Wimbledon run.

For each round: fetch results so far, update skills with only information
available BEFORE each match (as-of-date guard — the cardinal anti-leakage rule),
convert skills to p_serve, simulate the remaining draw, emit title/reach-round
probabilities, log them for calibration tracking.
"""

from __future__ import annotations


def skills_to_pserve(belief_a, belief_b, surface: str = "grass"):
    """Map two players' skill beliefs to their serve point-win probs (p_a, p_b).

    TODO: apply the logistic link with the grass term; optionally propagate
    posterior uncertainty by sampling for credible intervals.
    """
    raise NotImplementedError


def forecast_remaining_draw(draw, beliefs, as_of_date):
    """Simulate the remaining bracket -> per-player title / reach-round probs.

    TODO: assert every belief used is dated <= as_of_date (no leakage), build
    the alive bracket, call simulate.simulate_tournament with p_serve.
    """
    raise NotImplementedError
