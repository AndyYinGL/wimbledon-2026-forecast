"""A-layer: dynamic serve/return skill via an approximate Kalman filter.

Each player has a latent state [serve_skill, return_skill] (+ a per-player grass
offset that shrinks toward overall skill). State drifts as a random walk (predict
step inflates covariance by gamma^2 * dt); each match's serve/return counts are a
binomial-logit observation, linearised for a Gaussian (EKF-style) update.
Szczecinski-Tihon "one-fits-all" approximate Kalman; Elo/Glicko/TrueSkill are
special cases.

This is the hardest, most error-prone module. Build it incrementally and lean on
tests/ (synthetic-data recovery; zero-drift == static logistic regression).

Link:  p_serve(A->B) = logistic(mu + serve_A - return_B + grass_term)
mu (the ~1.29 anchor) and a zero-centred grass prior keep skills identifiable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SkillBelief:
    """Gaussian belief over a player's latent skills (mean vector + covariance)."""
    # TODO: mean (serve, return, grass offsets), covariance, last-update date.


def predict(belief: "SkillBelief", dt: float, gamma: float) -> "SkillBelief":
    """Random-walk predict step: inflate covariance by gamma**2 * dt.

    TODO: implement; dt is elapsed CALENDAR time, not match count.
    """
    raise NotImplementedError


def update(belief_server, belief_returner, points_won: int, points_played: int):
    """EKF-style update from one binomial serve observation.

    TODO: linearise the binomial-logit likelihood, compute Kalman gain, update
    both players' beliefs (server's serve skill, returner's return skill). Guard
    extreme counts with pseudo-counts.
    """
    raise NotImplementedError


def run_filter(observations, gamma: float, tau: float):
    """Forward pass over all observations in date order -> beliefs per player.

    TODO: maintain a dict player_id -> SkillBelief; predict to each match date,
    then update. tau controls grass-offset shrinkage (prior variance).
    """
    raise NotImplementedError
