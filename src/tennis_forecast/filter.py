"""A-layer: dynamic serve/return skill via an approximate Kalman filter.

STAGE 1: minimal working filter (no drift, no grass offset yet).
  link:  p_serve(A serving to B) = logistic(MU + serve_A - return_B)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MU = math.log(0.64 / 0.36)


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SkillBelief:
    """Gaussian belief over one player's [serve_skill, return_skill]."""
    serve_mean: float = 0.0
    return_mean: float = 0.0
    serve_var: float = 1.0
    return_var: float = 1.0


def update_one_observation(server, returner, points_won, points_played):
    """Approximate-Kalman update from one serve observation (in place)."""
    n = points_played
    if n <= 0:
        return

    z_mean = server.serve_mean - returner.return_mean
    p = _logistic(MU + z_mean)
    p = min(max(p, 1e-6), 1 - 1e-6)

    z_var = server.serve_var + returner.return_var
    obs_var = 1.0 / (n * p * (1 - p))
    innovation = (points_won - n * p) / (n * p * (1 - p))

    K = z_var / (z_var + obs_var)
    delta = K * innovation
    new_z_var = (1 - K) * z_var

    w_serve = server.serve_var / z_var
    w_return = returner.return_var / z_var

    server.serve_mean += w_serve * delta
    returner.return_mean -= w_return * delta

    server.serve_var = server.serve_var - w_serve * (z_var - new_z_var)
    returner.return_var = returner.return_var - w_return * (z_var - new_z_var)


def run_filter(observations):
    """Forward pass over observations -> {player_id: SkillBelief}."""
    beliefs = {}
    for row in observations.itertuples(index=False):
        s = beliefs.setdefault(row.server_id, SkillBelief())
        r = beliefs.setdefault(row.returner_id, SkillBelief())
        update_one_observation(s, r, int(row.points_won), int(row.points_played))
    return beliefs