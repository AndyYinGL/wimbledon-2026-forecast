"""A-layer: dynamic serve/return skill via an approximate Kalman filter.

STAGE 3: serve/return skill + random-walk drift + grass offset with shrinkage.
  link (non-grass): p = logistic(MU + serve_A - return_B)
  link (grass):     p = logistic(MU + (serve_A + serve_grass_A)
                                     - (return_B + return_grass_B))
Shrinkage: grass offset has a small prior variance (tau2); little grass data
=> offset stays near 0 (uses overall skill). tau2 is the shrinkage knob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MU = math.log(0.64 / 0.36)
TAU2 = 0.05  # default grass-offset prior variance


def _logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SkillBelief:
    serve_mean: float = 0.0
    return_mean: float = 0.0
    serve_var: float = 1.0
    return_var: float = 1.0
    serve_grass_mean: float = 0.0
    return_grass_mean: float = 0.0
    serve_grass_var: float = 0.05
    return_grass_var: float = 0.05
    last_date: object = None


def predict(belief, date, gamma):
    """Random-walk drift on overall skill: variance grows by gamma^2 * years."""
    if belief.last_date is not None:
        dt_years = (date - belief.last_date).days / 365.25
        if dt_years > 0:
            bump = gamma * gamma * dt_years
            belief.serve_var += bump
            belief.return_var += bump
    belief.last_date = date


def update_one_observation(server, returner, points_won, points_played, grass):
    """Approximate-Kalman update from one serve observation (in place)."""
    n = points_played
    if n <= 0:
        return

    if grass:
        z_mean = (server.serve_mean + server.serve_grass_mean) \
               - (returner.return_mean + returner.return_grass_mean)
        z_var = (server.serve_var + server.serve_grass_var
                 + returner.return_var + returner.return_grass_var)
    else:
        z_mean = server.serve_mean - returner.return_mean
        z_var = server.serve_var + returner.return_var

    p = _logistic(MU + z_mean)
    p = min(max(p, 1e-6), 1 - 1e-6)

    obs_var = 1.0 / (n * p * (1 - p))
    innovation = (points_won - n * p) / (n * p * (1 - p))

    K = z_var / (z_var + obs_var)
    delta = K * innovation
    new_z_var = (1 - K) * z_var
    var_reduction = z_var - new_z_var

    w = server.serve_var / z_var
    server.serve_mean += w * delta
    server.serve_var -= w * var_reduction

    w = returner.return_var / z_var
    returner.return_mean -= w * delta
    returner.return_var -= w * var_reduction

    if grass:
        w = server.serve_grass_var / z_var
        server.serve_grass_mean += w * delta
        server.serve_grass_var -= w * var_reduction

        w = returner.return_grass_var / z_var
        returner.return_grass_mean -= w * delta
        returner.return_grass_var -= w * var_reduction


def run_filter(observations, gamma=0.1, tau2=0.05):
    """Forward pass with drift + grass shrinkage -> {player_id: SkillBelief}.

    `observations` needs columns: tourney_date, surface, server_id,
    returner_id, points_won, points_played.
    """
    beliefs = {}
    for row in observations.itertuples(index=False):
        s = beliefs.setdefault(row.server_id,
                               SkillBelief(serve_grass_var=tau2, return_grass_var=tau2))
        r = beliefs.setdefault(row.returner_id,
                               SkillBelief(serve_grass_var=tau2, return_grass_var=tau2))
        predict(s, row.tourney_date, gamma)
        predict(r, row.tourney_date, gamma)
        is_grass = (row.surface == "Grass")
        update_one_observation(s, r, int(row.points_won), int(row.points_played), is_grass)
    return beliefs