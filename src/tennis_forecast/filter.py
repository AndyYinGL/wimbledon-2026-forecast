"""A-layer: dynamic serve/return skill via an approximate Kalman filter.

STAGE 3: serve/return skill + random-walk drift + per-player GRASS offset
that shrinks toward overall skill (the project's core idea).

State per player (4 latent dims):
  serve_mean, return_mean              -- overall skill (fed by ALL surfaces)
  serve_grass_mean, return_grass_mean  -- grass-specific offset (shrinks to 0)

Link:
  non-grass: p = logistic(MU + serve_A - return_B)
  grass:     p = logistic(MU + (serve_A + serve_grass_A)
                              - (return_B + return_grass_B))

Shrinkage is implemented by the grass offset's small prior variance TAU2:
with little grass data the offset stays near 0 (= use overall skill); with
more grass data it is allowed to move. TAU2 is the shrinkage knob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MU = math.log(0.64 / 0.36)

# Prior variance of the grass offset. Small => strong shrinkage toward overall
# skill (good when grass data is sparse). This is a key hyperparameter (TAU2).
TAU2 = 0.05


def _logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SkillBelief:
    """Gaussian belief over a player's overall + grass-offset serve/return skill."""
    serve_mean: float = 0.0
    return_mean: float = 0.0
    serve_var: float = 1.0
    return_var: float = 1.0
    # Grass offsets: prior mean 0, prior variance TAU2 (set in __post_init__).
    serve_grass_mean: float = 0.0
    return_grass_mean: float = 0.0
    serve_grass_var: float = TAU2
    return_grass_mean_var: float = TAU2  # (named below; see __post_init__)
    last_date: object = None

    def __post_init__(self):
        # Ensure grass variances start at the prior TAU2.
        self.serve_grass_var = TAU2
        self.return_grass_var = TAU2


def predict(belief, date, gamma):
    """Random-walk drift on OVERALL skill only (grass offset is treated as
    stationary; it represents a stable surface trait)."""
    if belief.last_date is not None:
        dt_years = (date - belief.last_date).days / 365.25
        if dt_years > 0:
            bump = gamma * gamma * dt_years
            belief.serve_var += bump
            belief.return_var += bump
    belief.last_date = date


def update_one_observation(server, returner, points_won, points_played, grass):
    """Approximate-Kalman update from one serve observation (in place).

    On grass, the latent serve advantage z uses overall + grass offset, and the
    correction is split across server.serve, returner.return, AND their grass
    offsets, in proportion to each component's variance. Off grass, only the
    overall skills are touched.
    """
    n = points_played
    if n <= 0:
        return

    if grass:
        z_mean = (server.serve_mean + server.serve_grass_mean) \
               - (returner.return_mean + returner.return_grass_mean)
        # Variance of z = sum of all four contributing component variances.
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
    var_reduction = z_var - new_z_var  # total variance removed, to share out

    # Distribute the mean correction & variance reduction across components,
    # each in proportion to its share of z_var. Small grass var => small share
    # => grass offset barely moves when grass data is thin (= shrinkage).
    def _apply(component_var, sign):
        w = component_var / z_var
        return sign * w * delta, w * var_reduction

    d, vr = _apply(server.serve_var, +1)
    server.serve_mean += d
    server.serve_var -= vr

    d, vr = _apply(returner.return_var, -1)
    returner.return_mean += d
    returner.return_var -= vr

    if grass:
        d, vr = _apply(server.serve_grass_var, +1)
        server.serve_grass_mean += d
        server.serve_grass_var -= vr

        d, vr = _apply(returner.return_grass_var, -1)
        returner.return_grass_mean += d
        returner.return_grass_var -= vr


def run_filter(observations, gamma=0.3):
    """Forward pass with drift + grass shrinkage -> {player_id: SkillBelief}.

    `observations` must have columns: tourney_date, surface, server_id,
    returner_id, points_won, points_played.
    """
    beliefs = {}
    for row in observations.itertuples(index=False):
        s = beliefs.setdefault(row.server_id, SkillBelief())
        r = beliefs.setdefault(row.returner_id, SkillBelief())
        predict(s, row.tourney_date, gamma)
        predict(r, row.tourney_date, gamma)
        is_grass = (row.surface == "Grass")
        update_one_observation(
            s, r, int(row.points_won), int(row.points_played), is_grass
        )
    return beliefs