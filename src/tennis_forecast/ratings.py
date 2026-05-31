"""Priors and cold-start seeding for players with little/no history.

Qualifiers, wildcards and comeback players need a sensible initial SkillBelief.
Seed from ATP ranking rather than a flat prior so the simulation isn't nonsense
for these players (several are in any Wimbledon main draw).
"""

from __future__ import annotations


def seed_from_ranking(ranking: int):
    """Map an ATP ranking to an initial SkillBelief (mean + wide covariance).

    TODO: monotone map ranking -> skill mean; large variance so data moves it
    fast. Calibrate the map so seeded players' implied match probs are sane.
    """
    raise NotImplementedError
