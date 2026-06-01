"""Live forecast pipeline: glue the layers into a round-by-round Wimbledon run.

For each round: fetch results so far, update skills with only information
available BEFORE each match (as-of-date guard — the cardinal anti-leakage rule),
convert skills to p_serve, simulate the remaining draw, emit title/reach-round
probabilities, log them for calibration tracking.
"""

from __future__ import annotations


def skills_to_pserve(server_belief, returner_belief, grass: bool = True) -> float:
    """Map two players' skill beliefs to the server's point-win prob p_serve.

    Reuses the exact link from filter.update_one_observation so the predicted
    p_serve is on the same scale the filter trained against:
      non-grass: logistic(MU + serve_S - return_R)
      grass:     logistic(MU + (serve_S + serve_grass_S)
                              - (return_R + return_grass_R))
    Point estimate only -- posterior variance is not propagated here.
    """
    from .filter import MU, _logistic

    if grass:
        z = (server_belief.serve_mean + server_belief.serve_grass_mean) \
            - (returner_belief.return_mean + returner_belief.return_grass_mean)
    else:
        z = server_belief.serve_mean - returner_belief.return_mean
    return _logistic(MU + z)


def forecast_remaining_draw(draw, beliefs, as_of_date):
    """Simulate the remaining bracket -> per-player title / reach-round probs.

    TODO: assert every belief used is dated <= as_of_date (no leakage), build
    the alive bracket, call simulate.simulate_tournament with p_serve.
    """
    raise NotImplementedError
