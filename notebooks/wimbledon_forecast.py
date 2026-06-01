"""Wimbledon forecast driver. Train the filter, build each player's grass-fitted
serve probability, and simulate the draw to get title odds.

For now it runs on a SYNTHETIC 128-player draw (real draw plugs in at tournament
time via data.load_wimbledon_draw). The whole engine is validated here so that
when the real draw lands, only the bracket needs swapping in.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import random
from tennis_forecast.data import load_atp_matches, serve_return_counts
from tennis_forecast.filter import run_filter, MU, _logistic
from tennis_forecast.simulate import simulate_tournament


def build_grass_serve_probs(beliefs, player_ids):
    """For each player, an *average* grass serve point-win prob vs a neutral
    (average) returner — used as their p_serve in the tournament sim.

    p_serve = logistic(MU + (serve + serve_grass) - 0)   [neutral returner]
    """
    probs = {}
    for pid in player_ids:
        b = beliefs.get(pid)
        if b is None:
            probs[pid] = _logistic(MU)  # unknown player -> league average
        else:
            probs[pid] = _logistic(MU + b.serve_mean + b.serve_grass_mean)
    return probs


# 1. Train the filter on all available history.
print("training filter...")
train = serve_return_counts(load_atp_matches(range(2010, 2027)))
beliefs = run_filter(train)

# 2. Build a SYNTHETIC 128-player draw from real, well-estimated players.
#    (At tournament time this is replaced by the real Wimbledon draw.)
matches = load_atp_matches(range(2023, 2027))
counts = serve_return_counts(matches)["server_id"].value_counts()
candidates = [pid for pid, c in counts.items() if c >= 100]
# trim to the largest power of two we can fill (64 for the synthetic test)
size = 64 if len(candidates) >= 64 else 32
top_players = candidates[:size]
print(f"synthetic draw size: {len(top_players)}")

id_to_name = (
    matches.drop_duplicates("winner_id")
    .set_index("winner_id")["winner_name"].to_dict()
)

# 3. Build grass serve probs and simulate.
p_serve = build_grass_serve_probs(beliefs, top_players)
rng = random.Random(0)
bracket = top_players[:]
rng.shuffle(bracket)  # random seeding for the synthetic test

result = simulate_tournament(bracket, p_serve, best_of=5, n_sims=5000)

# 4. Show the title-odds leaderboard.
champ = sorted(result["champion"].items(), key=lambda x: -x[1])
print("\n=== Title odds (synthetic draw) ===")
for pid, p in champ[:15]:
    print(f"  {id_to_name.get(pid, pid):24s} {p*100:5.1f}%")