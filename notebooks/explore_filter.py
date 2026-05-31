"""Quick sanity check: run the Stage-3 filter on real ATP data and look at
the top players' estimated serve & return skill, and grass offsets."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tennis_forecast.data import load_atp_matches, serve_return_counts
from tennis_forecast.filter import run_filter

# 1. Load real data and build observations.
matches = load_atp_matches(range(2010, 2027))
obs = serve_return_counts(matches)
print(f"observations: {len(obs)}   players: {obs['server_id'].nunique()}")

# 2. Run the filter (default gamma=0.3).
beliefs = run_filter(obs)

# 3. Map player ids back to names (use the most recent name seen per id).
id_to_name = (
    matches[["winner_id", "winner_name"]]
    .rename(columns={"winner_id": "id", "winner_name": "name"})
    .drop_duplicates("id")
    .set_index("id")["name"]
    .to_dict()
)

# 4. Count how many serve observations each player actually has.
counts = obs["server_id"].value_counts().to_dict()

rows = []
for pid, b in beliefs.items():
    rows.append({
        "name": id_to_name.get(pid, pid),
        "serve": b.serve_mean,
        "return": b.return_mean,
        "overall": b.serve_mean + b.return_mean,
        "grass_serve": b.serve_grass_mean,
        "n_obs": counts.get(pid, 0),
    })
df = pd.DataFrame(rows)

# Keep players with enough real matches (not low variance — see notes).
df = df[df["n_obs"] >= 200]
print(f"\nplayers with >=200 serve observations: {len(df)}")

# Center skills on the average tour-level player (n_obs>=200 subgroup) so the
# absolute level is interpretable: >0 = above tour-average, <0 = below.
# This is a display rescaling only — it does NOT change relative ordering or
# any prediction (match outcomes depend on serve_A - return_B differences).
serve_offset = df["serve"].mean()
return_offset = df["return"].mean()
df["serve"] = df["serve"] - serve_offset
df["return"] = df["return"] - return_offset
df["overall"] = df["serve"] + df["return"]
print(f"(centered: serve_offset={serve_offset:.3f}, return_offset={return_offset:.3f})")
print("\n=== Top 15 by overall (serve+return) ===")
print(df.sort_values("overall", ascending=False).head(15)
        .round(3).to_string(index=False))

print("\n=== Top 10 serve (raw) ===")
print(df.sort_values("serve", ascending=False).head(10)[["name","serve","n_obs"]]
        .round(3).to_string(index=False))

print("\n=== Top 10 grass-serve specialists ===")
print(df.sort_values("grass_serve", ascending=False).head(10)[["name","grass_serve","n_obs"]]
        .round(3).to_string(index=False))
