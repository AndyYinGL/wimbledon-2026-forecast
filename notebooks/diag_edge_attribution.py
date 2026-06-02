import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path.home() / "wimbledon-2026-forecast" / "src"))

d = pd.read_parquet(Path.home()/"wimbledon-2026-forecast"/"data/processed/mispricing_matches.parquet")
d = d.dropna(subset=["model_p1","market_p1","p1_won","odds_p1"]).copy()

def ll(p, y):
    p = np.clip(p, 1e-9, 1-1e-9)
    return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())

edge = d["model_p1"] - d["market_p1"]
bet = d[edge > 0.05]
print(f"下注场 n={len(bet)}")
print(f"  model log-loss : {ll(bet['model_p1'], bet['p1_won']):.4f}")
print(f"  market log-loss: {ll(bet['market_p1'], bet['p1_won']):.4f}")
print(f"  下注场 p1 实际胜率: {bet['p1_won'].mean():.4f}")
print(f"  下注场 model_p1 均值: {bet['model_p1'].mean():.4f}")
print(f"  下注场 market_p1 均值: {bet['market_p1'].mean():.4f}")
print()
# 全样本对比
print(f"全样本 n={len(d)}")
print(f"  model log-loss : {ll(d['model_p1'], d['p1_won']):.4f}")
print(f"  market log-loss: {ll(d['market_p1'], d['p1_won']):.4f}")