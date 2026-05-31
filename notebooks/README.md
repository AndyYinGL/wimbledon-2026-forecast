# Notebooks

Analysis & narrative only — import engines from `tennis_forecast`, never define
stateful/recursive logic here.

- `explore_filter.py` — sanity-check serve/return & grass rankings on real data
- `evaluate_predictions.py` — walk-forward out-of-sample evaluation (accuracy,
  log-loss, calibration)
- `tune_gamma.py` / `tune_tau.py` / `tune_joint.py` — hyperparameter studies
- `make_figures.py` — regenerate the README calibration figures
- `03_calibration_study.ipynb` — the headline calibration study (reliability
  diagrams, tuning curves)