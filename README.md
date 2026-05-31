# Wimbledon 2026 — Men's Singles Forecasting & Pricing Engine

A live, forward-simulation engine that prices the **2026 Wimbledon men's singles**
as the tournament unfolds. From the current draw it produces the full
distribution of outcomes — each player's probability of reaching every round and
of winning the title — quotes markets two-sided like a liquidity provider, and
tracks how well-**calibrated** its probabilities are against both the actual
results and the betting market.

The guiding question is a market-maker's, not a punter's: *how should the
distribution of game events be priced, and where does the market's pricing
break down?* — evaluated live and out-of-sample rather than on backtested
betting returns.

## Why grass, why this year

Grass is the most data-sparse of the three surfaces — the grass season is short,
so any player's grass-specific record is thin. That makes naive surface-splitting
(the approach that *added* noise in earlier work) actively harmful here, and it
is exactly why the model treats grass skill as a **per-player offset that shrinks
toward overall skill** when grass data is scarce. The hard problem and the
modelling choice are matched on purpose.

## Method

A two-layer design, chosen so the same point-level structure serves as both the
fitting likelihood and the forecasting simulator:

**Strength layer (A).** Each player carries a latent *serve* skill and *return*
skill that drift over time (random walk), plus a per-player grass offset that
shrinks toward overall skill. These are inferred online with a principled
approximate Kalman filter (a regular linearisation of the binomial–logit
likelihood, in the Szczecinski–Tihon "one-fits-all" sense; Elo, Glicko and
TrueSkill are all special cases of this family). Observations are **per-match
serve/return point counts** (SPW/RPW), which identify serve and return ability
separately.

**Distribution layer (B).** A hierarchical Markov model (Klaassen–Magnus) turns
the two players' serve point-win probabilities into exact win probabilities at
game / set / match level and — crucially — from *any* score, the basis for
in-play pricing. A Monte Carlo simulator reproduces the same scoring rules
exactly and validates the analytical engine; it also rolls the remaining draw
forward many times to produce the tournament-level distribution.

**Pricing & calibration.** Probabilities become two-sided quotes with a correct
bilateral overround; the headline deliverable is a calibration study (reliability
curves, log-loss, Brier) on a large historical backtest, with Wimbledon serving
as the live, genuinely out-of-sample demonstration.

## Architecture

Engines live in importable, tested modules; analysis and narrative live in notebooks that import them. (Stateful, recursive code does not belong in notebook cells.)

    src/tennis_forecast/
      markov.py     # analytical game/set/match/tiebreak win prob — built + tested
      simulate.py   # Monte Carlo match sim + tournament sim — built + tested
      filter.py     # approximate-Kalman serve/return skill filter (A-layer)
      pipeline.py   # live round-by-round update + as-of-date leakage guards
      ratings.py    # priors / cold-start seeding from ranking
      data.py       # Sackmann history, live draw/results, market odds
      pricing.py    # de-vig, two-sided quotes, log-loss / Brier / reliability — built
    tests/          # cross-validation: analytical == Monte Carlo, and more
    notebooks/      # data exploration, calibration study, live Wimbledon forecast
## Status

- Analytical Markov engine, validated against an independent Monte Carlo
  simulator (`tests/`).
- Tournament Monte Carlo + pricing / calibration utilities.
- Approximate-Kalman serve/return filter (A-layer).
- Data pipeline (Sackmann + live source), calibration backtest, live forecast.

## Data

- **History:** Jeff Sackmann's open ATP datasets (`tennis_atp`) — match-level
  serve/return statistics, the de facto standard for tennis research. *(CC
  BY-NC-SA; research / non-commercial use.)*
- **Live:** a tennis data API for the Wimbledon draw, results and in-tournament
  serve stats (Sackmann lags real time).
- **Market:** historical closing odds for the model-vs-market calibration study.

## Key references

Klaassen & Magnus (2003); Barnett & Clarke (2005); Knottenbelt et al. (2012);
Ingram (2019); Kovalchik & Reid (2018, 2019); Angelini, Candila & De Angelis
(2022); Duffield, Power & Rimella (2023/24); Szczecinski & Tihon (2023);
Gollub (2021).

## License

Code: MIT. Tennis data remains under its original license (Sackmann: CC BY-NC-SA).
