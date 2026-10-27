# Football Match Analysis & Prediction

Pre-match predictions and post-match analysis for Europe's top 5 leagues, with honest,
reproducible evaluation.

**Status: Week 1 of 6** — data pipeline and baseline models.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

python -m fa.data.load_results          # ~22k matches + odds -> data/football.duckdb
python -m fa.run_baselines              # tune Elo, evaluate, write reports/
python -m pytest -q                     # unit tests
```

If football-data.co.uk is unreachable, use `--source openfootball` (results only, no
odds, so the bookmaker baseline is skipped). Downloads are cached in `data/raw/`.

## Data

| Source | What | Notes |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | Results + market odds, 2014-15 → now | Default. Check their terms before any commercial use. |
| [openfootball](https://github.com/openfootball/football.json) | Results only | Fallback. 2024-25 La Liga and Serie A each miss ~10 matches. |

Table `matches`: one row per match (`league, season, date, home_team, away_team,
home_goals, away_goals, result, odds_*`). Table `predictions`: every model's
H/D/A probabilities for scored seasons.

## Evaluation protocol

- **Time-based split, never shuffled:** train 2014-15 → 2023-24, tune on 2024-25,
  test once on 2025-26. Using the future to predict the past is the #1 football-model bug.
- **Elo predictions use only pre-match ratings** (enforced by `test_elo_has_no_lookahead`).
- **Metrics:** log loss, Ranked Probability Score (the football standard), Brier,
  accuracy, and calibration plots. Accuracy alone is misleading.
- **Significance:** paired bootstrap intervals on score differences.

## Baselines (week 1)

| Model | What it knows |
|---|---|
| Uniform | Nothing (1/3 each) |
| League base rates | Home advantage and draw rate per league |
| Elo | Team strength, updated after every match |
| Bookmakers | The market (margin removed) — the hardest bar |

Full results: [`reports/week1_baselines.md`](reports/week1_baselines.md).

## Layout

```
src/fa/
  config.py              leagues, seasons, split
  data/load_results.py   download + validate + load into DuckDB
  models/baselines.py    uniform, base rates, bookmaker
  models/elo.py          Elo ratings + rating-gap -> H/D/A model
  eval/metrics.py        log loss, RPS, Brier, bootstrap, calibration
  run_baselines.py       week 1 experiment
tests/test_week1.py
```

## Roadmap

1. ✅ Data + baselines
2. Goals model (Poisson / Dixon–Coles), then gradient boosting with form features
3. Post-match analysis from StatsBomb event data (xG timeline, shot maps, pass networks)
4. LLM-written previews and reports, grounded in computed numbers, checked for invented stats
5. Publish predictions before kickoff; score them after (public track record)
6. Write-up

*This project is for analysis and learning, not betting advice.*
