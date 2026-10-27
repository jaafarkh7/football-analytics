# Week 2d: Comparing clubs across leagues (Champions League)

Our domestic ratings are per league, so a 1700 in Ligue 1 and a 1700 in the Premier
League aren't the same. Here all clubs share **one rating pool**: domestic matches move
points only within a league, so only cross-league matches (the Champions League) can
change a league's average. Clubs from outside the top 5 start below the top-5 average
and learn from their European results.

Baseline: identical code, but European results never move top-5 clubs, so every league
keeps the same average ("leagues equal"). The difference isolates league strength.

Data: openfootball Champions League results 2014-15..2019-20 and 2024-25 (932 matches;
2020-21..2023-24 are missing from the source). Tuned on 2016-17..2019-20, tested once on 2024-25.

## Results (Champions League matches only)

| Model | Log loss ↓ | RPS ↓ | Brier ↓ | Accuracy ↑ | Matches |
|---|---:|---:|---:|---:|---:|
| Leagues equal (baseline) | 0.9311 | 0.2108 | 0.5485 | 56.4% | 188 |
| Cross-league Elo | 0.9304 | 0.2098 | 0.5464 | 56.4% | 188 |

Validation (2016-17..2019-20):

| Model | Log loss ↓ | RPS ↓ | Brier ↓ | Accuracy ↑ | Matches |
|---|---:|---:|---:|---:|---:|
| Leagues equal (baseline) | 0.9439 | 0.1911 | 0.5540 | 58.5% | 494 |
| Cross-league Elo | 0.9398 | 0.1900 | 0.5516 | 58.1% | 494 |

**Cross-league Elo vs leagues equal, test log loss:** -0.0008 (95% interval [-0.0194, +0.0177]).

Matches between clubs of two *different* top-5 leagues (69 in 2024-25), where
league strength matters most: cross-league Elo 0.9474 vs leagues equal 0.9593.

## League strength

![League strength](league_strength.png)

Average club rating relative to the five-league average, end of 2025-26:

| League | Rating vs average |
|---|---:|
| EPL | +16 |
| BUNDESLIGA | +2 |
| SERIEA | +2 |
| LALIGA | -5 |
| LIGUE1 | -15 |

Roughly, 100 rating points ≈ the difference between a 64% and a 50% expected score.
In seasons without Champions League data (shaded: 2020-21..2023-24 and 2025-26) the
lines still move, but only because promoted clubs enter below a league's average and
relegated clubs leave it. That movement is bookkeeping, not evidence about league strength.

## Settings (validation)

Cross-league Elo: Champions League K **30**, home advantage **130**,
external clubs start **100** below the top-5 average.

|   k_cl |   home_adv_cl |   external_offset |   valid_log_loss |
|-------:|--------------:|------------------:|-----------------:|
|     30 |           130 |               100 |           0.9398 |
|     30 |           100 |               100 |           0.9399 |
|     30 |            70 |               100 |           0.9401 |
|     30 |           130 |                 0 |           0.9407 |
|     30 |           100 |                 0 |           0.9408 |
|     30 |            70 |                 0 |           0.9410 |

## Caveats

- ~120–190 matches per season: every comparison here is noisier than for league matches.
- Clubs outside the top 5 are only seen in Europe (6–12 matches a season), so their
  ratings are rough; Champions League qualifying rounds aren't in the data.
- Predictions are Elo-only: there's no shots data for European matches.
