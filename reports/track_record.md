# Track record · updated 2026-11-14

Every forecast here was saved *before* kickoff and scored afterwards. Only the latest pre-kickoff forecast per match counts; nothing is excluded after the fact.

Scored matches: **127** · forecasts waiting for results: 29

## Overall

| Model | Matches | Log loss ↓ | RPS ↓ | Result right | Exact score | Over 2.5 right | Both score right |
|---|---:|---:|---:|---:|---:|---:|---:|
| League model (Dixon–Coles + Elo) | 109 | 0.9119 | 0.1967 | 59% | 14% | 55% | 56% |
| Champions League model (cross-league Elo) | 18 | 0.8132 | 0.1966 | 67% | — | — | — |

For reference: guessing 1/3 each gives log loss 1.0986. Over a full season our test-season log loss was ~0.99 and the market's ~0.98. With few matches, expect big swings: one weekend says almost nothing.

## Model vs market (matches with odds)

| | Matches | Log loss ↓ | Result right |
|---|---:|---:|---:|
| Model | 53 | 0.8793 | 60% |
| Market | 53 | 0.8682 | 58% |

Model minus market: +0.0112 (95% interval [-0.0168, +0.0403]).

## By competition

| Competition | Matches | Log loss ↓ | Result right |
|---|---:|---:|---:|
| BUNDESLIGA | 18 | 0.8535 | 61% |
| EPL | 20 | 0.9890 | 55% |
| LALIGA | 20 | 0.9090 | 65% |
| LIGUE1 | 21 | 0.9014 | 57% |
| SERIEA | 30 | 0.9049 | 57% |
| UCL | 18 | 0.8132 | 67% |

## By week

| Week of | Matches | Log loss ↓ | Result right |
|---|---:|---:|---:|
| 2026-10-26 | 56 | 0.9028 | 61% |
| 2026-11-02 | 71 | 0.8941 | 59% |

## Calibration (all outcomes)

When the model says X%, does it happen about X% of the time?

| Predicted | Happened | Outcomes |
|---:|---:|---:|
| 15% | 4% | 51 |
| 27% | 25% | 215 |
| 48% | 54% | 89 |
| 69% | 88% | 25 |
| 84% | 100% | 1 |

## Biggest surprises

| Date | Match | Model H / D / A | Result |
|---|---|---|---|
| 2026-10-30 | Villarreal vs Espanol | 64% / 22% / 14% | 0-1 |
| 2026-11-02 | Atalanta vs Parma | 62% / 23% / 16% | 0-3 |
| 2026-10-31 | Man City vs Brighton | 56% / 24% / 20% | 0-1 |
| 2026-11-04 | Fenerbahçe vs Liverpool | 20% / 20% / 59% | 3-2 |
| 2026-10-28 | Genoa vs Juventus | 21% / 26% / 54% | 2-0 |
