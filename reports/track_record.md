# Track record · updated 2026-11-08

Every forecast here was saved *before* kickoff and scored afterwards. Only the latest pre-kickoff forecast per match counts; nothing is excluded after the fact.

Scored matches: **79** · forecasts waiting for results: 48

## Overall

| Model | Matches | Log loss ↓ | RPS ↓ | Result right | Exact score | Over 2.5 right | Both score right |
|---|---:|---:|---:|---:|---:|---:|---:|
| League model (Dixon–Coles + Elo) | 61 | 0.9317 | 0.2151 | 57% | 11% | 56% | 56% |
| Champions League model (cross-league Elo) | 18 | 0.8132 | 0.1966 | 67% | — | — | — |

For reference: guessing 1/3 each gives log loss 1.0986. Over a full season our test-season log loss was ~0.99 and the market's ~0.98. With few matches, expect big swings: one weekend says almost nothing.

## Model vs market (matches with odds)

| | Matches | Log loss ↓ | Result right |
|---|---:|---:|---:|
| Model | 10 | 0.8799 | 60% |
| Market | 10 | 0.9259 | 60% |

Fewer than 30 matches with odds: too few for a meaningful comparison yet.

## By competition

| Competition | Matches | Log loss ↓ | Result right |
|---|---:|---:|---:|
| BUNDESLIGA | 9 | 1.0989 | 33% |
| EPL | 10 | 1.0631 | 50% |
| LALIGA | 10 | 0.8376 | 80% |
| LIGUE1 | 12 | 0.8574 | 58% |
| SERIEA | 20 | 0.8822 | 60% |
| UCL | 18 | 0.8132 | 67% |

## By week

| Week of | Matches | Log loss ↓ | Result right |
|---|---:|---:|---:|
| 2026-10-26 | 56 | 0.9028 | 61% |
| 2026-11-02 | 23 | 0.9091 | 57% |

## Biggest surprises

| Date | Match | Model H / D / A | Result |
|---|---|---|---|
| 2026-10-30 | Villarreal vs Espanol | 64% / 22% / 14% | 0-1 |
| 2026-11-02 | Atalanta vs Parma | 62% / 23% / 16% | 0-3 |
| 2026-10-31 | Man City vs Brighton | 56% / 24% / 20% | 0-1 |
| 2026-11-04 | Fenerbahçe vs Liverpool | 20% / 20% / 59% | 3-2 |
| 2026-10-28 | Genoa vs Juventus | 21% / 26% / 54% | 2-0 |
