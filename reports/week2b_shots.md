# Week 2b: Does shot data help?

Data source: **football-data** · tune on **2024-25**, test once on **2025-26** · weekly walk-forward refits.

**Idea.** Goals are a noisy measure of team quality; shots on target are less noisy.
The shots model learns team strengths from
`goal_weight × goals + (1 − goal_weight) × 0.35·shots on target` (the conversion rate
is fitted per league on each training window), while still predicting real goals.
`goal_weight = 1` is exactly the week 2 model, so the comparison is clean.

## Test season (2025-26)

| Model | Log loss ↓ | RPS ↓ | Brier ↓ | Accuracy ↑ | Matches |
|---|---:|---:|---:|---:|---:|
| Elo | 0.9960 | 0.2032 | 0.5942 | 51.5% | 1752 |
| Dixon–Coles (goals) | 0.9989 | 0.2037 | 0.5956 | 51.5% | 1752 |
| Dixon–Coles (goals + shots) | 0.9953 | 0.2024 | 0.5933 | 52.3% | 1752 |
| Blend: Elo + DC (goals) | 0.9945 | 0.2027 | 0.5932 | 51.7% | 1752 |
| Blend: Elo + DC (goals + shots) | 0.9924 | 0.2019 | 0.5918 | 52.0% | 1752 |
| Bookmakers (market) | 0.9784 | 0.1977 | 0.5825 | 53.4% | 1752 |

**Gap to the market (test log loss):** Elo +0.0177 → goals blend +0.0161 → goals + shots blend +0.0140 (21% of Elo's gap closed).

## Validation season (2024-25)

| Model | Log loss ↓ | RPS ↓ | Brier ↓ | Accuracy ↑ | Matches |
|---|---:|---:|---:|---:|---:|
| Elo | 0.9846 | 0.2004 | 0.5869 | 53.1% | 1752 |
| Dixon–Coles (goals) | 0.9851 | 0.2009 | 0.5870 | 53.0% | 1752 |
| Dixon–Coles (goals + shots) | 0.9829 | 0.2001 | 0.5856 | 52.3% | 1752 |
| Blend: Elo + DC (goals) | 0.9823 | 0.2000 | 0.5854 | 52.9% | 1752 |
| Blend: Elo + DC (goals + shots) | 0.9806 | 0.1994 | 0.5842 | 52.7% | 1752 |
| Bookmakers (market) | 0.9640 | 0.1945 | 0.5731 | 53.6% | 1752 |

## Is the difference real? (paired bootstrap, log loss, test season)

| Comparison | Mean difference | 95% interval | Verdict |
|---|---:|---|---|
| Dixon–Coles (goals + shots) vs Dixon–Coles (goals) | -0.0036 | [-0.0080, +0.0008] | no clear difference |
| Dixon–Coles (goals + shots) vs Elo | -0.0006 | [-0.0079, +0.0066] | no clear difference |
| Blend: Elo + DC (goals + shots) vs Blend: Elo + DC (goals) | -0.0021 | [-0.0045, +0.0003] | no clear difference |
| Blend: Elo + DC (goals + shots) vs Elo | -0.0036 | [-0.0077, +0.0004] | no clear difference |
| Blend: Elo + DC (goals) vs Bookmakers (market) | +0.0161 | [+0.0096, +0.0225] | second is better |
| Blend: Elo + DC (goals + shots) vs Bookmakers (market) | +0.0140 | [+0.0082, +0.0198] | second is better |

## Settings (chosen on validation only)

- Goals-only DC: half-life 365 days.
- Goals + shots DC: half-life **180 days**, goal weight **0.55** (so 45% of the signal comes from shots).
- Blend weights on DC: goals 50%, goals + shots 55%.

|   half_life_days |   goal_weight |   valid_log_loss |
|-----------------:|--------------:|-----------------:|
|              180 |          0.55 |           0.9829 |
|              180 |          0.70 |           0.9834 |
|              180 |          0.40 |           0.9835 |
|              365 |          0.70 |           0.9838 |
|              365 |          0.85 |           0.9840 |
|              365 |          0.55 |           0.9843 |
|              180 |          0.85 |           0.9850 |
|              180 |          0.25 |           0.9851 |

## Notes

- Shots off target get a conversion rate of ~0 (they can't score by definition), so in
  this simple version only shots on target carry information. Shot-location xG
  (StatsBomb, week 3) is the natural upgrade.
- If the shots model's best half-life is shorter than the goals model's, that's the
  expected sign that shots are a less noisy signal: the model can afford to forget faster.
