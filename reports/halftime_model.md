# Half-time win probabilities

Pre-match forecasts replayed week by week, updated with the real half-time score.
Tuned on 2024-25: second-half goal share **0.60**, game-state effect **0.20** (trailing side scores 20% more, leader 20% less).

| Season | Matches | Pre-match | Half-time score only | Half-time model |
|---|---:|---:|---:|---:|
| 2024-25 (tuning) | 1751 | 0.9803 | 0.8721 | 0.8214 |
| 2025-26 (test) | 1752 | 0.9921 | 0.8472 | 0.8133 |

Log loss, lower is better. 1.0986 = no information.

## Calibration on the test season

| Predicted | Matches | Predicted avg | Observed |
|---|---:|---:|---:|
| 0%–10% | 811 | 4.5% | 3.7% |
| 10%–20% | 817 | 15.2% | 13.8% |
| 20%–30% | 1083 | 25.1% | 22.1% |
| 30%–40% | 1110 | 35.0% | 36.5% |
| 40%–50% | 382 | 44.7% | 43.5% |
| 50%–60% | 267 | 54.5% | 60.3% |
| 60%–70% | 254 | 64.6% | 67.3% |
| 70%–80% | 199 | 74.7% | 78.9% |
| 80%–90% | 155 | 84.9% | 89.7% |
| 90%–100% | 178 | 95.4% | 96.1% |

## Test season by half-time state

| State | Matches | Log loss | Leader went on to win: predicted / observed |
|---|---:|---:|---:|
| Home leading | 602 | 0.5977 | 74.2% / 79.6% |
| Level | 703 | 1.0337 | draw 35.2% / 38.0% |
| Away leading | 447 | 0.7571 | 65.7% / 68.0% |
