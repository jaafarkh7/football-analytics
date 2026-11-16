# Learning from results

Every Monday the model replays the last year of league matches as it would have forecast them, searches its settings on the older weeks, and adopts new ones only if they also forecast the most recent 8 weeks better. Log loss: lower is better.

**In use since start:** goals model 55% / ratings model 45%, confidence ×1.00, draws +0.0%.

| Date | Decision | Mix (goals model) | Confidence | Draws | Recent weeks: current → candidate |
|---|---|---:|---:|---:|---|
| 2026-11-16 | kept current | 80% | ×1.10 | +12.5% | 0.9344 → 0.9378 (311 matches) |
| 2026-11-09 | kept current | 70% | ×1.05 | +10.0% | 0.9603 → 0.9635 (306 matches) |
