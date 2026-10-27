"""Week 2: Dixon–Coles goals model vs Elo vs the market.

    python -m fa.run_week2

Tunes the goals model on 2024-25 with weekly walk-forward refits, then scores
2025-26 once. Writes reports/week2_goals_model.md and reports/calibration_week2.png.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from fa.config import REPORTS_DIR, TEST_SEASON, VALID_SEASON, season_label
from fa.eval.metrics import all_metrics, bootstrap_diff, log_loss
from fa.models.poisson import DCParams, walk_forward
from fa.run_baselines import evaluate, fmt_table, load_matches, plot_calibration, tune_elo

DC_GRID = {"half_life_days": [90, 180, 365, 730], "alpha": [1e-4, 1e-3, 1e-2]}
PCOLS = ["p_home", "p_draw", "p_away"]


def tune_dc(matches: pd.DataFrame):
    ev = matches[matches.season == VALID_SEASON].set_index("match_id")
    rows, t0 = [], time.time()
    for hl, alpha in itertools.product(*DC_GRID.values()):
        p = walk_forward(matches, VALID_SEASON, DCParams(half_life_days=hl, alpha=alpha))
        rows.append({"half_life_days": hl, "alpha": alpha,
                     "valid_log_loss": log_loss(p[PCOLS].to_numpy(), ev.result.loc[p.index])})
    grid = pd.DataFrame(rows).sort_values("valid_log_loss").reset_index(drop=True)
    print(f"Tuned Dixon–Coles over {len(grid)} settings in {time.time() - t0:.0f}s")
    best = DCParams(half_life_days=float(grid.loc[0, "half_life_days"]),
                    alpha=float(grid.loc[0, "alpha"]))
    return best, grid


def blend_weight(p_elo, p_dc, outcomes) -> float:
    """Weight on Dixon–Coles in a linear pool with Elo, chosen on validation."""
    ws = np.linspace(0, 1, 21)
    scores = [log_loss(w * p_dc + (1 - w) * p_elo, outcomes) for w in ws]
    return float(ws[int(np.argmin(scores))])


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    matches = matches[matches.season <= TEST_SEASON]
    source = matches.source.iloc[0]

    elo_params, _ = tune_elo(matches)
    _, preds, _ = evaluate(matches, elo_params)
    dc_params, grid = tune_dc(matches)
    print("Best Dixon–Coles:", dc_params)

    out, extra = {}, {}
    for split, season in [("valid", VALID_SEASON), ("test", TEST_SEASON)]:
        ev, models, has_odds = preds[split]
        dc = walk_forward(matches, season, dc_params).loc[ev.match_id]
        plain = walk_forward(matches, season,
                             DCParams(dc_params.half_life_days, dc_params.alpha,
                                      dixon_coles=False)).loc[ev.match_id]
        models["Poisson (no ρ correction)"] = plain[PCOLS].to_numpy()
        models["Dixon–Coles"] = dc[PCOLS].to_numpy()
        extra[split] = dc.reset_index()
        out[split] = (ev, models, has_odds)

    # Blend weight is learned on validation only, then frozen for the test season.
    ev_v, mv, _ = out["valid"]
    w = blend_weight(mv["Elo"], mv["Dixon–Coles"], ev_v.result)
    for split in out:
        _, models, _ = out[split]
        models["Blend (Elo + Dixon–Coles)"] = w * models["Dixon–Coles"] + (1 - w) * models["Elo"]

    order = ["League base rates", "Elo", "Poisson (no ρ correction)", "Dixon–Coles",
             "Blend (Elo + Dixon–Coles)", "Bookmakers (market)"]
    tables = {}
    for split, (ev, models, has_odds) in out.items():
        rows = []
        for name in order:
            if name not in models:
                continue
            m = has_odds if name.startswith("Bookmakers") else np.ones(len(ev), bool)
            rows.append({"model": name, **all_metrics(models[name][m], ev.result[m])})
        tables[split] = pd.DataFrame(rows)

    ev, models, has_odds = out["test"]
    y = ev.result.to_numpy()
    comparisons = [("Dixon–Coles", "Elo"), ("Blend (Elo + Dixon–Coles)", "Elo"),
                   ("Dixon–Coles", "Poisson (no ρ correction)")]
    if "Bookmakers (market)" in models:
        comparisons += [("Dixon–Coles", "Bookmakers (market)"),
                        ("Blend (Elo + Dixon–Coles)", "Bookmakers (market)")]
    cis = []
    for a, b in comparisons:
        m = has_odds if "Bookmakers" in b else np.ones(len(ev), bool)
        cis.append((a, b, bootstrap_diff(models[a][m], models[b][m], y[m], log_loss)))

    plot_calibration(ev, models, has_odds, REPORTS_DIR / "calibration_week2.png",
                     include=("Elo", "Dixon–Coles", "Bookmakers (market)"))

    # Example output for the write-up: the most lopsided and the closest test matches.
    ex = extra["test"].merge(ev[["match_id", "date", "league", "home_team", "away_team",
                                 "home_goals", "away_goals"]], on="match_id")
    ex["gap"] = (ex.p_home - ex.p_away).abs()
    examples = pd.concat([ex.nlargest(3, "gap"), ex.nsmallest(3, "gap")])

    report = f"""# Week 2: Dixon–Coles goals model

Data source: **{source}** · same split as week 1: tune on **{season_label(VALID_SEASON)}**, test once on **{season_label(TEST_SEASON)}**.

The goals model is refit every 7 days per league during each scored season, using only
matches played before the refit. No prediction ever sees its own match or any later one.

## Results on the test season ({season_label(TEST_SEASON)})

{fmt_table(tables['test'])}

## Validation season ({season_label(VALID_SEASON)})

{fmt_table(tables['valid'])}

## Is the difference real? (paired bootstrap, log loss, test season)

| Comparison | Mean difference | 95% interval | Verdict |
|---|---:|---|---|
""" + "\n".join(
        f"| {a} vs {b} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
        f"{'first is better' if hi < 0 else ('second is better' if lo > 0 else 'no clear difference')} |"
        for a, b, (d, lo, hi) in cis) + f"""

Negative = the first model has the lower (better) log loss.

## Settings (chosen on validation only)

- Dixon–Coles: half-life **{dc_params.half_life_days:.0f} days**, L2 shrinkage **{dc_params.alpha:g}**, 3-season window, weekly refits.
- Blend: **{w:.0%} Dixon–Coles + {1 - w:.0%} Elo** (linear pool of probabilities).

{grid.head(6).to_markdown(index=False, floatfmt=('.0f', 'g', '.4f'))}

## Example predictions (test season)

| Date | Match | xG | P(H / D / A) | Most likely score | Actual |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {r.date:%Y-%m-%d} | {r.home_team} vs {r.away_team} | {r.xg_home:.2f}–{r.xg_away:.2f} | "
        f"{r.p_home:.0%} / {r.p_draw:.0%} / {r.p_away:.0%} | {r.top_score} ({r.top_score_p:.0%}) | "
        f"{r.home_goals}–{r.away_goals} |" for r in examples.itertuples()) + """

## Calibration

![Calibration](calibration_week2.png)

## Notes

- "xG" here is the model's *expected goals from team strengths*, not shot-based xG.
- The most likely single scoreline rarely has more than ~15% probability. Football is noisy.
- The ρ correction is a small effect and varies by league (strongly negative in the
  Bundesliga and Serie A, near zero in the recent Premier League).
- Promoted teams with no top-flight history start with the strength of the 3 weakest teams.
"""
    (REPORTS_DIR / "week2_goals_model.md").write_text(report)
    for split in ("valid", "test"):
        print(f"\n{split}:\n", tables[split].round(4).to_string(index=False))
    for a, b, (d, lo, hi) in cis:
        print(f"{a} vs {b}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"Blend weight on Dixon–Coles: {w:.2f}")
    print(f"\nReport: {REPORTS_DIR / 'week2_goals_model.md'}")


if __name__ == "__main__":
    main()
