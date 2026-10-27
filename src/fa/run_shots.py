"""Week 2b: does shot data close the gap to the market?

    python -m fa.run_shots

Compares the goals-only Dixon–Coles model from week 2 with a version that learns
team strengths partly from shots on target. Same split, same rules: everything is
tuned on 2024-25 and scored once on 2025-26. Writes reports/week2b_shots.md.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from fa.config import REPORTS_DIR, TEST_SEASON, VALID_SEASON, season_label
from fa.eval.metrics import all_metrics, bootstrap_diff, log_loss
from fa.models.poisson import DCParams, walk_forward
from fa.run_baselines import evaluate, fmt_table, load_matches, tune_elo
from fa.run_week2 import PCOLS, blend_weight, tune_dc

SHOTS_GRID = {"half_life_days": [90, 180, 365], "goal_weight": [0.85, 0.7, 0.55, 0.4, 0.25]}


def tune_shots(matches: pd.DataFrame, alpha: float):
    ev = matches[matches.season == VALID_SEASON].set_index("match_id")
    rows, t0 = [], time.time()
    for hl, gw in itertools.product(*SHOTS_GRID.values()):
        p = walk_forward(matches, VALID_SEASON,
                         DCParams(half_life_days=hl, alpha=alpha, goal_weight=gw))
        rows.append({"half_life_days": hl, "goal_weight": gw,
                     "valid_log_loss": log_loss(p[PCOLS].to_numpy(), ev.result.loc[p.index])})
    grid = pd.DataFrame(rows).sort_values("valid_log_loss").reset_index(drop=True)
    print(f"Tuned shots model over {len(grid)} settings in {time.time() - t0:.0f}s")
    best = DCParams(half_life_days=float(grid.loc[0, "half_life_days"]), alpha=alpha,
                    goal_weight=float(grid.loc[0, "goal_weight"]))
    return best, grid


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    matches = matches[matches.season <= TEST_SEASON]
    if "home_sot" not in matches or matches.home_sot.isna().mean() > 0.2:
        raise SystemExit("No shots data. Reload: python -m fa.data.load_results "
                         "(football-data or github-mirror source)")
    source = matches.source.iloc[0]

    elo_params, _ = tune_elo(matches)
    _, preds, _ = evaluate(matches, elo_params)
    goals_params, _ = tune_dc(matches)
    shots_params, grid = tune_shots(matches, goals_params.alpha)
    print("Goals-only:", goals_params, "\nWith shots:", shots_params)

    out = {}
    for split, season in [("valid", VALID_SEASON), ("test", TEST_SEASON)]:
        ev, models, has_odds = preds[split]
        models["Dixon–Coles (goals)"] = (walk_forward(matches, season, goals_params)
                                         .loc[ev.match_id, PCOLS].to_numpy())
        models["Dixon–Coles (goals + shots)"] = (walk_forward(matches, season, shots_params)
                                                 .loc[ev.match_id, PCOLS].to_numpy())
        out[split] = (ev, models, has_odds)

    ev_v, mv, _ = out["valid"]
    w_goals = blend_weight(mv["Elo"], mv["Dixon–Coles (goals)"], ev_v.result)
    w_shots = blend_weight(mv["Elo"], mv["Dixon–Coles (goals + shots)"], ev_v.result)
    for split in out:
        _, m, _ = out[split]
        m["Blend: Elo + DC (goals)"] = w_goals * m["Dixon–Coles (goals)"] + (1 - w_goals) * m["Elo"]
        m["Blend: Elo + DC (goals + shots)"] = (w_shots * m["Dixon–Coles (goals + shots)"]
                                               + (1 - w_shots) * m["Elo"])

    order = ["Elo", "Dixon–Coles (goals)", "Dixon–Coles (goals + shots)",
             "Blend: Elo + DC (goals)", "Blend: Elo + DC (goals + shots)", "Bookmakers (market)"]
    tables = {}
    for split, (ev, models, has_odds) in out.items():
        rows = []
        for name in order:
            if name in models:
                m = has_odds if name.startswith("Bookmakers") else np.ones(len(ev), bool)
                rows.append({"model": name, **all_metrics(models[name][m], ev.result[m])})
        tables[split] = pd.DataFrame(rows)

    ev, models, has_odds = out["test"]
    y = ev.result.to_numpy()
    comparisons = [("Dixon–Coles (goals + shots)", "Dixon–Coles (goals)"),
                   ("Dixon–Coles (goals + shots)", "Elo"),
                   ("Blend: Elo + DC (goals + shots)", "Blend: Elo + DC (goals)"),
                   ("Blend: Elo + DC (goals + shots)", "Elo")]
    if "Bookmakers (market)" in models:
        comparisons += [("Blend: Elo + DC (goals)", "Bookmakers (market)"),
                        ("Blend: Elo + DC (goals + shots)", "Bookmakers (market)")]
    cis = []
    for a, b in comparisons:
        m = has_odds if "Bookmakers" in b else np.ones(len(ev), bool)
        cis.append((a, b, bootstrap_diff(models[a][m], models[b][m], y[m], log_loss)))

    gap_line = ""
    if "Bookmakers (market)" in models:
        t = tables["test"].set_index("model").log_loss
        mk = t["Bookmakers (market)"]
        g_old, g_new = t["Blend: Elo + DC (goals)"] - mk, t["Blend: Elo + DC (goals + shots)"] - mk
        g_elo = t["Elo"] - mk
        gap_line = (f"\n**Gap to the market (test log loss):** Elo {g_elo:+.4f} → "
                    f"goals blend {g_old:+.4f} → goals + shots blend {g_new:+.4f} "
                    f"({1 - g_new / g_elo:.0%} of Elo's gap closed).\n")

    report = f"""# Week 2b: Does shot data help?

Data source: **{source}** · tune on **{season_label(VALID_SEASON)}**, test once on **{season_label(TEST_SEASON)}** · weekly walk-forward refits.

**Idea.** Goals are a noisy measure of team quality; shots on target are less noisy.
The shots model learns team strengths from
`goal_weight × goals + (1 − goal_weight) × 0.35·shots on target` (the conversion rate
is fitted per league on each training window), while still predicting real goals.
`goal_weight = 1` is exactly the week 2 model, so the comparison is clean.

## Test season ({season_label(TEST_SEASON)})

{fmt_table(tables['test'])}
{gap_line}
## Validation season ({season_label(VALID_SEASON)})

{fmt_table(tables['valid'])}

## Is the difference real? (paired bootstrap, log loss, test season)

| Comparison | Mean difference | 95% interval | Verdict |
|---|---:|---|---|
""" + "\n".join(
        f"| {a} vs {b} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
        f"{'first is better' if hi < 0 else ('second is better' if lo > 0 else 'no clear difference')} |"
        for a, b, (d, lo, hi) in cis) + f"""

## Settings (chosen on validation only)

- Goals-only DC: half-life {goals_params.half_life_days:.0f} days.
- Goals + shots DC: half-life **{shots_params.half_life_days:.0f} days**, goal weight **{shots_params.goal_weight:.2f}** (so {1 - shots_params.goal_weight:.0%} of the signal comes from shots).
- Blend weights on DC: goals {w_goals:.0%}, goals + shots {w_shots:.0%}.

{grid.head(8).to_markdown(index=False, floatfmt=('.0f', '.2f', '.4f'))}

## Notes

- Shots off target get a conversion rate of ~0 (they can't score by definition), so in
  this simple version only shots on target carry information. Shot-location xG
  (StatsBomb, week 3) is the natural upgrade.
- If the shots model's best half-life is shorter than the goals model's, that's the
  expected sign that shots are a less noisy signal: the model can afford to forget faster.
"""
    (REPORTS_DIR / "week2b_shots.md").write_text(report)
    for split in ("valid", "test"):
        print(f"\n{split}:\n", tables[split].round(4).to_string(index=False))
    for a, b, (d, lo, hi) in cis:
        print(f"{a} vs {b}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(gap_line.strip())
    print(f"\nReport: {REPORTS_DIR / 'week2b_shots.md'}")


if __name__ == "__main__":
    main()
