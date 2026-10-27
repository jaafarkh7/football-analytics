"""Week 2c: does recent form or this-season performance improve the predictions?

    python -m fa.run_form

A logistic regression combines the existing signals (Elo rating gap, Dixon–Coles
expected goals) with form features. We compare feature sets trained identically on
2016-17..2023-24, choose on 2024-25, and score 2025-26 once. Writes reports/week2c_form.md.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fa.config import (BLEND_WEIGHT_DC, DC_SETTINGS, ELO_SETTINGS, REPORTS_DIR, TEST_SEASON,
                       VALID_SEASON, season_label)
from fa.eval.metrics import OUTCOMES, all_metrics, bootstrap_diff, log_loss
from fa.features.form import match_features
from fa.models.elo import EloOutcomeModel, EloParams, run_elo
from fa.models.poisson import DCParams, walk_forward
from fa.run_baselines import fmt_table, load_matches

FIRST_STACK_SEASON = 2016          # Dixon–Coles needs ~2 seasons of history first
PCOLS = ["p_home", "p_draw", "p_away"]

BASE = ["elo_gap", "xg_ratio", "xg_total"]
FEATURE_SETS = {
    "Base (Elo + DC signals)": BASE,
    "+ last-5 form": BASE + ["d_form5_pts", "d_form5_gd", "d_form5_sotd"],
    "+ this season so far": BASE + ["d_season_pts", "d_season_gd", "season_pts_w", "season_gd_w"],
    "+ rest days": BASE + ["d_rest_days"],
    "+ everything": BASE + ["d_form5_pts", "d_form5_gd", "d_form5_sotd", "d_season_pts",
                            "d_season_gd", "season_pts_w", "season_gd_w", "d_rest_days"],
}


def build_table(matches: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    rated = run_elo(matches, EloParams(**ELO_SETTINGS)).set_index("match_id")
    dc = pd.concat([walk_forward(matches, s, DCParams(**DC_SETTINGS))
                    for s in range(FIRST_STACK_SEASON, TEST_SEASON + 1)])
    feats = match_features(matches)
    df = rated.join(dc, how="inner").join(feats)
    df["elo_gap"] = df.elo_diff / 100
    df["xg_ratio"] = np.log(df.xg_home / df.xg_away)
    df["xg_total"] = np.log(df.xg_home * df.xg_away)
    # Season-so-far numbers mean little after 2 games and a lot after 15: let the model
    # weight them by how much of the season has been played.
    w = np.minimum(df.season_n_min, 15) / 15
    df["season_pts_w"] = df.d_season_pts * w
    df["season_gd_w"] = df.d_season_gd * w
    print(f"Built {len(df):,} rows with all signals in {time.time() - t0:.0f}s")
    return df.reset_index()


def fit_stack(train: pd.DataFrame, cols: list[str]):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(train[cols].to_numpy(), train.result.to_numpy())
    return model


def proba(model, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    p = model.predict_proba(df[cols].to_numpy())
    classes = list(model[-1].classes_)
    return p[:, [classes.index(o) for o in OUTCOMES]]


def production_blend(df: pd.DataFrame, train_seasons: range, ev: pd.DataFrame) -> np.ndarray:
    """The model currently used by fa.predict, for reference."""
    elo = EloOutcomeModel().fit(df[df.season.isin(train_seasons)])
    return BLEND_WEIGHT_DC * ev[PCOLS].to_numpy() + (1 - BLEND_WEIGHT_DC) * elo.predict_proba(ev)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    matches = matches[matches.season <= TEST_SEASON]
    df = build_table(matches)

    results, preds = [], {}
    for split, season in [("valid", VALID_SEASON), ("test", TEST_SEASON)]:
        train = df[df.season.between(FIRST_STACK_SEASON, season - 1)]
        ev = df[df.season == season].reset_index(drop=True)
        preds[split] = {"ev": ev}
        p_prod = production_blend(df, range(FIRST_STACK_SEASON - 1, season), ev)
        preds[split]["Current model (55/45 blend)"] = p_prod
        results.append({"split": split, "model": "Current model (55/45 blend)",
                        **all_metrics(p_prod, ev.result)})
        for name, cols in FEATURE_SETS.items():
            p = proba(fit_stack(train, cols), ev, cols)
            preds[split][name] = p
            results.append({"split": split, "model": name, **all_metrics(p, ev.result)})
    res = pd.DataFrame(results)

    # Which features matter? Standardised coefficients (home-win minus away-win logit).
    full_cols = FEATURE_SETS["+ everything"]
    full = fit_stack(df[df.season.between(FIRST_STACK_SEASON, VALID_SEASON - 1)], full_cols)
    lr = full[-1]
    ci = {c: i for i, c in enumerate(lr.classes_)}
    coef = pd.Series(lr.coef_[ci["H"]] - lr.coef_[ci["A"]], index=full_cols)
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)

    # Early vs late season on the test set: is season form worth more later on?
    ev = preds["test"]["ev"]
    phase = np.where(ev.season_n_min < 10, "first 10 games", "rest of season")
    by_phase = []
    for ph in ["first 10 games", "rest of season"]:
        m = phase == ph
        row = {"phase": ph, "matches": int(m.sum())}
        for name in ["Base (Elo + DC signals)", "+ this season so far", "+ everything"]:
            row[name] = log_loss(preds["test"][name][m], ev.result[m])
        by_phase.append(row)
    by_phase = pd.DataFrame(by_phase)

    y = ev.result.to_numpy()
    cis = []
    for a, b in [("+ everything", "Base (Elo + DC signals)"),
                 ("+ last-5 form", "Base (Elo + DC signals)"),
                 ("+ this season so far", "Base (Elo + DC signals)"),
                 ("+ everything", "Current model (55/45 blend)")]:
        cis.append((a, b, bootstrap_diff(preds["test"][a], preds["test"][b], y, log_loss)))

    def table(split):
        return fmt_table(res[res.split == split])

    report = f"""# Week 2c: Does form or this-season performance help?

Same data and rules as before: features use only earlier matches; stacked logistic
regression trained on {season_label(FIRST_STACK_SEASON)}–{season_label(VALID_SEASON - 1)}, compared on
**{season_label(VALID_SEASON)}**, scored once on **{season_label(TEST_SEASON)}** (retrained including {season_label(VALID_SEASON)}).

Base signals: Elo rating gap, Dixon–Coles expected-goals ratio and total. Form features
are home-minus-away differences: last-5 points, goal difference and shots-on-target
difference per game; season-so-far points and goal difference per game (also weighted
by how far into the season we are); rest days since the previous league match.

## Validation ({season_label(VALID_SEASON)})

{table('valid')}

## Test ({season_label(TEST_SEASON)})

{table('test')}

## Is the difference real? (paired bootstrap, log loss, test season)

| Comparison | Mean difference | 95% interval | Verdict |
|---|---:|---|---|
""" + "\n".join(
        f"| {a} vs {b} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
        f"{'first is better' if hi < 0 else ('second is better' if lo > 0 else 'no clear difference')} |"
        for a, b, (d, lo, hi) in cis) + f"""

## Early vs late season (test, log loss)

{by_phase.to_markdown(index=False, floatfmt='.4f')}

## What the model leans on (standardised coefficients, home win vs away win)

Bigger = more influence, given everything else in the model. Features overlap a lot
(a team in good form usually also has a high rating), so small coefficients mean
"adds little beyond the others", not "irrelevant to football".

{coef.to_frame('weight').to_markdown(floatfmt='+.3f')}
"""
    (REPORTS_DIR / "week2c_form.md").write_text(report)
    for split in ("valid", "test"):
        print(f"\n{split}:\n", res[res.split == split].drop(columns="split").round(4).to_string(index=False))
    print("\nby phase (test):\n", by_phase.round(4).to_string(index=False))
    for a, b, (d, lo, hi) in cis:
        print(f"{a} vs {b}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("\ncoefficients:\n", coef.round(3).to_string())
    print(f"\nReport: {REPORTS_DIR / 'week2c_form.md'}")


if __name__ == "__main__":
    main()
