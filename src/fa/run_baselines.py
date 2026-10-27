"""Week 1: tune Elo on the validation season, score every baseline on the test season.

    python -m fa.run_baselines

Writes reports/week1_baselines.md, reports/calibration_home.png and a
`predictions` table in DuckDB (reused later for the public track record).
"""
from __future__ import annotations

import itertools
import time

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fa.config import (DB_PATH, REPORTS_DIR, TEST_SEASON, TRAIN_SEASONS, VALID_SEASON,
                       season_label)
from fa.eval.metrics import all_metrics, bootstrap_diff, calibration_table, log_loss, rps
from fa.models.baselines import LeagueBaseRates, bookmaker, uniform
from fa.models.elo import EloOutcomeModel, EloParams, run_elo

WARMUP_SEASONS = 1   # ratings all start at 1500; don't learn from that first season

GRID = {
    "k": [15, 20, 25],
    "home_adv": [60, 75, 90, 105],
    "season_regress": [0.0, 0.05, 0.1, 0.2],
    "new_team_offset": [50, 75, 100],
}


def load_matches() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM matches ORDER BY date, match_id").df()
    con.close()
    return df


def tune_elo(matches: pd.DataFrame) -> tuple[EloParams, pd.DataFrame]:
    first = min(TRAIN_SEASONS) + WARMUP_SEASONS
    results = []
    combos = list(itertools.product(*GRID.values()))
    t0 = time.time()
    for combo in combos:
        params = EloParams(**dict(zip(GRID, combo)))
        rated = run_elo(matches, params)
        train = rated[rated.season.between(first, max(TRAIN_SEASONS))]
        valid = rated[rated.season == VALID_SEASON]
        model = EloOutcomeModel(params).fit(train)
        results.append({**dict(zip(GRID, combo)),
                        "valid_log_loss": log_loss(model.predict_proba(valid), valid.result)})
    grid = pd.DataFrame(results).sort_values("valid_log_loss").reset_index(drop=True)
    print(f"Tuned Elo over {len(combos)} settings in {time.time() - t0:.0f}s")
    best = EloParams(**{k: grid.loc[0, k].item() for k in GRID})
    return best, grid


def evaluate(matches: pd.DataFrame, params: EloParams):
    rated = run_elo(matches, params)
    first = min(TRAIN_SEASONS) + WARMUP_SEASONS
    rows, preds = [], {}

    for split, eval_season in [("valid", VALID_SEASON), ("test", TEST_SEASON)]:
        # Fit only on seasons strictly before the one being scored.
        fit_rows = rated[rated.season.between(first, eval_season - 1)]
        ev = rated[rated.season == eval_season].reset_index(drop=True)
        models = {
            "Uniform (1/3 each)": uniform(ev),
            "League base rates": LeagueBaseRates().fit(fit_rows).predict_proba(ev),
            "Elo": EloOutcomeModel(params).fit(fit_rows).predict_proba(ev),
        }
        has_odds = ev[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1).to_numpy()
        if has_odds.mean() > 0.9:
            models["Bookmakers (market)"] = bookmaker(ev)

        for name, p in models.items():
            m = has_odds if name.startswith("Bookmakers") else np.ones(len(ev), bool)
            rows.append({"split": split, "season": season_label(eval_season), "model": name,
                         **all_metrics(p[m], ev.result[m])})
        rows.append({"split": split, "season": season_label(eval_season),
                     "model": "Always pick home win", "log_loss": np.nan, "brier": np.nan,
                     "rps": np.nan, "accuracy": float((ev.result == "H").mean()), "n": len(ev)})
        preds[split] = (ev, models, has_odds)
    return pd.DataFrame(rows), preds, rated


def plot_calibration(ev, models, has_odds, path):
    surface, ink, ink2, grid_c = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"Elo": "#2a78d6", "Bookmakers (market)": "#eb6834"}
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot([0, 1], [0, 1], color=ink2, lw=1, ls=(0, (4, 4)), zorder=1,
            label="Perfect calibration")
    for name, color in colors.items():
        if name not in models:
            continue
        m = has_odds if name.startswith("Bookmakers") else np.ones(len(ev), bool)
        tbl = calibration_table(models[name][m, 0], (ev.result[m] == "H").to_numpy())
        x, y, n = zip(*tbl)
        ax.plot(x, y, color=color, lw=2, zorder=3, label=name)
        ax.scatter(x, y, s=36, color=color, edgecolor=surface, linewidth=2, zorder=4)
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, -2), textcoords="offset points",
                    color=ink, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability of a home win", color=ink2, fontsize=9)
    ax.set_ylabel("How often the home team actually won", color=ink2, fontsize=9)
    ax.set_title(f"Calibration, home wins, {preds_season_label(ev)} (test season)",
                 color=ink, fontsize=11, loc="left", pad=12)
    ax.grid(color=grid_c, lw=0.8)
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def preds_season_label(ev) -> str:
    return season_label(int(ev.season.iloc[0]))


def save_predictions(preds: dict, params: EloParams) -> None:
    frames = []
    for split, (ev, models, _) in preds.items():
        for name, p in models.items():
            frames.append(pd.DataFrame({"model": name, "split": split, "match_id": ev.match_id,
                                        "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2]}))
    df = pd.concat(frames).dropna()
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM df")
    con.close()


def fmt_table(df: pd.DataFrame) -> str:
    lines = ["| Model | Log loss ↓ | RPS ↓ | Brier ↓ | Accuracy ↑ | Matches |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in df.itertuples():
        f = lambda v, d=4: "—" if pd.isna(v) else f"{v:.{d}f}"   # noqa: E731
        lines.append(f"| {r.model} | {f(r.log_loss)} | {f(r.rps)} | {f(r.brier)} | "
                     f"{r.accuracy:.1%} | {r.n} |")
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    matches = matches[matches.season <= TEST_SEASON]       # ongoing season isn't scored
    source = matches.source.iloc[0]

    best, grid = tune_elo(matches)
    print("Best Elo:", best)
    table, preds, rated = evaluate(matches, best)

    ev, models, has_odds = preds["test"]
    cis = [("Elo", "League base rates", bootstrap_diff(models["Elo"], models["League base rates"],
                                                        ev.result, log_loss))]
    cis.append(("Elo", "League base rates (RPS)",
                bootstrap_diff(models["Elo"], models["League base rates"], ev.result, rps)))
    if "Bookmakers (market)" in models:
        m = has_odds
        cis.append(("Elo", "Bookmakers", bootstrap_diff(models["Elo"][m],
                                                       models["Bookmakers (market)"][m],
                                                       ev.result[m].to_numpy(), log_loss)))

    plot_calibration(ev, models, has_odds, REPORTS_DIR / "calibration_home.png")
    save_predictions(preds, best)

    # Sanity check: who does Elo think is best right now?
    last = rated.sort_values("date").groupby("league").tail(1).league
    top = []
    for lg in sorted(last):
        sub = rated[rated.league == lg]
        latest = pd.concat([sub[["date", "home_team", "elo_home"]].set_axis(["date", "team", "elo"], axis=1),
                            sub[["date", "away_team", "elo_away"]].set_axis(["date", "team", "elo"], axis=1)])
        latest = latest[latest.date.dt.year >= TEST_SEASON + 1]
        latest = latest.sort_values("date").groupby("team").tail(1).nlargest(3, "elo")
        top.append(f"| {lg} | " + ", ".join(f"{t} ({e:.0f})" for t, e in zip(latest.team, latest.elo)) + " |")

    report = f"""# Week 1: Baselines

Data source: **{source}** · {len(matches):,} matches · top 5 leagues, {season_label(int(matches.season.min()))} to {season_label(TEST_SEASON)}

Split (by time, never shuffled): train {season_label(min(TRAIN_SEASONS))}–{season_label(max(TRAIN_SEASONS))} (first season is Elo warm-up) · tune on **{season_label(VALID_SEASON)}** · test once on **{season_label(TEST_SEASON)}**.

## Results on the test season ({season_label(TEST_SEASON)})

{fmt_table(table[table.split == 'test'])}

## Validation season ({season_label(VALID_SEASON)}, used for tuning)

{fmt_table(table[table.split == 'valid'])}

## Is the difference real? (paired bootstrap, 2,000 resamples, test season)

| Comparison | Mean difference | 95% interval | Verdict |
|---|---:|---|---|
""" + "\n".join(
        f"| {a} vs {b} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
        f"{'A is better' if hi < 0 else ('B is better' if lo > 0 else 'no clear difference')} |"
        for a, b, (d, lo, hi) in cis) + f"""

Negative means the first model has the lower (better) score.

## Tuned Elo settings

`k={best.k}`, `home_adv={best.home_adv}`, `season_regress={best.season_regress}`, `new_team_offset={best.new_team_offset}`, goal-margin multiplier on.

Top 5 settings by validation log loss:

{grid.head(5).to_markdown(index=False, floatfmt=('.0f', '.0f', '.2f', '.0f', '.4f'))}

If these scores are within ~0.001 of each other, the settings are effectively tied:
don't read meaning into the exact winner.

## Sanity check: current top-rated teams

| League | Top 3 by Elo now |
|---|---|
""" + "\n".join(top) + """

## Calibration

![Calibration](calibration_home.png)

Points near the dashed line mean "when the model says 60%, it happens about 60% of the time".

## How to read these numbers

- **Log loss / RPS / Brier**: lower is better. RPS is the football standard because it treats H < D < A as ordered.
- **Uniform log loss is ln 3 ≈ 1.0986.** Anything above that is worse than guessing.
- **Accuracy is misleading on its own.** Draws are ~25% of matches but almost never the most likely outcome, so every model rarely predicts them.
- **The bar for a real model** (week 2): beat Elo with a clear bootstrap interval, and close as much of the gap to the bookmakers as possible.
"""
    (REPORTS_DIR / "week1_baselines.md").write_text(report)
    print(table.to_string(index=False))
    for a, b, (d, lo, hi) in cis:
        print(f"{a} vs {b}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"\nReport: {REPORTS_DIR / 'week1_baselines.md'}")


if __name__ == "__main__":
    main()
