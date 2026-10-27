"""Week 2d: comparing clubs across leagues (Champions League).

    python -m fa.data.load_european     # once
    python -m fa.run_cl

Tunes the cross-league Elo on Champions League 2016-17..2019-20 and scores 2024-25 once,
against the same model with every league held at the same average ("leagues equal").
Writes reports/week2d_champions_league.md and reports/league_strength.png.
"""
from __future__ import annotations

import itertools

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fa.config import DB_PATH, ELO_SETTINGS, REPORTS_DIR, TEST_SEASON
from fa.eval.metrics import all_metrics, bootstrap_diff, log_loss
from fa.models.cross_elo import CLOutcomeModel, CrossEloParams, run_cross_elo
from fa.run_baselines import fmt_table

VALID_CL = [2016, 2017, 2018, 2019]     # 2014-15 is warm-up, 2015-16 trains the first map
TEST_CL = 2024
GRID = {"k_cl": [10, 20, 30, 60], "home_adv_cl": [70, 100, 130], "external_offset": [0, 100, 200]}
BASE = dict(k=ELO_SETTINGS["k"], home_adv=ELO_SETTINGS["home_adv"],
            new_team_offset=ELO_SETTINGS["new_team_offset"])


def seasonal_probs(rated: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Each season predicted with an outcome map fitted only on earlier CL seasons."""
    parts = []
    for s in seasons:
        prior = rated[(rated.season < s) & (rated.season >= 2015)]
        ev = rated[rated.season == s].copy()
        ev[["p_home", "p_draw", "p_away"]] = CLOutcomeModel().fit(prior).predict_proba(ev)
        parts.append(ev)
    return pd.concat(parts)


def score(dom, cl, params, seasons) -> float:
    pr = seasonal_probs(run_cross_elo(dom, cl, params), seasons)
    return log_loss(pr[["p_home", "p_draw", "p_away"]].to_numpy(), pr.result)


def tune(dom, cl, league_strength: bool):
    rows = []
    for combo in itertools.product(*GRID.values()):
        params = CrossEloParams(**BASE, **dict(zip(GRID, combo)), league_strength=league_strength)
        rows.append({**dict(zip(GRID, combo)), "valid_log_loss": score(dom, cl, params, VALID_CL)})
    grid = pd.DataFrame(rows).sort_values("valid_log_loss").reset_index(drop=True)
    best = CrossEloParams(**BASE, **{k: float(grid.loc[0, k]) for k in GRID},
                          league_strength=league_strength)
    return best, grid


def league_strength_over_time(dom, cl, params) -> pd.DataFrame:
    """League average rating at the end of each season (cross-league model)."""
    rows = []
    for s in sorted(dom.season.unique()):
        d = dom[dom.season <= s]
        c = cl[cl.season <= s]
        r = run_cross_elo(d, c, params)
        for lg, mean in r.attrs["league_means"].items():
            rows.append({"season": s, "league": lg, "rating": mean})
    t = pd.DataFrame(rows).pivot(index="season", columns="league", values="rating")
    return t.sub(t.mean(axis=1), axis=0)          # relative to the average of the five


def plot_strength(t: pd.DataFrame, cl_seasons: set[int], path) -> None:
    surface, ink, ink2, grid_c = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"EPL": "#2a78d6", "LALIGA": "#eb6834", "SERIEA": "#1baf7a",
              "BUNDESLIGA": "#eda100", "LIGUE1": "#e87ba4"}
    names = {"EPL": "Premier League", "LALIGA": "La Liga", "SERIEA": "Serie A",
             "BUNDESLIGA": "Bundesliga", "LIGUE1": "Ligue 1"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    x = [f"{s}-{(s + 1) % 100:02d}" for s in t.index]
    for i, s in enumerate(t.index):                       # shade seasons without CL data
        if s not in cl_seasons:
            ax.axvspan(i - 0.5, i + 0.5, color=grid_c, alpha=0.55, lw=0)
    lgs = [lg for lg in ["EPL", "LALIGA", "BUNDESLIGA", "SERIEA", "LIGUE1"] if lg in t]
    for lg in lgs:
        ax.plot(x, t[lg], color=colors[lg], lw=2, label=names[lg], zorder=3)
        ax.scatter(x[-1], t[lg].iloc[-1], s=36, color=colors[lg], edgecolor=surface, lw=2, zorder=4)
    # Direct labels at the right edge, nudged apart so they never overlap.
    ys = sorted(((t[lg].iloc[-1], lg) for lg in lgs), reverse=True)
    span = t.values.max() - t.values.min()
    placed = []
    for y, lg in ys:
        y_lab = min(y, placed[-1] - 0.07 * span) if placed else y
        placed.append(y_lab)
        ax.annotate(names[lg], (len(x) - 1, y), xytext=(len(x) - 0.6, y_lab),
                    textcoords="data", va="center", fontsize=8, color=ink)
    ax.axhline(0, color=ink2, lw=0.8, ls=(0, (4, 4)))
    ax.set_ylabel("Average club rating vs the five-league average", color=ink2, fontsize=8.5)
    fig.suptitle("League strength learned from Champions League results", x=0.02, ha="left",
                 color=ink, fontsize=11)
    ax.set_title("Shaded seasons: no Champions League data, so lines move only through "
                 "promotions and relegations", loc="left", color=ink2, fontsize=7.5, pad=26)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=5, labelcolor=ink, handlelength=1.5, borderaxespad=0.3)
    ax.grid(axis="y", color=grid_c, lw=0.8)
    ax.tick_params(colors=ink2, labelsize=7.5, length=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.5, len(x) + 1.6)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    dom = con.execute(f"SELECT * FROM matches WHERE season <= {TEST_SEASON} ORDER BY date").df()
    try:
        cl = con.execute("SELECT * FROM cl_matches ORDER BY date").df()
    except duckdb.CatalogException:
        raise SystemExit("No Champions League data: run `python -m fa.data.load_european` first.")
    con.close()

    best_eq, grid_eq = tune(dom, cl, league_strength=False)
    best_ls, grid_ls = tune(dom, cl, league_strength=True)
    print("Leagues equal  :", best_eq)
    print("League strength:", best_ls)

    rows, preds = [], {}
    for name, params in [("Leagues equal (baseline)", best_eq), ("Cross-league Elo", best_ls)]:
        rated = run_cross_elo(dom, cl, params)
        for split, seasons in [("valid", VALID_CL), ("test", [TEST_CL])]:
            pr = seasonal_probs(rated, seasons)
            preds[(name, split)] = pr
            rows.append({"split": split, "model": name,
                         **all_metrics(pr[["p_home", "p_draw", "p_away"]].to_numpy(), pr.result)})
    res = pd.DataFrame(rows)

    a = preds[("Cross-league Elo", "test")]
    b = preds[("Leagues equal (baseline)", "test")]
    P = ["p_home", "p_draw", "p_away"]
    d, lo, hi = bootstrap_diff(a[P].to_numpy(), b[P].to_numpy(), a.result.to_numpy(), log_loss)

    # Where does it matter? Matches between clubs of two *different* top-5 leagues.
    cross = (a.home_league.notna() & a.away_league.notna() & (a.home_league != a.away_league)).to_numpy()
    ll_cross = (log_loss(a[P].to_numpy()[cross], a.result[cross]),
                log_loss(b[P].to_numpy()[cross], b.result[cross]))

    strength = league_strength_over_time(dom, cl, best_ls)
    plot_strength(strength, set(cl.season.unique()), REPORTS_DIR / "league_strength.png")
    final = strength.iloc[-1].sort_values(ascending=False)

    report = f"""# Week 2d: Comparing clubs across leagues (Champions League)

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

{fmt_table(res[res.split == 'test'])}

Validation (2016-17..2019-20):

{fmt_table(res[res.split == 'valid'])}

**Cross-league Elo vs leagues equal, test log loss:** {d:+.4f} (95% interval [{lo:+.4f}, {hi:+.4f}]).

Matches between clubs of two *different* top-5 leagues ({int(cross.sum())} in 2024-25), where
league strength matters most: cross-league Elo {ll_cross[0]:.4f} vs leagues equal {ll_cross[1]:.4f}.

## League strength

![League strength](league_strength.png)

Average club rating relative to the five-league average, end of {strength.index[-1]}-{(strength.index[-1] + 1) % 100:02d}:

| League | Rating vs average |
|---|---:|
""" + "\n".join(f"| {lg} | {v:+.0f} |" for lg, v in final.items()) + f"""

Roughly, 100 rating points ≈ the difference between a 64% and a 50% expected score.
In seasons without Champions League data (shaded: 2020-21..2023-24 and 2025-26) the
lines still move, but only because promoted clubs enter below a league's average and
relegated clubs leave it. That movement is bookkeeping, not evidence about league strength.

## Settings (validation)

Cross-league Elo: Champions League K **{best_ls.k_cl:.0f}**, home advantage **{best_ls.home_adv_cl:.0f}**,
external clubs start **{best_ls.external_offset:.0f}** below the top-5 average.

{grid_ls.head(6).to_markdown(index=False, floatfmt=(".0f", ".0f", ".0f", ".4f"))}

## Caveats

- ~120–190 matches per season: every comparison here is noisier than for league matches.
- Clubs outside the top 5 are only seen in Europe (6–12 matches a season), so their
  ratings are rough; Champions League qualifying rounds aren't in the data.
- Predictions are Elo-only: there's no shots data for European matches.
"""
    (REPORTS_DIR / "week2d_champions_league.md").write_text(report)
    print(res.round(4).to_string(index=False))
    print(f"\nCross-league vs equal (test): {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"Cross-league matches only ({int(cross.sum())}): {ll_cross[0]:.4f} vs {ll_cross[1]:.4f}")
    print("\nLeague strength now (vs average):", final.round(0).to_dict())
    print(f"Report: {REPORTS_DIR / 'week2d_champions_league.md'}")


if __name__ == "__main__":
    main()
