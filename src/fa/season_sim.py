"""Simulate the rest of each league season: title, top-4 and relegation chances.

    python -m fa.season_sim                      # current season, all leagues
    python -m fa.season_sim --backtest           # check the simulator against past seasons

Every top-5 league is a double round-robin, so the remaining fixtures are simply the
home/away pairs not yet played. Each remaining match gets probabilities from the
production model (fa.predict); each simulation samples a result, then a scoreline given
that result (from the goals model), so goal difference can break ties.

Ties are broken by points, then goal difference, then goals scored, then at random.
(Some leagues use head-to-head first; that's a small approximation.)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import poisson

from fa.config import DB_PATH, REPORTS_DIR

N_SIMS = 10_000
MAX_GOALS = 8
RELEGATION_SPOTS = 3            # bottom 3 (in Germany and France, 3rd-from-bottom goes to a playoff)
TOP_SPOTS = 4

SIMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS season_sims (
    created_at TIMESTAMP, as_of DATE, league VARCHAR, season INTEGER, team VARCHAR,
    played INTEGER, points INTEGER, goal_diff INTEGER, exp_points DOUBLE, exp_rank DOUBLE,
    p_title DOUBLE, p_top4 DOUBLE, p_relegation DOUBLE
);
"""


def season_so_far(matches: pd.DataFrame, league: str, season: int, as_of: pd.Timestamp):
    lg = matches[(matches.league == league) & (matches.season == season)]
    played = lg[lg.date < as_of]
    teams = sorted(set(lg.home_team) | set(lg.away_team))
    return played, teams


def remaining_fixtures(played: pd.DataFrame, teams: list[str], league: str,
                       as_of: pd.Timestamp) -> pd.DataFrame:
    done = set(zip(played.home_team, played.away_team))
    rows = [(h, a) for h in teams for a in teams if h != a and (h, a) not in done]
    fx = pd.DataFrame(rows, columns=["home_team", "away_team"])
    fx["league"], fx["date"] = league, as_of
    fx["match_id"] = league + "_sim_" + fx.home_team.str.replace(" ", "") + "_" + fx.away_team.str.replace(" ", "")
    return fx


def table(played: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    t = pd.DataFrame(0, index=teams, columns=["played", "points", "gf", "ga"])
    for r in played.itertuples():
        h, a, hg, ag = r.home_team, r.away_team, r.home_goals, r.away_goals
        t.loc[[h, a], "played"] += 1
        t.loc[h, "gf"] += hg; t.loc[h, "ga"] += ag
        t.loc[a, "gf"] += ag; t.loc[a, "ga"] += hg
        t.loc[h, "points"] += 3 if hg > ag else (1 if hg == ag else 0)
        t.loc[a, "points"] += 3 if ag > hg else (1 if hg == ag else 0)
    return t


def sample_scores(probs: np.ndarray, lam: np.ndarray, mu: np.ndarray, n: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """For each match: n results drawn from `probs`, then a scoreline consistent with each
    result drawn from independent Poisson goals. Returns (home_goals, away_goals), (m, n)."""
    g = np.arange(MAX_GOALS + 1)
    hx, ax = np.meshgrid(g, g, indexing="ij")
    masks = [hx > ax, hx == ax, hx < ax]
    hg = np.empty((len(probs), n), dtype=np.int16)
    ag = np.empty((len(probs), n), dtype=np.int16)
    for m in range(len(probs)):
        mat = np.outer(poisson.pmf(g, lam[m]), poisson.pmf(g, mu[m]))
        outcome = rng.choice(3, size=n, p=probs[m] / probs[m].sum())
        for o in range(3):
            sel = outcome == o
            k = int(sel.sum())
            if not k:
                continue
            w = np.where(masks[o], mat, 0).ravel()
            idx = rng.choice(w.size, size=k, p=w / w.sum())
            hg[m, sel], ag[m, sel] = np.unravel_index(idx, mat.shape)
    return hg, ag


def simulate(matches: pd.DataFrame, league: str, season: int, as_of: pd.Timestamp,
             n: int = N_SIMS, seed: int = 0) -> pd.DataFrame:
    from fa.predict import predict
    played, teams = season_so_far(matches, league, season, as_of)
    base = table(played, teams)
    fx = remaining_fixtures(played, teams, league, as_of)

    pts = np.repeat(base.points.to_numpy()[:, None], n, axis=1).astype(np.int32)
    gd = np.repeat((base.gf - base.ga).to_numpy()[:, None], n, axis=1).astype(np.int32)
    gf = np.repeat(base.gf.to_numpy()[:, None], n, axis=1).astype(np.int32)
    if len(fx):
        pr = predict(matches[matches.date < as_of], fx, as_of)
        rng = np.random.default_rng(seed)
        hg, ag = sample_scores(pr[["p_home", "p_draw", "p_away"]].to_numpy(),
                               pr.xg_home.to_numpy(), pr.xg_away.to_numpy(), n, rng)
        idx = {t: i for i, t in enumerate(teams)}
        hi = fx.home_team.map(idx).to_numpy()
        ai = fx.away_team.map(idx).to_numpy()
        home_pts = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        away_pts = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        np.add.at(pts, hi, home_pts)
        np.add.at(pts, ai, away_pts)
        np.add.at(gd, hi, hg - ag)
        np.add.at(gd, ai, ag - hg)
        np.add.at(gf, hi, hg)
        np.add.at(gf, ai, ag)

    # Rank each simulation: points, then GD, then GF, then a random tiebreak.
    noise = np.random.default_rng(seed + 1).random(pts.shape)
    key = pts * 1e6 + (gd + 500) * 1e3 + gf + noise * 0.5
    rank = (-key).argsort(axis=0).argsort(axis=0) + 1          # 1 = champion
    nt = len(teams)
    out = pd.DataFrame({
        "team": teams, "played": base.played.to_numpy(), "points": base.points.to_numpy(),
        "goal_diff": (base.gf - base.ga).to_numpy(),
        "exp_points": pts.mean(axis=1), "exp_rank": rank.mean(axis=1),
        "p_title": (rank == 1).mean(axis=1), "p_top4": (rank <= TOP_SPOTS).mean(axis=1),
        "p_relegation": (rank > nt - RELEGATION_SPOTS).mean(axis=1),
    })
    out.insert(0, "league", league)
    out.insert(1, "season", season)
    return out.sort_values(["exp_points", "points"], ascending=False).reset_index(drop=True)


def final_table(matches: pd.DataFrame, league: str, season: int) -> pd.DataFrame:
    played, teams = season_so_far(matches, league, season, pd.Timestamp.max)
    t = table(played, teams)
    t["gd"] = t.gf - t.ga
    t = t.sort_values(["points", "gd", "gf"], ascending=False)
    t["rank"] = np.arange(1, len(t) + 1)
    return t


# ---------------- backtest ----------------

def backtest(matches: pd.DataFrame, seasons=(2024, 2025), dates=("01-01", "03-01"), n=4000):
    from fa.eval.metrics import calibration_table
    rows = []
    for season in seasons:
        for md in dates:
            as_of = pd.Timestamp(f"{season + 1}-{md}")
            for league in sorted(matches.league.unique()):
                sim = simulate(matches, league, season, as_of, n=n)
                fin = final_table(matches, league, season)
                nt = len(fin)
                sim["won_title"] = sim.team.map(fin["rank"] == 1)
                sim["top4"] = sim.team.map(fin["rank"] <= TOP_SPOTS)
                sim["relegated"] = sim.team.map(fin["rank"] > nt - RELEGATION_SPOTS)
                sim["final_points"] = sim.team.map(fin.points)
                sim["as_of"] = as_of
                rows.append(sim)
                print(f"  {league} {season}-{(season + 1) % 100:02d} from {as_of:%d %b %Y}: done")
    bt = pd.concat(rows, ignore_index=True)
    summary = []
    for p, y, name in [("p_title", "won_title", "Title"), ("p_top4", "top4", "Top 4"),
                       ("p_relegation", "relegated", "Relegation")]:
        pr, out = bt[p].to_numpy(), bt[y].astype(float).to_numpy()
        base = np.full_like(pr, out.mean())
        summary.append({"event": name, "brier": float(np.mean((pr - out) ** 2)),
                        "brier_no_skill": float(np.mean((base - out) ** 2)),
                        "calibration": calibration_table(pr, out, bins=5)})
    pts_err = float(np.mean(np.abs(bt.exp_points - bt.final_points)))
    return bt, summary, pts_err


def backtest_report(bt, summary, pts_err) -> str:
    lines = ["# Season simulator backtest", "",
             "The simulator was run from 1 January and 1 March of the 2024-25 and 2025-26 "
             "seasons, using only results available on those dates, then compared with how "
             "the seasons really ended.", "",
             f"Teams × starting points checked: **{len(bt)}**. Average error in final points: "
             f"**{pts_err:.1f} points**.", "",
             "| Event | Brier score ↓ | Without the model ↓ |", "|---|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['event']} | {s['brier']:.4f} | {s['brier_no_skill']:.4f} |")
    lines += ["", "\"Without the model\" gives every team the same chance (the base rate).", ""]
    for s in summary:
        lines += [f"## Calibration: {s['event'].lower()}", "",
                  "| Predicted | Happened | Teams |", "|---:|---:|---:|"]
        lines += [f"| {p:.0%} | {h:.0%} | {n} |" for p, h, n in s["calibration"]]
        lines.append("")
    lines += ["Calibration bins with fewer than 20 teams are left out (too noisy to read)."]
    return "\n".join(lines) + "\n"


# ---------------- current season ----------------

def save_snapshot(sims: pd.DataFrame, as_of: pd.Timestamp) -> None:
    con = duckdb.connect(str(DB_PATH))
    con.execute(SIMS_SCHEMA)
    rows = sims.assign(created_at=datetime.now(timezone.utc).replace(tzinfo=None), as_of=as_of)
    cols = ["created_at", "as_of", "league", "season", "team", "played", "points", "goal_diff",
            "exp_points", "exp_rank", "p_title", "p_top4", "p_relegation"]
    con.register("sims_df", rows[cols])
    con.execute(f"INSERT INTO season_sims ({', '.join(cols)}) SELECT {', '.join(cols)} FROM sims_df")
    con.close()


def previous_snapshot(league: str, season: int, before: pd.Timestamp) -> pd.DataFrame | None:
    """The latest saved simulation for this league made at least 2 days earlier."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("SELECT * FROM season_sims WHERE league = ? AND season = ?",
                         [league, season]).df()
    except duckdb.CatalogException:
        return None
    finally:
        con.close()
    df = df[pd.to_datetime(df.as_of) <= before - pd.Timedelta(days=2)]
    if df.empty:
        return None
    latest = df.as_of.max()
    return df[df.as_of == latest].set_index("team")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--as-of", help="simulate from this date (default: today)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    matches = con.execute("SELECT * FROM matches ORDER BY date").df()
    con.close()
    matches["date"] = pd.to_datetime(matches.date)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.backtest:
        bt, summary, err = backtest(matches)
        (REPORTS_DIR / "season_sim_backtest.md").write_text(backtest_report(bt, summary, err))
        for s in summary:
            print(f"{s['event']:<11} Brier {s['brier']:.4f} (no-skill {s['brier_no_skill']:.4f})")
        print(f"Average final-points error: {err:.1f}")
        print(f"Report: {REPORTS_DIR / 'season_sim_backtest.md'}")
        return

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now().normalize()
    lines = [f"# Season simulations · {as_of:%Y-%m-%d}", "",
             f"{N_SIMS:,} simulations of the rest of each season with the production model.", ""]
    for league in sorted(matches.league.unique()):
        season = int(matches[(matches.league == league) & (matches.date < as_of)].season.max())
        sim = simulate(matches, league, season, as_of)
        prev = previous_snapshot(league, season, as_of)
        if not args.no_save:
            save_snapshot(sim, as_of)
        lines += [f"## {league} {season}-{(season + 1) % 100:02d}", "",
                  "| Team | Pts | Exp. pts | Title | Top 4 | Relegation |" + (" Title Δ |" if prev is not None else ""),
                  "|---|---:|---:|---:|---:|---:|" + ("---:|" if prev is not None else "")]
        for r in sim.itertuples():
            row = (f"| {r.team} | {r.points} | {r.exp_points:.0f} | {r.p_title:.0%} | "
                   f"{r.p_top4:.0%} | {r.p_relegation:.0%} |")
            if prev is not None and r.team in prev.index:
                row += f" {100 * (r.p_title - prev.loc[r.team, 'p_title']):+.0f} |"
            elif prev is not None:
                row += " — |"
            lines.append(row)
        lines.append("")
        top = sim.iloc[0]
        print(f"{league}: favourite {top.team} ({top.p_title:.0%} title), "
              f"{len(sim)} teams, {int(sim.played.sum() / 2)} matches played")
    path = REPORTS_DIR / "season_sims.md"
    path.write_text("\n".join(lines))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
