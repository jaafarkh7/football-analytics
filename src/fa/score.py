"""Score saved forecasts against what actually happened: the live track record.

    python -m fa.data.load_results     # bring in the latest results first
    python -m fa.score

Rules (fixed before any result was seen):
  * For each match and model, only the LATEST forecast made BEFORE KICKOFF counts.
  * Forecasts saved without a kickoff time (the very first ones) count only if they
    were made at least 2 hours before the match day started.
  * Every eligible forecast is scored. Nothing is dropped for being embarrassing.

Writes reports/track_record.md and reports/track_record_matches.csv.
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from fa.config import DB_PATH, REPORTS_DIR
from fa.eval.metrics import accuracy, all_metrics, bootstrap_diff, log_loss

PCOLS = ["p_home", "p_draw", "p_away"]
MCOLS = ["market_home", "market_draw", "market_away"]
MODEL_NAMES = {"blend_v1": "League model (Dixon–Coles + Elo)",
               "cl_elo_v1": "Champions League model (cross-league Elo)"}


def eligible_forecasts(fc: pd.DataFrame) -> pd.DataFrame:
    """Latest forecast per (model, match) made before kickoff."""
    fc = fc.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    if "kickoff_utc" not in fc:
        fc["kickoff_utc"] = pd.NaT
    fc["kickoff_utc"] = pd.to_datetime(fc["kickoff_utc"])
    deadline = fc.kickoff_utc.where(fc.kickoff_utc.notna(), fc.date - pd.Timedelta(hours=2))
    ok = fc[pd.to_datetime(fc.created_at) < deadline]
    return (ok.sort_values("created_at")
              .groupby(["model", "match_id"], as_index=False).tail(1)
              .reset_index(drop=True))


def attach_results(fc: pd.DataFrame, results: pd.DataFrame, max_days: int = 3) -> pd.DataFrame:
    """Join forecasts to results on league + teams, with dates within `max_days`.

    Match ids can differ by a day between sources (UK vs Paris local dates), so we don't
    join on them. A fixture between two clubs happens once per league season, so teams
    plus a nearby date identify it.
    """
    r = results[["league", "home_team", "away_team", "date", "home_goals", "away_goals",
                 "result"]].rename(columns={"date": "result_date"})
    m = fc.merge(r, on=["league", "home_team", "away_team"], how="inner")
    m = m[(m.result_date - m.date).abs() <= pd.Timedelta(days=max_days)]
    return m.drop_duplicates(["model", "match_id"])


def champions_league_results(use_api: bool) -> pd.DataFrame:
    if not use_api:
        return pd.DataFrame()
    try:
        from fa.data.fixtures_api import fetch_cl_results
        cl = fetch_cl_results()
    except (SystemExit, Exception) as e:
        print(f"Champions League results unavailable ({type(e).__name__}); skipping them")
        return pd.DataFrame()
    if cl.empty:
        return cl
    return cl.assign(league="UCL", home_team=cl.home_display, away_team=cl.away_display)


def summarize(df: pd.DataFrame) -> dict:
    out = all_metrics(df[PCOLS].to_numpy(), df.result)
    out["uniform_log_loss"] = float(np.log(3))
    score = df.top_score.notna()
    if score.any():
        actual = df.home_goals.astype(int).astype(str) + "-" + df.away_goals.astype(int).astype(str)
        out["exact_score"] = float((df.top_score[score] == actual[score]).mean())
    goals = df.home_goals + df.away_goals
    o = df.p_over25.notna() if "p_over25" in df else pd.Series(False, index=df.index)
    if o.any():
        out["over25_acc"] = float(((df.p_over25[o] > 0.5) == (goals[o] >= 3)).mean())
    b = df.p_btts.notna() if "p_btts" in df else pd.Series(False, index=df.index)
    if b.any():
        both = (df.home_goals > 0) & (df.away_goals > 0)
        out["btts_acc"] = float(((df.p_btts[b] > 0.5) == both[b]).mean())
    return out


def fmt(x, pct=False, d=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.0%}" if pct else f"{x:.{d}f}"


def build_report(scored: pd.DataFrame, pending: int, generated: pd.Timestamp) -> str:
    lines = [f"# Track record · updated {generated:%Y-%m-%d}", "",
             "Every forecast here was saved *before* kickoff and scored afterwards. Only the "
             "latest pre-kickoff forecast per match counts; nothing is excluded after the fact.", "",
             f"Scored matches: **{len(scored)}** · forecasts waiting for results: {pending}", ""]
    if scored.empty:
        return "\n".join(lines + ["No results yet. Check back after the next round."])

    lines += ["## Overall", "",
              "| Model | Matches | Log loss ↓ | RPS ↓ | Result right | Exact score | Over 2.5 right | Both score right |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, g in scored.groupby("model"):
        s = summarize(g)
        lines.append(f"| {MODEL_NAMES.get(model, model)} | {s['n']} | {fmt(s['log_loss'])} | "
                     f"{fmt(s['rps'])} | {fmt(s['accuracy'], True)} | {fmt(s.get('exact_score'), True)} | "
                     f"{fmt(s.get('over25_acc'), True)} | {fmt(s.get('btts_acc'), True)} |")
    lines += ["", f"For reference: guessing 1/3 each gives log loss {np.log(3):.4f}. Over a full "
              "season our test-season log loss was ~0.99 and the market's ~0.98. "
              "With few matches, expect big swings: one weekend says almost nothing.", ""]

    with_market = scored.dropna(subset=MCOLS)
    if len(with_market):
        p, m, y = with_market[PCOLS].to_numpy(), with_market[MCOLS].to_numpy(), with_market.result
        lines += ["## Model vs market (matches with odds)", "",
                  "| | Matches | Log loss ↓ | Result right |", "|---|---:|---:|---:|",
                  f"| Model | {len(y)} | {log_loss(p, y):.4f} | {accuracy(p, y):.0%} |",
                  f"| Market | {len(y)} | {log_loss(m, y):.4f} | {accuracy(m, y):.0%} |", ""]
        if len(y) >= 30:
            d, lo, hi = bootstrap_diff(p, m, y.to_numpy(), log_loss)
            lines += [f"Model minus market: {d:+.4f} (95% interval [{lo:+.4f}, {hi:+.4f}]).", ""]
        else:
            lines += ["Fewer than 30 matches with odds: too few for a meaningful comparison yet.", ""]

    lines += ["## By competition", "", "| Competition | Matches | Log loss ↓ | Result right |",
              "|---|---:|---:|---:|"]
    for lg, g in scored.groupby("league"):
        lines.append(f"| {lg} | {len(g)} | {log_loss(g[PCOLS].to_numpy(), g.result):.4f} | "
                     f"{accuracy(g[PCOLS].to_numpy(), g.result):.0%} |")

    scored = scored.assign(week=scored.date.dt.to_period("W-SUN").dt.start_time)
    lines += ["", "## By week", "", "| Week of | Matches | Log loss ↓ | Result right |",
              "|---|---:|---:|---:|"]
    for wk, g in scored.groupby("week"):
        lines.append(f"| {wk:%Y-%m-%d} | {len(g)} | {log_loss(g[PCOLS].to_numpy(), g.result):.4f} | "
                     f"{accuracy(g[PCOLS].to_numpy(), g.result):.0%} |")

    if len(scored) >= 100:
        p = scored[PCOLS].to_numpy().ravel()
        hit = np.stack([(scored.result == o).to_numpy() for o in "HDA"], axis=1).ravel()
        bins = pd.cut(p, [0, .2, .4, .6, .8, 1], include_lowest=True)
        cal = pd.DataFrame({"p": p, "hit": hit, "bin": bins}).groupby("bin", observed=True).agg(
            predicted=("p", "mean"), happened=("hit", "mean"), n=("p", "size"))
        lines += ["", "## Calibration (all outcomes)", "",
                  "When the model says X%, does it happen about X% of the time?", "",
                  "| Predicted | Happened | Outcomes |", "|---:|---:|---:|"]
        lines += [f"| {r.predicted:.0%} | {r.happened:.0%} | {r.n} |" for r in cal.itertuples()]

    biggest = scored.assign(
        surprise=-np.log(np.choose(scored.result.map({"H": 0, "D": 1, "A": 2}).to_numpy(),
                                   scored[PCOLS].to_numpy().T))).nlargest(5, "surprise")
    lines += ["", "## Biggest surprises", "",
              "| Date | Match | Model H / D / A | Result |", "|---|---|---|---|"]
    lines += [f"| {r.date:%Y-%m-%d} | {r.home_team} vs {r.away_team} | "
              f"{r.p_home:.0%} / {r.p_draw:.0%} / {r.p_away:.0%} | {int(r.home_goals)}-{int(r.away_goals)} |"
              for r in biggest.itertuples()]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true", help="skip fetching Champions League results")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        fc = con.execute("SELECT * FROM forecasts").df()
    except duckdb.CatalogException:
        raise SystemExit("No forecasts yet: run `python -m fa.predict` first.")
    league = con.execute("SELECT * FROM matches").df()
    con.close()

    total = fc.groupby(["model", "match_id"]).ngroups
    fc_ok = eligible_forecasts(fc)
    late = total - len(fc_ok)

    results = [league]
    if (fc_ok.model == "cl_elo_v1").any():
        cl = champions_league_results(not args.no_api)
        if len(cl):
            results.append(cl)
    res = pd.concat(results, ignore_index=True)
    res["date"] = pd.to_datetime(res["date"])

    scored = attach_results(fc_ok, res).sort_values(["date", "league"])
    pending = len(fc_ok) - len(scored)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now()
    (REPORTS_DIR / "track_record.md").write_text(build_report(scored, pending, now))
    keep = ["date", "league", "model", "home_team", "away_team", "p_home", "p_draw", "p_away",
            "top_score", "home_goals", "away_goals", "result", "market_home", "market_draw",
            "market_away", "created_at"]
    scored[[c for c in keep if c in scored]].to_csv(REPORTS_DIR / "track_record_matches.csv",
                                                    index=False)

    print(f"Forecasts: {total} matches forecast · {len(fc_ok)} eligible (made before kickoff)"
          + (f" · {late} not eligible" if late else ""))
    print(f"Scored: {len(scored)} · waiting for results: {pending}")
    for model, g in scored.groupby("model"):
        s = summarize(g)
        print(f"  {MODEL_NAMES.get(model, model)}: {s['n']} matches, log loss {s['log_loss']:.4f}, "
              f"result right {s['accuracy']:.0%}")
    print(f"Report: {REPORTS_DIR / 'track_record.md'}")


if __name__ == "__main__":
    main()
