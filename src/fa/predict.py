"""Pre-match predictions for upcoming fixtures.

    python -m fa.predict                    # next 8 days, fixtures from football-data.co.uk
    python -m fa.predict --days 4
    python -m fa.predict --fixtures my.csv --as-of 2026-05-15   # backtest / offline

Run `python -m fa.data.load_results` first so the latest results are in.

Model: 55% Dixon–Coles (goals + shots) + 45% Elo, with the settings tuned in weeks 1–2b
(see config.py). Every run is *appended* to the `forecasts` table with a timestamp, so
forecasts can be scored after the matches are played (the public track record).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from fa.config import (BLEND_WEIGHT_DC, DB_PATH, DC_SETTINGS, ELO_SETTINGS, FORECASTS_DIR,
                       LAST_SEASON, LEAGUES)
from fa.data.load_results import ODDS_COLUMNS, _get, _parse_dates
from fa.models.elo import EloOutcomeModel, EloParams, ratings_for_fixtures, run_elo
from fa.models.poisson import DCParams, DixonColes

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DIV_TO_LEAGUE = {codes[0]: league for league, codes in LEAGUES.items()}
PCOLS = ["p_home", "p_draw", "p_away"]

FORECASTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    created_at  TIMESTAMP NOT NULL,
    model       VARCHAR   NOT NULL,
    match_id    VARCHAR   NOT NULL,
    league      VARCHAR, date DATE, home_team VARCHAR, away_team VARCHAR,
    p_home DOUBLE, p_draw DOUBLE, p_away DOUBLE,
    xg_home DOUBLE, xg_away DOUBLE,
    top_score VARCHAR, top_score_p DOUBLE,   -- predicted score: best scoreline within the predicted result
    market_home DOUBLE, market_draw DOUBLE, market_away DOUBLE
);
"""


def read_fixtures(source: str) -> pd.DataFrame:
    """football-data.co.uk fixtures CSV (or a local file in the same format)."""
    if source.startswith("http"):
        raw = _get(source)
        if raw is None:
            raise SystemExit(f"Could not download fixtures from {source}")
        import io
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    else:
        df = pd.read_csv(source, encoding="latin-1", on_bad_lines="skip")
    df.columns = [c.strip().lstrip("﻿").lstrip("ï»¿") for c in df.columns]
    df = df[df["Div"].isin(DIV_TO_LEAGUE)].dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    out = pd.DataFrame({
        "league": df["Div"].map(DIV_TO_LEAGUE),
        "date": _parse_dates(df["Date"]),
        "time": df["Time"] if "Time" in df.columns else "",
        "home_team": df["HomeTeam"].str.strip(),
        "away_team": df["AwayTeam"].str.strip(),
    })
    out[["market_home", "market_draw", "market_away"]] = np.nan
    for cols in ODDS_COLUMNS:
        if all(c in df.columns for c in cols):
            inv = 1 / df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy()
            implied = inv / inv.sum(axis=1, keepdims=True)
            mask = out["market_home"].isna().to_numpy()
            out.loc[mask, ["market_home", "market_draw", "market_away"]] = implied[mask]
    out["match_id"] = (out.league + "_" + out.date.dt.strftime("%Y%m%d") + "_"
                       + out.home_team.str.replace(" ", "") + "_"
                       + out.away_team.str.replace(" ", ""))
    return out.reset_index(drop=True)


def predict(matches: pd.DataFrame, fixtures: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Blend of Elo and goals+shots Dixon–Coles, using only matches before `as_of`."""
    played = matches[matches.date < as_of]

    # Elo: ratings after the last played match; outcome map fitted on completed seasons.
    elo_params = EloParams(**ELO_SETTINGS)
    rated = run_elo(played, elo_params)
    fit_rows = rated[rated.season.between(rated.season.min() + 1, LAST_SEASON)]
    elo_model = EloOutcomeModel(elo_params).fit(fit_rows)
    fx = ratings_for_fixtures(rated, fixtures, elo_params)
    p_elo = elo_model.predict_proba(fx)

    # Dixon–Coles with shots, one fit per league, as of today.
    dc_parts = []
    for league, group in fixtures.groupby("league"):
        model = DixonColes(DCParams(**DC_SETTINGS)).fit(played[played.league == league], as_of)
        dc_parts.append(model.predict(group))
    dc = pd.concat(dc_parts).loc[fixtures.index]

    out = fixtures.copy()
    out[["elo_home_p", "elo_draw_p", "elo_away_p"]] = p_elo
    out[["dc_home_p", "dc_draw_p", "dc_away_p"]] = dc[PCOLS].to_numpy()
    out[PCOLS] = BLEND_WEIGHT_DC * dc[PCOLS].to_numpy() + (1 - BLEND_WEIGHT_DC) * p_elo
    for c in ["xg_home", "xg_away", "p_over25", "p_btts"]:
        out[c] = dc[c].to_numpy()
    # Predicted score: the most likely scoreline *within* the most likely result, so it
    # never contradicts the probabilities (see DixonColes.predict for why that matters).
    pick = np.array(["H", "D", "A"])[out[PCOLS].to_numpy().argmax(axis=1)]
    out["top_score"] = [dc[f"score_{o}"].iloc[k] for k, o in enumerate(pick)]
    out["top_score_p"] = [dc[f"score_{o}_p"].iloc[k] for k, o in enumerate(pick)]
    return out


def check_team_names(matches: pd.DataFrame, fixtures: pd.DataFrame) -> list[str]:
    """Teams with no recent matches are either promoted or a name mismatch. Say so."""
    recent = matches[matches.date >= matches.date.max() - pd.Timedelta(days=400)]
    known = set(zip(recent.league, recent.home_team)) | set(zip(recent.league, recent.away_team))
    unknown = {(lg, t) for lg, h, a in zip(fixtures.league, fixtures.home_team, fixtures.away_team)
               for t in (h, a) if (lg, t) not in known}
    return sorted(f"{t} ({lg})" for lg, t in unknown)


def save(preds: pd.DataFrame, created_at: datetime, model: str = "blend_v1") -> None:
    rows = preds.assign(created_at=created_at, model=model)
    for c in ("market_home", "market_draw", "market_away"):
        if c not in rows:
            rows[c] = np.nan
    cols = ["created_at", "model", "match_id", "league", "date", "home_team", "away_team",
            "p_home", "p_draw", "p_away", "xg_home", "xg_away", "top_score", "top_score_p",
            "market_home", "market_draw", "market_away"]
    con = duckdb.connect(str(DB_PATH))
    con.execute(FORECASTS_SCHEMA)
    con.register("new_rows", rows[cols])
    con.execute(f"INSERT INTO forecasts SELECT {', '.join(cols)} FROM new_rows")
    con.close()


def to_markdown(preds: pd.DataFrame, as_of: pd.Timestamp, unknown: list[str]) -> str:
    lines = [f"# Predictions · generated {as_of:%Y-%m-%d}", "",
             "Model: 55% Dixon–Coles (goals + shots) + 45% Elo. "
             "Probabilities are home / draw / away. For analysis, not betting advice.", "",
             "*Predicted score* is the most likely scoreline within the most likely result. "
             "Any single scoreline is unlikely (usually 8–14%), so read the percentages first.", ""]
    has_market = preds.market_home.notna().any()
    for league, g in preds.sort_values(["league", "date"]).groupby("league"):
        lines += [f"## {league}", "",
                  "| Date | Match | Model H / D / A | Expected goals | Predicted score | "
                  "Over 2.5 | Both score" + (" | Market H / D / A |" if has_market else " |"),
                  "|---|---|---|---|---|---|---" + ("|---|" if has_market else "|")]
        for r in g.itertuples():
            row = (f"| {r.date:%a %d %b} | {r.home_team} vs {r.away_team} | "
                   f"{r.p_home:.0%} / {r.p_draw:.0%} / {r.p_away:.0%} | "
                   f"{r.xg_home:.1f}–{r.xg_away:.1f} | {r.top_score} | "
                   f"{r.p_over25:.0%} | {r.p_btts:.0%} |")
            if has_market:
                row += (f" {r.market_home:.0%} / {r.market_draw:.0%} / {r.market_away:.0%} |"
                        if not np.isnan(r.market_home) else " — |")
            lines.append(row)
        lines.append("")
    if has_market:
        d = preds.dropna(subset=["market_home"]).copy()
        d["gap"] = d.p_home - d.market_home
        top = d.reindex(d.gap.abs().sort_values(ascending=False).index).head(5)
        lines += ["## Where the model disagrees most with the market", "",
                  "The market is usually right; these are the matches to watch to learn *why*.", "",
                  "| Match | Model home win | Market home win | Difference |", "|---|---|---|---|"]
        lines += [f"| {r.home_team} vs {r.away_team} | {r.p_home:.0%} | {r.market_home:.0%} | "
                  f"{r.gap:+.0%} |" for r in top.itertuples()]
        lines.append("")
    if unknown:
        lines += ["## Data warnings", "",
                  "No recent matches for: " + ", ".join(unknown)
                  + ". Promoted teams are expected here; anything else is a team-name mismatch.", ""]
    return "\n".join(lines)


def get_fixtures(args, as_of: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Fixtures in [as_of, as_of + days). Returns (fixtures, notes for the user)."""
    end = as_of + pd.Timedelta(days=args.days)
    source = args.source
    if source == "auto":
        from fa.data.fixtures_api import api_key
        try:
            api_key()
            source = "api"
        except SystemExit:
            source = "csv"
    notes = []

    csv_fx = pd.DataFrame()
    if source == "csv" or args.fixtures != FIXTURES_URL:
        csv_fx = read_fixtures(args.fixtures)
        fx = csv_fx[(csv_fx.date >= as_of) & (csv_fx.date < end)].reset_index(drop=True)
        if fx.empty:
            span = (f"{csv_fx.date.min():%Y-%m-%d} → {csv_fx.date.max():%Y-%m-%d}"
                    if len(csv_fx) else "nothing for our leagues")
            raise SystemExit(f"No fixtures in the next {args.days} days. The fixtures file "
                             f"covers {span}; football-data.co.uk updates it closer to each "
                             "weekend. Try again later, or use the API (--source api).")
        return fx, ["Fixtures: football-data.co.uk fixtures.csv"]

    from fa.data.fixtures_api import fetch_fixtures
    fx = fetch_fixtures(as_of, end - pd.Timedelta(days=1), refresh_teams=args.refresh_teams)
    if fx.empty:
        raise SystemExit(f"football-data.org lists no upcoming matches in the next {args.days} days.")
    fx = fx[(fx.date >= as_of) & (fx.date < end)].reset_index(drop=True)
    notes.append("Fixtures: football-data.org API")
    # Market odds, where football-data.co.uk already lists the same match.
    try:
        odds = read_fixtures(FIXTURES_URL).set_index("match_id")
        mcols = ["market_home", "market_draw", "market_away"]
        hit = fx.match_id.isin(odds.index)
        fx.loc[hit, mcols] = odds.loc[fx.match_id[hit], mcols].to_numpy()
        notes.append(f"Market odds found for {int(hit.sum())} of {len(fx)} matches")
    except (SystemExit, Exception) as e:                     # odds are optional
        notes.append(f"No market odds ({type(e).__name__})")
    return fx, notes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=8, help="predict matches in the next N days")
    p.add_argument("--source", choices=["auto", "api", "csv"], default="auto",
                   help="fixtures from football-data.org (needs a key) or the CSV file")
    p.add_argument("--fixtures", default=FIXTURES_URL, help="URL or local CSV (csv source)")
    p.add_argument("--refresh-teams", action="store_true", help="re-match API team names")
    p.add_argument("--as-of", help="pretend today is this date (YYYY-MM-DD), for backtests")
    p.add_argument("--no-save", action="store_true", help="don't record in the forecasts table")
    args = p.parse_args()

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now().normalize()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    matches = con.execute("SELECT * FROM matches ORDER BY date, match_id").df()
    con.close()

    fixtures, notes = get_fixtures(args, as_of)
    cups = fixtures[~fixtures.league.isin(LEAGUES)]          # e.g. Champions League
    fixtures = fixtures[fixtures.league.isin(LEAGUES)].reset_index(drop=True)
    for n in notes:
        print(n)

    latest = matches[matches.date < as_of].date.max()
    if (as_of - latest).days > 10:
        print(f"Warning: latest result in the database is {latest:%Y-%m-%d}. "
              "Run `python -m fa.data.load_results` first?")

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    path = FORECASTS_DIR / f"{as_of:%Y-%m-%d}.md"
    report = ""
    if len(fixtures):
        unknown = check_team_names(matches[matches.date < as_of], fixtures)
        preds = predict(matches, fixtures, as_of)
        if not args.no_save:
            save(preds, datetime.now(timezone.utc).replace(tzinfo=None))
        report = to_markdown(preds, as_of, unknown)
        show = preds[["league", "date", "home_team", "away_team", "p_home", "p_draw", "p_away",
                      "top_score", "p_over25", "p_btts"]].rename(
            columns={"top_score": "score", "p_over25": "over2.5", "p_btts": "btts"}).copy()
        show[["over2.5", "btts"]] = (show[["over2.5", "btts"]] * 100).round().astype(int)
        show[PCOLS] = (show[PCOLS] * 100).round().astype(int)
        print(show.to_string(index=False))
        if unknown:
            print("\nNo recent matches for:", ", ".join(unknown))
        print(f"\n{len(preds)} league predictions{'' if args.no_save else ' saved'}")
    if len(cups):
        cl_preds = predict_champions_league(matches, cups, as_of)
        if cl_preds is None:
            report += cups_markdown(cups)
            print(f"{len(cups)} Champions League fixtures listed without predictions "
                  "(run `python -m fa.data.load_european` to enable them)")
        else:
            if not args.no_save:
                save(cl_preds.assign(xg_home=np.nan, xg_away=np.nan, top_score=None,
                                     top_score_p=np.nan), datetime.now(timezone.utc).replace(tzinfo=None),
                     model="cl_elo_v1")
            report += cl_markdown(cl_preds)
            show = cl_preds[["date", "home_team", "away_team"] + PCOLS].copy()
            show[PCOLS] = (show[PCOLS] * 100).round().astype(int)
            print("\nChampions League:\n" + show.to_string(index=False))
            print(f"{len(cl_preds)} Champions League predictions{'' if args.no_save else ' saved'}")
    path.write_text(report or f"# No fixtures · {as_of:%Y-%m-%d}\n")
    print(f"Report: {path}")


def predict_champions_league(matches: pd.DataFrame, cups: pd.DataFrame, as_of: pd.Timestamp,
                             use_api: bool = True) -> pd.DataFrame | None:
    """Cross-league Elo predictions, or None if the Champions League history isn't loaded."""
    from fa.models.cl_predict import predict_cl
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        cl_hist = con.execute("SELECT * FROM cl_matches ORDER BY date").df()
    except duckdb.CatalogException:
        return None
    finally:
        con.close()
    recent = pd.DataFrame()
    if use_api:
        from fa.data.fixtures_api import fetch_cl_results
        try:
            recent = fetch_cl_results()
            print(f"Champions League results from the API: {len(recent)} matches")
        except Exception as e:                                # predictions still work without
            print(f"Couldn't fetch recent Champions League results ({type(e).__name__}); "
                  "ratings use older seasons only")
    cups = cups[cups.league == "UCL"].reset_index(drop=True)
    out = predict_cl(matches, cl_hist, recent, cups, as_of)
    unknown = sorted({n for n, ok in zip(out.home_team, out.known_home) if not ok} |
                     {n for n, ok in zip(out.away_team, out.known_away) if not ok})
    if unknown:
        print("New to our Champions League data (rated as an average outside club): "
              + ", ".join(unknown))
    out.attrs["unknown"] = unknown
    return out


def cl_markdown(cl: pd.DataFrame) -> str:
    league_names = {"EPL": "ENG", "LALIGA": "ESP", "SERIEA": "ITA", "BUNDESLIGA": "GER",
                    "LIGUE1": "FRA"}

    def tag(league):
        return f" ({league_names[league]})" if isinstance(league, str) and league in league_names else ""

    lines = ["## Champions League", "",
             "Cross-league Elo: one rating pool for all clubs, with league strength learned "
             "from European results. Elo only (no shots data for European matches), so "
             "treat these as rougher than the league predictions.", "",
             "| Date | Match | Model H / D / A |", "|---|---|---|"]
    for r in cl.sort_values(["date", "time"]).itertuples():
        lines.append(f"| {r.date:%a %d %b} {r.time} | {r.home_team}{tag(r.home_league)} vs "
                     f"{r.away_team}{tag(r.away_league)} | "
                     f"{r.p_home:.0%} / {r.p_draw:.0%} / {r.p_away:.0%} |")
    unknown = cl.attrs.get("unknown", [])
    if unknown:
        lines += ["", "No European history in our data for: " + ", ".join(unknown)
                  + " (rated as an average outside club)."]
    return "\n".join(lines) + "\n"


def cups_markdown(cups: pd.DataFrame) -> str:
    lines = ["## Champions League (fixtures only)", "",
             "Load the Champions League history (`python -m fa.data.load_european`) to get "
             "predictions here.", "",
             "| Date | Match | Stage |", "|---|---|---|"]
    for r in cups.sort_values(["date", "time"]).itertuples():
        lines.append(f"| {r.date:%a %d %b} {r.time} | {r.home_team} vs {r.away_team} | "
                     f"{str(r.stage).replace('_', ' ').title()} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
