"""Fact sheets for the written match analysis: every number the writer may use, computed
here from our data and already formatted, so the model only has to copy, never compute.

Pre-match: each team's season so far (points, goals, chances created and conceded,
finishing), recent form, home or away record, league ranks, injuries (Premier League)
and our forecast. Post-match: the match's numbers next to each team's season averages,
who deserved it, scorers and cards (Premier League), forecast right or wrong, and what
the matchday changed in the season simulations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fa.cards import LEAGUE_NAMES
from fa.cards_season import conversion_rate

LEAGUES = LEAGUE_NAMES["en"]


def f1(x) -> str:
    return f"{x:.1f}"


def pct(x) -> str:
    return f"{int(round(100 * x))}%"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def finishing(diff: float, verb: str) -> str:
    """Goals against what the chances were worth, in words: the writer copies this phrase
    instead of a signed number it might misread."""
    shots = ("their shots on target are usually worth" if verb == "scored"
             else "the shots on target they faced are usually worth")
    if abs(diff) < 0.5:
        return f"{verb} about as many goals as {shots}"
    more = "more" if diff > 0 else "fewer"
    return f"{verb} {abs(diff):.1f} {more} goals than {shots}"


def team_games(matches: pd.DataFrame, league: str, season: int, team: str, before) -> pd.DataFrame:
    """This season's league games of `team` before `before`, from the team's point of view."""
    m = matches[(matches.league == league) & (matches.season == season) & (matches.date < before)]
    rows = []
    for venue, side, other in (("home", "home", "away"), ("away", "away", "home")):
        g = m[m[f"{side}_team"] == team]
        rows.append(pd.DataFrame({
            "date": g.date, "venue": venue, "opponent": g[f"{other}_team"],
            "gf": g[f"{side}_goals"], "ga": g[f"{other}_goals"],
            "sot_for": g.get(f"{side}_sot"), "sot_against": g.get(f"{other}_sot"),
            "shots_for": g.get(f"{side}_shots"), "shots_against": g.get(f"{other}_shots")}))
    df = pd.concat(rows).sort_values("date")
    df["pts"] = np.select([df.gf > df.ga, df.gf == df.ga], [3, 1], 0)
    df["res"] = np.select([df.gf > df.ga, df.gf == df.ga], ["W", "D"], "L")
    return df


def league_ranks(matches: pd.DataFrame, league: str, season: int, before) -> pd.DataFrame:
    """Per-team season averages and their rank in the league (1 = most)."""
    m = matches[(matches.league == league) & (matches.season == season) & (matches.date < before)]
    teams = sorted(set(m.home_team) | set(m.away_team))
    rows = []
    for t in teams:
        g = team_games(matches, league, season, t, before)
        if g.empty:
            continue
        rows.append({"team": t, "pts": g.pts.sum(), "gd": (g.gf - g.ga).sum(), "gf": g.gf.sum(),
                     "sot_for_pg": g.sot_for.mean(), "sot_against_pg": g.sot_against.mean(),
                     "gf_pg": g.gf.mean(), "ga_pg": g.ga.mean()})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = df.sort_values(["pts", "gd", "gf"], ascending=False).team.tolist()
    df["position"] = df.team.map({t: i + 1 for i, t in enumerate(order)})
    df["sot_for_rank"] = df.sot_for_pg.rank(ascending=False, method="min").astype(int)
    df["sot_against_rank"] = df.sot_against_pg.rank(ascending=True, method="min").astype(int)
    return df.set_index("team")


def team_profile(matches: pd.DataFrame, league: str, season: int, team: str, venue: str,
                 before, ranks: pd.DataFrame, conv: float) -> dict:
    g = team_games(matches, league, season, team, before)
    if g.empty:
        return {"name": team, "games_played": "0"}
    n = len(g)
    last5 = g.tail(5)
    at = g[g.venue == venue]
    sxg_for, sxg_against = conv * g.sot_for.sum(), conv * g.sot_against.sum()
    p = {
        "name": team,
        "league_position": ordinal(int(ranks.loc[team, "position"])) if team in ranks.index else None,
        "teams_in_league": str(len(ranks)),
        "games_played": str(n),
        "points": str(int(g.pts.sum())),
        "goals_for": str(int(g.gf.sum())), "goals_against": str(int(g.ga.sum())),
        "goals_for_per_game": f1(g.gf.mean()), "goals_against_per_game": f1(g.ga.mean()),
        "last_5_results_oldest_first": " ".join(last5.res),
        "last_5_points": str(int(last5.pts.sum())),
        f"{venue}_record_w_d_l": f"{(at.res == 'W').sum()}-{(at.res == 'D').sum()}-{(at.res == 'L').sum()}",
    }
    if g.sot_for.notna().any():
        p.update({
            "shots_on_target_per_game": f1(g.sot_for.mean()),
            "shots_on_target_conceded_per_game": f1(g.sot_against.mean()),
            "shots_per_game": f1(g.shots_for.mean()),
            "rank for shots on target created": ordinal(int(ranks.loc[team, "sot_for_rank"])),
            "rank for fewest shots on target conceded": ordinal(int(ranks.loc[team, "sot_against_rank"])),
            "finishing": finishing(g.gf.sum() - sxg_for, "scored"),
            "defending and goalkeeping": finishing(g.ga.sum() - sxg_against, "conceded"),
        })
    return {k: v for k, v in p.items() if v is not None}


def preview_facts(row, matches: pd.DataFrame, news: dict | None = None) -> dict:
    """Facts for a match that hasn't been played yet (`row` is a forecast)."""
    date = pd.Timestamp(row.date)
    lg = matches[(matches.league == row.league) & (matches.date < date)]
    season = int(lg.season.max()) if len(lg) else None
    facts = {"type": "preview", "competition": LEAGUES.get(row.league, row.league),
             "date": f"{date:%A %-d %B}", "home_team": row.home_team, "away_team": row.away_team}
    if season is not None and row.league in LEAGUES and row.league != "UCL":
        ranks = league_ranks(matches, row.league, season, date)
        conv = conversion_rate(matches, row.league, date)
        facts["home"] = team_profile(matches, row.league, season, row.home_team, "home", date, ranks, conv)
        facts["away"] = team_profile(matches, row.league, season, row.away_team, "away", date, ranks, conv)
    fc = {"home_win": pct(row.p_home), "draw": pct(row.p_draw), "away_win": pct(row.p_away)}
    if isinstance(row.get("top_score"), str):
        fc["most_likely_score"] = row.top_score
    if pd.notna(row.get("p_over25")):
        fc["over_2_5_goals"] = pct(row.p_over25)
        fc["both_teams_score"] = pct(row.p_btts)
    if pd.notna(row.get("xg_home")):
        fc["expected_goals"] = f"{f1(row.xg_home)} - {f1(row.xg_away)}"
    facts["our_forecast"] = fc
    if news:
        facts["team_news"] = {side: {k: v for k, v in news[side].items() if v}
                              for side in ("home", "away") if news.get(side)}
    return facts


def review_facts(r, matches: pd.DataFrame, events: dict | None = None,
                 changes: list | None = None) -> dict:
    """Facts for a played match (`r` is a row from cards_review.review_rows)."""
    date = pd.Timestamp(r.result_date)
    called = "HDA"[int(np.argmax([r.p_home, r.p_draw, r.p_away]))] == r.result
    facts = {"type": "review", "competition": LEAGUES.get(r.league, r.league),
             "date": f"{date:%A %-d %B}", "home_team": r.home_team, "away_team": r.away_team,
             "final_score": f"{int(r.home_goals)}-{int(r.away_goals)}",
             "result": {"H": f"{r.home_team} won", "D": "draw", "A": f"{r.away_team} won"}[r.result],
             "match": {"shots": f"{int(r.home_shots)}-{int(r.away_shots)}",
                       "shots_on_target": f"{int(r.home_sot)}-{int(r.away_sot)}",
                       "expected_goals_from_shots_on_target": f"{f1(r.sxg_home)} - {f1(r.sxg_away)}"},
             "deserved_result_from_chances": {"home_win": pct(r.d_home), "draw": pct(r.d_draw),
                                              "away_win": pct(r.d_away)},
             "our_pre_match_forecast": {"home_win": pct(r.p_home), "draw": pct(r.p_draw),
                                        "away_win": pct(r.p_away),
                                        "verdict": "right" if called else "wrong"}}
    lg = matches[(matches.league == r.league) & (matches.date < date)]
    if len(lg):
        season = int(lg.season.max())
        for side, team in (("home", r.home_team), ("away", r.away_team)):
            g = team_games(matches, r.league, season, team, date)
            if len(g) and g.sot_for.notna().any():
                facts[f"{side}_season_average_before_this_match"] = {
                    "games": str(len(g)), "shots_on_target_per_game": f1(g.sot_for.mean()),
                    "shots_on_target_conceded_per_game": f1(g.sot_against.mean()),
                    "goals_per_game": f1(g.gf.mean())}
    if events:
        facts["scorers_assists_cards"] = {r.home_team: events["home"], r.away_team: events["away"],
                                          "note": "no minutes are known for these events"}
    if changes:
        names = {"p_title": "title chance", "p_top4": "top-4 chance", "p_relegation": "relegation chance"}
        facts["season_outlook_change"] = [f"{t}: {names[ev]} {pct(b)} before, {pct(a)} after"
                                          for t, ev, b, a in changes]
    return facts
