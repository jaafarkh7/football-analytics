"""Scorers, assists and cards of finished Premier League matches, from the official Fantasy
Premier League API (free, no key). Fetched by `python -m fa.data.team_news`.

The API says who scored / assisted / was booked, but not in which minute. Other leagues
have no free source for this (football-data.org keeps it for a paid plan), so their
review cards simply have no events section.
"""
from __future__ import annotations

import pandas as pd
import requests

from fa.config import ROOT

FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
EVENTS_PATH = ROOT / "records" / "match_events.csv"
COLUMNS = ["league", "date", "home_team", "away_team", "home_goals", "away_goals", "side",
           "kind", "player", "count"]
STATS = {"goals_scored": "goal", "own_goals": "own_goal", "assists": "assist",
         "yellow_cards": "yellow", "red_cards": "red"}
KIND_ORDER = {"goal": 0, "own_goal": 1, "assist": 2, "yellow": 3, "red": 4}


def fetch_fixtures() -> list[dict]:
    r = requests.get(FIXTURES_URL, timeout=60, headers={"User-Agent": "football-analytics (github)"})
    r.raise_for_status()
    return r.json()


def parse(bootstrap: dict, fixtures: list[dict], our_teams: list[str]) -> pd.DataFrame:
    """One row per (match, team, kind, player). `side` is the team the event counts for:
    an own goal counts for the other team."""
    from fa.data.team_news import team_mapping
    teams, _ = team_mapping(bootstrap, our_teams)
    players = {e["id"]: e.get("web_name") or e.get("second_name") for e in bootstrap.get("elements", [])}
    rows = []
    for f in fixtures:
        if not (f.get("finished") or f.get("finished_provisional")) or not f.get("kickoff_time"):
            continue
        home, away = teams.get(f.get("team_h")), teams.get(f.get("team_a"))
        if not home or not away:
            continue
        date = (pd.Timestamp(f["kickoff_time"]).tz_convert("Europe/London").tz_localize(None)
                .normalize())
        base = {"league": "EPL", "date": date, "home_team": home, "away_team": away,
                "home_goals": f.get("team_h_score"), "away_goals": f.get("team_a_score")}
        for stat in f.get("stats", []):
            kind = STATS.get(stat.get("identifier"))
            if not kind:
                continue
            for side_key, side in (("h", "home"), ("a", "away")):
                for item in stat.get(side_key, []):
                    name = players.get(item.get("element"))
                    if not name or not item.get("value"):
                        continue
                    credited = side if kind != "own_goal" else ("away" if side == "home" else "home")
                    rows.append({**base, "side": credited, "kind": kind, "player": name,
                                 "count": int(item["value"])})
    return pd.DataFrame(rows, columns=COLUMNS)


def save(df: pd.DataFrame) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EVENTS_PATH, index=False)


def load() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(EVENTS_PATH, parse_dates=["date"])


def for_match(events: pd.DataFrame, league: str, home: str, away: str, date,
              home_goals=None, away_goals=None) -> dict | None:
    """{'home': {'goals': [...], 'assists': [...], 'yellow': [...], 'red': [...]}, 'away': {...}},
    or None when the match isn't covered."""
    if events.empty:
        return None
    e = events[(events.league == league) & (events.home_team == home) & (events.away_team == away)]
    e = e[(e.date - pd.Timestamp(date)).abs() <= pd.Timedelta(days=1)]
    if e.empty:
        # A finished match with no events at all (0-0, no cards) is still "covered" only if
        # it appears in the file; parse() keeps no row for it, so we can't tell. Show nothing.
        return None
    if home_goals is not None and (e.home_goals.iloc[0] != home_goals or e.away_goals.iloc[0] != away_goals):
        return None                      # sources disagree on the score: show nothing rather than guess
    out = {}
    for side in ("home", "away"):
        s = e[e.side == side].sort_values(["kind", "count", "player"], ascending=[True, False, True],
                                          key=lambda c: c.map(KIND_ORDER) if c.name == "kind" else c)
        goals = [f"{r.player} {r.count}" if r.count > 1 else r.player
                 for r in s[s.kind == "goal"].itertuples()]
        goals += [f"{r.player} (og)" for r in s[s.kind == "own_goal"].itertuples()]
        out[side] = {"goals": goals,
                     "assists": [f"{r.player} {r.count}" if r.count > 1 else r.player
                                 for r in s[s.kind == "assist"].itertuples()],
                     "yellow": list(s[s.kind == "yellow"].player),
                     "red": list(s[s.kind == "red"].player)}
    return out
