"""Injuries and suspensions for Premier League clubs, from the official Fantasy Premier League API.

    python -m fa.data.team_news

Free, no key. Every player has a status (a = available, d = doubtful, i = injured,
s = suspended, n = not available, u = unavailable: left the club or on loan) and a news
line written by the Premier League ("Hamstring injury - Expected back 08 Nov"). Only
d / i / s / n are kept. The latest snapshot goes to the `team_news` table and to
records/team_news.csv, so the scheduled run can draw it on the match cards.

Other leagues have no free source like this yet: their cards simply have no team news.
"""
from __future__ import annotations

import duckdb
import pandas as pd
import requests

from fa.config import DB_PATH, ROOT
from fa.data.team_names import assign

FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
NEWS_PATH = ROOT / "records" / "team_news.csv"
KEEP = {"d": "doubt", "i": "out", "s": "out", "n": "out"}
COLUMNS = ["fetched_at", "league", "team", "player", "kind", "status", "chance", "news",
           "news_added", "minutes"]


def fetch_fpl() -> dict:
    r = requests.get(FPL_URL, timeout=60, headers={"User-Agent": "football-analytics (github)"})
    r.raise_for_status()
    return r.json()


def team_mapping(payload: dict, our_teams: list[str]) -> tuple[dict[int, str], list[str]]:
    """FPL team id -> our team name (confident matches only), plus what couldn't be matched."""
    theirs = [{"id": t["id"], "name": t["name"], "shortName": t.get("short_name"),
               "tla": t.get("short_name")} for t in payload.get("teams", [])]
    mapping, problems = {}, []
    for m in assign(theirs, our_teams, "EPL"):
        if m["our_name"] and m["status"] in ("alias", "auto"):
            mapping[m["id"]] = m["our_name"]
        else:
            problems.append(f"{m['their_name']} -> {m['our_name']} ({m['status']})")
    return mapping, problems


def parse(payload: dict, our_teams: list[str], fetched_at: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """FPL payload -> one row per unavailable or doubtful player, with our team names."""
    mapping, problems = team_mapping(payload, our_teams)

    rows = []
    for e in payload.get("elements", []):
        kind = KEEP.get(e.get("status"))
        team = mapping.get(e.get("team"))
        if not kind or not team:
            continue
        chance = e.get("chance_of_playing_next_round")
        rows.append({"fetched_at": fetched_at, "league": "EPL", "team": team,
                     "player": e.get("web_name") or e.get("second_name"), "kind": kind,
                     "status": e["status"], "chance": chance,
                     "news": (e.get("news") or "").strip(), "news_added": e.get("news_added"),
                     "minutes": int(e.get("minutes") or 0)})
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["team", "kind", "minutes"], ascending=[True, False, False]), problems


def save(df: pd.DataFrame) -> None:
    NEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(NEWS_PATH, index=False)
    con = duckdb.connect(str(DB_PATH))
    con.register("news_df", df)
    con.execute("CREATE OR REPLACE TABLE team_news AS SELECT * FROM news_df")
    con.close()


def load(max_age_days: float = 2, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """The latest snapshot, if it's recent enough to trust (empty otherwise)."""
    if not NEWS_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(NEWS_PATH, parse_dates=["fetched_at"])
    now = now if now is not None else pd.Timestamp.now("UTC").tz_localize(None)
    if df.empty or (now - df.fetched_at.max()) > pd.Timedelta(days=max_age_days):
        return pd.DataFrame(columns=COLUMNS)
    return df


def for_team(news: pd.DataFrame, team: str, limit: int = 4) -> dict:
    """{'out': [...], 'doubt': [...]} for one club, most-used players first."""
    g = news[news.team == team]
    res = {}
    for kind in ("out", "doubt"):
        k = g[g.kind == kind].sort_values("minutes", ascending=False)
        names = [f"{r.player} {int(r.chance)}%" if kind == "doubt" and pd.notna(r.chance)
                 else str(r.player) for r in k.itertuples()]
        res[kind] = names[:limit] + ([f"+{len(names) - limit}"] if len(names) > limit else [])
    return res


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    season = con.execute("SELECT max(season) FROM matches WHERE league = 'EPL'").fetchall()[0][0]
    ours = [r[0] for r in con.execute(
        "SELECT DISTINCT home_team FROM matches WHERE league = 'EPL' AND season = ?", [season]).fetchall()]
    con.close()
    try:
        payload = fetch_fpl()
    except requests.RequestException as e:
        raise SystemExit(f"Fantasy Premier League API unreachable ({type(e).__name__}); "
                         "cards will have no team news today.")
    df, problems = parse(payload, ours, pd.Timestamp.now("UTC").tz_localize(None).floor("min"))
    for p in problems:
        print("Team not matched, skipped:", p, "· fix in src/fa/data/team_aliases.csv")
    save(df)
    print(f"Premier League team news: {(df.kind == 'out').sum()} out, {(df.kind == 'doubt').sum()} "
          f"doubtful, across {df.team.nunique()} clubs -> {NEWS_PATH.relative_to(ROOT)}")
    for team, g in df.groupby("team"):
        n = for_team(df, team)
        print(f"  {team:<15} out: {', '.join(n['out']) or '-'} · doubt: {', '.join(n['doubt']) or '-'}")

    # Scorers, assists and cards of finished matches (same source, one more request).
    from fa.data import match_events
    try:
        events = match_events.parse(payload, match_events.fetch_fixtures(), ours)
    except requests.RequestException as e:
        print(f"Match events unavailable ({type(e).__name__}); review cards will show none.")
        return
    match_events.save(events)
    print(f"Match events: {len(events)} rows for {events[['home_team', 'away_team']].drop_duplicates().shape[0]} "
          f"finished matches -> {match_events.EVENTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
