"""Upcoming fixtures from the football-data.org API (free tier).

Setup (once): get a free key at https://www.football-data.org/client/register and put
it in a file called `.env` in the project root (git ignores it):

    FOOTBALL_DATA_API_KEY=your-key-here

Team names are mapped to ours (football-data.co.uk) once per league with
fa.data.team_names and cached in data/team_map.csv; Champions League teams are mapped
through their API team id, so a club has the same name in every competition.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from fa.config import DB_PATH, ROOT
from fa.data.team_names import assign

API = "https://api.football-data.org/v4"
COMPETITIONS = {"EPL": "PL", "LALIGA": "PD", "SERIEA": "SA", "BUNDESLIGA": "BL1",
                "LIGUE1": "FL1", "UCL": "CL"}
DOMESTIC = [lg for lg in COMPETITIONS if lg != "UCL"]
TEAM_MAP_PATH = ROOT / "records" / "team_map.csv"     # committed: scheduled runs reuse it
OLD_TEAM_MAP_PATH = ROOT / "data" / "team_map.csv"     # where earlier versions kept it
UPCOMING = {"SCHEDULED", "TIMED"}          # TIMED = scheduled with a confirmed kickoff time
MIN_GAP_S = 6.5                             # free tier: 10 requests per minute


def api_key() -> str:
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("FOOTBALL_DATA_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("No API key. Add FOOTBALL_DATA_API_KEY=... to a .env file in "
                         f"{ROOT} (free key: https://www.football-data.org/client/register)")
    return key


class Client:
    def __init__(self, key: str | None = None):
        self.session = requests.Session()
        self.session.headers["X-Auth-Token"] = key or api_key()
        self._last = 0.0

    def get(self, path: str, **params) -> dict:
        for attempt in range(4):
            wait = MIN_GAP_S - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.session.get(f"{API}{path}", params=params, timeout=30)
            if r.status_code == 429:                       # rate limited: back off
                time.sleep(int(r.headers.get("X-RequestCounter-Reset", 60)) + 1)
                continue
            if r.status_code in (401, 403):
                raise SystemExit(f"football-data.org refused {path} ({r.status_code}): "
                                 f"{r.text[:200]}\nCheck the key, and that this competition "
                                 "is in your plan.")
            r.raise_for_status()
            return r.json()
        raise SystemExit(f"football-data.org kept rate-limiting {path}")


# ---------- team names ----------

def our_current_teams(con, league: str) -> list[str]:
    season = con.execute("SELECT MAX(season) FROM matches WHERE league = ?", [league]).fetchone()[0]
    rows = con.execute("""SELECT DISTINCT team FROM (
            SELECT home_team AS team FROM matches WHERE league = ? AND season = ?
            UNION SELECT away_team FROM matches WHERE league = ? AND season = ?)""",
                       [league, season, league, season]).fetchall()
    return sorted(r[0] for r in rows)


def build_team_map(client: Client, con) -> pd.DataFrame:
    rows = []
    for league in DOMESTIC:
        teams = client.get(f"/competitions/{COMPETITIONS[league]}/teams")["teams"]
        rows += assign(teams, our_current_teams(con, league), league)
    tm = pd.DataFrame(rows)
    TEAM_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    tm.to_csv(TEAM_MAP_PATH, index=False)
    return tm


def load_team_map(client: Client, con, refresh: bool = False) -> pd.DataFrame:
    if not TEAM_MAP_PATH.exists() and OLD_TEAM_MAP_PATH.exists() and not refresh:
        TEAM_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        TEAM_MAP_PATH.write_bytes(OLD_TEAM_MAP_PATH.read_bytes())
    if TEAM_MAP_PATH.exists() and not refresh:
        return pd.read_csv(TEAM_MAP_PATH)
    tm = build_team_map(client, con)
    shaky = tm[tm.status != "auto"]
    print(f"Matched {len(tm)} team names (saved to {TEAM_MAP_PATH}).")
    if len(shaky):
        print("Please check these; fix any wrong ones in src/fa/data/team_aliases.csv:")
        print(shaky[["league", "their_name", "our_name", "score", "status"]].to_string(index=False))
    return tm


# ---------- fixtures ----------

def matches_to_frame(payload: dict, league: str, id_map: dict[int, tuple[str, str]],
                     statuses: set[str] = UPCOMING) -> pd.DataFrame:
    """API matches -> our rows. Clubs outside our leagues keep their API name
    (short name for display, full name in *_full for linking to older data)."""
    rows = []
    for m in payload.get("matches", []):
        if m.get("status") not in statuses:
            continue
        kickoff = pd.Timestamp(m["utcDate"]).tz_convert("Europe/Paris")
        h, a = m["homeTeam"], m["awayTeam"]
        if not h.get("id") or not a.get("id"):
            continue                                    # knockout tie with teams not yet known
        h_league, h_name = id_map.get(h["id"], (None, h.get("shortName") or h.get("name")))
        a_league, a_name = id_map.get(a["id"], (None, a.get("shortName") or a.get("name")))
        score = m.get("score") or {}
        # Ratings use the 90-minute result; fullTime includes extra time when there was some.
        ft = (score.get("regularTime") if score.get("duration") not in (None, "REGULAR")
              and score.get("regularTime") else score.get("fullTime")) or {}
        rows.append({"league": league, "date": kickoff.tz_localize(None).normalize(),
                     "kickoff_utc": pd.Timestamp(m["utcDate"]).tz_convert("UTC").tz_localize(None),
                     "time": kickoff.strftime("%H:%M"), "home_team": h_name, "away_team": a_name,
                     "home_full": h_name if h_league else (h.get("name") or h_name),
                     "away_full": a_name if a_league else (a.get("name") or a_name),
                     "home_league": h_league, "away_league": a_league,
                     "stage": m.get("stage"), "matchday": m.get("matchday"),
                     "home_goals": ft.get("home"), "away_goals": ft.get("away"),
                     "duration": score.get("duration")})
    return pd.DataFrame(rows)


_client: Client | None = None


def get_client() -> Client:
    """One shared client per run, so the rate limiter sees every request."""
    global _client
    if _client is None:
        _client = Client()
    return _client


def _id_map(client: Client, refresh_teams: bool = False) -> dict[int, tuple[str, str]]:
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    tm = load_team_map(client, con, refresh=refresh_teams)
    con.close()
    tm = tm.dropna(subset=["our_name"])
    return {int(i): (lg, name) for i, lg, name in zip(tm.id, tm.league, tm.our_name)}


def fetch_fixtures(date_from: pd.Timestamp, date_to: pd.Timestamp,
                   refresh_teams: bool = False, client: Client | None = None) -> pd.DataFrame:
    client = client or get_client()
    id_map = _id_map(client, refresh_teams)
    frames = []
    for league, code in COMPETITIONS.items():
        payload = client.get(f"/competitions/{code}/matches",
                             dateFrom=f"{date_from:%Y-%m-%d}", dateTo=f"{date_to:%Y-%m-%d}")
        frames.append(matches_to_frame(payload, league, id_map))
    fx = pd.concat(frames, ignore_index=True)
    if fx.empty:
        return fx
    fx["match_id"] = (fx.league + "_" + fx.date.dt.strftime("%Y%m%d") + "_"
                      + fx.home_team.str.replace(" ", "") + "_" + fx.away_team.str.replace(" ", ""))
    for c in ("market_home", "market_draw", "market_away"):
        fx[c] = float("nan")
    return fx


def fetch_cl_results(client: Client | None = None) -> pd.DataFrame:
    """Finished Champions League matches of the current season (and the previous one,
    if the plan allows), in the same shape as the cl_matches table."""
    client = client or get_client()
    id_map = _id_map(client)
    frames = []
    for season in (None, "previous"):
        params = {"status": "FINISHED"}
        if season == "previous":
            current = client.get("/competitions/CL").get("currentSeason", {})
            start = str(current.get("startDate", ""))[:4]
            if not start.isdigit():
                continue
            params["season"] = int(start) - 1
        try:
            payload = client.get("/competitions/CL/matches", **params)
        except SystemExit as e:                      # older seasons may not be in the free plan
            print(f"  (Champions League {params.get('season', 'current')} season unavailable: "
                  f"{str(e).splitlines()[0][:80]})")
            continue
        frames.append(matches_to_frame(payload, "UCL", id_map, statuses={"FINISHED"}))
    if not frames or all(f.empty for f in frames):
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).dropna(subset=["home_goals", "away_goals"])
    df["season"] = [d.year if d.month >= 7 else d.year - 1 for d in df.date]
    df["neutral"] = df.stage.eq("FINAL")
    # Use the club's full API name for clubs outside our leagues (better for linking);
    # keep the display name too, which is what forecasts use.
    df["home_display"], df["away_display"] = df.home_team, df.away_team
    df["home_team"] = df.home_full
    df["away_team"] = df.away_full
    df["home_goals"] = df.home_goals.astype(int)
    df["away_goals"] = df.away_goals.astype(int)
    df["result"] = ["H" if h > a else ("A" if h < a else "D")
                    for h, a in zip(df.home_goals, df.away_goals)]
    df["round"] = df.stage
    df["source"] = "football-data.org"
    df["match_id"] = ("UCL_" + df.date.dt.strftime("%Y%m%d") + "_" + df.home_team.str.replace(" ", "")
                      + "_" + df.away_team.str.replace(" ", ""))
    return df.drop_duplicates("match_id")
