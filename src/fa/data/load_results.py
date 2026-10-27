"""Download historical match results into DuckDB.

Sources
-------
football-data  (default) football-data.co.uk CSVs: results + bookmaker odds.
github-mirror  GitHub copy of football-data.co.uk: results + shots, no odds.
openfootball   GitHub JSON: results only.

Usage
-----
    python -m fa.data.load_results                    # football-data.co.uk
    python -m fa.data.load_results --source openfootball
"""
from __future__ import annotations

import argparse
import io
import time

import duckdb
import numpy as np
import pandas as pd
import requests

from fa.config import DB_PATH, FIRST_SEASON, LAST_SEASON, LEAGUES, RAW_DIR, season_label

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id    VARCHAR PRIMARY KEY,
    league      VARCHAR NOT NULL,
    season      INTEGER NOT NULL,     -- starting year, 2023 = 2023-24
    date        DATE    NOT NULL,
    home_team   VARCHAR NOT NULL,
    away_team   VARCHAR NOT NULL,
    home_goals  INTEGER NOT NULL,
    away_goals  INTEGER NOT NULL,
    result      VARCHAR NOT NULL,     -- 'H', 'D', 'A'
    odds_home   DOUBLE,               -- closing/average decimal odds (NULL if unknown)
    odds_draw   DOUBLE,
    odds_away   DOUBLE,
    home_shots  INTEGER,              -- shots and shots on target (NULL if unknown)
    away_shots  INTEGER,
    home_sot    INTEGER,
    away_sot    INTEGER,
    source      VARCHAR NOT NULL
);
"""

# Preference order for odds columns in football-data.co.uk files.
# Avg* = market average (2019-20 onwards), BbAv* = older market average, then Bet365, Pinnacle.
ODDS_COLUMNS = [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"),
                ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA")]


def _get(url: str, retries: int = 3) -> bytes | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"  ! failed {url}: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def _cached(name: str, url: str, refresh: bool = False) -> bytes | None:
    """Download once, then reuse the local copy. `refresh=True` always re-downloads
    (used for the season in progress, which gains new matches every week)."""
    path = RAW_DIR / name
    if path.exists() and not refresh:
        return path.read_bytes()
    content = _get(url)
    if content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return content


def _result(hg: int, ag: int) -> str:
    return "H" if hg > ag else ("A" if hg < ag else "D")


SHOT_COLUMNS = {"HS": "home_shots", "AS": "away_shots", "HST": "home_sot", "AST": "away_sot"}


def _ongoing(season: int) -> bool:
    return season > LAST_SEASON


def _parse_dates(col: pd.Series) -> pd.Series:
    """football-data.co.uk uses 11/08/2023 (day first); the mirror uses 2023-08-11.
    `dayfirst=True` on ISO dates silently swaps day and month, so detect the format."""
    col = col.astype(str).str.strip()
    iso = col.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns]")
    out[iso] = pd.to_datetime(col[iso], format="%Y-%m-%d")
    out[~iso] = pd.to_datetime(col[~iso], dayfirst=True, format="mixed")
    return out


def _parse_fd_csv(raw: bytes, source: str) -> pd.DataFrame:
    """football-data.co.uk CSV format (also used by the GitHub mirror)."""
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame({
        "date": _parse_dates(df["Date"]),
        "home_team": df["HomeTeam"].str.strip(),
        "away_team": df["AwayTeam"].str.strip(),
        "home_goals": df["FTHG"].astype(int),
        "away_goals": df["FTAG"].astype(int),
    })
    out[["odds_home", "odds_draw", "odds_away"]] = np.nan
    for cols in ODDS_COLUMNS:          # fill gaps from the next-best bookmaker
        if all(c in df.columns for c in cols):
            vals = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy()
            mask = out["odds_home"].isna().to_numpy()
            out.loc[mask, ["odds_home", "odds_draw", "odds_away"]] = vals[mask]
    for col, name in SHOT_COLUMNS.items():
        out[name] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    out["source"] = source
    return out


def fetch_football_data(league: str, season: int) -> pd.DataFrame:
    fd_code = LEAGUES[league][0]
    yy = f"{season % 100:02d}{(season + 1) % 100:02d}"
    url = f"https://www.football-data.co.uk/mmz4281/{yy}/{fd_code}.csv"
    raw = _cached(f"football-data/{yy}_{fd_code}.csv", url, refresh=_ongoing(season))
    return pd.DataFrame() if raw is None else _parse_fd_csv(raw, "football-data")


MIRROR_SLUGS = {"EPL": "premier-league", "LALIGA": "la-liga", "SERIEA": "serie-a",
                "BUNDESLIGA": "bundesliga", "LIGUE1": "ligue-1"}


def fetch_github_mirror(league: str, season: int) -> pd.DataFrame:
    """Mirror of football-data.co.uk on GitHub (datasets/football-datasets): results and
    shots, but no odds. A fallback when football-data.co.uk itself is unreachable."""
    yy = f"{season % 100:02d}{(season + 1) % 100:02d}"
    url = ("https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/"
           f"{MIRROR_SLUGS[league]}/season-{yy}.csv")
    raw = _cached(f"github-mirror/{yy}_{league}.csv", url, refresh=_ongoing(season))
    return pd.DataFrame() if raw is None else _parse_fd_csv(raw, "github-mirror")


def fetch_openfootball(league: str, season: int) -> pd.DataFrame:
    of_code = LEAGUES[league][1]
    s = season_label(season)
    url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{s}/{of_code}.json"
    raw = _cached(f"openfootball/{s}_{of_code}.json", url, refresh=_ongoing(season))
    if raw is None:
        return pd.DataFrame()
    import json
    rows = []
    for m in json.loads(raw).get("matches", []):
        score = m.get("score")
        ft = score.get("ft") if isinstance(score, dict) else score  # both formats exist
        if not ft or len(ft) != 2:
            continue                      # not played yet / postponed
        rows.append({"date": pd.to_datetime(m["date"]), "home_team": m["team1"],
                     "away_team": m["team2"], "home_goals": int(ft[0]),
                     "away_goals": int(ft[1])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out[["odds_home", "odds_draw", "odds_away"]] = np.nan
    out[list(SHOT_COLUMNS.values())] = np.nan
    out["source"] = "openfootball"
    return out


FETCHERS = {"football-data": fetch_football_data, "github-mirror": fetch_github_mirror,
            "openfootball": fetch_openfootball}


def load(source: str, last_season: int, leagues: list[str]) -> pd.DataFrame:
    frames = []
    for league in leagues:
        for season in range(FIRST_SEASON, last_season + 1):
            df = FETCHERS[source](league, season)
            if df.empty:
                print(f"  - {league} {season_label(season)}: no data")
                continue
            df["league"], df["season"] = league, season
            frames.append(df)
            print(f"  + {league} {season_label(season)}: {len(df)} matches")
    df = pd.concat(frames, ignore_index=True)
    df["result"] = [_result(h, a) for h, a in zip(df.home_goals, df.away_goals)]
    df["match_id"] = (df.league + "_" + df.date.dt.strftime("%Y%m%d") + "_"
                      + df.home_team.str.replace(" ", "") + "_"
                      + df.away_team.str.replace(" ", ""))
    df = df.drop_duplicates("match_id").sort_values(["date", "league"]).reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> None:
    """Cheap sanity checks: catch broken downloads before they poison a model."""
    assert df.match_id.is_unique, "duplicate matches"
    assert (df.home_team != df.away_team).all(), "team playing itself"
    assert df.home_goals.between(0, 15).all() and df.away_goals.between(0, 15).all()
    # A season runs roughly Aug-May; allow June-August slack (Covid pushed 2019-20 into Aug 2020).
    start = pd.to_datetime(df.season.astype(str) + "-06-01")
    end = pd.to_datetime((df.season + 1).astype(str) + "-08-31")
    bad = (df.date < start) | (df.date > end)
    assert not bad.any(), f"{bad.sum()} matches dated outside their season (date parsing?)"
    per_season = df.groupby(["league", "season"]).size()
    small = per_season[per_season < 250]
    if not small.empty:
        print("  note: seasons with < 250 matches (in progress, Covid-shortened, or partial):")
        print("   ", small.to_dict())
    odds = df[["odds_home", "odds_draw", "odds_away"]].dropna()
    if len(odds):
        overround = (1 / odds).sum(axis=1)
        assert overround.between(0.95, 1.30).mean() > 0.99, "odds look corrupted"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=FETCHERS, default="football-data")
    p.add_argument("--last-season", type=int, default=2026,
                   help="starting year of the last season to fetch (current season included)")
    p.add_argument("--leagues", nargs="+", default=list(LEAGUES))
    args = p.parse_args()

    print(f"Downloading from {args.source} ...")
    df = load(args.source, args.last_season, args.leagues)
    validate(df)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS matches")
    con.execute(SCHEMA)
    cols = ["match_id", "league", "season", "date", "home_team", "away_team", "home_goals",
            "away_goals", "result", "odds_home", "odds_draw", "odds_away",
            "home_shots", "away_shots", "home_sot", "away_sot", "source"]
    con.register("df", df[cols])
    con.execute(f"INSERT INTO matches SELECT {', '.join(cols)} FROM df")
    summary = con.execute("""
        SELECT league, COUNT(*) AS matches, MIN(date) AS first, MAX(date) AS last,
               ROUND(AVG((result = 'H')::INT) * 100, 1) AS home_win_pct,
               ROUND(AVG((odds_home IS NOT NULL)::INT) * 100, 1) AS with_odds_pct,
               ROUND(AVG((home_sot IS NOT NULL)::INT) * 100, 1) AS with_shots_pct
        FROM matches GROUP BY league ORDER BY league""").df()
    con.close()
    print(f"\nSaved {len(df)} matches to {DB_PATH}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
