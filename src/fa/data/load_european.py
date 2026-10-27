"""Champions League results, so clubs from different leagues can be compared.

    python -m fa.data.load_european          (after fa.data.load_results)

Source: openfootball (GitHub), which has 2014-15..2019-20 and 2024-25 and tags every club
with its country, e.g. "Celtic FC (SCO)". A club tagged ENG/ESP/ITA/GER/FRA is matched
to that season's teams in *that* league only (so "Celtic" can never become "Celta");
every other club keeps its own name as an "external" club.
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd

from fa.config import DB_PATH
from fa.data.load_results import _cached
from fa.data.team_names import assign, same_club

CL_SEASONS = [2014, 2015, 2016, 2017, 2018, 2019, 2024]
COUNTRY_LEAGUE = {"ENG": "EPL", "ESP": "LALIGA", "ITA": "SERIEA", "GER": "BUNDESLIGA",
                  "FRA": "LIGUE1", "MCO": "LIGUE1"}          # Monaco plays in Ligue 1

SCHEMA = """
CREATE TABLE cl_matches (
    match_id     VARCHAR PRIMARY KEY,
    season       INTEGER NOT NULL,
    date         DATE    NOT NULL,
    round        VARCHAR,
    neutral      BOOLEAN NOT NULL,      -- finals, and the 2020 Lisbon tournament
    home_team    VARCHAR NOT NULL,      -- our name for top-5 clubs, "Name (CTRY)" otherwise
    away_team    VARCHAR NOT NULL,
    home_league  VARCHAR,               -- NULL for clubs outside our 5 leagues
    away_league  VARCHAR,
    home_goals   INTEGER NOT NULL,      -- after 90 minutes
    away_goals   INTEGER NOT NULL,
    result       VARCHAR NOT NULL,
    source       VARCHAR NOT NULL
);
"""


def split_country(team: str) -> tuple[str, str]:
    name, _, rest = team.rpartition(" (")
    return (name, rest.rstrip(")")) if name else (team, "")


def is_neutral(season: int, rnd: str) -> bool:
    r = rnd.lower()
    if r.endswith("final") and "semi" not in r and "quarter" not in r:
        return True                                   # the final itself
    return season == 2019 and ("quarter" in r or "semi" in r)   # Lisbon, August 2020


def season_matches(season: int) -> list[dict]:
    label = f"{season}-{(season + 1) % 100:02d}"
    url = ("https://raw.githubusercontent.com/openfootball/football.json/master/"
           f"{label}/uefa.cl.json")
    raw = _cached(f"openfootball/{label}_uefa.cl.json", url)
    return [] if raw is None else json.loads(raw)["matches"]


def build_name_map(season: int, clubs: set[str], con) -> dict[str, tuple[str, str | None]]:
    """openfootball club -> (our name, league) for top-5 clubs, or (key, None) otherwise."""
    out, by_league = {}, {}
    for club in clubs:
        name, country = split_country(club)
        league = COUNTRY_LEAGUE.get(country)
        if league:
            by_league.setdefault(league, []).append(club)
        else:
            out[club] = (club, None)
    for league, members in by_league.items():
        ours = [r[0] for r in con.execute(
            "SELECT DISTINCT home_team FROM matches WHERE league = ? AND season = ?",
            [league, season]).fetchall()]
        theirs = [{"id": c, "name": split_country(c)[0]} for c in members]
        for r in assign(theirs, ours, league):
            if r["status"] != "auto" and r["status"] != "alias":
                raise ValueError(f"Unsure how to match {r['id']} in {league} {season}: "
                                 f"best guess {r['our_name']} ({r['score']}). "
                                 "Add a line to src/fa/data/team_aliases.csv.")
            out[r["id"]] = (r["our_name"], league)
    return out


def canonical_externals(df: pd.DataFrame) -> dict[str, str]:
    """The source spells some clubs differently across seasons ("SL Benfica (POR)" vs
    "Sport Lisboa e Benfica (POR)"). Merge spellings of the same club within a country,
    keeping the most recent spelling, so a club keeps one rating history."""
    ext = pd.concat([df.loc[df.home_league.isna(), ["home_team", "date"]].set_axis(["team", "date"], axis=1),
                     df.loc[df.away_league.isna(), ["away_team", "date"]].set_axis(["team", "date"], axis=1)])
    latest = ext.groupby("team").date.max().sort_values(ascending=False)
    canon: dict[str, str] = {}
    keep: list[str] = []
    for team in latest.index:                       # newest spelling first
        name, country = split_country(team)
        match = [k for k in keep if split_country(k)[1] == country
                 and same_club(name, split_country(k)[0])]
        if len(match) == 1:
            canon[team] = match[0]
        else:
            keep.append(team)
            canon[team] = team
    return canon


def load(con) -> pd.DataFrame:
    rows = []
    for season in CL_SEASONS:
        ms = season_matches(season)
        played = []
        for m in ms:
            score = m.get("score")
            ft = score.get("ft") if isinstance(score, dict) else score
            if ft and len(ft) == 2:
                played.append((m, ft))
        names = build_name_map(season, {t for m, _ in played for t in (m["team1"], m["team2"])}, con)
        for m, ft in played:
            (h, hl), (a, al) = names[m["team1"]], names[m["team2"]]
            hg, ag = int(ft[0]), int(ft[1])
            rows.append({"season": season, "date": pd.Timestamp(m["date"]), "round": m["round"],
                         "neutral": is_neutral(season, m["round"]),
                         "home_team": h, "away_team": a, "home_league": hl, "away_league": al,
                         "home_goals": hg, "away_goals": ag,
                         "result": "H" if hg > ag else ("A" if hg < ag else "D"),
                         "source": "openfootball"})
    df = pd.DataFrame(rows)
    canon = canonical_externals(df)
    merged = {k: v for k, v in canon.items() if k != v}
    if merged:
        print(f"Merged {len(merged)} alternative spellings of clubs outside the top 5, e.g. "
              + "; ".join(f"{k} → {v}" for k, v in list(merged.items())[:3]))
    for side in ("home", "away"):
        ext = df[f"{side}_league"].isna()
        df.loc[ext, f"{side}_team"] = df.loc[ext, f"{side}_team"].map(canon)
    df["match_id"] = ("UCL_" + df.date.dt.strftime("%Y%m%d") + "_" + df.home_team.str.replace(" ", "")
                      + "_" + df.away_team.str.replace(" ", ""))
    return df.drop_duplicates("match_id").sort_values("date").reset_index(drop=True)


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    df = load(con)
    con.execute("DROP TABLE IF EXISTS cl_matches")
    con.execute(SCHEMA)
    cols = ["match_id", "season", "date", "round", "neutral", "home_team", "away_team",
            "home_league", "away_league", "home_goals", "away_goals", "result", "source"]
    con.register("cl_df", df[cols])
    con.execute(f"INSERT INTO cl_matches SELECT {', '.join(cols)} FROM cl_df")
    summary = con.execute("""
        SELECT season, COUNT(*) AS matches,
               SUM((home_league IS NOT NULL AND away_league IS NOT NULL
                    AND home_league <> away_league)::INT) AS top5_vs_top5_other_league,
               SUM((home_league IS NULL OR away_league IS NULL)::INT) AS vs_external,
               SUM(neutral::INT) AS neutral
        FROM cl_matches GROUP BY season ORDER BY season""").df()
    con.close()
    print(f"Saved {len(df)} Champions League matches to {DB_PATH}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
