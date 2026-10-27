"""Run with:  PYTHONPATH=src python -m pytest -q

The real API isn't called: FakeClient returns payloads in football-data.org's v4 format.
"""
import duckdb
import pandas as pd
import pytest

import fa.data.fixtures_api as fa_api
from fa.data.team_names import assign, similarity

PL_TEAMS = [
    {"id": 57, "name": "Arsenal FC", "shortName": "Arsenal", "tla": "ARS"},
    {"id": 65, "name": "Manchester City FC", "shortName": "Man City", "tla": "MCI"},
    {"id": 66, "name": "Manchester United FC", "shortName": "Man United", "tla": "MUN"},
    {"id": 351, "name": "Nottingham Forest FC", "shortName": "Nottingham", "tla": "NOT"},
    {"id": 76, "name": "Wolverhampton Wanderers FC", "shortName": "Wolverhampton", "tla": "WOL"},
    {"id": 397, "name": "Brighton & Hove Albion FC", "shortName": "Brighton Hove", "tla": "BHA"},
]
OUR_PL = ["Arsenal", "Brighton", "Man City", "Man United", "Nott'm Forest", "Wolves"]


def _match(mid, utc, home, away, status="TIMED", stage="REGULAR_SEASON"):
    return {"id": mid, "utcDate": utc, "status": status, "stage": stage, "matchday": 10,
            "homeTeam": home, "awayTeam": away}


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, path, **params):
        self.calls.append(path)
        if path.endswith("/PL/teams"):
            return {"teams": PL_TEAMS}
        if path.endswith("/teams"):
            return {"teams": []}
        if path.endswith("/PL/matches"):
            return {"matches": [
                _match(1, "2026-10-31T15:00:00Z", PL_TEAMS[1], PL_TEAMS[2]),
                _match(2, "2026-10-31T23:30:00Z", PL_TEAMS[0], PL_TEAMS[3]),  # after midnight in Paris
                _match(3, "2026-10-25T15:00:00Z", PL_TEAMS[4], PL_TEAMS[5], status="FINISHED"),
            ]}
        if path.endswith("/CL/matches"):
            benfica = {"id": 1903, "name": "Sport Lisboa e Benfica", "shortName": "Benfica"}
            return {"matches": [_match(9, "2026-10-28T20:00:00Z", PL_TEAMS[0], benfica,
                                       stage="LEAGUE_STAGE"),
                                _match(10, "2027-02-17T20:00:00Z", {"id": None}, {"id": None},
                                       status="SCHEDULED", stage="LAST_16")]}
        return {"matches": []}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    db = tmp_path / "f.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE matches (league VARCHAR, season INT, home_team VARCHAR, away_team VARCHAR)")
    for i, t in enumerate(OUR_PL):
        con.execute("INSERT INTO matches VALUES ('EPL', 2026, ?, ?)", [t, OUR_PL[(i + 1) % 6]])
    con.close()
    monkeypatch.setattr(fa_api, "DB_PATH", db)
    monkeypatch.setattr(fa_api, "TEAM_MAP_PATH", tmp_path / "team_map.csv")
    monkeypatch.setattr(fa_api, "OLD_TEAM_MAP_PATH", tmp_path / "old_team_map.csv")
    monkeypatch.setattr(fa_api, "assign", lambda t, o, lg: assign(t, o, lg, aliases={}))


def test_assignment_resolves_look_alikes():
    res = {r["their_name"]: r["our_name"] for r in assign(PL_TEAMS, OUR_PL, "EPL", aliases={})}
    assert res["Manchester City FC"] == "Man City"
    assert res["Manchester United FC"] == "Man United"
    assert res["Nottingham Forest FC"] == "Nott'm Forest"
    assert res["Wolverhampton Wanderers FC"] == "Wolves"
    assert res["Brighton & Hove Albion FC"] == "Brighton"


def test_alias_overrides_automatic_match():
    res = assign(PL_TEAMS, OUR_PL, "EPL", aliases={("EPL", "Arsenal FC"): "Man City"})
    by_name = {r["their_name"]: r for r in res}
    assert by_name["Arsenal FC"]["our_name"] == "Man City"
    assert by_name["Arsenal FC"]["status"] == "alias"
    assert by_name["Manchester City FC"]["our_name"] != "Man City"   # each team used once


def test_accents_and_prefixes_ignored():
    assert similarity("FC Bayern München", "Bayern Munich") > 0.6
    assert similarity("1. FC Köln", "FC Koln") > 0.9


def test_fetch_fixtures_end_to_end(isolated):
    client = FakeClient()
    fx = fa_api.fetch_fixtures(pd.Timestamp("2026-10-27"), pd.Timestamp("2026-11-02"),
                               client=client)
    league = fx[fx.league == "EPL"].set_index("home_team")
    assert set(league.index) == {"Man City", "Arsenal"}            # FINISHED match dropped
    assert league.loc["Man City", "away_team"] == "Man United"
    # 23:30 UTC on the 31st is 00:30 on 1 Nov in Paris: dated by local kickoff
    assert league.loc["Arsenal", "date"] == pd.Timestamp("2026-11-01")
    assert league.loc["Arsenal", "match_id"] == "EPL_20261101_Arsenal_Nott'mForest"

    cl = fx[fx.league == "UCL"]
    assert len(cl) == 1                                            # undecided knockout tie dropped
    assert cl.home_team.iloc[0] == "Arsenal"                       # mapped via team id
    assert cl.home_league.iloc[0] == "EPL"
    assert cl.away_team.iloc[0] == "Benfica" and pd.isna(cl.away_league.iloc[0])


def test_team_map_is_cached(isolated):
    fa_api.fetch_fixtures(pd.Timestamp("2026-10-27"), pd.Timestamp("2026-11-02"), client=FakeClient())
    second = FakeClient()
    fa_api.fetch_fixtures(pd.Timestamp("2026-10-27"), pd.Timestamp("2026-11-02"), client=second)
    assert not any(c.endswith("/teams") for c in second.calls)
