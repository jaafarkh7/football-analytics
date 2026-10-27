"""Run with:  PYTHONPATH=src python -m pytest -q"""
import numpy as np
import pandas as pd

import fa.data.fixtures_api as fa_api
from fa.data.load_european import canonical_externals, is_neutral
from fa.data.team_names import same_club
from fa.models.cl_predict import link_external_names, predict_cl
from fa.models.cross_elo import CrossEloParams, run_cross_elo


# ---------- names ----------

def test_same_club_rules():
    assert same_club("SL Benfica", "Sport Lisboa e Benfica")
    assert same_club("PSV", "PSV Eindhoven")
    assert same_club("Feyenoord", "Feyenoord Rotterdam")
    assert not same_club("AC Sparta Praha", "SK Slavia Praha")
    assert not same_club("Dinamo Kiev", "Dinamo Zagreb")


def test_canonical_externals_merge_within_country_only():
    df = pd.DataFrame({
        "home_team": ["SL Benfica (POR)", "Sport Lisboa e Benfica (POR)", "AC Sparta Praha (CZE)",
                      "Slavia Praha (CZE)", "Dinamo Kiev (UKR)"],
        "away_team": ["X"] * 5, "home_league": [None] * 5, "away_league": ["EPL"] * 5,
        "date": pd.to_datetime(["2015-09-01", "2024-09-01", "2024-09-01", "2019-09-01", "2016-09-01"]),
    })
    canon = canonical_externals(df)
    assert canon["SL Benfica (POR)"] == "Sport Lisboa e Benfica (POR)"   # newest spelling kept
    assert canon["Slavia Praha (CZE)"] == "Slavia Praha (CZE)"
    assert canon["AC Sparta Praha (CZE)"] == "AC Sparta Praha (CZE)"


def test_linking_refuses_ambiguous_names():
    hist = {"FC Porto (POR)", "Dinamo Zagreb (CRO)", "Dinamo Kiev (UKR)"}
    link = link_external_names({"FC Porto", "Dinamo"}, hist)
    assert link == {"FC Porto": "FC Porto (POR)"}                       # "Dinamo" is ambiguous


def test_neutral_venues():
    assert is_neutral(2018, "Final") and is_neutral(2024, "Finals, Final")
    assert not is_neutral(2018, "Semifinals")
    assert is_neutral(2019, "Quarterfinals")                            # Lisbon 2020


# ---------- league strength ----------

def _two_leagues(n_cl=30, seed=0):
    """Two leagues of 4 equal clubs; in Europe, league A always beats league B."""
    rng = np.random.default_rng(seed)
    dom, d0 = [], pd.Timestamp("2018-08-01")
    for lg, teams in (("A", ["A1", "A2", "A3", "A4"]), ("B", ["B1", "B2", "B3", "B4"])):
        for i in range(240):
            h, a = rng.choice(teams, 2, replace=False)
            g = rng.integers(0, 3, 2)
            dom.append({"match_id": f"{lg}{i}", "date": d0 + pd.Timedelta(days=i * 3), "season": 2020,
                        "league": lg, "home_team": h, "away_team": a,
                        "home_goals": g[0], "away_goals": g[1]})
    cl = []
    for i in range(n_cl):
        a, b = f"A{i % 4 + 1}", f"B{(i + 1) % 4 + 1}"
        home_a = i % 2 == 0
        if i % 5 == 4:                                                  # some draws, like real data
            cl.append({"match_id": f"cl{i}", "date": d0 + pd.Timedelta(days=i * 6 + 1),
                       "season": 2020, "neutral": False, "home_team": a, "away_team": b,
                       "home_league": "A", "away_league": "B", "home_goals": 1,
                       "away_goals": 1, "result": "D"})
            continue
        cl.append({"match_id": f"cl{i}", "date": d0 + pd.Timedelta(days=i * 6 + 1), "season": 2020,
                   "neutral": False, "home_team": a if home_a else b, "away_team": b if home_a else a,
                   "home_league": "A" if home_a else "B", "away_league": "B" if home_a else "A",
                   "home_goals": 2 if home_a else 0, "away_goals": 0 if home_a else 2,
                   "result": "H" if home_a else "A"})
    return pd.DataFrame(dom), pd.DataFrame(cl)


def test_cross_league_results_move_league_averages():
    dom, cl = _two_leagues()
    strong = run_cross_elo(dom, cl, CrossEloParams()).attrs["league_means"]
    equal = run_cross_elo(dom, cl, CrossEloParams(league_strength=False)).attrs["league_means"]
    assert strong["A"] - strong["B"] > 50
    assert abs(equal["A"] - equal["B"]) < 1e-9


def test_predict_cl_favours_stronger_league_and_ignores_future():
    dom, cl = _two_leagues()
    dom["result"] = np.where(dom.home_goals > dom.away_goals, "H",
                             np.where(dom.home_goals < dom.away_goals, "A", "D"))
    hist = cl[cl.match_id.isin([f"cl{i}" for i in range(20)])]
    early = hist.assign(season=2015, match_id=hist.match_id + "_15",       # outcome map needs 2015+
                        date=hist.date - pd.Timedelta(days=400))
    hist = pd.concat([early, hist])
    fixture = pd.DataFrame({"league": ["UCL"], "date": [pd.Timestamp("2021-06-01")], "time": ["21:00"],
                            "home_team": ["B1"], "away_team": ["A1"], "home_league": ["B"],
                            "away_league": ["A"], "stage": ["LEAGUE_STAGE"], "match_id": ["f1"]})
    as_of = pd.Timestamp("2021-05-01")
    p = predict_cl(dom, hist, pd.DataFrame(), fixture, as_of)
    assert p.p_away.iloc[0] > p.p_home.iloc[0]                           # A club favoured, even away
    later = pd.concat([hist, cl.assign(date=pd.Timestamp("2021-05-20"), match_id="late",
                                       home_goals=0, away_goals=9, result="A").head(1)])
    p2 = predict_cl(dom, later, pd.DataFrame(), fixture, as_of)
    np.testing.assert_allclose(p[["p_home", "p_draw", "p_away"]], p2[["p_home", "p_draw", "p_away"]])


# ---------- API results ----------

def test_extra_time_uses_90_minute_score_and_final_is_neutral(monkeypatch):
    class Fake:
        def get(self, path, **params):
            if path == "/competitions/CL":
                return {"currentSeason": {"startDate": "2026-09-01"}}
            if params.get("season"):
                raise SystemExit("not in plan")
            team = lambda i, n: {"id": i, "name": n, "shortName": n}
            return {"matches": [
                {"utcDate": "2026-10-21T19:00:00Z", "status": "FINISHED", "stage": "LEAGUE_STAGE",
                 "homeTeam": team(1, "Club One"), "awayTeam": team(2, "Club Two"),
                 "score": {"duration": "REGULAR", "fullTime": {"home": 2, "away": 1}}},
                {"utcDate": "2027-05-29T19:00:00Z", "status": "FINISHED", "stage": "FINAL",
                 "homeTeam": team(1, "Club One"), "awayTeam": team(2, "Club Two"),
                 "score": {"duration": "EXTRA_TIME", "fullTime": {"home": 2, "away": 1},
                           "regularTime": {"home": 1, "away": 1}}},
            ]}
    monkeypatch.setattr(fa_api, "_id_map", lambda client, refresh_teams=False: {})
    df = fa_api.fetch_cl_results(client=Fake()).set_index("stage")
    assert (df.loc["LEAGUE_STAGE", "home_goals"], df.loc["LEAGUE_STAGE", "result"]) == (2, "H")
    assert (df.loc["FINAL", "home_goals"], df.loc["FINAL", "result"]) == (1, "D")
    assert df.loc["FINAL", "neutral"] and not df.loc["LEAGUE_STAGE", "neutral"]
