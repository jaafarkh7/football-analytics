"""Run with:  PYTHONPATH=src python -m pytest -q"""
import numpy as np
import pandas as pd

from fa.models.elo import EloParams, ratings_for_fixtures, run_elo
from fa.predict import predict, read_fixtures
from test_week2 import _league
from test_week2b import _with_shots


def _fixtures(rows):
    return pd.DataFrame(rows, columns=["league", "date", "home_team", "away_team"]).assign(
        date=lambda d: pd.to_datetime(d.date))


def test_read_fixtures_filters_and_converts_odds(tmp_path):
    csv = tmp_path / "fixtures.csv"
    csv.write_text("﻿Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\n"
                   "E0,01/11/2026,15:00,Arsenal,Chelsea,2.0,3.5,4.0\n"
                   "SC0,01/11/2026,15:00,Celtic,Rangers,1.8,3.6,4.5\n")   # not our league
    fx = read_fixtures(str(csv))
    assert list(fx.league) == ["EPL"]
    assert fx.date.iloc[0] == pd.Timestamp("2026-11-01")
    implied = fx[["market_home", "market_draw", "market_away"]].iloc[0]
    assert abs(implied.sum() - 1) < 1e-9                       # bookmaker margin removed
    assert fx.market_home.iloc[0] > fx.market_away.iloc[0]


def test_fixtures_do_not_affect_each_other():
    played = _league()
    rated = run_elo(played, EloParams())
    one = _fixtures([("L", "2023-08-01", "Strong", "Weak")])
    two = _fixtures([("L", "2023-08-01", "Strong", "Weak"), ("L", "2023-08-02", "Weak", "Strong")])
    a = ratings_for_fixtures(rated, one).elo_diff.iloc[0]
    b = ratings_for_fixtures(rated, two).elo_diff.iloc[0]
    assert a == b


def test_promoted_team_starts_below_average():
    rated = run_elo(_league(), EloParams(new_team_offset=100))
    fx = ratings_for_fixtures(rated, _fixtures([("L", "2023-08-01", "Newcomer", "T1")]),
                              EloParams(new_team_offset=100))
    assert fx.elo_home.iloc[0] < fx.elo_away.iloc[0]


def test_predict_ignores_results_after_as_of():
    df = _with_shots(_league())
    df["result"] = np.where(df.home_goals > df.away_goals, "H",
                            np.where(df.home_goals < df.away_goals, "A", "D"))
    as_of = pd.Timestamp("2022-06-01")
    fx = _fixtures([("L", "2022-06-02", "Strong", "Weak"), ("L", "2022-06-03", "T1", "T2")])
    base = predict(df, fx, as_of)
    future = df.copy()
    late = future.date >= as_of
    future.loc[late, ["home_goals", "away_goals"]] = [0, 9]     # nonsense future results
    after = predict(future, fx, as_of)
    pd.testing.assert_frame_equal(base, after)
    np.testing.assert_allclose(base[["p_home", "p_draw", "p_away"]].sum(axis=1), 1)
    assert base.p_home.iloc[0] > 0.6                             # Strong at home to Weak


def test_predicted_score_agrees_with_predicted_result():
    df = _with_shots(_league())
    df["result"] = np.where(df.home_goals > df.away_goals, "H",
                            np.where(df.home_goals < df.away_goals, "A", "D"))
    teams = sorted(set(df.home_team))
    fx = _fixtures([("L", "2023-08-01", h, a) for h in teams for a in teams if h != a])
    out = predict(df, fx, pd.Timestamp("2023-07-01"))
    for r in out.itertuples():
        h, a = map(int, r.top_score.split("-"))
        best = ["H", "D", "A"][int(np.argmax([r.p_home, r.p_draw, r.p_away]))]
        assert (h > a, h == a, h < a)[["H", "D", "A"].index(best)], (r.home_team, r.top_score)
    assert out.p_over25.between(0, 1).all() and out.p_btts.between(0, 1).all()
