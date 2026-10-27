"""Run with:  PYTHONPATH=src python -m pytest -q"""
import pandas as pd
import pytest

from fa.features.form import match_features, team_form


def _m(i, date, season, h, a, hg, ag):
    return {"match_id": f"m{i}", "league": "L", "season": season, "date": pd.Timestamp(date),
            "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag,
            "home_sot": hg + 2, "away_sot": ag + 2}


@pytest.fixture
def games():
    return pd.DataFrame([
        _m(1, "2025-08-10", 2025, "A", "B", 3, 0),    # A wins
        _m(2, "2025-08-17", 2025, "B", "A", 1, 1),    # draw
        _m(3, "2025-08-24", 2025, "A", "B", 0, 2),    # B wins
        _m(4, "2026-08-15", 2026, "A", "B", 1, 0),    # new season
    ])


def test_first_match_has_neutral_form(games):
    f = match_features(games)
    assert f.loc["m1"].abs().drop("season_n_min").max() == 0


def test_form_uses_only_previous_matches(games):
    tv = team_form(games).set_index(["match_id", "team"])
    # Before m3, A has: W (3 pts, +3), D (1 pt, 0)  -> 2.0 pts/game, +1.5 gd/game
    assert tv.loc[("m3", "A"), "form5_pts"] == pytest.approx(2.0)
    assert tv.loc[("m3", "A"), "form5_gd"] == pytest.approx(1.5)
    # m3's own result (A lost 0-2) must not be in it
    assert tv.loc[("m3", "B"), "form5_pts"] == pytest.approx((0 + 1) / 2)


def test_season_stats_reset_but_form_carries_over(games):
    tv = team_form(games).set_index(["match_id", "team"])
    assert tv.loc[("m4", "A"), "season_n"] == 0
    assert tv.loc[("m4", "A"), "season_pts"] == pytest.approx(1.37)        # neutral: no games yet
    assert tv.loc[("m4", "A"), "form5_pts"] == pytest.approx((3 + 1 + 0) / 3)


def test_changing_a_result_only_affects_later_matches(games):
    before = match_features(games)
    edited = games.copy()
    edited.loc[edited.match_id == "m2", ["home_goals", "away_goals"]] = [5, 0]
    after = match_features(edited)
    pd.testing.assert_series_equal(before.loc["m2"], after.loc["m2"])
    assert not before.loc["m3"].equals(after.loc["m3"])


def test_rest_days(games):
    tv = team_form(games).set_index(["match_id", "team"])
    assert tv.loc[("m2", "A"), "rest_days"] == 7
    assert tv.loc[("m4", "A"), "rest_days"] == 14                          # capped
