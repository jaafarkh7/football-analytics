"""Run with:  PYTHONPATH=src python -m pytest -q"""
import math

import numpy as np
import pandas as pd
import pytest

from fa.eval.metrics import accuracy, brier, log_loss, rps
from fa.models.baselines import bookmaker
from fa.models.elo import EloParams, run_elo


# ---------- metrics ----------

def test_uniform_log_loss_is_ln3():
    p = np.full((4, 3), 1 / 3)
    assert log_loss(p, ["H", "D", "A", "H"]) == pytest.approx(math.log(3))


def test_perfect_forecast_scores_zero():
    p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    y = ["H", "D", "A"]
    assert brier(p, y) == pytest.approx(0)
    assert rps(p, y) == pytest.approx(0)
    assert accuracy(p, y) == 1


def test_rps_rewards_near_misses():
    """Home win happened: predicting a draw is less wrong than predicting an away win."""
    said_draw = np.array([[0, 1, 0.0]])
    said_away = np.array([[0, 0, 1.0]])
    assert rps(said_draw, ["H"]) < rps(said_away, ["H"])
    assert brier(said_draw, ["H"]) == brier(said_away, ["H"])  # Brier can't tell them apart


def test_bookmaker_removes_margin():
    df = pd.DataFrame({"odds_home": [2.0], "odds_draw": [3.4], "odds_away": [3.8]})
    p = bookmaker(df)
    assert p.sum() == pytest.approx(1)
    assert p[0, 0] > p[0, 2]


# ---------- Elo ----------

def _toy_matches(n=60, seed=1):
    rng = np.random.default_rng(seed)
    teams = list("ABCDEF")
    rows = []
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        hg, ag = rng.integers(0, 4, 2)
        rows.append({"match_id": f"m{i:03d}", "league": "L", "season": 2020 + i // 30,
                     "date": pd.Timestamp("2020-08-01") + pd.Timedelta(days=i),
                     "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag})
    return pd.DataFrame(rows)


def test_elo_has_no_lookahead():
    """Changing match i's result must not change any prediction up to and including match i."""
    df = _toy_matches()
    base = run_elo(df, EloParams(season_regress=0))
    i = 40
    df2 = df.copy()
    df2.loc[i, ["home_goals", "away_goals"]] = [9, 0]
    changed = run_elo(df2, EloParams(season_regress=0))
    np.testing.assert_allclose(base.elo_diff[: i + 1], changed.elo_diff[: i + 1])
    assert not np.allclose(base.elo_diff[i + 1:], changed.elo_diff[i + 1:])


def test_first_season_teams_start_equal_not_as_promoted():
    """Teams first seen in the first season must not get the promoted-team penalty."""
    df = _toy_matches(n=30)
    rated = run_elo(df, EloParams(new_team_offset=200))
    first = rated.iloc[0]
    assert first.elo_home == first.elo_away == 1500


def test_winners_go_up():
    df = pd.DataFrame([{"match_id": f"m{i}", "league": "L", "season": 2020,
                        "date": pd.Timestamp("2020-08-01") + pd.Timedelta(days=i),
                        "home_team": "Strong", "away_team": "Weak", "home_goals": 3,
                        "away_goals": 0} for i in range(10)])
    rated = run_elo(df)
    assert rated.elo_home.is_monotonic_increasing
    assert rated.elo_away.is_monotonic_decreasing
