"""Run with:  PYTHONPATH=src python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from fa.data.load_results import _parse_dates
from fa.models.poisson import DCParams, DixonColes
from test_week2 import _league


def _with_shots(df, seed=0):
    """Add shots: every goal was a shot on target, plus extra saved/missed shots."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    for side in ("home", "away"):
        df[f"{side}_sot"] = df[f"{side}_goals"] + rng.poisson(2.5, len(df))
        df[f"{side}_shots"] = df[f"{side}_sot"] + rng.poisson(6, len(df))
    return df


def test_dates_both_formats():
    """Regression test: dayfirst=True used to turn 2023-08-11 into 2023-11-08."""
    d = _parse_dates(pd.Series(["2023-08-11", "11/08/2023", "11/08/23"]))
    assert (d == pd.Timestamp("2023-08-11")).all()


def test_goal_weight_one_ignores_shots():
    df = _with_shots(_league())
    as_of = df.date.max() + pd.Timedelta(days=1)
    a = DixonColes(DCParams(goal_weight=1.0)).fit(df, as_of).predict(df.head(10))
    b = DixonColes(DCParams(goal_weight=1.0)).fit(
        df.drop(columns=["home_shots", "away_shots", "home_sot", "away_sot"]), as_of
    ).predict(df.head(10))
    pd.testing.assert_frame_equal(a, b)


def test_shots_required_when_used():
    df = _league()
    with pytest.raises(ValueError, match="shots"):
        DixonColes(DCParams(goal_weight=0.5)).fit(df, df.date.max() + pd.Timedelta(days=1))


def test_conversion_rate_is_sensible():
    df = _with_shots(_league())
    m = DixonColes(DCParams(goal_weight=0.5)).fit(df, df.date.max() + pd.Timedelta(days=1))
    assert 0.2 < m.conversion["on_target"] < 0.6
    assert m.conversion["off_target"] < 0.05


def test_missing_shot_rows_fall_back_to_goals():
    df = _with_shots(_league())
    df.loc[df.index[:50], ["home_shots", "home_sot", "away_shots", "away_sot"]] = np.nan
    m = DixonColes(DCParams(goal_weight=0.5)).fit(df, df.date.max() + pd.Timedelta(days=1))
    p = m.predict(df.head(5))
    assert np.isfinite(p[["p_home", "p_draw", "p_away"]].to_numpy()).all()
