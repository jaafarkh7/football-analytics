"""Run with:  PYTHONPATH=src python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

import fa.telegram as tg
from fa.cards_season import biggest_movers, deserved, deserved_card, luckiest_win, table_card
from fa.season_sim import remaining_fixtures, sample_scores, simulate, table
from test_week2 import _league
from test_week2b import _with_shots


def _played(rows):
    return pd.DataFrame(rows, columns=["home_team", "away_team", "home_goals", "away_goals"])


def test_remaining_fixtures_are_the_unplayed_pairs():
    played = _played([("A", "B", 1, 0), ("B", "C", 2, 2)])
    fx = remaining_fixtures(played, ["A", "B", "C"], "L", pd.Timestamp("2026-01-01"))
    assert set(zip(fx.home_team, fx.away_team)) == {("A", "C"), ("B", "A"), ("C", "A"), ("C", "B")}


def test_table_points_and_goals():
    t = table(_played([("A", "B", 2, 0), ("B", "A", 1, 1)]), ["A", "B"])
    assert t.loc["A", "points"] == 4 and t.loc["B", "points"] == 1
    assert (t.loc["A", "gf"], t.loc["A", "ga"]) == (3, 1) and t.loc["A", "played"] == 2


def test_sampled_scores_match_sampled_results_and_probabilities():
    rng = np.random.default_rng(0)
    probs = np.array([[0.6, 0.25, 0.15]])
    hg, ag = sample_scores(probs, np.array([1.8]), np.array([0.9]), 20000, rng)
    res = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))[0]
    freq = np.bincount(res, minlength=3) / res.size
    np.testing.assert_allclose(freq, probs[0], atol=0.015)


@pytest.fixture(scope="module")
def league():
    df = _with_shots(_league(n_seasons=3))
    df["result"] = np.where(df.home_goals > df.away_goals, "H",
                            np.where(df.home_goals < df.away_goals, "A", "D"))
    return df


def test_simulation_is_consistent(league):
    as_of = league[league.season == 2022].date.sort_values().iloc[45]
    sim = simulate(league, "L", 2022, as_of, n=2000)
    n = len(sim)
    assert sim.p_title.sum() == pytest.approx(1, abs=1e-9)
    assert sim.p_top4.sum() == pytest.approx(min(4, n), abs=1e-9)
    assert sim.p_relegation.sum() == pytest.approx(3, abs=1e-9)
    assert (sim.exp_points >= sim.points).all()                  # points can only go up
    assert sim.set_index("team").loc["Strong", "p_title"] > sim.set_index("team").loc["Weak", "p_title"]


def test_simulation_at_season_end_is_certain(league):
    sim = simulate(league, "L", 2022, pd.Timestamp("2030-01-01"), n=100)
    assert set(sim.p_title.round(6)) <= {0.0, 1.0}
    assert (sim.exp_points == sim.points).all()


def test_simulation_ignores_future_results(league):
    as_of = league[league.season == 2022].date.sort_values().iloc[45]
    a = simulate(league, "L", 2022, as_of, n=500)
    changed = league.copy()
    changed.loc[changed.date >= as_of, ["home_goals", "away_goals"]] = [9, 0]
    b = simulate(changed, "L", 2022, as_of, n=500)
    pd.testing.assert_frame_equal(a, b)


def test_biggest_movers():
    now = pd.DataFrame({"team": ["A", "B", "C"], "p_title": [0.6, 0.3, 0.1],
                        "p_top4": [1, 0.9, 0.5], "p_relegation": [0, 0, 0.2]})
    prev = now.assign(p_title=[0.4, 0.5, 0.1], p_relegation=[0, 0, 0.18]).set_index("team")
    m = biggest_movers(now, prev)
    assert {(t, e) for t, e, _ in m} == {("A", "title"), ("B", "title")}   # C moved < 5 points


def test_deserved_and_luckiest(league):
    recent = deserved(league, league.date.max() - pd.Timedelta(days=10), league.date.max() + pd.Timedelta(days=1))
    assert len(recent) > 0
    assert np.allclose(recent[["d_home", "d_draw", "d_away"]].sum(axis=1), 1)
    rows = pd.DataFrame({"result": ["D", "H", "A"], "p_actual": [0.1, 0.3, 0.6]})
    assert luckiest_win(rows).result == "H"                         # draws never count as lucky wins
    assert luckiest_win(rows[rows.result == "D"]) is None


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_new_cards_render(lang, league):
    as_of = league[league.season == 2022].date.sort_values().iloc[45]
    sim = simulate(league, "L", 2022, as_of, n=300)
    img, _ = table_card(sim, "EPL", lang, sim.set_index("team"))
    assert img.size[0] == 1080
    recent = deserved(league, league.date.max() - pd.Timedelta(days=10), league.date.max() + pd.Timedelta(days=1))
    img, _ = deserved_card(recent.head(5), "EPL", lang)
    assert img.size[0] == 1080


def test_telegram_plans_new_card_kinds(tmp_path):
    for name in ["table_EPL_en", "deserved_SERIEA_ar"]:
        (tmp_path / f"{name}.png").write_bytes(b"x")
    kinds = {(p["kind"], p["lang"]) for p in tg.plan(tmp_path)}
    assert kinds == {("table", "en"), ("deserved", "ar")}
