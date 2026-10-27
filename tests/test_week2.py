"""Run with:  PYTHONPATH=src python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from fa.models.poisson import DCParams, DixonColes, tau, walk_forward


def _league(n_seasons=3, seed=0):
    """Synthetic league: 'Strong' scores a lot, 'Weak' concedes a lot."""
    rng = np.random.default_rng(seed)
    teams = ["Strong", "Weak"] + [f"T{i}" for i in range(8)]
    attack = {t: 0.0 for t in teams} | {"Strong": 0.6, "Weak": -0.4}
    weak = {t: 0.0 for t in teams} | {"Strong": -0.4, "Weak": 0.5}
    rows, day = [], pd.Timestamp("2020-08-01")
    for s in range(n_seasons):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lh = np.exp(0.2 + 0.15 + attack[h] + weak[a])
                la = np.exp(0.2 + attack[a] + weak[h])
                rows.append({"match_id": f"{s}_{h}_{a}", "league": "L", "season": 2020 + s,
                             "date": day, "home_team": h, "away_team": a,
                             "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la)})
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def league():
    return _league()


def test_probabilities_sum_to_one(league):
    m = DixonColes().fit(league, league.date.max() + pd.Timedelta(days=1))
    p = m.predict(league.head(20))
    np.testing.assert_allclose(p[["p_home", "p_draw", "p_away"]].sum(axis=1), 1, atol=1e-9)


def test_recovers_strong_and_weak_teams(league):
    m = DixonColes(DCParams(alpha=1e-4)).fit(league, league.date.max() + pd.Timedelta(days=1))
    strength = dict(zip(m.teams, m.attack - m.weakness))
    assert max(strength, key=strength.get) == "Strong"
    assert min(strength, key=strength.get) == "Weak"
    assert 0.05 < m.home_adv < 0.3                       # true value 0.15


def test_rho_zero_is_plain_poisson():
    x, y = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    assert np.all(tau(x, y, np.ones_like(x, float), np.ones_like(x, float), 0.0) == 1)


def test_negative_rho_adds_draws():
    m = DixonColes()
    m.rho = 0.0
    plain = m.score_matrix(1.2, 1.0)
    m.rho = -0.1
    adj = m.score_matrix(1.2, 1.0)
    assert adj[0, 0] > plain[0, 0] and adj[1, 1] > plain[1, 1]


def test_unseen_team_is_treated_as_weak(league):
    m = DixonColes().fit(league, league.date.max() + pd.Timedelta(days=1))
    fixture = pd.DataFrame({"home_team": ["Newcomer", "T1"], "away_team": ["T1", "Newcomer"]})
    p = m.predict(fixture)
    assert p.p_home.iloc[0] < p.p_away.iloc[0] + 0.25    # newcomer not favoured much at home
    assert p.p_home.iloc[1] > p.p_away.iloc[1]          # and loses away


def test_walk_forward_has_no_lookahead(league):
    """Changing a result can only affect predictions of *later* matches."""
    params = DCParams(window_days=2000)
    base = walk_forward(league, 2022, params)
    target = league[league.season == 2022].sort_values("date")
    changed_row = target.index[60]
    when = league.loc[changed_row, "date"]
    edited = league.copy()
    edited.loc[changed_row, ["home_goals", "away_goals"]] = [9, 0]
    after = walk_forward(edited, 2022, params)
    upto = target[target.date <= when].match_id
    pd.testing.assert_frame_equal(base.loc[upto], after.loc[upto])
    later = target[target.date > when + pd.Timedelta(days=7)].match_id
    assert not np.allclose(base.loc[later, "p_home"], after.loc[later, "p_home"])
