"""Run with:  PYTHONPATH=src python -m pytest -q"""
import json

import numpy as np
import pandas as pd

import fa.learn as learn

DEFAULT = dict(learn.DEFAULT)


def test_default_settings_are_the_old_fixed_blend():
    p_dc, p_elo = np.array([[0.5, 0.3, 0.2]]), np.array([[0.4, 0.3, 0.3]])
    expected = DEFAULT["blend_weight_dc"] * p_dc + (1 - DEFAULT["blend_weight_dc"]) * p_elo
    assert np.allclose(learn.combine(p_dc, p_elo, DEFAULT), expected)


def test_adjustments_do_what_they_say():
    p = np.array([[0.5, 0.3, 0.2]])
    sharper = learn.combine(p, p, {**DEFAULT, "temperature": 1.2})
    assert sharper[0, 0] > 0.5 and np.isclose(sharper.sum(), 1)
    drawier = learn.combine(p, p, {**DEFAULT, "draw_boost": 0.1})
    assert drawier[0, 1] > 0.3 and np.isclose(drawier.sum(), 1)


def _season(n, draw_rate, rng):
    """Matches where the models under-predict draws when draw_rate is high."""
    dates = pd.date_range("2025-08-01", periods=n, freq="8h")
    rest = 1 - draw_rate
    res = rng.choice(list("HDA"), size=n, p=[0.6 * rest, draw_rate, 0.4 * rest])
    p = np.tile([0.46, 0.24, 0.30], (n, 1))
    return pd.DataFrame({"date": dates, "result": res,
                         "dc_home_p": p[:, 0], "dc_draw_p": p[:, 1], "dc_away_p": p[:, 2],
                         "elo_home_p": p[:, 0], "elo_draw_p": p[:, 1], "elo_away_p": p[:, 2]})


def test_learns_a_real_bias_and_ignores_a_fluke():
    rng = np.random.default_rng(0)
    real = _season(4000, 0.34, rng)                       # draws underrated all year long
    d = learn.decide(real, DEFAULT, real.date.iloc[3000])
    assert d["adopted"] and d["candidate"]["draw_boost"] > 0
    fluke = pd.concat([_season(3000, 0.34, rng), _season(1000, 0.18, rng)])   # bias gone lately
    fluke["date"] = pd.date_range("2025-08-01", periods=len(fluke), freq="8h")
    d = learn.decide(fluke, DEFAULT, fluke.date.iloc[3000])
    assert not d["adopted"]                               # the recent weeks say no


def test_decisions_are_saved_and_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "SETTINGS_PATH", tmp_path / "learned.json")
    monkeypatch.setattr(learn, "LOG_PATH", tmp_path / "log.csv")
    assert learn.load_settings() == DEFAULT
    new = {**DEFAULT, "draw_boost": 0.05}
    d = {"current": DEFAULT, "candidate": new, "adopted": True, "tune_matches": 1, "check_matches": 1,
         "tune_current": 1.0, "tune_candidate": 0.99, "check_current": 1.0, "check_candidate": 0.99}
    learn.record(d, pd.Timestamp("2026-11-09"))
    assert learn.load_settings() == new
    assert json.loads((tmp_path / "learned.json").read_text())["since"] == "2026-11-09"
    learn.record({**d, "current": new, "candidate": DEFAULT, "adopted": False}, pd.Timestamp("2026-11-16"))
    assert learn.load_settings() == new                   # kept, and "since" unchanged
    assert json.loads((tmp_path / "learned.json").read_text())["since"] == "2026-11-09"
    text = learn.report(pd.Timestamp("2026-11-16"))
    assert "| 2026-11-09 | adopted |" in text and "| 2026-11-16 | kept current |" in text
    assert len(pd.read_csv(tmp_path / "log.csv")) == 2
