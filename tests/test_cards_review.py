"""Run with:  PYTHONPATH=src python -m pytest -q"""
import pandas as pd
import pytest

import fa.telegram as tg
from fa.cards_review import (biggest_change, make_reviews, posted_review_ids, review_card,
                             review_rows)

NOW = pd.Timestamp("2026-11-02 06:30")


def _matches():
    rows = []
    # History so the "deserved" model can learn how often shots on target become goals.
    for i in range(60):
        rows.append(dict(match_id=f"h{i}", league="EPL", season=2026,
                         date=pd.Timestamp("2026-08-20") + pd.Timedelta(days=i), home_team="X",
                         away_team="Y", home_goals=1, away_goals=1, result="D", home_shots=10,
                         away_shots=10, home_sot=4, away_sot=4))
    rows += [
        dict(match_id="r1", league="EPL", season=2026, date=pd.Timestamp("2026-11-01"),
             home_team="Liverpool", away_team="Arsenal", home_goals=0, away_goals=2, result="A",
             home_shots=15, away_shots=6, home_sot=6, away_sot=3),
        dict(match_id="r2", league="EPL", season=2026, date=pd.Timestamp("2026-11-01"),
             home_team="Chelsea", away_team="Man United", home_goals=1, away_goals=1, result="D",
             home_shots=None, away_shots=None, home_sot=None, away_sot=None),   # no shots yet
        dict(match_id="r3", league="EPL", season=2026, date=pd.Timestamp("2026-11-01"),
             home_team="Man City", away_team="Brighton", home_goals=3, away_goals=0, result="H",
             home_shots=20, away_shots=4, home_sot=9, away_sot=1),
    ]
    return pd.DataFrame(rows)


def _forecasts():
    base = dict(model="blend_v1", league="EPL", date=pd.Timestamp("2026-11-01"),
                kickoff_utc=pd.Timestamp("2026-11-01 16:30"), p_home=0.5, p_draw=0.25, p_away=0.25)
    return pd.DataFrame([
        {**base, "match_id": "EPL_20261101_Liverpool_Arsenal", "home_team": "Liverpool",
         "away_team": "Arsenal", "created_at": pd.Timestamp("2026-10-30 06:30")},
        {**base, "match_id": "EPL_20261101_Chelsea_ManUnited", "home_team": "Chelsea",
         "away_team": "Man United", "created_at": pd.Timestamp("2026-10-30 06:30")},
        # Saved after kickoff: never reviewed (it isn't a real forecast).
        {**base, "match_id": "EPL_20261101_ManCity_Brighton", "home_team": "Man City",
         "away_team": "Brighton", "created_at": pd.Timestamp("2026-11-01 18:00")},
    ])


def test_only_pre_kickoff_forecasts_with_shot_data():
    rows = review_rows(_matches(), _forecasts(), NOW)
    assert list(rows.match_id) == ["EPL_20261101_Liverpool_Arsenal"]
    r = rows.iloc[0]
    assert r.home_goals == 0 and r.away_goals == 2 and r.home_sot == 6
    assert abs(r.d_home + r.d_draw + r.d_away - 1) < 1e-6
    assert r.d_home > r.d_away                       # 6 shots on target vs 3


def test_already_posted_matches_are_skipped(tmp_path):
    log = tmp_path / "posted.csv"
    log.write_text("2026-11-02T06:40:00+00:00,@c|2026-11-02/review_EPL_20261101_Liverpool_Arsenal_en.png"
                   "|2026-11-02/review_EPL_20261101_Other_Club_en.png\n")
    ids = posted_review_ids(log)
    assert ids == {"EPL_20261101_Liverpool_Arsenal", "EPL_20261101_Other_Club"}
    assert review_rows(_matches(), _forecasts(), NOW, ids).empty


def test_too_old_matches_are_not_reviewed():
    assert review_rows(_matches(), _forecasts(), NOW + pd.Timedelta(days=5)).empty


def test_biggest_change_picks_the_largest_real_move():
    before = pd.DataFrame({"p_title": [0.01], "p_top4": [0.41], "p_relegation": [0.0]}, index=["Milan"])
    after = pd.DataFrame({"p_title": [0.02], "p_top4": [0.52], "p_relegation": [0.0]}, index=["Milan"])
    assert biggest_change("Milan", before, after) == ("p_top4", 0.41, 0.52)
    assert biggest_change("Milan", before, before) is None          # nothing moved
    assert biggest_change("Unknown", before, after) is None


@pytest.mark.parametrize("lang", ["en", "ar"])
@pytest.mark.parametrize("changes", [None, [("Arsenal", "p_title", 0.30, 0.41),
                                            ("Liverpool", "p_relegation", 0.0, 0.04)]])
def test_review_card_renders(lang, changes):
    r = review_rows(_matches(), _forecasts(), NOW).iloc[0]
    img, missing = review_card(r, lang, (70, 120), "@test", changes)
    assert img.size == (1080, 1180 + (100 + 60 * len(changes) if changes else 0))
    assert not missing


def test_make_reviews_and_telegram_album(tmp_path):
    made, _ = make_reviews(_matches(), _forecasts(), NOW, tmp_path, None,
                           {"en": "", "ar": ""}, tmp_path / "none.csv", with_season=False)
    assert sorted(f.name for f in made) == ["review_EPL_20261101_Liverpool_Arsenal_ar.png",
                                            "review_EPL_20261101_Liverpool_Arsenal_en.png"]
    posts = [p for p in tg.plan(tmp_path) if p["kind"] == "review"]
    assert {p["lang"] for p in posts} == {"en", "ar"} and all(len(p["files"]) == 1 for p in posts)
