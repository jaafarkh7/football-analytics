"""Run with:  PYTHONPATH=src python -m pytest -q   (the real FPL API isn't called)"""
import sys
from pathlib import Path

import pandas as pd

from fa.cards_review import review_card, review_rows
from fa.data import match_events as me

sys.path.insert(0, str(Path(__file__).parent))
from test_cards_review import NOW, _forecasts, _matches   # noqa: E402

BOOT = {"teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"},
                  {"id": 12, "name": "Liverpool", "short_name": "LIV"},
                  {"id": 14, "name": "Man Utd", "short_name": "MUN"}],
        "elements": [{"id": 100, "web_name": "Saka"}, {"id": 101, "web_name": "Ødegaard"},
                     {"id": 102, "web_name": "Rice"}, {"id": 103, "web_name": "Gabriel"},
                     {"id": 107, "web_name": "Konaté"}, {"id": 108, "web_name": "Van Dijk"}]}


def _fixture(**kw):
    f = {"finished": True, "kickoff_time": "2026-11-01T16:30:00Z", "team_h": 12, "team_a": 1,
         "team_h_score": 0, "team_a_score": 2, "stats": [
             {"identifier": "goals_scored", "h": [], "a": [{"value": 1, "element": 100}]},
             {"identifier": "own_goals", "h": [{"value": 1, "element": 107}], "a": []},
             {"identifier": "assists", "h": [], "a": [{"value": 1, "element": 101}]},
             {"identifier": "yellow_cards", "h": [{"value": 1, "element": 108}],
              "a": [{"value": 1, "element": 102}]},
             {"identifier": "red_cards", "h": [], "a": [{"value": 1, "element": 103}]},
             {"identifier": "bonus", "h": [], "a": [{"value": 3, "element": 100}]}]}
    f.update(kw)
    return f


TEAMS = ["Liverpool", "Arsenal", "Man United"]


def test_parse_credits_own_goals_to_the_other_team():
    ev = me.parse(BOOT, [_fixture()], TEAMS)
    og = ev[ev.kind == "own_goal"].iloc[0]
    assert og.player == "Konaté" and og.side == "away"        # Liverpool player, Arsenal's goal
    assert set(ev.kind) == {"goal", "own_goal", "assist", "yellow", "red"}   # bonus ignored
    assert ev.date.iloc[0] == pd.Timestamp("2026-11-01")


def test_unfinished_and_unmapped_fixtures_are_skipped():
    ev = me.parse(BOOT, [_fixture(finished=False), _fixture(team_h=99)], TEAMS)
    assert ev.empty


def test_for_match_lists_and_checks_the_score():
    ev = me.parse(BOOT, [_fixture()], TEAMS)
    e = me.for_match(ev, "EPL", "Liverpool", "Arsenal", "2026-11-01", 0, 2)
    assert e["away"] == {"goals": ["Saka", "Konaté (og)"], "assists": ["Ødegaard"],
                         "yellow": ["Rice"], "red": ["Gabriel"]}
    assert e["home"]["yellow"] == ["Van Dijk"] and e["home"]["goals"] == []
    assert me.for_match(ev, "EPL", "Liverpool", "Arsenal", "2026-11-01", 1, 2) is None  # score mismatch
    assert me.for_match(ev, "EPL", "Chelsea", "Arsenal", "2026-11-01") is None


def test_braces_are_counted():
    f = _fixture(team_h_score=0, team_a_score=3)
    f["stats"][0]["a"] = [{"value": 2, "element": 100}]
    e = me.for_match(me.parse(BOOT, [f], TEAMS), "EPL", "Liverpool", "Arsenal", "2026-11-01")
    assert e["away"]["goals"][0] == "Saka 2"


def test_card_grows_by_one_row_per_event_kind():
    r = review_rows(_matches(), _forecasts(), NOW).iloc[0]
    e = me.for_match(me.parse(BOOT, [_fixture()], TEAMS), "EPL", "Liverpool", "Arsenal",
                     r.result_date, r.home_goals, r.away_goals)
    for lang in ("en", "ar"):
        img, _ = review_card(r, lang, None, "", None, e)
        assert img.size == (1080, 1180 + 30 + 42 * 4)              # goals, assists, yellow, red
