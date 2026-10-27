"""Run with:  PYTHONPATH=src python -m pytest -q   (the real FPL API isn't called)"""
import pandas as pd
import pytest

import fa.data.team_news as tn
from fa.cards import match_card

NOW = pd.Timestamp("2026-10-27 06:30")
PAYLOAD = {
    "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"},
              {"id": 12, "name": "Liverpool", "short_name": "LIV"},
              {"id": 14, "name": "Man Utd", "short_name": "MUN"}],
    "elements": [
        {"team": 1, "web_name": "Saka", "status": "i", "chance_of_playing_next_round": 0,
         "news": "Hamstring injury - Expected back 08 Nov", "minutes": 700},
        {"team": 1, "web_name": "Rice", "status": "s", "chance_of_playing_next_round": 0,
         "news": "Suspended until 02 Nov", "minutes": 800},
        {"team": 1, "web_name": "Odegaard", "status": "d", "chance_of_playing_next_round": 75,
         "news": "Knock - 75% chance of playing", "minutes": 650},
        {"team": 1, "web_name": "Raya", "status": "a", "minutes": 900},
        {"team": 12, "web_name": "Loaned", "status": "u", "news": "Joined X on loan", "minutes": 0},
        {"team": 14, "web_name": "Mount", "status": "i", "news": "Calf injury", "minutes": 100},
    ],
}


def test_parse_keeps_only_absences_and_maps_names():
    df, problems = tn.parse(PAYLOAD, ["Arsenal", "Liverpool", "Man United"], NOW)
    assert not problems
    assert set(df.player) == {"Saka", "Rice", "Odegaard", "Mount"}       # available + loaned dropped
    assert df.set_index("player").loc["Mount", "team"] == "Man United"   # "Man Utd" mapped
    n = tn.for_team(df, "Arsenal")
    assert n["out"] == ["Rice", "Saka"]                                  # most minutes first
    assert n["doubt"] == ["Odegaard 75%"]
    assert tn.for_team(df, "Liverpool") == {"out": [], "doubt": []}


def test_old_snapshot_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(tn, "NEWS_PATH", tmp_path / "team_news.csv")
    df, _ = tn.parse(PAYLOAD, ["Arsenal", "Liverpool", "Man United"], NOW)
    df.to_csv(tn.NEWS_PATH, index=False)
    assert len(tn.load(now=NOW + pd.Timedelta(hours=20))) == 4
    assert tn.load(now=NOW + pd.Timedelta(days=3)).empty                 # stale: no news on cards


def test_long_list_is_capped():
    many = {"teams": PAYLOAD["teams"], "elements": [
        {"team": 1, "web_name": f"P{i}", "status": "i", "minutes": i} for i in range(7)]}
    df, _ = tn.parse(many, ["Arsenal", "Liverpool", "Man United"], NOW)
    assert tn.for_team(df, "Arsenal")["out"] == ["P6", "P5", "P4", "P3", "+3"]


@pytest.mark.parametrize("lang", ["en", "ar"])
@pytest.mark.parametrize("goals", [True, False])
def test_card_with_news_renders(lang, goals):
    df, _ = tn.parse(PAYLOAD, ["Arsenal", "Liverpool", "Man United"], NOW)
    news = {"as_of": NOW, "home": tn.for_team(df, "Liverpool"), "away": tn.for_team(df, "Arsenal")}
    row = pd.Series(dict(match_id="m", league="EPL", date=pd.Timestamp("2026-11-01"),
                         kickoff_utc=pd.Timestamp("2026-11-01 16:30"), home_team="Liverpool",
                         away_team="Arsenal", p_home=.29, p_draw=.29, p_away=.42, top_score="0-1",
                         xg_home=1.1, xg_away=1.4, p_over25=.44 if goals else float("nan"),
                         p_btts=.5))
    img, _ = match_card(row, lang, (70, 120), "@test", news)
    assert img.size == (1080, (1240 if goals else 900) + 300)
