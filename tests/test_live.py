"""Run with:  PYTHONPATH=src python -m pytest -q   (no network: API and Telegram are faked)"""
import numpy as np
import pandas as pd
import pytest

import fa.live as live
from fa.models.live import halftime_probs, outcome_probs, rates_from_probs

NOW = pd.Timestamp("2026-10-28 18:20")          # Venezia-Inter kicked off 17:30 UTC


def test_rates_reproduce_the_forecast():
    lam, mu = rates_from_probs(0.52, 0.22)
    p = outcome_probs(lam, mu)
    assert abs(p[0] - 0.52) < 1e-3 and abs(p[2] - 0.22) < 1e-3


def test_halftime_probs_behave():
    level = halftime_probs(0.45, 0.30, 0, 0)
    assert abs(level.sum() - 1) < 1e-9
    assert level[1] > 0.25                                   # 0-0 at half-time: more draws
    lead1, lead2 = halftime_probs(0.45, 0.30, 1, 0), halftime_probs(0.45, 0.30, 2, 0)
    assert level[0] < lead1[0] < lead2[0] < 1                # bigger lead, safer win
    assert halftime_probs(0.45, 0.30, 0, 1)[2] > lead1[2] * 0 + 0.5   # away lead favours away
    # Only the goal difference matters: 1-1 is like 0-0.
    assert np.allclose(halftime_probs(0.45, 0.30, 1, 1), level)


def _api_match(mid, status, utc, ht, home_id=1, away_id=2, code="SA"):
    return {"id": mid, "status": status, "utcDate": utc, "competition": {"code": code},
            "homeTeam": {"id": home_id, "shortName": "H"}, "awayTeam": {"id": away_id, "shortName": "A"},
            "score": {"halfTime": {"home": ht[0], "away": ht[1]} if ht else {"home": None, "away": None}}}


IDS = {1: "Venezia", 2: "Inter", 3: "Milan", 4: "Bologna", 5: "Genoa", 6: "Juventus"}


def test_only_matches_at_halftime_are_picked():
    payload = {"matches": [
        _api_match(1, "PAUSED", "2026-10-28T17:30:00Z", (1, 0)),                    # half-time
        _api_match(2, "IN_PLAY", "2026-10-28T18:00:00Z", (0, 0), 3, 4),            # first half (20')
        _api_match(3, "IN_PLAY", "2026-10-28T16:50:00Z", (0, 1), 5, 6),             # 90 min: too late
        _api_match(4, "TIMED", "2026-10-28T19:45:00Z", None, 3, 4),                 # not started
        _api_match(5, "PAUSED", "2026-10-28T17:30:00Z", (0, 0), 7, 8, code="ELC"),  # not our league
    ]}
    ht = live.at_halftime(payload, IDS, NOW)
    assert list(ht.home_team) == ["Venezia"] and ht.ht_home.iloc[0] == 1
    # Second half just started (kickoff + 65') still counts.
    later = live.at_halftime({"matches": [_api_match(1, "IN_PLAY", "2026-10-28T17:30:00Z", (1, 0))]},
                             IDS, pd.Timestamp("2026-10-28 18:35"))
    assert len(later) == 1


def _forecasts(tmp_path):
    fc = pd.DataFrame([dict(created_at="2026-10-27 10:31", model="blend_v1",
                            match_id="SERIEA_20261028_Venezia_Inter", league="SERIEA",
                            date="2026-10-28", home_team="Venezia", away_team="Inter",
                            p_home=0.07, p_draw=0.14, p_away=0.79, xg_home=0.7, xg_away=2.3,
                            kickoff_utc="2026-10-28 17:30:00")])
    path = tmp_path / "forecasts.csv"
    fc.to_csv(path, index=False)
    return path


def test_card_and_attachment(tmp_path):
    fc = live.latest_forecasts(_forecasts(tmp_path))
    ht = live.at_halftime({"matches": [_api_match(1, "PAUSED", "2026-10-28T17:30:00Z", (1, 0))]}, IDS, NOW)
    j = live.attach_forecasts(ht, fc)
    assert len(j) == 1
    row = j.iloc[0]
    probs = live.update(row)
    assert probs["now"][2] < probs["before"][2]              # Inter behind: chance falls
    img, missing = live.halftime_card(row, probs, "en", "@test")
    assert img.size == (1080, 1120) and not missing


def test_main_posts_each_halftime_once(tmp_path, monkeypatch):
    import fa.records as rec
    import fa.telegram as tg
    monkeypatch.setattr(rec, "RECORDS", tmp_path)
    _forecasts(tmp_path)
    monkeypatch.setattr(tg, "POSTED_LOG", tmp_path / "posted.csv")
    monkeypatch.setattr(tg.already_posted, "__defaults__", (tmp_path / "posted.csv",))
    monkeypatch.setattr(tg.record, "__defaults__", (tmp_path / "posted.csv",))
    monkeypatch.setattr(live, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(live, "team_ids", lambda: IDS)
    monkeypatch.setattr(tg, "chats", lambda: {"en": "@test"})
    monkeypatch.setattr(tg, "env", lambda name: "TOKEN" if name == "TELEGRAM_BOT_TOKEN" else None)

    class FakeClient:
        def get(self, path, **params):
            return {"matches": [_api_match(1, "PAUSED", "2026-10-28T17:30:00Z", (1, 0))]}
    import fa.data.fixtures_api as api
    monkeypatch.setattr(api, "get_client", lambda: FakeClient())

    sent = []

    class FakeBot:
        def __init__(self, token):
            pass

        def send(self, chat, files, caption):
            sent.append((chat, [f.name for f in files], caption))
    monkeypatch.setattr(tg, "Bot", FakeBot)
    monkeypatch.setattr(live.time, "sleep", lambda s: None)

    for _ in range(2):                                       # two runs 15 minutes apart
        monkeypatch.setattr("sys.argv", ["fa.live", "--send", "--now", "2026-10-28 18:20"])
        live.main()
    assert len(sent) == 1
    assert sent[0][1] == ["halftime_SERIEA_20261028_Venezia_Inter_en.png"]
    assert "Venezia 1-0 Inter" in sent[0][2]
