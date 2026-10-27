"""Run with:  PYTHONPATH=src python -m pytest -q"""
import json

import pandas as pd
import pytest

import fa.telegram as tg
from fa.cards import Names, match_card, round_card, runs, scorecard


def _row(**kw):
    base = dict(match_id="EPL_20261101_Liverpool_Arsenal", league="EPL",
                date=pd.Timestamp("2026-11-01"), kickoff_utc=pd.Timestamp("2026-11-01 16:30"),
                home_team="Liverpool", away_team="Arsenal", p_home=0.29, p_draw=0.29, p_away=0.42,
                top_score="0-1", xg_home=1.1, xg_away=1.4, p_over25=0.44, p_btts=0.50)
    base.update(kw)
    return pd.Series(base)


# ---------------- cards ----------------

def test_runs_split_arabic_and_numbers():
    assert runs("أكثر من 2.5 هدف") == [("أكثر من", True), ("2.5", False), ("هدف", True)]
    assert runs("Over 2.5 goals") == [("Over 2.5 goals", False)]


def test_unknown_club_falls_back_to_english_and_is_reported():
    names = Names("ar")
    assert names.team("Liverpool") == "ليفربول"
    assert names.team("Made Up FC") == "Made Up FC"
    assert names.missing == {"Made Up FC"}


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_cards_render(lang):
    img, missing = match_card(_row(), lang, record=(70, 120), handle="@test")
    assert img.size == (1080, 1240) and not missing
    rows = pd.DataFrame([_row(), _row(home_team="Chelsea", away_team="Man United", match_id="m2")])
    img, _ = round_card(rows, "EPL", lang)
    assert img.size[0] == 1080 and img.size[1] > 400
    scored = rows.assign(home_goals=[0, 2], away_goals=[2, 1], result=["A", "H"])
    img, _ = scorecard(scored, lang)
    assert img.size == (1080, 1080)


def test_card_without_goal_markets_still_renders():
    """Champions League forecasts have no scoreline model: no xG or goal markets."""
    img, _ = match_card(_row(league="UCL", top_score=None, xg_home=float("nan"),
                             xg_away=float("nan"), p_over25=float("nan"), p_btts=float("nan")), "ar")
    assert img.size == (1080, 900)                                # shorter: no goal tiles


# ---------------- telegram ----------------

class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        self._body = body if body is not None else {"ok": True, "result": {}}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def post(self, url, data=None, files=None, timeout=None):
        self.calls.append((url.rsplit("/", 1)[-1], dict(data), sorted(files)))
        return self.responses.pop(0) if self.responses else FakeResponse()


@pytest.fixture
def cards(tmp_path):
    folder = tmp_path / "2026-10-30"
    folder.mkdir()
    for name in ["match_a_en", "match_b_en", "match_a_ar", "round_EPL_en", "round_EPL_ar",
                 "scorecard_en"]:
        (folder / f"{name}.png").write_bytes(b"\x89PNG fake")
    return folder


def test_plan_groups_albums_per_language(cards):
    posts = tg.plan(cards)
    kinds = [(p["lang"], p["kind"], len(p["files"])) for p in posts]
    assert ("en", "match", 2) in kinds and ("ar", "match", 1) in kinds
    assert ("en", "round", 1) in kinds and ("en", "scorecard", 1) in kinds
    assert ("ar", "scorecard", 1) not in kinds
    ar_round = next(p for p in posts if p["lang"] == "ar" and p["kind"] == "round")
    assert "الدوري الإنجليزي" in ar_round["caption"]


def test_album_uses_media_group_with_caption_on_first_photo(cards):
    s = FakeSession()
    bot = tg.Bot("TOKEN", session=s)
    post = next(p for p in tg.plan(cards) if p["kind"] == "match" and p["lang"] == "en")
    bot.send("@chan", post["files"], post["caption"])
    method, data, files = s.calls[0]
    media = json.loads(data["media"])
    assert method == "sendMediaGroup" and files == ["p0", "p1"]
    assert "caption" in media[0] and "caption" not in media[1]


def test_rate_limit_is_retried_and_errors_are_raised(cards, monkeypatch):
    monkeypatch.setattr(tg.time, "sleep", lambda s: None)
    s = FakeSession([FakeResponse(429, {"ok": False, "parameters": {"retry_after": 1}}),
                     FakeResponse()])
    tg.Bot("T", session=s).send("@c", [next(cards.glob("round_EPL_en.png"))], "x")
    assert [c[0] for c in s.calls] == ["sendPhoto", "sendPhoto"]
    bad = FakeSession([FakeResponse(400, {"ok": False, "description": "chat not found"})])
    with pytest.raises(tg.TelegramError, match="chat not found"):
        tg.Bot("T", session=bad).send("@c", [next(cards.glob("round_EPL_en.png"))], "x")


def test_nothing_is_posted_twice(cards, tmp_path, monkeypatch):
    log = tmp_path / "posted.csv"
    post = tg.plan(cards)[0]
    key = tg.post_key(post, "@chan")
    assert key not in tg.already_posted(log)
    tg.record(key, log)
    assert key in tg.already_posted(log)


def test_chat_routing(monkeypatch, tmp_path):
    monkeypatch.setattr(tg, "ROOT", tmp_path)                    # no .env here
    for k in ("TELEGRAM_CHAT", "TELEGRAM_CHAT_EN", "TELEGRAM_CHAT_AR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT", "@both")
    assert tg.chats() == {"en": "@both", "ar": "@both"}
    monkeypatch.setenv("TELEGRAM_CHAT_AR", "@arabic")
    assert tg.chats() == {"en": "@both", "ar": "@arabic"}


def test_merged_env_lines_are_refused_without_printing_the_value(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tg, "ROOT", tmp_path)
    for k in ("TELEGRAM_CHAT", "TELEGRAM_CHAT_EN", "TELEGRAM_CHAT_AR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_EN", "@mychannel FOOTBALL_DATA_API_KEY=secret123")
    with pytest.raises(SystemExit) as e:
        tg.chats()
    assert "secret123" not in str(e.value)
    monkeypatch.setenv("TELEGRAM_CHAT_EN", "@mychannel")
    assert tg.chats() == {"en": "@mychannel"}


def test_network_errors_never_show_the_token(cards, monkeypatch):
    import requests
    monkeypatch.setattr(tg.time, "sleep", lambda s: None)
    monkeypatch.setattr(tg, "_force_ipv4", lambda: None)

    class DownSession:
        calls = 0

        def post(self, url, **kw):
            DownSession.calls += 1
            raise requests.ConnectionError(f"Max retries exceeded with url: {url} (unreachable)")

    bot = tg.Bot("123456:SECRET-TOKEN", session=DownSession())
    with pytest.raises(tg.TelegramError) as e:
        bot.send("@c", [next(cards.glob("round_EPL_en.png"))], "x")
    assert "SECRET-TOKEN" not in str(e.value) and e.value.__cause__ is None
    assert DownSession.calls == 3


def test_retry_resends_the_whole_file(cards, monkeypatch):
    import requests
    monkeypatch.setattr(tg.time, "sleep", lambda s: None)
    monkeypatch.setattr(tg, "_force_ipv4", lambda: None)
    sizes = []

    class FlakySession:
        def post(self, url, data=None, files=None, timeout=None):
            sizes.append(len(files["photo"][1].read()))
            if len(sizes) == 1:
                raise requests.ConnectionError("blip")
            return FakeResponse()

    tg.Bot("T", session=FlakySession()).send("@c", [next(cards.glob("round_EPL_en.png"))], "x")
    assert sizes[0] == sizes[1] > 0
