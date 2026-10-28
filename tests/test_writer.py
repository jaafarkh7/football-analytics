"""Run with:  PYTHONPATH=src python -m pytest -q   (Gemini and Telegram are faked)"""
import json

import pandas as pd

import fa.telegram as tg
from fa.writer import Gemini, check, write

FACTS = {"type": "preview", "competition": "Serie A", "date": "Sunday 17 May",
         "home_team": "Roma", "away_team": "Lazio",
         "home": {"name": "Roma", "league_position": "4th", "points": "67",
                  "last_5_points": "13", "home_record_w_d_l": "12-3-3",
                  "shots_on_target_per_game": "4.6", "goals_minus_expected_from_shots_on_target": "+4.2"},
         "away": {"name": "Lazio", "league_position": "9th", "points": "51",
                  "shots_on_target_per_game": "4.0"},
         "our_forecast": {"home_win": "52%", "draw": "26%", "away_win": "22%", "most_likely_score": "1-0"},
         "team_news": {"home": {"out": ["Dybala"]}}}

GOOD = ("Roma host Lazio with 13 points from their last five games and a 12-3-3 home record, "
        "which has carried them to 4th on 67 points. They create 4.6 shots on target per game "
        "against Lazio's 4.0, and they have scored 4.2 more goals than those chances suggest, "
        "so their finishing has been sharp. Lazio sit 9th on 51 points. Roma are without "
        "Dybala, yet our model still makes them 52% favourites, with the draw at 26%, Lazio at "
        "22% and a 1-0 home win as the single most likely score.")


def test_good_paragraph_passes():
    assert check(GOOD, FACTS) == []


def test_invented_numbers_are_caught_by_kind():
    bad = GOOD.replace("52% favourites", "55% favourites").replace("4.6 shots", "4.7 shots")
    problems = check(bad, FACTS)
    assert "the number 55% is not in the facts" in problems
    assert "the number 4.7 is not in the facts" in problems
    # 67 is a points total, not a percentage: "67%" must fail.
    assert any("67%" in p for p in check(GOOD.replace("52% favourites", "67% favourites"), FACTS))


def test_invented_names_and_unsupported_claims_are_caught():
    bad = GOOD.replace("Roma are without Dybala", "Roma are without Pellegrini") \
              .replace("so their finishing", "attacking down the left flank, so their finishing")
    problems = check(bad, FACTS)
    assert "'Pellegrini' is not in the facts" in problems
    assert any("left" in p for p in problems) and any("flank" in p for p in problems)
    assert any("odds" in p for p in check(GOOD + " The odds agree.", FACTS))


def test_length_and_format():
    assert any("length" in p for p in check("Roma are 52% favourites.", FACTS))
    assert any("single plain paragraph" in p for p in check(GOOD + "\n\n" + GOOD[:50], FACTS))


class FakeLLM:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def generate(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_rejected_paragraph_is_retried_with_the_problems():
    llm = FakeLLM([GOOD.replace("52%", "58%"), GOOD])
    text, problems = write(llm, FACTS)
    assert text == GOOD and problems == []
    assert "58% is not in the facts" in llm.prompts[1]          # the model was told what to fix


def test_gives_up_after_three_bad_paragraphs():
    text, problems = write(FakeLLM([GOOD.replace("52%", "58%")] * 3), FACTS)
    assert text is None and problems


class FakeResponse:
    def __init__(self, body, status=200):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self):
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        gen = ["generateContent"]
        return FakeResponse({"models": [
            {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": gen},
            {"name": "models/gemini-3.6-flash", "supportedGenerationMethods": gen},
            {"name": "models/gemini-3.6-flash-lite", "supportedGenerationMethods": gen},
            {"name": "models/gemini-3.7-flash-preview", "supportedGenerationMethods": gen},
            {"name": "models/gemini-3.6-pro", "supportedGenerationMethods": gen},
            {"name": "models/text-embedding-9", "supportedGenerationMethods": ["embedContent"]}]})

    def post(self, url, json=None, timeout=None):
        self.url = url
        return FakeResponse({"candidates": [{"content": {"parts": [
            {"text": "thinking...", "thought": True}, {"text": GOOD}]}}]})


def test_picks_newest_stable_flash_and_skips_thoughts(monkeypatch):
    import fa.writer as w
    monkeypatch.setattr(w, "MIN_GAP_S", 0)
    s = FakeSession()
    g = Gemini("KEY", session=s)
    assert g.model == "gemini-3.6-flash"                         # not lite, preview or pro
    assert s.headers["x-goog-api-key"] == "KEY"                  # key in a header, not the URL
    assert g.generate("facts") == GOOD and "KEY" not in s.url


def test_notes_are_posted_as_text_after_the_album(tmp_path):
    (tmp_path / "match_SERIEA_20260517_Roma_Lazio_en.png").write_bytes(b"png")
    notes = [{"match_id": "m", "title": "Roma vs Lazio", "text": GOOD + " <tag> & co"}] * 9
    (tmp_path / "notes_match_en.json").write_text(json.dumps(notes))
    posts = [p for p in tg.plan(tmp_path) if p["kind"] == "match"]
    assert posts[0]["files"] and all(p.get("text") for p in posts[1:])
    assert len(posts) == 3                                       # 9 notes split over 2 messages
    assert all(len(p["text"]) <= 4096 for p in posts[1:])
    assert "&lt;tag&gt; &amp; co" in posts[1]["text"]            # escaped for Telegram HTML
    keys = {tg.post_key(p, "@c", tmp_path) for p in posts[1:]}
    assert len(keys) == 2 and all(tmp_path.name in k for k in keys)


class SlowThenFineSession(FakeSession):
    """The newest model times out; the next one answers. It rejects thinkingConfig first."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def post(self, url, json=None, timeout=None):
        import requests
        self.calls.append((url.split("/models/")[1].split(":")[0], "thinkingConfig" in json["generationConfig"]))
        if "3.6-flash" in url:
            raise requests.ReadTimeout("slow")
        if "thinkingConfig" in json["generationConfig"]:
            return FakeResponse({"error": {"message": "Unknown name thinkingConfig"}}, status=400)
        return FakeResponse({"candidates": [{"content": {"parts": [{"text": GOOD}]}}]})


def test_slow_model_falls_back_to_the_next_one(monkeypatch):
    import fa.writer as w
    monkeypatch.setattr(w, "MIN_GAP_S", 0)
    monkeypatch.setattr(w.time, "sleep", lambda s: None)
    s = SlowThenFineSession()
    g = Gemini("KEY", session=s)
    assert g.models[:2] == ["gemini-3.6-flash", "gemini-3.5-flash"]
    assert g.generate("facts") == GOOD and g.model == "gemini-3.5-flash" and g.thinking is None
    assert s.calls[:2] == [("gemini-3.6-flash", True)] * 2         # two timeouts, then switch


def test_busy_model_switches_and_time_budget_keeps_earlier_notes(monkeypatch):
    import fa.writer as w
    monkeypatch.setattr(w, "MIN_GAP_S", 0)
    monkeypatch.setattr(w.time, "sleep", lambda s: None)

    class Busy(FakeSession):
        def post(self, url, json=None, timeout=None):
            if "3.6-flash" in url:
                return FakeResponse({"error": "overloaded"}, status=503)
            return super().post(url, json, timeout)
    g = Gemini("KEY", session=Busy())
    assert g.generate("facts") == GOOD and g.model == "gemini-3.5-flash"
    g.deadline = 0                                     # budget used up
    text, problems = w.safe_write(g, FACTS)
    assert text is None and problems == ["time budget used up"]


def test_cut_off_paragraphs_are_rejected():
    cut = GOOD.rsplit(" ", 6)[0]                               # ends mid-sentence, like the real run
    assert any("cut off" in p for p in check(cut, FACTS))


def test_fact_sheet_jargon_is_rejected():
    robotic = GOOD.replace("so their finishing has been sharp",
                           "posting a goals minus expected of 4.2")
    assert any("jargon" in p for p in check(robotic, FACTS))
    assert any("jargon" in p for p in check(GOOD.replace("4th", "league_position 4th"), FACTS))


def test_finishing_is_given_in_words():
    from fa.facts import finishing
    assert finishing(-1.94, "scored") == "scored 1.9 fewer goals than their shots on target are usually worth"
    assert finishing(4.2, "conceded") == "conceded 4.2 more goals than the shots on target they faced are usually worth"
    assert finishing(0.3, "scored").startswith("scored about as many")


def test_warm_up_starts_with_the_first_model_that_answers(monkeypatch):
    import requests

    class FirstSlow(FakeSession):
        def post(self, url, json=None, timeout=None):
            if "3.6-flash" in url:
                raise requests.ReadTimeout("slow")
            return super().post(url, json, timeout)
    g = Gemini("KEY", session=FirstSlow())
    assert g.warm_up() == "gemini-3.5-flash" and g.model == "gemini-3.5-flash"
