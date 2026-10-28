"""Written match analysis: one paragraph per match card, by a Gemini model, from our facts
only, checked before anything is posted.

    python -m fa.writer                 # write notes for the newest cards folder, print them
    python -m fa.writer --folder reports/cards/2026-10-28

Needs GEMINI_API_KEY (free key from Google AI Studio) in .env or as a GitHub secret;
without it this step does nothing. The model is the newest stable Gemini Flash the key
can use, unless GEMINI_MODEL names one.

The checker rejects a paragraph that
  - contains a number not in the fact sheet (compared by value: "79%" matches "79"),
  - names someone or something not in the fact sheet (capitalised words mid-sentence),
  - makes claims our data can't support (tactics, flanks, possession, minutes, betting),
  - is too short or too long.
A rejected paragraph is retried with the problems listed; after 3 failures the match
simply gets no paragraph. Notes go to <folder>/notes_<kind>_en.json for fa.telegram.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

from fa.config import DB_PATH, REPORTS_DIR

API = "https://generativelanguage.googleapis.com/v1beta"
MIN_GAP_S = 5                     # free tier: about 15 requests per minute
MIN_WORDS, MAX_WORDS = 50, 140

SYSTEM = """You are the analyst of a football statistics channel. Write ONE paragraph of 70 to 110 words in plain English about the match described by the JSON fact sheet.

Rules:
- Use only the facts given. Copy numbers exactly as written in the facts; never calculate, round or estimate new numbers.
- Mention only the teams and players named in the facts.
- Our data has no tactics, formations, possession, passing, crosses, corners, pressing, attacking sides or flanks, and no minutes of goals or cards: do not mention any of these.
- No betting language (odds, bets, tips, value, locks).
- "shots on target" are our measure of chances; "expected goals from shots on target" estimates how many goals those chances were worth. A positive "goals_minus_expected" means a team has scored more than its chances suggest (clinical or lucky); a negative "goals_conceded_minus_expected..." means it conceded fewer than the chances it faced suggest (good goalkeeping or luck).
- Previews: compare the two teams' form, chances created and conceded, and finishing, then give our forecast. Reviews: say who created more, whether the result was deserved, how it compares with the teams' season averages, whether our forecast was right, and any season-outlook change.
- No headline, no bullet points, no emojis, no hashtags. Write it as a knowledgeable, neutral analyst."""

BANNED = re.compile(
    r"\b(bets?|betting|odds|tips?|banker|value bet|locks?|guarantee\w*|formations?|flanks?|"
    r"wings?|wingers?|wide areas?|possession|pressing|high press|counter[- ]?attack\w*|"
    r"set[- ]pieces?|corners?|cross(es|ing)?|left|right[- ]side|minutes?|heat ?maps?|"
    r"4-4-2|4-3-3|3-5-2|back (three|four|five))\b", re.I)
CAPITALISED = re.compile(r"\b[A-ZÀ-ÖØ-Þ][\w'’.-]*")
ALWAYS_OK = {"I", "A", "The", "xG", "Premier", "League", "La", "Liga", "Serie", "Bundesliga",
             "Ligue", "Champions", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday", "January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"}


# ---------------- checking ----------------

def _strings(obj) -> list[str]:
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in [str(k)] + _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return [str(obj)]


NUMBER_IN_CONTEXT = re.compile(r"(?<![\w.])[+-]?(\d+(?:\.\d+)?)(\s*%)?")


def _kind(raw: str, percent: bool) -> str:
    return "percent" if percent else ("decimal" if "." in raw else "integer")


def allowed_numbers(facts: dict) -> dict[str, set[float]]:
    """Numbers in the facts by kind: a percentage must match a percentage, a decimal a
    decimal, so a made-up "55%" can't pass because some team scored 55 goals."""
    nums = {"percent": set(), "decimal": set(), "integer": set()}
    for s in _strings(facts):
        for raw, pc in NUMBER_IN_CONTEXT.findall(s):
            nums[_kind(raw, bool(pc))].add(float(raw))
        for a, b in re.findall(r"(\d+)-(\d+)", s):       # scores and records: "2-0", "12-3-3"
            nums["integer"].update({float(a), float(b)})
    return nums


def allowed_words(facts: dict) -> set[str]:
    words = set(ALWAYS_OK)
    for s in _strings(facts):
        for w in re.findall(r"[\w'’.-]+", s):
            words.add(w.strip(".-"))
    return words


def check(text: str, facts: dict) -> list[str]:
    """Problems with a paragraph (empty list = OK to post)."""
    problems = []
    words = len(text.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        problems.append(f"length is {words} words; it must be between {MIN_WORDS} and {MAX_WORDS}")
    if "\n" in text.strip() or text.lstrip().startswith(("#", "-", "*")):
        problems.append("write a single plain paragraph")
    nums = allowed_numbers(facts)
    for raw, pc in NUMBER_IN_CONTEXT.findall(text):
        if float(raw) not in nums[_kind(raw, bool(pc))]:
            problems.append(f"the number {raw}{'%' if pc else ''} is not in the facts")
    vocab = allowed_words(facts)
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        tokens = CAPITALISED.findall(sentence)
        first = sentence.split()[0] if sentence.split() else ""
        for tok in tokens:
            t = tok.strip(".'’-")
            if not t or t == first.strip(".'’-\"") or t in vocab:
                continue
            if re.sub(r"['’]s$", "", t) in vocab:            # possessive: "Inter's"
                continue
            problems.append(f"'{t}' is not in the facts")
    for m in BANNED.finditer(text):
        problems.append(f"'{m.group(0)}' is not something our data covers")
    return sorted(set(problems))


# ---------------- Gemini ----------------

class GeminiError(RuntimeError):
    """Safe to print: never contains the key."""


class Gemini:
    TIMEOUT_S = 180               # thinking models can take a while on long prompts

    def __init__(self, key: str, model: str | None = None, session: requests.Session | None = None):
        self.s = session or requests.Session()
        self.s.headers["x-goog-api-key"] = key
        self._last = 0.0
        self.models = [model] if model else self.list_models()
        self.model = self.models[0]
        self.thinking = {"thinkingLevel": "low"}         # dropped if a model doesn't accept it

    def _wait(self):
        gap = MIN_GAP_S - (time.time() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.time()

    def list_models(self) -> list[str]:
        """Stable 'flash' models that support generateContent, newest first."""
        try:
            r = self.s.get(f"{API}/models", params={"pageSize": 200}, timeout=30)
        except requests.RequestException as e:
            raise GeminiError(f"couldn't reach Gemini ({type(e).__name__})") from None
        if r.status_code != 200:
            raise GeminiError(f"listing models failed ({r.status_code}): {r.text[:200]}")
        found = []
        for m in r.json().get("models", []):
            name = m.get("name", "").removeprefix("models/")
            if ("generateContent" not in m.get("supportedGenerationMethods", []) or "flash" not in name
                    or re.search(r"lite|image|tts|audio|live|preview|exp|thinking|latest", name)):
                continue
            v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
            found.append(((float(v.group(1)) if v else 0.0, -len(name)), name))
        if not found:
            raise GeminiError("no Gemini Flash model available for this key")
        return [name for _, name in sorted(found, reverse=True)]

    def pick_model(self) -> str:
        return self.list_models()[0]

    def _next_model(self) -> bool:
        i = self.models.index(self.model)
        if i + 1 < len(self.models):
            self.model = self.models[i + 1]
            print(f"  (switching to {self.model})")
            return True
        return False

    def generate(self, prompt: str, system: str = SYSTEM) -> str:
        timeouts = 0
        for attempt in range(6):
            config = {"temperature": 0.4, "maxOutputTokens": 2048}
            if self.thinking:
                config["thinkingConfig"] = self.thinking
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": config}
            self._wait()
            try:
                r = self.s.post(f"{API}/models/{self.model}:generateContent", json=body,
                                timeout=self.TIMEOUT_S)
            except requests.RequestException as e:
                timeouts += 1
                if timeouts >= 2 and not self._next_model():   # this model keeps timing out
                    raise GeminiError(f"couldn't reach Gemini ({type(e).__name__})") from None
                continue
            if r.status_code == 400 and self.thinking and "think" in r.text.lower():
                self.thinking = None                          # this model has no such setting
                continue
            if r.status_code in (429, 500, 503):
                time.sleep(20 * (attempt + 1))                # quota or overload: back off
                continue
            if r.status_code == 404 and self._next_model():
                continue
            if r.status_code != 200:
                raise GeminiError(f"Gemini refused ({r.status_code}): {r.text[:300]}")
            parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            return " ".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        raise GeminiError("Gemini is rate-limiting, overloaded or too slow; try later")


def write(llm, facts: dict, tries: int = 3) -> tuple[str | None, list[str]]:
    """A checked paragraph, or None with the last problems found."""
    prompt = "Fact sheet:\n" + json.dumps(facts, ensure_ascii=False, indent=1)
    problems: list[str] = []
    for _ in range(tries):
        text = " ".join(llm.generate(prompt).split())
        problems = check(text, facts)
        if not problems:
            return text, []
        prompt = ("Fact sheet:\n" + json.dumps(facts, ensure_ascii=False, indent=1)
                  + "\n\nYour previous paragraph was rejected:\n" + text
                  + "\n\nProblems:\n- " + "\n- ".join(problems)
                  + "\n\nWrite the paragraph again, fixing every problem.")
    return None, problems


# ---------------- building notes for a cards folder ----------------

def _ids(folder, kind: str) -> list[str]:
    return [f.stem[len(kind) + 1:-3] for f in sorted(folder.glob(f"{kind}_*_en.png"))]


def notes_for_folder(folder, llm, now: pd.Timestamp) -> dict[str, list[dict]]:
    from fa.cards_review import (biggest_change, review_rows, season_change)
    from fa.data import match_events, team_news
    from fa.facts import preview_facts, review_facts
    con = duckdb.connect(str(DB_PATH), read_only=True)
    matches = con.execute("SELECT * FROM matches").df()
    fc = con.execute("SELECT * FROM forecasts").df()
    con.close()
    matches["date"] = pd.to_datetime(matches.date)
    out = {}

    ids = _ids(folder, "match")
    if ids:
        latest = (fc.assign(date=pd.to_datetime(fc.date)).sort_values("created_at")
                  .groupby(["model", "match_id"]).tail(1))
        latest = latest[latest.match_id.isin(ids)]
        latest = latest.sort_values("model").drop_duplicates("match_id")    # blend_v1 first
        snapshot = team_news.load(now=now)
        notes = []
        for _, row in latest.iterrows():
            news = None
            if len(snapshot) and row.league in set(snapshot.league):
                news = {"home": team_news.for_team(snapshot, row.home_team),
                        "away": team_news.for_team(snapshot, row.away_team)}
            facts = preview_facts(row, matches, news)
            text, problems = write(llm, facts)
            title = f"{row.home_team} vs {row.away_team}"
            print(f"{'✓' if text else '✗'} preview {title}" + ("" if text else f": {problems[:3]}"))
            if text:
                notes.append({"match_id": row.match_id, "title": title, "text": text})
        out["match"] = notes

    ids = _ids(folder, "review")
    if ids:
        rows = review_rows(matches, fc, now, days=4)
        rows = rows[rows.match_id.isin(ids)]
        events = match_events.load()
        cache, notes = {}, []
        for _, r in rows.iterrows():
            ev = match_events.for_match(events, r.league, r.home_team, r.away_team, r.result_date,
                                        r.home_goals, r.away_goals)
            sims = season_change(matches, r.league, pd.Timestamp(r.result_date), cache)
            changes = [(t, *c) for t in (r.home_team, r.away_team)
                       for c in [biggest_change(t, *sims) if sims else None] if c]
            facts = review_facts(r, matches, ev, changes)
            text, problems = write(llm, facts)
            title = f"{r.home_team} {int(r.home_goals)}-{int(r.away_goals)} {r.away_team}"
            print(f"{'✓' if text else '✗'} review {title}" + ("" if text else f": {problems[:3]}"))
            if text:
                notes.append({"match_id": r.match_id, "title": title, "text": text})
        out["review"] = notes
    return out


def main() -> None:
    from fa.telegram import env
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", help="cards folder (default: the newest in reports/cards)")
    ap.add_argument("--now", help="pretend it's this time (UTC), for testing")
    ap.add_argument("--ping", action="store_true", help="one tiny request: is Gemini working?")
    args = ap.parse_args()
    key = env("GEMINI_API_KEY")
    if not key:
        print("No GEMINI_API_KEY: skipping the written analysis.")
        return
    if args.ping:
        try:
            llm = Gemini(key, env("GEMINI_MODEL"))
            print(f"Models available: {', '.join(llm.models)}")
            t = time.time()
            reply = llm.generate("Reply with the single word: ready", system="Answer in one word.")
            print(f"{llm.model} answered {reply!r} in {time.time() - t:.0f} s")
        except GeminiError as e:
            raise SystemExit(f"Gemini not working: {e}") from None
        return
    root = REPORTS_DIR / "cards"
    folders = sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []
    folder = Path(args.folder) if args.folder else (folders[-1] if folders else None)
    if folder is None or not folder.exists():
        raise SystemExit("No cards folder: run `python -m fa.cards` first.")
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now("UTC").tz_localize(None)
    try:
        llm = Gemini(key, env("GEMINI_MODEL"))
        print(f"Writing with {llm.model}")
        notes = notes_for_folder(folder, llm, now)
    except GeminiError as e:
        raise SystemExit(f"Written analysis skipped: {e}") from None
    for kind, items in notes.items():
        path = folder / f"notes_{kind}_en.json"
        path.write_text(json.dumps(items, ensure_ascii=False, indent=1))
        print(f"\n{len(items)} {kind} notes -> {path}")
        for n in items:
            print(f"\n{n['title']}\n{n['text']}")


if __name__ == "__main__":
    main()
