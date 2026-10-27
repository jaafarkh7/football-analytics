"""Post cards to Telegram.

    python -m fa.cards                       # make the cards first
    python -m fa.telegram                    # dry run: shows what would be posted
    python -m fa.telegram --send             # actually post

Setup (once):
  1. In Telegram, talk to @BotFather, send /newbot, and copy the token it gives you.
  2. Create a channel, open its settings, and add your bot as an administrator.
  3. Add to `.env` in the project root (never commit it):
        TELEGRAM_BOT_TOKEN=123456:ABC...
        TELEGRAM_CHAT=@your_channel          # one channel for both languages
     or, for one channel per language:
        TELEGRAM_CHAT_EN=@your_channel_en
        TELEGRAM_CHAT_AR=@your_channel_ar

Every posted card is recorded in records/posted.csv, so nothing is ever posted twice.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from fa.config import REPORTS_DIR, ROOT

POSTED_LOG = ROOT / "records" / "posted.csv"
ALBUM_MAX = 10                      # Telegram's limit for one media group

CAPTIONS = {
    "en": {"match": "Big matches this week: our model's forecasts.\n"
                    "Every forecast is saved before kickoff and scored afterwards.",
           "round": "{league}: forecasts for the next matches (home / draw / away).",
           "scorecard": "How last week's forecasts did, misses included.",
           "table": "{league}: title, top-4 and relegation chances from 10,000 simulations of the rest of the season.",
           "deserved": "{league}: who deserved to win? Chances estimated from shots on target.",
           "tail": "\nAnalysis, not betting advice."},
    "ar": {"match": "أبرز مباريات الأسبوع: توقعات النموذج.\n"
                    "كل توقع يُحفظ قبل انطلاق المباراة ويُقيَّم بعدها.",
           "round": "{league}: توقعات المباريات القادمة (فوز صاحب الأرض / تعادل / فوز الضيف).",
           "scorecard": "أداء توقعات الأسبوع الماضي، بما فيها الأخطاء.",
           "table": "{league}: فرص اللقب والمراكز الأربعة الأولى والهبوط من 10,000 محاكاة لبقية الموسم.",
           "deserved": "{league}: من استحق الفوز؟ الفرص محسوبة من التسديدات على المرمى.",
           "tail": "\nتحليل وليس نصيحة للمراهنة."},
}


def env(name: str) -> str | None:
    value = os.environ.get(name)
    path = ROOT / ".env"
    if value is None and path.exists():
        for line in path.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
    return value or None


def chats() -> dict[str, str]:
    """Language -> chat. One shared chat, or one per language."""
    shared = env("TELEGRAM_CHAT")
    out = {lang: env(f"TELEGRAM_CHAT_{lang.upper()}") or shared for lang in ("en", "ar")}
    return {lang: chat for lang, chat in out.items() if chat}


# ---------------- what to post ----------------

def plan(folder: Path) -> list[dict]:
    """Group the cards in `folder` into posts: one album per kind and language."""
    from fa.cards import LEAGUE_NAMES
    posts = []
    for lang in ("en", "ar"):
        cap = CAPTIONS[lang]
        matches = sorted(folder.glob(f"match_*_{lang}.png"))
        for i in range(0, len(matches), ALBUM_MAX):
            posts.append({"lang": lang, "kind": "match", "files": matches[i:i + ALBUM_MAX],
                          "caption": cap["match"] + cap["tail"]})
        for kind in ("round", "deserved", "table"):
            for f in sorted(folder.glob(f"{kind}_*_{lang}.png")):
                league = f.stem[len(kind) + 1:-len(f"_{lang}")]
                posts.append({"lang": lang, "kind": kind, "files": [f],
                              "caption": cap[kind].format(league=LEAGUE_NAMES[lang].get(league, league))
                              + cap["tail"]})
        sc = folder / f"scorecard_{lang}.png"
        if sc.exists():
            posts.append({"lang": lang, "kind": "scorecard", "files": [sc],
                          "caption": cap["scorecard"] + cap["tail"]})
    return posts


def post_key(post: dict, chat: str) -> str:
    return chat + "|" + "|".join(f"{f.parent.name}/{f.name}" for f in post["files"])


def already_posted(path: Path = POSTED_LOG) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {row["key"] for row in csv.DictReader(f)}


def record(key: str, path: Path = POSTED_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["posted_at", "key"])
        if new:
            w.writeheader()
        w.writerow({"posted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "key": key})


# ---------------- sending ----------------

class TelegramError(RuntimeError):
    pass


class Bot:
    def __init__(self, token: str, session: requests.Session | None = None):
        self.base = f"https://api.telegram.org/bot{token}"
        self.s = session or requests.Session()

    def _call(self, method: str, data: dict, files: dict) -> dict:
        for attempt in range(3):
            r = self.s.post(f"{self.base}/{method}", data=data, files=files, timeout=60)
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if r.status_code == 429:                     # flood control: wait as told
                time.sleep(int(body.get("parameters", {}).get("retry_after", 5)) + 1)
                continue
            if not body.get("ok"):
                raise TelegramError(f"{method} failed ({r.status_code}): "
                                    f"{body.get('description', r.text[:200])}")
            return body
        raise TelegramError(f"{method}: still rate limited after retries")

    def send(self, chat: str, files: list[Path], caption: str) -> None:
        handles = [f.open("rb") for f in files]
        try:
            if len(files) == 1:
                self._call("sendPhoto", {"chat_id": chat, "caption": caption},
                           {"photo": (files[0].name, handles[0], "image/png")})
            else:
                import json
                media = [{"type": "photo", "media": f"attach://p{i}"} for i in range(len(files))]
                media[0]["caption"] = caption            # an album's caption sits on its first photo
                self._call("sendMediaGroup", {"chat_id": chat, "media": json.dumps(media)},
                           {f"p{i}": (f.name, h, "image/png") for i, (f, h) in enumerate(zip(files, handles))})
        finally:
            for h in handles:
                h.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="really post (default: dry run)")
    ap.add_argument("--folder", help="cards folder (default: the newest in reports/cards)")
    args = ap.parse_args()

    root = REPORTS_DIR / "cards"
    folders = sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []
    folder = Path(args.folder) if args.folder else (folders[-1] if folders else None)
    if folder is None or not folder.exists():
        raise SystemExit("No cards found: run `python -m fa.cards` first.")

    targets = chats()
    if not targets:
        raise SystemExit("No Telegram chat set. Add TELEGRAM_CHAT=@your_channel to .env "
                         "(see the top of src/fa/telegram.py for setup).")
    token = env("TELEGRAM_BOT_TOKEN")
    if args.send and not token:
        raise SystemExit("No TELEGRAM_BOT_TOKEN in .env (get one from @BotFather).")

    done = already_posted()
    bot = Bot(token) if args.send else None
    sent = skipped = 0
    for post in plan(folder):
        chat = targets.get(post["lang"])
        if not chat:
            continue
        key = post_key(post, chat)
        if key in done:
            skipped += 1
            continue
        names = ", ".join(f.name for f in post["files"])
        if not args.send:
            print(f"[dry run] {chat} ({post['lang']}, {post['kind']}): {names}")
            continue
        bot.send(chat, post["files"], post["caption"])
        record(key)
        sent += 1
        print(f"posted to {chat}: {names}")
        time.sleep(1.5)                                   # stay well under Telegram's limits
    if args.send:
        print(f"{sent} posts sent, {skipped} already posted earlier")
    else:
        print(f"Dry run only ({skipped} already posted). Add --send to post for real.")


if __name__ == "__main__":
    main()
