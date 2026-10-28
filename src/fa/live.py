"""Half-time updates: when a match we forecast reaches half-time, post its score and the
updated win chances.

    python -m fa.live            # dry run: what would be posted now
    python -m fa.live --send     # post (GitHub Actions runs this every 15 min in match hours)

Light on purpose: it reads records/forecasts.csv and records/team_map.csv (no database),
makes one football-data.org request for today's matches, and posts each half-time once
(records/posted.csv).

The free football-data.org plan shows scores a few minutes late. A match counts as at
half-time when its status is PAUSED, or when it is back IN_PLAY but the second half is
young (kickoff + 62 to 80 minutes); later than that, the update is no longer news and is
skipped.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from PIL import Image

from fa.cards import (CARD, INK2, MUTED, RULE, SURFACE, TEXT, W, Names, Painter,
                      outcome_bar, pct, when)
from fa.cards_review import ARROW, _pieces
from fa.config import REPORTS_DIR
from fa.data.fixtures_api import COMPETITIONS, TEAM_MAP_PATH
from fa.models.live import halftime_probs

CODE_TO_LEAGUE = {code: lg for lg, code in COMPETITIONS.items()}
LATE_MIN, STALE_MIN = 62, 80          # minutes after kickoff: second half started / too late
LIVE = {"PAUSED", "IN_PLAY"}
T = {"en": {"ht": "Half-time", "now": "Win chances now", "before": "Before kickoff",
            "caption": "Half-time: {home} {h}-{a} {away}. Updated win chances from our model "
                       "(pre-match forecast + the score, tested on 1,750 past matches).",
            "draw": "Draw",
            "method": "Pre-match forecast updated with the score · second half simulated"}}


# ---------------- data ----------------

def latest_forecasts(path) -> pd.DataFrame:
    """The latest forecast saved before kickoff for each match (league model first)."""
    from fa.score import eligible_forecasts
    fc = pd.read_csv(path, parse_dates=["created_at", "date", "kickoff_utc"])
    fc = eligible_forecasts(fc)
    fc["rank"] = fc.model.map({"blend_v1": 0}).fillna(1)
    return fc.sort_values("rank").drop_duplicates("match_id").drop(columns="rank")


def team_ids(path=TEAM_MAP_PATH) -> dict[int, str]:
    tm = pd.read_csv(path).dropna(subset=["our_name"])
    return {int(i): name for i, name in zip(tm.id, tm.our_name)}


def at_halftime(payload: dict, ids: dict[int, str], now: pd.Timestamp) -> pd.DataFrame:
    """Matches at (or just after) half-time, with our team names and the half-time score."""
    rows = []
    for m in payload.get("matches", []):
        league = CODE_TO_LEAGUE.get((m.get("competition") or {}).get("code"))
        if not league or m.get("status") not in LIVE:
            continue
        ht = ((m.get("score") or {}).get("halfTime") or {})
        if ht.get("home") is None or ht.get("away") is None:
            continue
        kickoff = pd.Timestamp(m["utcDate"]).tz_convert("UTC").tz_localize(None)
        minutes = (now - kickoff).total_seconds() / 60
        if m["status"] == "IN_PLAY" and not (LATE_MIN <= minutes <= STALE_MIN):
            continue                                   # first half still on, or too late
        h, a = m.get("homeTeam") or {}, m.get("awayTeam") or {}
        home = ids.get(h.get("id"), h.get("shortName") or h.get("name"))
        away = ids.get(a.get("id"), a.get("shortName") or a.get("name"))
        rows.append({"league": league, "home_team": home, "away_team": away,
                     "kickoff_utc": kickoff, "ht_home": int(ht["home"]), "ht_away": int(ht["away"]),
                     "status": m["status"], "api_id": m.get("id")})
    return pd.DataFrame(rows)


def attach_forecasts(live: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Match live games to forecasts on league + teams + kickoff within a day."""
    if live.empty:
        return live
    j = live.merge(fc.drop(columns=["kickoff_utc"]).assign(fc_date=fc.date),
                   on=["league", "home_team", "away_team"], how="inner")
    j = j[(j.fc_date - j.kickoff_utc.dt.normalize()).abs() <= pd.Timedelta(days=1)]
    return j.drop_duplicates("api_id")


def update(row) -> dict:
    hint = (row.xg_home, row.xg_away) if pd.notna(row.get("xg_home")) else (1.45, 1.15)
    now = halftime_probs(row.p_home, row.p_away, row.ht_home, row.ht_away, hint=hint)
    return {"now": now, "before": np.array([row.p_home, row.p_draw, row.p_away])}


# ---------------- card ----------------

def headline(names: Names, row, probs: dict, t: dict):
    """The outcome whose chance moved most, as pieces: 'Arsenal win: 42% → 74%'."""
    change = probs["now"] - probs["before"]
    k = int(np.argmax(np.abs(change)))
    label = [TEXT["en"]["win"].format(names.team(row.home_team)), t["draw"],
             TEXT["en"]["win"].format(names.team(row.away_team))][k]
    fill = "#1D6B3A" if change[k] > 0 else "#A3361F"
    return [(label + ":", INK2, True), (pct(probs["before"][k]), INK2, False),
            (ARROW, MUTED, False), (pct(probs["now"][k]), fill, True)]


def halftime_card(row, probs: dict, lang: str = "en", handle: str = "") -> tuple[Image.Image, set]:
    names = Names(lang)
    t, tt = TEXT[lang], T[lang]
    H = 1120
    img = Image.new("RGB", (W, H), SURFACE)
    p = Painter(img, rtl=(lang == "ar"))
    p.rect((40, 40, W - 40, H - 40), CARD, r=36)
    start, end = (90, W - 90) if not p.rtl else (W - 90, 90)
    p.text(start, 130, f"{names.league(row.league)} · {tt['ht']}", 30, bold=True, fill=INK2)
    p.text(end, 130, when(row.fc_date, row.kickoff_utc, lang), 28, fill=INK2, align="end")

    home, away = names.team(row.home_team), names.team(row.away_team)
    side_w = 360
    size = min(p.fit(home, side_w, 56, True), p.fit(away, side_w, 56, True))
    p.text(90 + side_w / 2, 290, home, size, bold=True, align="center")
    p.text(W - 90 - side_w / 2, 290, away, size, bold=True, align="center")
    p.text(W / 2, 300, f"{row.ht_home} – {row.ht_away}", 84, bold=True, align="center")

    p.text(start, 430, tt["now"], 28, bold=True, align="start")
    labels = [t["win"].format(home), t["draw"], t["win"].format(away)]
    labels = [lb if p.width(lb, 24) < 300 else (t["draw"] if i == 1 else lb.split(" ")[0])
              for i, lb in enumerate(labels)]
    outcome_bar(p, 90, 460, W - 180, 110, list(probs["now"]), labels, size=36)

    p.text(start, 700, tt["before"], 24, fill=INK2, align="start")
    outcome_bar(p, 90, 725, W - 180, 56, list(probs["before"]), None, size=22)

    _pieces(p, start, 860, headline(names, row, probs, tt), 30)
    y = H - 190                       # footer: no track-record line (no database in live runs)
    p.d.line((70, y, W - 70, y), fill=RULE, width=2)
    left, right = (70, W - 70) if not p.rtl else (W - 70, 70)
    p.text(left, y + 46, tt["method"], 22, fill=INK2, align="start")
    p.text(left, y + 86, t["disclaimer"], 22, fill=MUTED, align="start")
    if handle:
        p.text(right, y + 86, handle, 22, bold=True, fill=INK2, align="end")
    return img, names.missing


# ---------------- main ----------------

def main() -> None:
    from fa.data.fixtures_api import get_client
    from fa.records import RECORDS
    from fa.telegram import (Bot, TelegramError, already_posted, chats, env, record)
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="really post (default: dry run)")
    ap.add_argument("--now", help="pretend it's this time (UTC), for testing")
    args = ap.parse_args()
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now("UTC").tz_localize(None)

    fc = latest_forecasts(RECORDS / "forecasts.csv")
    if not len(fc[(fc.date >= now.normalize() - pd.Timedelta(days=1)) & (fc.date <= now.normalize() + pd.Timedelta(days=1))]):
        print("No forecast matches today: nothing to do.")
        return
    payload = get_client().get("/matches", dateFrom=f"{now - pd.Timedelta(hours=4):%Y-%m-%d}",
                               dateTo=f"{now:%Y-%m-%d}")
    live = attach_forecasts(at_halftime(payload, team_ids(), now), fc)
    if live.empty:
        print(f"{now:%H:%M} UTC: no forecast match at half-time right now.")
        return

    chat = chats().get("en")
    if not chat:
        raise SystemExit("No Telegram chat set (TELEGRAM_CHAT_EN).")
    token = env("TELEGRAM_BOT_TOKEN")
    if args.send and not token:
        raise SystemExit("No TELEGRAM_BOT_TOKEN set.")
    done = already_posted()
    bot = Bot(token) if args.send else None
    out = REPORTS_DIR / "cards" / f"{now:%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    handle = chat if chat.startswith("@") else ""
    for _, row in live.iterrows():
        key = f"{chat}|halftime/{row.match_id}"
        if key in done:
            continue
        probs = update(row)
        img, _ = halftime_card(row, probs, "en", handle)
        path = out / f"halftime_{row.match_id}_en.png"
        img.save(path)
        caption = T["en"]["caption"].format(home=row.home_team, h=row.ht_home, a=row.ht_away,
                                            away=row.away_team) + "\nAnalysis, not betting advice."
        line = (f"{row.home_team} {row.ht_home}-{row.ht_away} {row.away_team}: "
                f"now {pct(probs['now'][0])} / {pct(probs['now'][1])} / {pct(probs['now'][2])}")
        if not args.send:
            print(f"[dry run] {chat}: {line}")
            continue
        try:
            bot.send(chat, [path], caption)
        except TelegramError as e:
            raise SystemExit(f"Posting stopped: {e}") from None
        record(key)
        print(f"posted: {line}")
        time.sleep(1.5)


if __name__ == "__main__":
    main()
