"""Shareable PNG cards (English and Arabic) from saved forecasts and results.

    python -m fa.cards                  # upcoming forecasts -> reports/cards/<date>/
    python -m fa.cards --featured 6     # how many single-match cards to make

Cards:
  match_<id>_<lang>.png     one big match: probabilities, predicted score, goal markets
  round_<league>_<lang>.png every forecast match of a competition in one image
  scorecard_<lang>.png      how last week's forecasts did

Arabic is laid out right to left. The Arabic font has no digits or Latin letters, so
every string is split into runs and each run is drawn with the font that has its glyphs.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, features

from fa.config import DB_PATH, ELO_SETTINGS, REPORTS_DIR, ROOT

FONTS = ROOT / "assets" / "fonts"
W = 1080
SURFACE, CARD, INK, INK2, MUTED, RULE = "#f6f5f1", "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
TILE = "#f1efe8"
# Outcome colours: soft fills with dark text of the same hue (readable, colour-blind safe
# thanks to labels and positions; never colour alone).
HOME = ("#B5D4F4", "#0C447C")
DRAW = ("#D3D1C7", "#2C2C2A")
AWAY = ("#F5C4B3", "#712B13")
RAQM = features.check("raqm")
ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

LEAGUE_NAMES = {
    "en": {"EPL": "Premier League", "LALIGA": "La Liga", "SERIEA": "Serie A",
           "BUNDESLIGA": "Bundesliga", "LIGUE1": "Ligue 1", "UCL": "Champions League"},
    "ar": {"EPL": "الدوري الإنجليزي", "LALIGA": "الدوري الإسباني", "SERIEA": "الدوري الإيطالي",
           "BUNDESLIGA": "الدوري الألماني", "LIGUE1": "الدوري الفرنسي", "UCL": "دوري أبطال أوروبا"},
}
TEXT = {
    "en": {"win": "{} win", "draw": "Draw", "score": "Predicted score", "over": "Over 2.5 goals",
           "btts": "Both teams score", "xg": "Expected goals", "vs": "vs",
           "disclaimer": "Analysis, not betting advice", "home": "home", "away": "away",
           "record": "Track record: {}% results right in {} matches",
           "record_new": "Track record starts this week: every forecast is scored",
           "round": "{} · next matches", "last_week": "How last week's forecasts did",
           "right": "results right", "matches": "matches scored", "surprise": "Biggest surprise",
           "model_gave": "model gave {}%", "logloss": "lower is better · 1.10 = pure guessing",
           "logloss_label": "Log loss", "no_goals": "Rating-based forecast, no scoreline model",
           "market": "market", "home_win": "Home win", "away_win": "Away win"},
    "ar": {"win": "فوز {}", "draw": "تعادل", "score": "النتيجة المتوقعة", "over": "أكثر من 2.5 هدف",
           "btts": "تسجيل الفريقين", "xg": "الأهداف المتوقعة", "vs": "ضد",
           "disclaimer": "تحليل وليس نصيحة للمراهنة", "home": "أرضه", "away": "ضيف",
           "record": "سجل التوقعات: {}% نتائج صحيحة في {} مباراة",
           "record_new": "يبدأ سجل التوقعات هذا الأسبوع: كل توقع يُقيَّم",
           "round": "{} · المباريات القادمة", "last_week": "أداء توقعات الأسبوع الماضي",
           "right": "نتائج صحيحة", "matches": "مباريات مقيَّمة", "surprise": "أكبر مفاجأة",
           "model_gave": "النموذج أعطى {}%", "logloss": "الأقل أفضل، والتخمين العشوائي يعطي 1.10",
           "logloss_label": "مقياس الخطأ", "no_goals": "توقع مبني على تصنيف الفرق، دون نموذج للنتيجة",
           "market": "السوق", "home_win": "فوز صاحب الأرض", "away_win": "فوز الضيف"},
}
AR_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس",
             "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
AR_DAYS = ["الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]


# ---------------- text with mixed scripts ----------------

_font_cache: dict = {}


def font(arabic: bool, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = ("NotoSansArabic" if arabic else "NotoSans") + ("-Bold" if bold else "-Regular") + ".ttf"
    key = (name, size)
    if key not in _font_cache:
        layout = ImageFont.Layout.RAQM if RAQM else ImageFont.Layout.BASIC
        _font_cache[key] = ImageFont.truetype(str(FONTS / name), size, layout_engine=layout)
    return _font_cache[key]


def _shape(text: str) -> str:
    """Without raqm, pre-shape Arabic and reverse it for display (needs two small packages)."""
    if RAQM:
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


_AR_GLYPHS: set[int] | None = None


def _arabic_has(ch: str) -> bool:
    """Does the Arabic font contain this character? (It lacks digits, Latin and brackets.)"""
    global _AR_GLYPHS
    if _AR_GLYPHS is None:
        from fontTools.ttLib import TTFont
        _AR_GLYPHS = set(TTFont(str(FONTS / "NotoSansArabic-Regular.ttf")).getBestCmap())
    return ord(ch) in _AR_GLYPHS


def runs(text: str) -> list[tuple[str, bool]]:
    """Split into (chunk, is_arabic) word by word. A word goes to the Arabic font only if
    that font has every one of its characters, so nothing can render as an empty box.
    Consecutive words in the same font form one run; the painter puts spaces between runs."""
    out: list[tuple[str, bool]] = []
    for word in text.split():
        is_ar = bool(ARABIC.search(word)) and all(_arabic_has(c) for c in word)
        if out and out[-1][1] == is_ar:
            out[-1] = (out[-1][0] + " " + word, is_ar)
        else:
            out.append((word, is_ar))
    return out


@dataclass
class Painter:
    img: Image.Image
    rtl: bool

    def __post_init__(self):
        self.d = ImageDraw.Draw(self.img)

    def _draw_run(self, x, y, t, is_ar, size, bold, fill):
        f = font(is_ar, size, bold)
        if is_ar:
            kw = {"direction": "rtl", "language": "ar"} if RAQM else {}
            self.d.text((x, y), _shape(t), font=f, fill=fill, anchor="ls", **kw)
            return self.d.textlength(_shape(t), font=f, **kw)
        self.d.text((x, y), t, font=f, fill=fill, anchor="ls")
        return self.d.textlength(t, font=f)

    def _space(self, size: int, bold: bool) -> float:
        return self.d.textlength(" ", font=font(False, size, bold))

    def width(self, text: str, size: int, bold: bool = False) -> float:
        parts = runs(text)
        total = self._space(size, bold) * max(len(parts) - 1, 0)   # one space between runs
        for t, is_ar in parts:
            f = font(is_ar, size, bold)
            kw = {"direction": "rtl", "language": "ar"} if (is_ar and RAQM) else {}
            total += self.d.textlength(_shape(t) if is_ar else t, font=f, **kw)
        return total

    def text(self, x: float, y: float, text: str, size: int, bold=False, fill=INK, align="start"):
        """Draw at baseline y. align: start / end / center, relative to reading direction."""
        w = self.width(text, size, bold)
        rtl = self.rtl and bool(ARABIC.search(text))
        if align == "center":
            left = x - w / 2
        elif (align == "start") != self.rtl:      # start in LTR, or end in RTL -> x is the left edge
            left = x
        else:
            left = x - w
        parts = runs(text)
        if rtl:
            parts = parts[::-1]                    # rightmost run is read first
        cur, space = left, self._space(size, bold)
        for t, is_ar in parts:
            cur += self._draw_run(cur, y, t, is_ar, size, bold, fill) + space
        return w

    def fit(self, text: str, max_w: float, size: int, bold=False, min_size=26) -> int:
        while size > min_size and self.width(text, size, bold) > max_w:
            size -= 2
        return size

    def rect(self, box, fill, r=18):
        self.d.rounded_rectangle(box, r, fill=fill)


# ---------------- names, dates, numbers ----------------

def load_arabic_names(path: Path = Path(__file__).parent / "data" / "team_names_ar.csv") -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        rows = csv.reader(line for line in f if not line.startswith("#"))
        next(rows, None)
        return {r[0].strip(): r[1].strip() for r in rows if len(r) >= 2}


class Names:
    def __init__(self, lang: str):
        self.lang = lang
        self.ar = load_arabic_names() if lang == "ar" else {}
        self.missing: set[str] = set()

    def team(self, name: str) -> str:
        if self.lang != "ar":
            return name
        if name not in self.ar:
            self.missing.add(name)
        return self.ar.get(name, name)

    def league(self, code: str) -> str:
        return LEAGUE_NAMES[self.lang].get(code, code)


def when(ts: pd.Timestamp, kickoff_utc, lang: str) -> str:
    """Local kickoff in Beirut time (the audience), or the date if no time is known."""
    if pd.notna(kickoff_utc):
        t = pd.Timestamp(kickoff_utc).tz_localize("UTC").tz_convert("Asia/Beirut")
        hhmm = t.strftime("%H:%M")
    else:
        t, hhmm = pd.Timestamp(ts), ""
    if lang == "ar":
        return f"{AR_DAYS[t.weekday()]} {t.day} {AR_MONTHS[t.month - 1]} {hhmm}".strip()
    return f"{t:%a %-d %b} {hhmm}".strip()


def pct(x: float) -> str:
    return f"{int(round(100 * x))}%"


# ---------------- card pieces ----------------

def outcome_bar(p: Painter, x0, y0, width, height, probs, labels, size=34):
    """Home / draw / away bar. In RTL the home segment sits on the right."""
    order = [0, 1, 2] if not p.rtl else [2, 1, 0]
    colors = [HOME, DRAW, AWAY]
    gap = 6
    usable = width - 2 * gap
    widths = [max(usable * probs[i], 70) for i in range(3)]
    scale = usable / sum(widths)
    widths = [w * scale for w in widths]
    x = x0
    for k, i in enumerate(order):
        w = widths[i]
        fill, ink = colors[i]
        p.rect((x, y0, x + w, y0 + height), fill, r=12)
        p.text(x + w / 2, y0 + height / 2 + size * 0.36, pct(probs[i]), size, bold=True,
               fill=ink, align="center")
        x += w + gap
    if labels:                                  # fixed thirds, so narrow segments can't collide
        col = width / 3
        for k, i in enumerate(order):
            cx = x0 + col * k + col / 2
            lw = p.width(labels[i], 24)
            sx = cx - lw / 2 - 28 if not p.rtl else cx + lw / 2 + 8   # swatch where reading starts
            p.rect((sx, y0 + height + 20, sx + 20, y0 + height + 40), colors[i][0], r=5)
            p.text(cx + 0, y0 + height + 40, labels[i], 24, fill=INK2, align="center")


def tile(p: Painter, box, label, value):
    p.rect(box, TILE, r=16)
    cx = (box[0] + box[2]) / 2
    p.text(cx, box[1] + 50, label, 24, fill=INK2, align="center")
    p.text(cx, box[1] + 118, value, 52, bold=True, align="center")


def footer(p: Painter, y, t: dict, record: tuple[int, int] | None, handle: str):
    p.d.line((70, y, W - 70, y), fill=RULE, width=2)
    left, right = (70, W - 70) if not p.rtl else (W - 70, 70)
    line = (t["record"].format(int(round(100 * record[0] / record[1])), record[1])
            if record and record[1] >= 30 else t["record_new"])
    p.text(left, y + 46, line, 24, fill=INK2, align="start")
    p.text(left, y + 86, t["disclaimer"], 22, fill=MUTED, align="start")
    if handle:
        p.text(right, y + 86, handle, 22, bold=True, fill=INK2, align="end")


# ---------------- cards ----------------

def match_card(row: pd.Series, lang: str, record=None, handle="") -> tuple[Image.Image, set]:
    names = Names(lang)
    t = TEXT[lang]
    has_goals = pd.notna(row.get("p_over25"))
    H = 1240 if has_goals else 900
    img = Image.new("RGB", (W, H), SURFACE)
    p = Painter(img, rtl=(lang == "ar"))
    p.rect((40, 40, W - 40, H - 40), CARD, r=36)
    start, end = (90, W - 90) if not p.rtl else (W - 90, 90)

    p.text(start, 130, names.league(row.league), 30, bold=True, fill=INK2)
    p.text(end, 130, when(row.date, row.get("kickoff_utc"), lang), 28, fill=INK2, align="end")

    home, away = names.team(row.home_team), names.team(row.away_team)
    side_w = 420
    hs, as_ = p.fit(home, side_w, 64, True), p.fit(away, side_w, 64, True)
    size = min(hs, as_)
    left_name, right_name = (home, away) if not p.rtl else (away, home)
    left_tag, right_tag = (t["home"], t["away"]) if not p.rtl else (t["away"], t["home"])
    p.text(90 + side_w / 2, 300, left_name, size, bold=True, align="center")
    p.text(W - 90 - side_w / 2, 300, right_name, size, bold=True, align="center")
    p.text(90 + side_w / 2, 350, left_tag, 24, fill=MUTED, align="center")
    p.text(W - 90 - side_w / 2, 350, right_tag, 24, fill=MUTED, align="center")
    p.text(W / 2, 300, t["vs"], 30, fill=MUTED, align="center")

    probs = [row.p_home, row.p_draw, row.p_away]
    labels = [t["win"].format(home), t["draw"], t["win"].format(away)]
    labels = [lb if p.width(lb, 24) < 300 else (t["draw"] if i == 1 else lb.split(" ")[0])
              for i, lb in enumerate(labels)]
    outcome_bar(p, 90, 430, W - 180, 110, probs, labels)

    if not has_goals:
        p.text(W / 2, 650, t["no_goals"], 24, fill=MUTED, align="center")
        footer(p, H - 190, t, record, handle)
        return img, names.missing

    tw = (W - 180 - 2 * 24) / 3
    score = row.top_score if isinstance(row.get("top_score"), str) else "–"
    if p.rtl and "-" in score:          # away team is on the left in Arabic: its goals go left
        score = "-".join(score.split("-")[::-1])
    tiles = [(t["score"], score)]
    if has_goals:
        tiles += [(t["over"], pct(row.p_over25)), (t["btts"], pct(row.p_btts))]
    if p.rtl:
        tiles = tiles[::-1] if len(tiles) == 3 else tiles
    for k, (label, value) in enumerate(tiles):
        x = 90 + k * (tw + 24)
        tile(p, (x, 660, x + tw, 820), label, value)

    if pd.notna(row.get("xg_home")):
        xg = (f"{row.xg_home:.1f} – {row.xg_away:.1f}" if not p.rtl
              else f"{row.xg_away:.1f} – {row.xg_home:.1f}")
        p.text(W / 2, 910, t["xg"], 26, fill=INK2, align="center")
        p.text(W / 2, 975, xg, 44, bold=True, align="center")

    footer(p, 1050, t, record, handle)
    return img, names.missing


def round_card(rows: pd.DataFrame, league: str, lang: str, record=None, handle="") -> tuple[Image.Image, set]:
    names = Names(lang)
    t = TEXT[lang]
    n = len(rows)
    row_h = 118
    H = 260 + n * row_h + 190
    img = Image.new("RGB", (W, H), SURFACE)
    p = Painter(img, rtl=(lang == "ar"))
    p.rect((40, 40, W - 40, H - 40), CARD, r=36)
    start = 90 if not p.rtl else W - 90
    p.text(start, 140, t["round"].format(names.league(league)), 40, bold=True)
    # Legend: colour swatch + label for each outcome, in reading order.
    items = [(HOME, t["home_win"]), (DRAW, t["draw"]), (AWAY, t["away_win"])]
    x = 90 if not p.rtl else W - 90
    for (fill, _), label in items:
        sw = 26
        box = (x, 168, x + sw, 194) if not p.rtl else (x - sw, 168, x, 194)
        p.rect(box, fill, r=6)
        tx = x + sw + 10 if not p.rtl else x - sw - 10
        w = p.text(tx, 190, label, 22, fill=INK2, align="start")
        x = tx + w + 30 if not p.rtl else tx - w - 30
    y = 240
    name_w = 300
    for _, r in rows.iterrows():
        home, away = names.team(r.home_team), names.team(r.away_team)
        left_name, right_name = (home, away) if not p.rtl else (away, home)
        ls = p.fit(left_name, name_w, 30, True, 20)
        rs = p.fit(right_name, name_w, 30, True, 20)
        p.text(90, y + 40, left_name, ls, bold=True, align="start" if not p.rtl else "end")
        p.text(W - 90, y + 40, right_name, rs, bold=True, align="end" if not p.rtl else "start")
        p.text(90 if not p.rtl else W - 90, y + 80, when(r.date, r.get("kickoff_utc"), lang), 20,
               fill=MUTED, align="start")
        outcome_bar(p, 90 + name_w + 20, y + 4, W - 180 - 2 * (name_w + 20), 56,
                    [r.p_home, r.p_draw, r.p_away], None, size=22)
        p.d.line((90, y + row_h - 14, W - 90, y + row_h - 14), fill=RULE, width=1)
        y += row_h
    footer(p, H - 170, t, record, handle)
    return img, names.missing


def scorecard(scored: pd.DataFrame, lang: str, record=None, handle="") -> tuple[Image.Image, set]:
    from fa.eval.metrics import log_loss
    names = Names(lang)
    t = TEXT[lang]
    img = Image.new("RGB", (W, 1080), SURFACE)
    p = Painter(img, rtl=(lang == "ar"))
    p.rect((40, 40, W - 40, 1040), CARD, r=36)
    start = 90 if not p.rtl else W - 90
    p.text(start, 150, t["last_week"], 44, bold=True)

    P = scored[["p_home", "p_draw", "p_away"]].to_numpy()
    right = int((np.array(list("HDA"))[P.argmax(axis=1)] == scored.result.to_numpy()).sum())
    tw = (W - 180 - 24) / 2
    boxes = [(90, 220, 90 + tw, 400), (90 + tw + 24, 220, W - 90, 400)]
    if p.rtl:
        boxes = boxes[::-1]
    tile(p, boxes[0], t["right"], f"{right} / {len(scored)}")
    tile(p, boxes[1], t["logloss_label"], f"{log_loss(P, scored.result):.3f}")
    p.text(W / 2, 450, t["logloss"], 22, fill=MUTED, align="center")

    idx = {"H": 0, "D": 1, "A": 2}
    gave = P[np.arange(len(P)), scored.result.map(idx).to_numpy()]
    s = scored.iloc[int(np.argmin(gave))]
    p.text(start, 560, t["surprise"], 30, bold=True, fill=INK2)
    home, away = names.team(s.home_team), names.team(s.away_team)
    hg, ag = int(s.home_goals), int(s.away_goals)
    # In Arabic the home team is read first, on the right, so the right-hand number is its goals.
    line = f"{home} {hg}-{ag} {away}" if not p.rtl else f"{home} {ag}-{hg} {away}"
    p.text(start, 640, line, p.fit(line, W - 180, 44, True), bold=True)
    p.text(start, 700, t["model_gave"].format(int(round(100 * gave.min()))), 28, fill=INK2)
    footer(p, 880, t, record, handle)
    return img, names.missing


# ---------------- selecting what to show ----------------

def latest_upcoming(con, now: pd.Timestamp) -> pd.DataFrame:
    fc = con.execute("SELECT * FROM forecasts").df()
    fc["date"] = pd.to_datetime(fc["date"])
    if "kickoff_utc" not in fc:
        fc["kickoff_utc"] = pd.NaT
    upcoming = fc[(fc.kickoff_utc > now) | (fc.kickoff_utc.isna() & (fc.date >= now.normalize()))]
    return (upcoming.sort_values("created_at").groupby(["model", "match_id"]).tail(1)
            .sort_values(["date", "kickoff_utc"]).reset_index(drop=True))


def featured(fc: pd.DataFrame, matches: pd.DataFrame, k: int) -> pd.DataFrame:
    """The k biggest matches: both clubs highly rated (the weaker club's rating decides)."""
    from fa.models.elo import EloParams, run_elo
    rated = run_elo(matches, EloParams(**ELO_SETTINGS))
    ratings = {team: r for (_, team), r in rated.attrs["final_ratings"].items()}
    fc = fc.assign(importance=[min(ratings.get(h, 1400), ratings.get(a, 1400))
                               for h, a in zip(fc.home_team, fc.away_team)])
    return fc.nlargest(k, "importance").sort_values(["date", "kickoff_utc"])


def track_record(con) -> tuple[int, int] | None:
    from fa.score import attach_results, eligible_forecasts
    try:
        fc = con.execute("SELECT * FROM forecasts").df()
    except duckdb.CatalogException:
        return None
    res = con.execute("SELECT * FROM matches").df()
    res["date"] = pd.to_datetime(res["date"])
    s = attach_results(eligible_forecasts(fc), res)
    if s.empty:
        return None
    P = s[["p_home", "p_draw", "p_away"]].to_numpy()
    right = int((np.array(list("HDA"))[P.argmax(axis=1)] == s.result.to_numpy()).sum())
    return right, len(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--featured", type=int, default=6)
    ap.add_argument("--handle", default="", help="channel name printed on all cards, e.g. @mychannel")
    ap.add_argument("--handle-en", default="", help="channel name on English cards")
    ap.add_argument("--handle-ar", default="", help="channel name on Arabic cards")
    ap.add_argument("--now", help="pretend it's this time (UTC), for testing")
    ap.add_argument("--days", type=float, default=8, help="only matches kicking off in the next N days")
    args = ap.parse_args()
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now("UTC").tz_localize(None)
    from fa.records import sync_in
    sync_in()
    from fa.telegram import chats
    channels = chats()          # TELEGRAM_CHAT / TELEGRAM_CHAT_EN / TELEGRAM_CHAT_AR from .env
    handle = {lang: (getattr(args, f"handle_{lang}") or args.handle
                     or (channels.get(lang, "") if channels.get(lang, "").startswith("@") else ""))
              for lang in ("en", "ar")}

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        fc = latest_upcoming(con, now)
        until = now + pd.Timedelta(days=args.days)
        fc = fc[(fc.kickoff_utc < until) | (fc.kickoff_utc.isna() & (fc.date < until.normalize()))]
    except duckdb.CatalogException:
        raise SystemExit("No forecasts yet: run `python -m fa.predict` first.")
    matches = con.execute("SELECT * FROM matches ORDER BY date").df()
    record = track_record(con)
    from fa.score import attach_results, eligible_forecasts
    all_fc = con.execute("SELECT * FROM forecasts").df()
    con.close()

    out = REPORTS_DIR / "cards" / f"{now:%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):   # an earlier run today (other --days) must not leak in
        old.unlink()
    made, missing = [], set()
    if fc.empty:
        print("No upcoming forecasts to draw.")
    else:
        league_fc = fc[fc.model == "blend_v1"]
        top = featured(league_fc, matches, args.featured) if len(league_fc) else league_fc
        for lang in ("en", "ar"):
            for _, row in top.iterrows():
                img, miss = match_card(row, lang, record, handle[lang])
                path = out / f"match_{row.match_id}_{lang}.png"
                img.save(path)
                made.append(path)
                missing |= miss
            for league, rows in fc.groupby("league", sort=False):
                img, miss = round_card(rows.head(12), league, lang, record, handle[lang])
                path = out / f"round_{league}_{lang}.png"
                img.save(path)
                made.append(path)
                missing |= miss

    # Scorecard: forecasts scored in the last 7 days.
    res = matches.assign(date=pd.to_datetime(matches.date))
    scored = attach_results(eligible_forecasts(all_fc), res)
    week = scored[scored.result_date > now - pd.Timedelta(days=7)] if len(scored) else scored
    if len(week) >= 5:
        for lang in ("en", "ar"):
            img, miss = scorecard(week, lang, record, handle[lang])
            path = out / f"scorecard_{lang}.png"
            img.save(path)
            made.append(path)
            missing |= miss

    # Season simulation tables (from the latest `python -m fa.season_sim` run).
    from fa.cards_season import deserved, deserved_card, table_card
    from fa.season_sim import previous_snapshot
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        sims = con.execute("SELECT * FROM season_sims").df()
    except duckdb.CatalogException:
        sims = pd.DataFrame()
    con.close()
    if len(sims):
        sims["as_of"] = pd.to_datetime(sims.as_of)
        for league, g in sims.groupby("league"):
            latest = g[g.as_of == g.as_of.max()]
            latest = latest[latest.created_at == latest.created_at.max()]
            latest = latest.sort_values(["exp_points", "points"], ascending=False)
            prev = previous_snapshot(league, int(latest.season.iloc[0]), latest.as_of.iloc[0])
            for lang in ("en", "ar"):
                img, miss = table_card(latest, league, lang, prev, handle[lang])
                path = out / f"table_{league}_{lang}.png"
                img.save(path)
                made.append(path)
                missing |= miss

    # Who deserved it: matches of the last 4 days that have shots data.
    recent = deserved(res, now.normalize() - pd.Timedelta(days=4), now)
    for league, g in (recent.groupby("league") if len(recent) else []):
        g = g.sort_values("date").tail(12)
        for lang in ("en", "ar"):
            img, miss = deserved_card(g, league, lang, handle[lang])
            path = out / f"deserved_{league}_{lang}.png"
            img.save(path)
            made.append(path)
            missing |= miss

    print(f"{len(made)} cards in {out}")
    if missing:
        print("No Arabic name yet (shown in English) for: " + ", ".join(sorted(missing))
              + "\nAdd them to src/fa/data/team_names_ar.csv")


if __name__ == "__main__":
    main()
