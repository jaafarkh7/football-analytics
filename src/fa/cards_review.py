"""Post-match review cards: the result next to our pre-match forecast, the result the
chances deserved (from shots on target), and what the matchday changed in the season
simulations. English + Arabic.

Only matches we forecast before kickoff are reviewed, and each match only once (the
posted log in records/posted.csv is checked), so a card can't appear twice when the
shot data arrives a day late.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from fa.cards import (CARD, HOME, AWAY, INK, INK2, MUTED, RULE, SURFACE, TEXT, W, Names,
                      Painter, featured, footer, outcome_bar, pct, when)
from fa.cards_season import T2, deserved

T3 = {
    "en": {"ft": "Full time", "shots": "Shots", "sot": "on target", "forecast": "Our forecast before kickoff",
           "right": "Forecast right", "wrong": "Forecast wrong",
           "deserved": "What the chances deserved",
           "xg": "Expected goals from shots on target: {}",
           "season": "Season outlook, before and after the matchday"},
    "ar": {"ft": "نهاية المباراة", "shots": "التسديدات", "sot": "على المرمى",
           "forecast": "توقعنا قبل المباراة", "right": "التوقع صحيح", "wrong": "التوقع خاطئ",
           "deserved": "ما استحقته الفرص",
           "xg": "الأهداف المتوقعة من التسديدات على المرمى: {}",
           "season": "توقعات الموسم، قبل الجولة وبعدها"},
}
EVENTS = ("p_title", "p_top4", "p_relegation")
EVENT_LABEL = {"p_title": "ev_title", "p_top4": "ev_top4", "p_relegation": "ev_releg"}
RIGHT, WRONG = "#1D6B3A", "#A3361F"
ARROW = "<arrow>"


# ---------------- data ----------------

def review_rows(matches: pd.DataFrame, forecasts: pd.DataFrame, now: pd.Timestamp,
                skip_ids: set[str] = frozenset(), days: int = 3) -> pd.DataFrame:
    """Played matches of the last `days` days that we forecast before kickoff and that
    have shot data, with the forecast, the result and the deserved probabilities."""
    from fa.score import attach_results, eligible_forecasts
    since, until = now.normalize() - pd.Timedelta(days=days), now
    fc = eligible_forecasts(forecasts)
    fc = fc[fc.model == "blend_v1"]
    if fc.empty:
        return fc
    played = attach_results(fc, matches)
    played = played[(played.result_date >= since) & (played.result_date < until)]
    played = played[~played.match_id.isin(skip_ids)]
    if played.empty:
        return played
    d = deserved(matches, since, until)
    if d.empty:
        return d
    d = d[["league", "home_team", "away_team", "date", "home_shots", "away_shots", "home_sot",
           "away_sot", "xg_home", "xg_away", "d_home", "d_draw", "d_away"]].rename(
        columns={"date": "result_date", "xg_home": "sxg_home", "xg_away": "sxg_away"})
    return played.merge(d, on=["league", "home_team", "away_team", "result_date"], how="inner")


def pick(rows: pd.DataFrame, matches: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """The biggest matches, plus the biggest upset if it isn't among them."""
    if len(rows) <= k:
        return rows.sort_values("result_date")
    prob = rows[["p_home", "p_draw", "p_away"]].to_numpy()
    actual = prob[np.arange(len(rows)), rows.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()]
    rows = rows.assign(p_actual_fc=actual)
    top = featured(rows, matches, k)
    upset = rows.loc[rows.p_actual_fc.idxmin()]
    if upset.p_actual_fc < 0.25 and upset.match_id not in set(top.match_id):
        top = pd.concat([top.iloc[:-1], upset.to_frame().T])
    return top.sort_values("result_date")


def season_change(matches: pd.DataFrame, league: str, date: pd.Timestamp,
                  cache: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Season simulations from just before and just after this matchday."""
    from fa.season_sim import simulate
    cache = {} if cache is None else cache
    key = (league, date)
    if key not in cache:
        lg = matches[(matches.league == league) & (matches.date <= date)]
        if lg.empty:
            cache[key] = None
        else:
            season = int(lg.season.max())
            before = simulate(matches, league, season, date).set_index("team")
            after = simulate(matches, league, season, date + pd.Timedelta(days=1)).set_index("team")
            cache[key] = (before, after)
    return cache[key]


def biggest_change(team: str, before: pd.DataFrame, after: pd.DataFrame, min_change: float = 0.03):
    """(event, before, after) with the largest move for this team, if it moved at all."""
    if team not in before.index or team not in after.index:
        return None
    best = None
    for ev in EVENTS:
        b, a = float(before.loc[team, ev]), float(after.loc[team, ev])
        if max(a, b) < 0.03 or abs(a - b) < min_change:
            continue
        if best is None or abs(a - b) > abs(best[2] - best[1]):
            best = (ev, b, a)
    return best


# ---------------- drawing ----------------

def _pieces(p: Painter, x: float, y: float, pieces, size: int):
    """Draw text pieces in reading order (keeps Arabic labels and numbers apart)."""
    step = 1 if not p.rtl else -1
    for s, fill, bold in pieces:
        if s == ARROW:                      # drawn, not typed: the fonts have no arrow glyph
            w = size * 1.3
            x0, x1 = (x, x + w) if not p.rtl else (x - w, x)
            mid = y - size * 0.35
            tip, back = (x1, x1 - size * 0.45) if not p.rtl else (x0, x0 + size * 0.45)
            p.d.line((x0, mid, x1, mid), fill=fill, width=3)
            p.d.polygon([(tip, mid), (back, mid - size * 0.3), (back, mid + size * 0.3)], fill=fill)
            x += step * (w + 12)
            continue
        p.text(x, y, s, size, bold=bold, fill=fill, align="start")
        x += step * (p.width(s, size, bold) + 12)


def _pair(p: Painter, home, away) -> str:
    left, right = (home, away) if not p.rtl else (away, home)
    return f"{int(left)} – {int(right)}"


def review_card(r, lang: str, record=None, handle: str = "", changes: list | None = None
                ) -> tuple[Image.Image, set]:
    names = Names(lang)
    t, t2, t3 = TEXT[lang], T2[lang], T3[lang]
    changes = [c for c in (changes or []) if c]
    H = 1180 + (100 + 60 * len(changes) if changes else 0)
    img = Image.new("RGB", (W, H), SURFACE)
    p = Painter(img, rtl=(lang == "ar"))
    p.rect((40, 40, W - 40, H - 40), CARD, r=36)
    start, end = (90, W - 90) if not p.rtl else (W - 90, 90)

    p.text(start, 130, f"{names.league(r.league)} · {t3['ft']}", 30, bold=True, fill=INK2)
    p.text(end, 130, when(pd.Timestamp(r.result_date), None, lang), 28, fill=INK2, align="end")

    home, away = names.team(r.home_team), names.team(r.away_team)
    side_w = 360
    left, right = (home, away) if not p.rtl else (away, home)
    size = min(p.fit(home, side_w, 56, True), p.fit(away, side_w, 56, True))
    p.text(90 + side_w / 2, 290, left, size, bold=True, align="center")
    p.text(W - 90 - side_w / 2, 290, right, size, bold=True, align="center")
    p.text(W / 2, 300, _pair(p, r.home_goals, r.away_goals), 84, bold=True, align="center")
    shots = (f"{t3['shots']} {_pair(p, r.home_shots, r.away_shots)}   ·   "
             f"{t3['sot']} {_pair(p, r.home_sot, r.away_sot)}")
    if p.rtl:   # Arabic labels and numbers drawn as separate pieces
        y = 380
        pieces = [(t3["shots"], INK2, False), (_pair(p, r.home_shots, r.away_shots), INK, True),
                  ("·", MUTED, False), (t3["sot"], INK2, False),
                  (_pair(p, r.home_sot, r.away_sot), INK, True)]
        w = sum(p.width(s, 26, b) + 12 for s, _, b in pieces) - 12
        _pieces(p, W / 2 + w / 2, y, pieces, 26)
    else:
        p.text(W / 2, 380, shots, 26, fill=INK2, align="center")

    # Forecast vs result.
    probs = [r.p_home, r.p_draw, r.p_away]
    called = "HDA"[int(np.argmax(probs))] == r.result
    p.text(start, 480, t3["forecast"], 26, bold=True, align="start")
    p.text(end, 480, t3["right"] if called else t3["wrong"], 26, bold=True,
           fill=RIGHT if called else WRONG, align="end")
    labels = [t["win"].format(home), t["draw"], t["win"].format(away)]
    labels = [lb if p.width(lb, 24) < 300 else (t["draw"] if i == 1 else lb.split(" ")[0])
              for i, lb in enumerate(labels)]
    outcome_bar(p, 90, 510, W - 180, 90, probs, labels, size=30)

    # Deserved result.
    p.text(start, 740, t3["deserved"], 26, bold=True, align="start")
    outcome_bar(p, 90, 770, W - 180, 90, [r.d_home, r.d_draw, r.d_away], None, size=30)
    lx, rx = (r.sxg_home, r.sxg_away) if not p.rtl else (r.sxg_away, r.sxg_home)
    xg = f"{lx:.1f} – {rx:.1f}"
    if p.rtl:
        _pieces(p, start, 910, [(t3["xg"].format("").rstrip(": ") + ":", INK2, False),
                                (xg, INK, True)], 24)
    else:
        p.text(start, 910, t3["xg"].format(xg), 24, fill=INK2, align="start")
    p.text(start, 950, t2["basis"], 20, fill=MUTED, align="start")

    y = 1000
    if changes:
        p.d.line((70, y, W - 70, y), fill=RULE, width=2)
        p.text(start, y + 50, t3["season"], 26, bold=True, align="start")
        y += 110
        for team, ev, b, a in changes:
            fill = HOME[1] if a > b else AWAY[1]
            if ev == "p_relegation":
                fill = AWAY[1] if a > b else HOME[1]          # relegation going up is bad news
            _pieces(p, start, y, [(names.team(team), INK, True), ("·", MUTED, False),
                                  (t2[EVENT_LABEL[ev]] + ":", INK2, False), (pct(b), INK2, False),
                                  (ARROW, MUTED, False), (pct(a), fill, True)], 26)
            y += 60
        y += 10
    footer(p, H - 190, t, record, handle)
    return img, names.missing


def posted_review_ids(log_path) -> set[str]:
    """Match ids whose review card was already posted (any day)."""
    if not log_path.exists():
        return set()
    ids = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        for part in line.split("|")[1:]:
            name = part.split("/")[-1]
            if name.startswith("review_"):
                ids.add(name[len("review_"):].rsplit("_", 1)[0])
    return ids


def make_reviews(matches: pd.DataFrame, forecasts: pd.DataFrame, now: pd.Timestamp, out,
                 record, handle: dict, log_path, k: int = 6, with_season: bool = True):
    rows = review_rows(matches, forecasts, now, posted_review_ids(log_path))
    made, missing = [], set()
    if rows.empty:
        return made, missing
    rows = pick(rows, matches, k)
    cache = {}
    for _, r in rows.iterrows():
        changes = []
        if with_season:
            sims = season_change(matches, r.league, pd.Timestamp(r.result_date), cache)
            if sims:
                for team in (r.home_team, r.away_team):
                    c = biggest_change(team, *sims)
                    if c:
                        changes.append((team, *c))
        for lang in ("en", "ar"):
            img, miss = review_card(r, lang, record, handle[lang], changes)
            path = out / f"review_{r.match_id}_{lang}.png"
            img.save(path)
            made.append(path)
            missing |= miss
    return made, missing
