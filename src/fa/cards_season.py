"""Season-simulation table cards and post-match "who deserved it" cards (English + Arabic)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import poisson

from fa.cards import (CARD, INK, INK2, MUTED, RULE, SURFACE, TEXT, W, Names, Painter,
                      footer, outcome_bar)

T2 = {
    "en": {"sim_title": "{}: how the season could end", "sim_sub": "10,000 simulations of the remaining matches",
           "team": "Team", "pts": "Pts", "exp": "Proj.", "title": "Title", "top4": "Top 4",
           "releg": "Down", "movers": "Biggest movers since last week",
           "ev_title": "title", "ev_top4": "top 4", "ev_releg": "relegation",
           "deserved_title": "{}: who deserved it?",
           "deserved_sub": "Chances from shots on target. Blue = home deserved the win.",
           "sot": "on target", "lucky": "Luckiest win", "basis": "Shots-on-target estimate, cruder than full xG"},
    "ar": {"sim_title": "{}: كيف قد ينتهي الموسم", "sim_sub": "10,000 محاكاة للمباريات المتبقية",
           "team": "الفريق", "pts": "النقاط", "exp": "المتوقع", "title": "اللقب", "top4": "أول 4",
           "releg": "الهبوط", "movers": "أكبر التغيرات منذ الأسبوع الماضي",
           "ev_title": "اللقب", "ev_top4": "المراكز الأربعة الأولى", "ev_releg": "الهبوط",
           "deserved_title": "{}: من استحق الفوز؟",
           "deserved_sub": "الفرص محسوبة من التسديدات على المرمى. الأزرق: صاحب الأرض استحق الفوز.",
           "sot": "على المرمى", "lucky": "أكثر فوز محظوظ", "basis": "تقدير من التسديدات على المرمى، أقل دقة من xG"},
}


def fmt_p(p: float) -> str:
    if p <= 0:
        return "–"
    if p < 0.01:
        return "<1%"
    if p > 0.99 and p < 1:
        return ">99%"
    return f"{int(round(100 * p))}%"


def table_card(sim: pd.DataFrame, league: str, lang: str, prev: pd.DataFrame | None = None,
               handle: str = "") -> tuple[Image.Image, set]:
    names = Names(lang)
    t, t2 = TEXT[lang], T2[lang]
    n = len(sim)
    row_h = 56
    H = 300 + n * row_h + (440 if prev is not None else 190)
    img = Image.new("RGB", (W, H), SURFACE)
    p = Painter(img, rtl=(lang == "ar"))
    p.rect((40, 40, W - 40, H - 40), CARD, r=36)
    start = 90 if not p.rtl else W - 90
    p.text(start, 140, t2["sim_title"].format(names.league(league)), 38, bold=True)
    p.text(start, 186, t2["sim_sub"], 24, fill=INK2)

    # Column x-centres, written left-to-right for English and mirrored for Arabic.
    cols = {"rank": 115, "team": 150, "pts": 560, "exp": 650, "title": 760, "top4": 870, "releg": 975}

    def X(key):
        return cols[key] if not p.rtl else W - cols[key]

    y = 250
    for key in ["pts", "exp", "title", "top4", "releg"]:
        p.text(X(key), y, t2[key], 22, fill=MUTED, align="center")
    p.text(X("team"), y, t2["team"], 22, fill=MUTED, align="start")
    y += 20
    p.d.line((90, y, W - 90, y), fill=RULE, width=2)

    for i, r in enumerate(sim.itertuples(), start=1):
        y += row_h
        if i % 2 == 0:
            p.d.rectangle((70, y - row_h + 12, W - 70, y + 12), fill="#faf9f6")
        p.text(X("rank"), y, str(i), 22, fill=MUTED, align="center")
        team = names.team(r.team)
        p.text(X("team"), y, team, p.fit(team, 360, 26, True, 18), bold=True, align="start")
        p.text(X("pts"), y, str(int(r.points)), 24, fill=INK2, align="center")
        p.text(X("exp"), y, f"{r.exp_points:.0f}", 24, align="center")
        for key, col in (("title", "p_title"), ("top4", "p_top4"), ("releg", "p_relegation")):
            val = getattr(r, col)
            p.text(X(key), y, fmt_p(val), 24, bold=val >= 0.5,
                   fill=INK if val > 0 else MUTED, align="center")
    movers = biggest_movers(sim, prev) if prev is not None else []
    if movers:
        y += 70
        p.text(start, y, t2["movers"], 24, bold=True, fill=INK2)
        for k, (team, ev, delta) in enumerate(movers):
            # Phrase first, then the change as its own piece, so digits never sit inside
            # an Arabic phrase (where bidirectional text would reorder them).
            yy = y + 44 * (k + 1)
            w = p.text(start, yy, f"{names.team(team)} · {t2['ev_' + ev]}", 24, fill=INK2)
            x = start + w + 16 if not p.rtl else start - w - 16
            p.text(x, yy, f"{'+' if delta > 0 else '−'}{abs(delta):.0f}", 24, bold=True,
                   fill=INK, align="start")
    footer(p, H - 170, t, None, handle)
    return img, names.missing


def biggest_movers(sim: pd.DataFrame, prev: pd.DataFrame, k: int = 3, min_change: float = 5):
    """Largest changes (in percentage points) in title, top-4 or relegation chances."""
    rows = []
    for r in sim.itertuples():
        if r.team not in prev.index:
            continue
        for ev, col in (("title", "p_title"), ("top4", "p_top4"), ("releg", "p_relegation")):
            d = 100 * (getattr(r, col) - prev.loc[r.team, col])
            if abs(d) >= min_change:
                rows.append((r.team, ev, d))
    rows.sort(key=lambda x: -abs(x[2]))
    seen, out = set(), []
    for team, ev, d in rows:                    # one line per team, its biggest change
        if team not in seen:
            seen.add(team)
            out.append((team, ev, d))
    return out[:k]


# ---------------- who deserved it ----------------

def conversion_rate(matches: pd.DataFrame, league: str, before: pd.Timestamp) -> float:
    """Goals per shot on target in this league over roughly the last two seasons."""
    m = matches[(matches.league == league) & (matches.date < before) &
                (matches.date >= before - pd.Timedelta(days=730))].dropna(subset=["home_sot", "away_sot"])
    sot = (m.home_sot + m.away_sot).sum()
    return float((m.home_goals + m.away_goals).sum() / sot) if sot else 0.33


def deserved(matches: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """Recent matches with shots: P(home/draw/away) if goals had followed chances."""
    m = matches[(matches.date >= since) & (matches.date < until)].dropna(subset=["home_sot", "away_sot"]).copy()
    if m.empty:
        return m
    g = np.arange(11)
    rows = []
    for r in m.itertuples():
        c = conversion_rate(matches, r.league, r.date)
        lh, la = max(c * r.home_sot, 0.05), max(c * r.away_sot, 0.05)
        mat = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        mat /= mat.sum()
        rows.append({"xg_home": lh, "xg_away": la, "d_home": np.tril(mat, -1).sum(),
                     "d_draw": np.trace(mat), "d_away": np.triu(mat, 1).sum()})
    out = pd.concat([m.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    idx = out.result.map({"H": "d_home", "D": "d_draw", "A": "d_away"})
    out["p_actual"] = [out.loc[i, c] for i, c in enumerate(idx)]     # how likely the real result was
    return out


def luckiest_win(rows: pd.DataFrame):
    """The winner whose chances said it least deserved to win (None if no match had a winner)."""
    wins = rows[rows.result != "D"]
    if wins.empty:
        return None
    return wins.loc[wins.p_actual.idxmin()]


def deserved_card(rows: pd.DataFrame, league: str, lang: str, handle: str = "") -> tuple[Image.Image, set]:
    names = Names(lang)
    t, t2 = TEXT[lang], T2[lang]
    n = len(rows)
    row_h = 128
    lucky = luckiest_win(rows)
    H = 280 + n * row_h + (330 if lucky is not None else 240)
    img = Image.new("RGB", (W, H), SURFACE)
    p = Painter(img, rtl=(lang == "ar"))
    p.rect((40, 40, W - 40, H - 40), CARD, r=36)
    start = 90 if not p.rtl else W - 90
    p.text(start, 140, t2["deserved_title"].format(names.league(league)), 38, bold=True)
    p.text(start, 186, t2["deserved_sub"], 22, fill=INK2)
    y = 240
    name_w = 260
    for r in rows.itertuples():
        home, away = names.team(r.home_team), names.team(r.away_team)
        left, right = (home, away) if not p.rtl else (away, home)
        lg, rg = (r.home_goals, r.away_goals) if not p.rtl else (r.away_goals, r.home_goals)
        ls, rs = (r.home_sot, r.away_sot) if not p.rtl else (r.away_sot, r.home_sot)
        p.text(90, y + 40, left, p.fit(left, name_w, 28, True, 18), bold=True,
               align="start" if not p.rtl else "end")
        p.text(W - 90, y + 40, right, p.fit(right, name_w, 28, True, 18), bold=True,
               align="end" if not p.rtl else "start")
        p.text(W / 2, y + 42, f"{int(lg)} - {int(rg)}", 34, bold=True, align="center")
        p.text(90, y + 80, f"{int(ls)} {t2['sot']}", 20, fill=MUTED,
               align="start" if not p.rtl else "end")
        p.text(W - 90, y + 80, f"{int(rs)} {t2['sot']}", 20, fill=MUTED,
               align="end" if not p.rtl else "start")
        outcome_bar(p, W / 2 - 170, y + 62, 340, 36, [r.d_home, r.d_draw, r.d_away], None, size=18)
        p.d.line((90, y + row_h - 12, W - 90, y + row_h - 12), fill=RULE, width=1)
        y += row_h
    if lucky is not None:
        lh, la = names.team(lucky.home_team), names.team(lucky.away_team)
        hg, ag = int(lucky.home_goals), int(lucky.away_goals)
        line = f"{lh} {hg}-{ag} {la}" if not p.rtl else f"{lh} {ag}-{hg} {la}"
        p.text(start, y + 50, t2["lucky"], 24, bold=True, fill=INK2)
        p.text(start, y + 92, line, 28, bold=True)
    p.text(W / 2, H - 196, t2["basis"], 20, fill=MUTED, align="center")
    footer(p, H - 170, t, None, handle)
    return img, names.missing
