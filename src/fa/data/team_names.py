"""Match team names between data sources.

football-data.co.uk says "Man City", "Ath Madrid", "M'gladbach"; football-data.org and
openfootball say "Manchester City FC", "Club Atlético de Madrid", "Borussia
Mönchengladbach". Within one league both sides list the same ~20 teams, so instead of
matching names one by one we solve a one-to-one *assignment*: each team is used exactly
once, which makes look-alikes ("Man City" / "Man United") unambiguous.

Manual fixes go in team_aliases.csv next to this file and always win.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ALIASES_PATH = Path(__file__).with_name("team_aliases.csv")
MIN_CONFIDENT = 0.55   # below this, a match is reported for a human to check

# Short forms used by football-data.co.uk, expanded before comparing.
EXPANSIONS = {
    "man": "manchester", "utd": "united", "nottm": "nottingham", "ath": "atletico",
    "ein": "eintracht", "mgladbach": "monchengladbach", "espanol": "espanyol",
    "sg": "saint germain", "st": "saint", "psg": "paris saint germain", "wolves": "wolverhampton",
    "spurs": "tottenham", "inter": "internazionale", "koln": "koln", "leverkusen": "leverkusen",
    "sociedad": "real sociedad", "betis": "real betis", "vallecano": "rayo vallecano",
}
# Words that carry no identity: club-type prefixes, founding years, etc.
STOPWORDS = {
    "fc", "afc", "cf", "ac", "as", "ss", "ssc", "sc", "us", "rc", "rcd", "cd", "ud", "sd",
    "ca", "club", "de", "del", "la", "le", "calcio", "sv", "vfb", "vfl", "tsg", "fsv", "bv",
    "ogc", "aj", "balompie", "futbol", "and", "hove", "albion", "the", "1", "fk", "sad",
}


def normalize(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = []
    for t in s.split():
        if t.isdigit():
            continue                       # 1899, 04, 1846 ...
        tokens.extend(EXPANSIONS.get(t, t).split())
    kept = [t for t in tokens if t not in STOPWORDS]
    return kept or tokens


def similarity(a: str, b: str) -> float:
    """0..1. Combines whole-string similarity with token overlap (prefix-tolerant)."""
    ta, tb = normalize(a), normalize(b)
    if not ta or not tb:
        return 0.0
    whole = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()

    def tok_match(x, y):
        return x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x)))

    hits = sum(any(tok_match(x, y) for y in tb) for x in ta)
    hits_b = sum(any(tok_match(y, x) for x in ta) for y in tb)
    overlap = max(hits / len(ta), hits_b / len(tb)) * min(1.0, min(len(ta), len(tb)) / 1)
    # A full-overlap on one side ("Bayern" in "Bayern Munchen") is strong but not perfect.
    overlap = min(overlap, 0.95) if hits / len(ta) != hits_b / len(tb) else overlap
    return max(whole, overlap)


def load_aliases(path: Path = ALIASES_PATH) -> dict[tuple[str, str], str]:
    """(league, other-source name) -> our name. Lines starting with # are comments."""
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        rows = (r for r in csv.reader(f) if r and not r[0].startswith("#"))
        for league, other, ours in rows:
            out[(league.strip(), other.strip())] = ours.strip()
    return out


def assign(theirs: list[dict], ours: list[str], league: str,
           aliases: dict | None = None) -> list[dict]:
    """Match each of `theirs` ({'id', 'name', 'shortName', ...}) to one of `ours`.

    Returns one dict per team in `theirs`: id, their name, our name (or None), score,
    and how it was matched ('alias', 'auto', 'low-confidence' or 'unmatched').
    """
    aliases = load_aliases() if aliases is None else aliases
    results, free_theirs, used = {}, [], set()
    for t in theirs:
        for key in (t.get("name"), t.get("shortName")):
            if key and (league, key) in aliases:
                results[t["id"]] = (aliases[(league, key)], 1.0, "alias")
                used.add(aliases[(league, key)])
                break
        else:
            free_theirs.append(t)
    free_ours = [o for o in ours if o not in used]

    if free_theirs and free_ours:
        score = np.array([[max(similarity(t.get(k) or "", o) for k in ("name", "shortName", "tla"))
                           for o in free_ours] for t in free_theirs])
        rows, cols = linear_sum_assignment(-score)
        for r, c in zip(rows, cols):
            s = float(score[r, c])
            status = "auto" if s >= MIN_CONFIDENT else "low-confidence"
            results[free_theirs[r]["id"]] = (free_ours[c], s, status)

    out = []
    for t in theirs:
        ours_name, s, status = results.get(t["id"], (None, 0.0, "unmatched"))
        out.append({"league": league, "id": t["id"], "their_name": t.get("name"),
                    "our_name": ours_name, "score": round(s, 3), "status": status})
    return out


def core_tokens(name: str) -> set[str]:
    """Distinctive words only (drops club-type prefixes and 1-2 letter tokens like 'SL')."""
    return {t for t in normalize(name) if len(t) > 2}


def same_club(a: str, b: str) -> bool:
    """True when one name's distinctive words are all contained in the other's:
    'PSV' / 'PSV Eindhoven', 'SL Benfica' / 'Sport Lisboa e Benfica'. Different clubs
    like 'Sparta Praha' / 'Slavia Praha' fail, since neither contains the other."""
    ta, tb = core_tokens(a), core_tokens(b)
    return bool(ta and tb) and (ta <= tb or tb <= ta)
