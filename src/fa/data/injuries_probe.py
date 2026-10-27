"""What does the API-Football free plan really give us? Uses ~8 of the 100 daily requests.

    python -m fa.data.injuries_probe

Needs API_FOOTBALL_KEY in .env (free key from dashboard.api-football.com). The key is
never printed.
"""
from __future__ import annotations

import re

import requests

from fa.telegram import env

BASE = "https://v3.football.api-sports.io"
LEAGUES = {"EPL": 39, "LALIGA": 140, "SERIEA": 135, "BUNDESLIGA": 78, "LIGUE1": 61, "UCL": 2}
KEY_SHAPE = re.compile(r"^[A-Za-z0-9]{20,64}$")


class ApiError(Exception):
    pass


def get(key: str, path: str, **params) -> dict:
    try:
        r = requests.get(BASE + path, params=params, headers={"x-apisports-key": key}, timeout=30)
    except requests.RequestException as e:
        raise ApiError(f"network error ({type(e).__name__})") from None
    if r.status_code != 200:
        raise ApiError(f"HTTP {r.status_code}")
    data = r.json()
    errors = data.get("errors")
    if errors:
        msgs = errors.values() if isinstance(errors, dict) else errors
        raise ApiError("; ".join(str(m) for m in msgs))
    return data


def main() -> None:
    key = env("API_FOOTBALL_KEY")
    if not key:
        raise SystemExit("No API_FOOTBALL_KEY in .env (add it with nano, never on the command line).")
    if not KEY_SHAPE.match(key):
        raise SystemExit("API_FOOTBALL_KEY doesn't look like a key (merged lines in .env?). Not shown.")

    try:
        status = get(key, "/status")["response"]          # free: doesn't count against the quota
        sub, req = status.get("subscription", {}), status.get("requests", {})
        print(f"Plan: {sub.get('plan')} · requests today: {req.get('current')}/{req.get('limit_day')}")

        lg = get(key, "/leagues", id=LEAGUES["EPL"], current="true")["response"]
        season = lg[0]["seasons"][0] if lg else {}
        year = season.get("year")
        print(f"Current Premier League season: {year} · coverage flags: "
              f"injuries={season.get('coverage', {}).get('injuries')}, "
              f"lineups={season.get('coverage', {}).get('fixtures', {}).get('lineups')}")
    except ApiError as e:
        raise SystemExit(f"API error: {e}")

    print()
    ok = 0
    for name, lid in LEAGUES.items():
        try:
            rows = get(key, "/injuries", league=lid, season=year)["response"]
        except ApiError as e:
            print(f"{name:<11} ✗ {e}")
            continue
        ok += 1
        latest = max((r["fixture"]["date"] for r in rows), default="-")[:10]
        print(f"{name:<11} ✓ {len(rows)} injury/suspension rows · latest fixture {latest}")
        for r in sorted(rows, key=lambda r: r["fixture"]["date"], reverse=True)[:3]:
            p = r["player"]
            print(f"{'':13}{r['fixture']['date'][:10]}  {r['team']['name']}: {p['name']} "
                  f"({p.get('type')}: {p.get('reason')})")

    print()
    if ok:
        print("Verdict: injuries for the current season ARE available on this plan.")
    else:
        print("Verdict: no current-season injuries on this plan; we'll use news feeds instead.")


if __name__ == "__main__":
    main()
