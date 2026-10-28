"""The whole pipeline in one command, for scheduled runs (GitHub Actions) or by hand.

    python -m fa.daily                 # everything, but post nothing (dry run)
    python -m fa.daily --post          # post what today's schedule says
    python -m fa.daily --post --kinds match,round   # override what gets posted

Every day: download results, refresh forecasts for the next days, score finished
matches. What gets posted depends on the day (Beirut time):

    Monday   : last week's scorecard, who-deserved-it cards, season tables
    Tuesday  : forecasts for the midweek matches
    Friday   : forecasts for the weekend
    other days: nothing (forecasts are still saved, so the track record stays complete)

Season simulations are run on Mondays (so the tables show week-to-week movement).
Each step runs as its own process, exactly like running the commands by hand.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import pandas as pd

from fa.config import ROOT

# Every day: reviews of matches played since the last run. Plus Mon / Tue / Fri specials.
POST_PLAN = {d: "review" for d in range(7)}
POST_PLAN.update({0: "review,scorecard,deserved,table", 1: "review,match,round", 4: "review,match,round"})
FORECAST_DAYS = {1: 4, 4: 4}        # Tuesday covers Tue-Fri, Friday covers Fri-Mon


def run(step: str, *args: str, optional: bool = False) -> bool:
    """Run `python -m fa.<step> ...`. Optional steps may fail without stopping the day."""
    cmd = [sys.executable, "-m", f"fa.{step}", *args]
    print(f"\n=== {' '.join(cmd[2:])}", flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    ok = result.returncode == 0
    print(f"=== {'ok' if ok else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)
    if not ok and not optional:
        raise SystemExit(f"Step fa.{step} failed; stopping (nothing posted).")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true", help="really post to Telegram")
    ap.add_argument("--kinds", help="post these card kinds instead of today's schedule")
    ap.add_argument("--weekday", type=int, help="pretend today is this weekday (0 = Monday)")
    ap.add_argument("--skip-download", action="store_true", help="use the data already downloaded")
    args = ap.parse_args()

    today = pd.Timestamp.now(tz="Asia/Beirut")
    weekday = today.weekday() if args.weekday is None else args.weekday
    kinds = args.kinds if args.kinds is not None else POST_PLAN.get(weekday, "")
    print(f"{today:%A %Y-%m-%d %H:%M} Beirut · posting today: {kinds or 'nothing'}")

    if not args.skip_download:
        run("data.load_results")
        run("data.load_european", optional=True)
    run("predict", "--days", str(FORECAST_DAYS.get(weekday, 8)), optional=True)  # no fixtures is fine
    run("score", optional=True)
    if weekday == 0 or "table" in kinds:
        run("season_sim")
    if not kinds:
        print("\nNothing to post today. Forecasts and scores are saved.")
        return
    run("data.team_news", optional=True)      # Premier League injuries; cards work without it
    run("cards", "--days", str(FORECAST_DAYS.get(weekday, 8)))
    run("telegram", "--kinds", kinds, *(["--send"] if args.post else []))


if __name__ == "__main__":
    main()
