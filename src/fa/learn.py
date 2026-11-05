"""Learning from results: a weekly self-check that adjusts how the forecast is put together.

    python -m fa.learn          # replay the last year, maybe adopt new settings (~1-2 min)

Team strengths already update every day from the latest results. This goes one step
further and learns from the model's own mistakes. It replays the last year of league
matches week by week, exactly as the bot would have forecast them (only data from before
each match), and searches three settings:

  blend_weight_dc   share of the goals model (Dixon–Coles) vs the ratings model (Elo)
  temperature       above 1: sharper, more confident forecasts; below 1: more cautious
  draw_boost        +0.05 makes draws 5% likelier before renormalising

The search uses the older part of the year. New settings are adopted only if they also
beat the current ones on the most recent 8 weeks, which the search never saw. Every
decision is logged in records/learning_log.csv and reports/learning.md; the settings in
use live in records/learned.json and are read by fa.predict.
"""
from __future__ import annotations

import itertools
import json

import duckdb
import numpy as np
import pandas as pd

from fa.config import BLEND_WEIGHT_DC, DB_PATH, REPORTS_DIR, ROOT

SETTINGS_PATH = ROOT / "records" / "learned.json"
LOG_PATH = ROOT / "records" / "learning_log.csv"
DEFAULT = {"blend_weight_dc": BLEND_WEIGHT_DC, "temperature": 1.0, "draw_boost": 0.0}
REPLAY_DAYS, CHECK_DAYS = 365, 56
MIN_GAIN = 0.0005                 # log loss: smaller "gains" are noise, not learning
GRID = {"blend_weight_dc": np.round(np.arange(0.30, 0.81, 0.05), 2),
        "temperature": np.round(np.arange(0.85, 1.16, 0.05), 2),
        "draw_boost": np.round(np.arange(-0.10, 0.151, 0.025), 3)}


def load_settings(path=None) -> dict:
    path = path or SETTINGS_PATH
    if not path.exists():
        return dict(DEFAULT)
    saved = json.loads(path.read_text())
    return {k: float(saved.get(k, v)) for k, v in DEFAULT.items()}


def combine(p_dc: np.ndarray, p_elo: np.ndarray, s: dict) -> np.ndarray:
    """Blend the two models, then apply the learned confidence and draw adjustments."""
    p = s["blend_weight_dc"] * p_dc + (1 - s["blend_weight_dc"]) * p_elo
    p = np.clip(p, 1e-9, 1) ** s["temperature"]
    p[:, 1] *= 1 + s["draw_boost"]
    return p / p.sum(axis=1, keepdims=True)


def log_loss(P: np.ndarray, result) -> float:
    idx = pd.Series(result).map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return float(-np.mean(np.log(np.clip(P[np.arange(len(P)), idx], 1e-12, 1))))


def replay(matches: pd.DataFrame, end: pd.Timestamp, days: int = REPLAY_DAYS) -> pd.DataFrame:
    """Both models' forecasts for every league match of the last `days`, refit weekly."""
    from fa.predict import predict
    start = end - pd.Timedelta(days=days)
    out = []
    for w in pd.date_range(start.normalize(), end, freq="7D"):
        week = matches[(matches.date >= w) & (matches.date < min(w + pd.Timedelta(days=7), end))]
        if week.empty:
            continue
        fx = week[["league", "date", "home_team", "away_team"]].assign(kickoff_utc=pd.NaT)
        pr = predict(matches[matches.date < w], fx, w)
        out.append(pr[["league", "date", "home_team", "away_team", "dc_home_p", "dc_draw_p",
                       "dc_away_p", "elo_home_p", "elo_draw_p", "elo_away_p"]]
                   .merge(week[["league", "date", "home_team", "away_team", "result"]],
                          on=["league", "date", "home_team", "away_team"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(df: pd.DataFrame, s: dict) -> float:
    P = combine(df[["dc_home_p", "dc_draw_p", "dc_away_p"]].to_numpy(),
                df[["elo_home_p", "elo_draw_p", "elo_away_p"]].to_numpy(), s)
    return log_loss(P, df.result)


def decide(df: pd.DataFrame, current: dict, check_from: pd.Timestamp) -> dict:
    """Search on the older part, confirm on the recent part. Returns the decision."""
    tune, check = df[df.date < check_from], df[df.date >= check_from]
    best, best_ll = current, evaluate(tune, current)
    for values in itertools.product(*GRID.values()):
        s = dict(zip(GRID.keys(), map(float, values)))
        ll = evaluate(tune, s)
        if ll < best_ll - 1e-12:
            best, best_ll = s, ll
    cur_tune, cur_check = evaluate(tune, current), evaluate(check, current)
    new_check = evaluate(check, best)
    adopt = best != current and new_check < cur_check - MIN_GAIN and best_ll < cur_tune - MIN_GAIN
    return {"current": current, "candidate": best, "adopted": bool(adopt),
            "tune_matches": len(tune), "check_matches": len(check),
            "tune_current": cur_tune, "tune_candidate": best_ll,
            "check_current": cur_check, "check_candidate": new_check}


def record(decision: dict, today: pd.Timestamp) -> None:
    s = decision["candidate"] if decision["adopted"] else decision["current"]
    previous = json.loads(SETTINGS_PATH.read_text()).get("since", "start") if SETTINGS_PATH.exists() else "start"
    since = f"{today:%Y-%m-%d}" if decision["adopted"] else previous
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps({**s, "since": since}, indent=1) + "\n")
    row = {"date": f"{today:%Y-%m-%d}", "adopted": decision["adopted"],
           **{f"current_{k}": v for k, v in decision["current"].items()},
           **{f"candidate_{k}": v for k, v in decision["candidate"].items()},
           **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in decision.items()
              if k.startswith(("tune_", "check_"))}}
    new = not LOG_PATH.exists()
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=new, index=False)


def report(today: pd.Timestamp) -> str:
    log = pd.read_csv(LOG_PATH)
    s = load_settings()
    lines = ["# Learning from results", "",
             "Every Monday the model replays the last year of league matches as it would have "
             "forecast them, searches its settings on the older weeks, and adopts new ones only "
             f"if they also forecast the most recent {CHECK_DAYS // 7} weeks better. Log loss: "
             "lower is better.", "",
             f"**In use since {json.loads(SETTINGS_PATH.read_text()).get('since', 'start')}:** "
             f"goals model {s['blend_weight_dc']:.0%} / ratings model {1 - s['blend_weight_dc']:.0%}, "
             f"confidence ×{s['temperature']:.2f}, draws {s['draw_boost']:+.1%}.", "",
             "| Date | Decision | Mix (goals model) | Confidence | Draws | Recent weeks: current → candidate |",
             "|---|---|---:|---:|---:|---|"]
    for r in log.iloc[::-1].itertuples():
        lines.append(f"| {r.date} | {'adopted' if r.adopted else 'kept current'} | "
                     f"{r.candidate_blend_weight_dc:.0%} | ×{r.candidate_temperature:.2f} | "
                     f"{r.candidate_draw_boost:+.1%} | {r.check_current:.4f} → {r.check_candidate:.4f} "
                     f"({int(r.check_matches)} matches) |")
    return "\n".join(lines) + "\n"


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    m = con.execute("SELECT * FROM matches ORDER BY date").df()
    con.close()
    m["date"] = pd.to_datetime(m.date)
    today = pd.Timestamp.now().normalize()
    end = min(today, m.date.max() + pd.Timedelta(days=1))
    df = replay(m, end)
    if len(df) < 500:
        print(f"Only {len(df)} replayed matches: not enough to learn from. Settings unchanged.")
        return
    current = load_settings()
    d = decide(df, current, end - pd.Timedelta(days=CHECK_DAYS))
    record(d, today)
    (REPORTS_DIR / "learning.md").write_text(report(today))
    c = d["candidate"]
    print(f"Replayed {len(df)} matches. Best on the older weeks: goals model {c['blend_weight_dc']:.0%}, "
          f"confidence ×{c['temperature']:.2f}, draws {c['draw_boost']:+.1%}")
    print(f"Last {CHECK_DAYS // 7} weeks ({d['check_matches']} matches): current {d['check_current']:.4f} "
          f"vs candidate {d['check_candidate']:.4f} -> "
          + ("ADOPTED" if d["adopted"] else "kept current settings"))


if __name__ == "__main__":
    main()
