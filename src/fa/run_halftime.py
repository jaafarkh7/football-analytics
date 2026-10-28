"""Experiment: how good are half-time win probabilities?

    python -m fa.run_halftime        # ~1-2 min

Pre-match forecasts are replayed week by week (the model only sees earlier matches),
then updated with each match's real half-time score. The second-half goal share and the
game-state effect are tuned on 2024-25 (validation) and reported once on 2025-26 (test).
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from fa.config import DB_PATH, REPORTS_DIR
from fa.models.live import halftime_probs, rates_from_probs
from fa.predict import predict

VALID, TEST = 2024, 2025


def replay(matches: pd.DataFrame, season: int) -> pd.DataFrame:
    """Pre-match forecasts for every match of `season`, refit weekly."""
    s = matches[matches.season == season].dropna(subset=["ht_home_goals", "ht_away_goals"])
    out = []
    for start in pd.date_range(s.date.min().normalize(), s.date.max() + pd.Timedelta(days=7), freq="7D"):
        week = s[(s.date >= start) & (s.date < start + pd.Timedelta(days=7))]
        if week.empty:
            continue
        fx = week[["league", "date", "home_team", "away_team"]].assign(kickoff_utc=pd.NaT)
        pr = predict(matches[matches.date < start], fx, start)
        out.append(pr.merge(week[["league", "date", "home_team", "away_team", "result",
                                  "ht_home_goals", "ht_away_goals"]],
                            on=["league", "date", "home_team", "away_team"]))
    return pd.concat(out, ignore_index=True)


def log_loss(P: np.ndarray, result: pd.Series) -> float:
    idx = result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return float(-np.mean(np.log(np.clip(P[np.arange(len(P)), idx], 1e-12, 1))))


def ht_matrix(df: pd.DataFrame, rates: list, share: float, game_state: float) -> np.ndarray:
    from fa.models.live import outcome_probs
    P = []
    for (lam, mu), h, a in zip(rates, df.ht_home_goals, df.ht_away_goals):
        lead = int(h) - int(a)
        l2, m2 = lam * share, mu * share
        if lead:
            s = np.sign(lead)
            l2, m2 = l2 * (1 - s * game_state), m2 * (1 + s * game_state)
        P.append(outcome_probs(l2, m2, lead))
    return np.array(P)


def score_only(df: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """Baseline: final-result frequencies by half-time goal difference (no team strength)."""
    lead = (train.ht_home_goals - train.ht_away_goals).clip(-3, 3)
    table = pd.crosstab(lead, train.result, normalize="index")[["H", "D", "A"]]
    d = (df.ht_home_goals - df.ht_away_goals).clip(-3, 3)
    return table.loc[d].to_numpy()


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    m = con.execute("SELECT * FROM matches ORDER BY date").df()
    con.close()
    if "ht_home_goals" not in m:
        raise SystemExit("No half-time scores yet: rerun `python -m fa.data.load_results`.")
    m["date"] = pd.to_datetime(m.date)

    data = {s: replay(m, s) for s in (VALID, TEST)}
    rates = {s: [rates_from_probs(r.p_home, r.p_away, (r.xg_home, r.xg_away))
                 for r in d.itertuples()] for s, d in data.items()}

    # Tune on validation.
    grid = [(sh, gs) for sh in np.arange(0.44, 0.72, 0.02) for gs in np.arange(0.0, 0.45, 0.05)]
    scores = {(sh, gs): log_loss(ht_matrix(data[VALID], rates[VALID], sh, gs), data[VALID].result)
              for sh, gs in grid}
    best = min(scores, key=scores.get)
    print(f"Best on {VALID}-{(VALID + 1) % 100:02d}: second-half share {best[0]:.2f}, "
          f"game state {best[1]:.2f} (log loss {scores[best]:.4f})")

    train = m[(m.season < TEST) & m.ht_home_goals.notna()]
    lines = ["# Half-time win probabilities", "",
             "Pre-match forecasts replayed week by week, updated with the real half-time score.",
             f"Tuned on {VALID}-{(VALID + 1) % 100:02d}: second-half goal share **{best[0]:.2f}**, "
             f"game-state effect **{best[1]:.2f}** (trailing side scores {best[1]:.0%} more, leader "
             f"{best[1]:.0%} less).", "",
             "| Season | Matches | Pre-match | Half-time score only | Half-time model |",
             "|---|---:|---:|---:|---:|"]
    for s, d in data.items():
        pre = log_loss(d[["p_home", "p_draw", "p_away"]].to_numpy(), d.result)
        so = log_loss(score_only(d, train if s == TEST else m[(m.season < VALID) & m.ht_home_goals.notna()]), d.result)
        ht = log_loss(ht_matrix(d, rates[s], *best), d.result)
        label = f"{s}-{(s + 1) % 100:02d}" + (" (test)" if s == TEST else " (tuning)")
        lines.append(f"| {label} | {len(d)} | {pre:.4f} | {so:.4f} | {ht:.4f} |")
        print(f"{label}: pre-match {pre:.4f} · half-time score only {so:.4f} · half-time model {ht:.4f}")

    # Calibration on the test season: predicted vs observed, for each outcome.
    d = data[TEST]
    P = ht_matrix(d, rates[TEST], *best)
    lines += ["", "Log loss, lower is better. 1.0986 = no information.", "",
              "## Calibration on the test season", "",
              "| Predicted | Matches | Predicted avg | Observed |", "|---|---:|---:|---:|"]
    flat_p = P.ravel()
    flat_y = np.column_stack([(d.result == k).to_numpy() for k in "HDA"]).ravel()
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (flat_p >= lo) & (flat_p < hi)
        if sel.sum():
            lines.append(f"| {lo:.0%}–{min(hi, 1):.0%} | {sel.sum()} | {flat_p[sel].mean():.1%} | "
                         f"{flat_y[sel].mean():.1%} |")
    # By state at half-time.
    lead = np.sign(d.ht_home_goals - d.ht_away_goals)
    lines += ["", "## Test season by half-time state", "",
              "| State | Matches | Log loss | Leader went on to win: predicted / observed |",
              "|---|---:|---:|---:|"]
    for name, k in (("Home leading", 1), ("Level", 0), ("Away leading", -1)):
        sel = (lead == k).to_numpy()
        ll = log_loss(P[sel], d.result[sel])
        if k == 0:
            extra = f"draw {P[sel][:, 1].mean():.1%} / {(d.result[sel] == 'D').mean():.1%}"
        else:
            col, res = (0, "H") if k == 1 else (2, "A")
            extra = f"{P[sel][:, col].mean():.1%} / {(d.result[sel] == res).mean():.1%}"
        lines.append(f"| {name} | {sel.sum()} | {ll:.4f} | {extra} |")
    path = REPORTS_DIR / "halftime_model.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
