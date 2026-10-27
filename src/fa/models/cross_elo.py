"""One Elo rating pool across the top-5 leagues and the Champions League.

Domestic matches only move points *between clubs of the same league*, so a league's
average rating can only change through matches against other leagues. Champions League
results therefore end up setting the relative strength of the leagues, which is what
lets us compare, say, Napoli with Leverkusen.

Clubs outside our 5 leagues ("external", e.g. Benfica, PSV) only play in our data in
the Champions League; they start below the top-5 average and learn from those matches.

With `league_strength=False` Champions League results never move top-5 clubs, so every
league stays centred on the same average: the "all leagues are equal" baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fa.eval.metrics import OUTCOMES
from fa.models.elo import _margin_multiplier


@dataclass(frozen=True)
class CrossEloParams:
    k: float = 20.0                # domestic update speed (week 1 tuning)
    home_adv: float = 105.0        # domestic home advantage (week 1 tuning)
    new_team_offset: float = 100.0
    k_cl: float = 30.0             # Champions League update speed
    home_adv_cl: float = 70.0      # home advantage in Europe (usually smaller)
    external_offset: float = 100.0 # new external clubs start this far below the top-5 mean
    league_strength: bool = True   # False = baseline: CL results don't move top-5 clubs
    initial: float = 1500.0


def run_cross_elo(domestic: pd.DataFrame, cl: pd.DataFrame,
                  params: CrossEloParams = CrossEloParams()) -> pd.DataFrame:
    """Walk through domestic and CL matches by date; return CL matches with pre-match ratings.

    The returned frame carries the final state in .attrs: ratings, club_league, league_means.
    """
    p = params
    if cl.match_id.duplicated().any():
        raise ValueError(f"duplicate Champions League match ids: "
                         f"{cl.match_id[cl.match_id.duplicated()].tolist()[:3]}")
    d = domestic[["match_id", "date", "season", "league", "home_team", "away_team",
                  "home_goals", "away_goals"]].assign(kind="dom", neutral=False,
                                                      home_league=domestic.league,
                                                      away_league=domestic.league)
    c = cl[["match_id", "date", "season", "home_team", "away_team", "home_goals", "away_goals",
            "neutral", "home_league", "away_league"]].assign(kind="cl", league="UCL")
    stream = pd.concat([d, c], ignore_index=True).sort_values(["date", "match_id"])

    ratings: dict[str, float] = {}
    club_league: dict[str, str | None] = {}
    last_season: dict[str, int] = {}
    first_season: dict[str, int] = {}
    pre_h, pre_a, ids = [], [], []

    def league_mean(league: str, season: int) -> float | None:
        vals = [ratings[t] for t, lg in club_league.items()
                if lg == league and last_season.get(t, -1) >= season - 1]
        return float(np.mean(vals)) if vals else None

    def top5_mean(season: int) -> float:
        vals = [ratings[t] for t, lg in club_league.items()
                if lg is not None and last_season.get(t, -1) >= season - 1]
        return float(np.mean(vals)) if vals else p.initial

    for r in stream.itertuples(index=False):
        for team, lg in ((r.home_team, r.home_league), (r.away_team, r.away_league)):
            lg = None if (lg is None or (isinstance(lg, float) and np.isnan(lg))) else lg
            if team not in ratings:
                if lg is None:                                   # external club
                    ratings[team] = top5_mean(r.season) - p.external_offset
                else:
                    first_season.setdefault(lg, r.season)
                    mean = league_mean(lg, r.season)
                    if r.season == first_season[lg] or mean is None:
                        ratings[team] = p.initial
                    else:                                        # promoted club
                        ratings[team] = mean - p.new_team_offset
            if r.kind == "dom" or team not in club_league:
                club_league[team] = lg
            if r.kind == "dom":
                last_season[team] = r.season

        rh, ra = ratings[r.home_team], ratings[r.away_team]
        if r.kind == "cl":
            pre_h.append(rh)
            pre_a.append(ra)
            ids.append(r.match_id)
            hadv, k = (0.0 if r.neutral else p.home_adv_cl), p.k_cl
        else:
            hadv, k = p.home_adv, p.k
        expected = 1 / (1 + 10 ** (-(rh + hadv - ra) / 400))
        gd = r.home_goals - r.away_goals
        actual = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        delta = k * _margin_multiplier(gd) * (actual - expected)

        if r.kind == "cl" and not p.league_strength:
            # Baseline: top-5 clubs are never moved by European results.
            if club_league.get(r.home_team) is None:
                ratings[r.home_team] = rh + delta
            if club_league.get(r.away_team) is None:
                ratings[r.away_team] = ra - delta
        else:
            ratings[r.home_team] = rh + delta
            ratings[r.away_team] = ra - delta

    out = cl.set_index("match_id").loc[ids].reset_index()
    out["elo_home"], out["elo_away"] = pre_h, pre_a
    out["elo_diff"] = out.elo_home - out.elo_away + np.where(out.neutral, 0.0, p.home_adv_cl)
    season = int(stream.season.max())
    out.attrs["ratings"] = dict(ratings)
    out.attrs["club_league"] = dict(club_league)
    out.attrs["league_means"] = {lg: league_mean(lg, season)
                                 for lg in sorted({v for v in club_league.values() if v})}
    return out


class CLOutcomeModel:
    """Rating gap -> P(H, D, A), fitted on earlier Champions League matches."""

    def __init__(self):
        self.clf = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, rated: pd.DataFrame) -> "CLOutcomeModel":
        self.clf.fit((rated.elo_diff.to_numpy() / 100)[:, None], rated.result.to_numpy())
        return self

    def predict_proba(self, rated: pd.DataFrame) -> np.ndarray:
        p = self.clf.predict_proba((rated.elo_diff.to_numpy() / 100)[:, None])
        classes = list(self.clf.classes_)
        # An outcome never seen in training gets probability 0 (can't happen with real data).
        return np.column_stack([p[:, classes.index(o)] if o in classes else np.zeros(len(p))
                                for o in OUTCOMES])
