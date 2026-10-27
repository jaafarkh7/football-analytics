"""Elo ratings for club football, plus a calibrated map from rating gap to H/D/A.

Elo by itself gives an *expected score* (win=1, draw=0.5), not three probabilities.
We record each match's pre-match rating gap, then fit a small multinomial logistic
regression (gap -> H/D/A) on training seasons only.

Every prediction uses ratings from *before* the match, so there is no look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fa.eval.metrics import OUTCOMES


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0               # update speed
    home_adv: float = 65.0        # rating points added to the home side
    season_regress: float = 0.25  # pull ratings this fraction toward the mean each summer
    new_team_offset: float = 75.0 # promoted teams start this far below the league mean
    goal_margin: bool = True      # bigger wins move ratings more (World Football Elo style)
    initial: float = 1500.0


def _margin_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def run_elo(matches: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Walk through matches in date order and return pre-match ratings for each.

    `matches` needs: match_id, league, season, date, home_team, away_team,
    home_goals, away_goals. Leagues are rated as separate pools.
    """
    df = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    out_home = np.empty(len(df))
    out_away = np.empty(len(df))

    ratings: dict[tuple[str, str], float] = {}
    league_season: dict[str, int] = {}
    league_first: dict[str, int] = {}

    for i, row in enumerate(df.itertuples(index=False)):
        lg = row.league
        # New season in this league: regress everyone toward the league mean.
        if league_season.get(lg) != row.season:
            pool = {k: v for k, v in ratings.items() if k[0] == lg}
            if pool:
                mean = float(np.mean(list(pool.values())))
                for k, v in pool.items():
                    ratings[k] = v + params.season_regress * (mean - v)
            league_season[lg] = row.season
            league_first.setdefault(lg, row.season)

        h, a = (lg, row.home_team), (lg, row.away_team)
        for team in (h, a):
            if team not in ratings:
                if row.season == league_first[lg]:
                    ratings[team] = params.initial      # first season: everyone starts equal
                else:                                   # later: a promoted team
                    pool = [v for k, v in ratings.items() if k[0] == lg]
                    ratings[team] = float(np.mean(pool)) - params.new_team_offset

        rh, ra = ratings[h], ratings[a]
        out_home[i], out_away[i] = rh, ra

        expected_home = 1 / (1 + 10 ** (-(rh + params.home_adv - ra) / 400))
        gd = row.home_goals - row.away_goals
        actual_home = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        mult = _margin_multiplier(gd) if params.goal_margin else 1.0
        delta = params.k * mult * (actual_home - expected_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta          # zero-sum

    df["elo_home"] = out_home
    df["elo_away"] = out_away
    df["elo_diff"] = out_home + params.home_adv - out_away
    df.attrs["final_ratings"] = dict(ratings)     # ratings after the last match
    return df


def ratings_for_fixtures(rated: pd.DataFrame, fixtures: pd.DataFrame,
                         params: EloParams = EloParams()) -> pd.DataFrame:
    """Pre-match Elo for upcoming fixtures, from the ratings after the last played match.

    Unseen (promoted) teams get the league mean minus `new_team_offset`, as in run_elo.
    Fixtures are rated independently: one upcoming match never affects another.
    """
    ratings = rated.attrs["final_ratings"]
    out = fixtures.copy()
    hs, as_ = [], []
    for row in fixtures.itertuples(index=False):
        pool = [v for (lg, _), v in ratings.items() if lg == row.league]
        default = float(np.mean(pool)) - params.new_team_offset if pool else params.initial
        hs.append(ratings.get((row.league, row.home_team), default))
        as_.append(ratings.get((row.league, row.away_team), default))
    out["elo_home"], out["elo_away"] = hs, as_
    out["elo_diff"] = out.elo_home + params.home_adv - out.elo_away
    return out


class EloOutcomeModel:
    """Maps a pre-match Elo gap to P(H), P(D), P(A)."""

    def __init__(self, params: EloParams = EloParams()):
        self.params = params
        self.clf = LogisticRegression(C=1e6, max_iter=1000)

    @staticmethod
    def _x(df: pd.DataFrame) -> np.ndarray:
        return (df["elo_diff"].to_numpy() / 100.0)[:, None]

    def fit(self, rated: pd.DataFrame) -> "EloOutcomeModel":
        self.clf.fit(self._x(rated), rated["result"].to_numpy())
        return self

    def predict_proba(self, rated: pd.DataFrame) -> np.ndarray:
        p = self.clf.predict_proba(self._x(rated))
        order = [list(self.clf.classes_).index(o) for o in OUTCOMES]   # -> H, D, A
        return p[:, order]
