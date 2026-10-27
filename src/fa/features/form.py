"""Form features: how a team has done recently, using only matches *before* each match.

For every match we compute, for the home and the away team:
    form5_pts, form5_gd, form5_sotd   last 5 league matches, per game
    season_pts, season_gd             this season so far, per game
    season_n                          matches played this season so far
    rest_days                         days since the previous league match (capped at 14)

and then home-minus-away differences, which is what the model uses.

Every rolling statistic is shifted by one match, so a match never sees its own result.
Rest days ignore cup and European matches (not in our data), so they're a rough proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORM_WINDOW = 5
FEATURES = ["form5_pts", "form5_gd", "form5_sotd", "season_pts", "season_gd", "rest_days"]


def team_view(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team): the match from that team's point of view."""
    m = matches
    has_sot = "home_sot" in m.columns
    common = ["match_id", "league", "season", "date"]
    home = m[common].assign(team=m.home_team, is_home=True,
                            gf=m.home_goals, ga=m.away_goals,
                            sotf=m.home_sot if has_sot else np.nan,
                            sota=m.away_sot if has_sot else np.nan)
    away = m[common].assign(team=m.away_team, is_home=False,
                            gf=m.away_goals, ga=m.home_goals,
                            sotf=m.away_sot if has_sot else np.nan,
                            sota=m.home_sot if has_sot else np.nan)
    tv = pd.concat([home, away], ignore_index=True)
    tv["pts"] = np.select([tv.gf > tv.ga, tv.gf == tv.ga], [3, 1], 0)
    tv["gd"] = tv.gf - tv.ga
    tv["sotd"] = tv.sotf - tv.sota
    return tv.sort_values(["league", "team", "date", "match_id"]).reset_index(drop=True)


def _prior_mean(group: pd.core.groupby.SeriesGroupBy, window: int | None) -> pd.Series:
    """Mean of the previous `window` values (all previous if None), excluding the current."""
    if window is None:
        return group.transform(lambda s: s.shift(1).expanding().mean())
    return group.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def team_form(matches: pd.DataFrame) -> pd.DataFrame:
    tv = team_view(matches)
    by_team = tv.groupby(["league", "team"], sort=False)
    by_season = tv.groupby(["league", "team", "season"], sort=False)

    tv["form5_pts"] = _prior_mean(by_team["pts"], FORM_WINDOW)
    tv["form5_gd"] = _prior_mean(by_team["gd"], FORM_WINDOW)
    tv["form5_sotd"] = _prior_mean(by_team["sotd"], FORM_WINDOW)
    tv["season_pts"] = _prior_mean(by_season["pts"], None)
    tv["season_gd"] = _prior_mean(by_season["gd"], None)
    tv["season_n"] = by_season.cumcount()
    tv["rest_days"] = by_team["date"].diff().dt.days.clip(upper=14)

    # Missing values: no previous match (season opener, new team). Use neutral values so
    # the difference between two teams with no information is 0.
    neutral = {"form5_pts": 1.37, "form5_gd": 0.0, "form5_sotd": 0.0,
               "season_pts": 1.37, "season_gd": 0.0, "rest_days": 14.0}
    tv = tv.fillna(neutral)
    return tv


def match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Per match: home-minus-away differences, plus how many matches the season has had."""
    tv = team_form(matches)
    cols = FEATURES + ["season_n"]
    h = tv[tv.is_home].set_index("match_id")[cols]
    a = tv[~tv.is_home].set_index("match_id")[cols]
    out = pd.DataFrame(index=h.index)
    for c in FEATURES:
        out[f"d_{c}"] = h[c] - a.loc[h.index, c]
    out["season_n_min"] = np.minimum(h.season_n, a.loc[h.index, "season_n"])
    return out
