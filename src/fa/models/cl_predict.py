"""Champions League predictions with the cross-league Elo (see run_cl.py for evaluation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fa.config import CL_SETTINGS, ELO_SETTINGS
from fa.data.team_names import same_club, similarity
from fa.models.cross_elo import CLOutcomeModel, CrossEloParams, run_cross_elo

LINK_THRESHOLD = 0.8     # below this, an outside club is treated as new rather than linked


def link_external_names(recent_names: set[str], historical_keys: set[str]) -> dict[str, str]:
    """Map current API names of non-top-5 clubs ("Benfica") to historical keys
    ("Sport Lisboa e Benfica (POR)"). Only confident, unambiguous links are made."""
    hist = {k: k.rpartition(" (")[0] or k for k in historical_keys}
    out = {}
    for name in recent_names:
        scores = sorted(((1.0 if same_club(name, h) else similarity(name, h), k)
                         for k, h in hist.items()), reverse=True)
        if not scores or scores[0][0] < LINK_THRESHOLD:
            continue
        if len(scores) > 1 and scores[1][0] >= scores[0][0] - 0.05:
            continue                                     # two plausible matches: don't guess
        out[name] = scores[0][1]
    return out


def predict_cl(domestic: pd.DataFrame, cl_hist: pd.DataFrame, cl_recent: pd.DataFrame,
               fixtures: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """P(H/D/A) for Champions League fixtures, from results before `as_of` only."""
    params = CrossEloParams(k=ELO_SETTINGS["k"], home_adv=ELO_SETTINGS["home_adv"],
                            new_team_offset=ELO_SETTINGS["new_team_offset"], **CL_SETTINGS)
    externals_hist = set(cl_hist.loc[cl_hist.home_league.isna(), "home_team"]) | \
        set(cl_hist.loc[cl_hist.away_league.isna(), "away_team"])

    recent = cl_recent.copy() if len(cl_recent) else cl_recent
    fx = fixtures.copy()
    names_now = set()
    for df, h, a in ((recent, "home", "away"), (fx, "home", "away")):
        if len(df):
            for side in (h, a):
                col = f"{side}_full" if f"{side}_full" in df else f"{side}_team"
                names_now |= set(df.loc[df[f"{side}_league"].isna(), col])
    link = link_external_names(names_now, externals_hist)

    def relabel(df):
        for side in ("home", "away"):
            col = f"{side}_full" if f"{side}_full" in df else f"{side}_team"
            ext = df[f"{side}_league"].isna()
            df.loc[ext, f"{side}_key"] = df.loc[ext, col].map(lambda n: link.get(n, n))
            df.loc[~ext, f"{side}_key"] = df.loc[~ext, f"{side}_team"]
        return df

    cols = ["match_id", "season", "date", "neutral", "home_team", "away_team", "home_league",
            "away_league", "home_goals", "away_goals", "result"]
    hist = cl_hist[cols]
    if len(recent):
        recent = relabel(recent)
        recent = recent.assign(home_team=recent.home_key, away_team=recent.away_key)[cols]
        hist = pd.concat([hist, recent[~recent.match_id.isin(hist.match_id)]], ignore_index=True)
    hist = hist[hist.date < as_of]
    played = domestic[domestic.date < as_of]

    rated = run_cross_elo(played, hist, params)
    ratings, club_league = rated.attrs["ratings"], rated.attrs["club_league"]
    fit = rated[rated.season >= 2015]
    outcome = CLOutcomeModel().fit(fit)

    fx = relabel(fx)
    top5 = [v for t, v in ratings.items() if club_league.get(t)]
    new_external = (np.mean(top5) if top5 else params.initial) - params.external_offset
    fx["elo_home"] = [ratings.get(k, new_external) for k in fx.home_key]
    fx["elo_away"] = [ratings.get(k, new_external) for k in fx.away_key]
    fx["known_home"] = [k in ratings for k in fx.home_key]
    fx["known_away"] = [k in ratings for k in fx.away_key]
    neutral = fx.stage.eq("FINAL") if "stage" in fx else pd.Series(False, index=fx.index)
    fx["neutral"] = neutral.to_numpy()
    fx["elo_diff"] = fx.elo_home - fx.elo_away + np.where(fx.neutral, 0.0, params.home_adv_cl)
    fx[["p_home", "p_draw", "p_away"]] = outcome.predict_proba(fx)
    fx.attrs["linked"] = link
    return fx
