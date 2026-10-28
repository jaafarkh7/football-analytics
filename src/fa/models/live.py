"""In-match win probabilities from the pre-match forecast and the current score.

The pre-match forecast is turned into two Poisson scoring rates (one per team) that
reproduce its home / away win probabilities. The rest of the match is then a Poisson
process: in the second half each team scores at its rate times SECOND_HALF_SHARE
(about 55% of goals come after the break), with a game-state tweak: the trailing team
scores a little more and the leading team a little less. Both numbers are tuned on
2024-25 and tested on 2025-26 in `python -m fa.run_halftime`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

SECOND_HALF_SHARE = 0.60     # tuned on 2024-25 (see reports/halftime_model.md)
GAME_STATE = 0.20
MAX_GOALS = 12


def outcome_probs(lam: float, mu: float, lead: int = 0) -> np.ndarray:
    """P(home win, draw, away win) when the home side leads by `lead` and the two teams
    score `lam` and `mu` more goals on average from now on."""
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    diff = g[:, None] - g[None, :] + lead
    p = np.array([m[diff > 0].sum(), m[diff == 0].sum(), m[diff < 0].sum()])
    return p / p.sum()


def rates_from_probs(p_home: float, p_away: float, hint: tuple[float, float] = (1.45, 1.15)
                     ) -> tuple[float, float]:
    """Poisson scoring rates (home, away) whose full-match win probabilities match."""
    def err(x):
        p = outcome_probs(np.exp(x[0]), np.exp(x[1]))
        return [p[0] - p_home, p[2] - p_away]
    sol = least_squares(err, np.log(np.maximum(hint, 0.05)), bounds=([-3, -3], [2, 2]))
    return float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))


def halftime_probs(p_home: float, p_away: float, ht_home: int, ht_away: int,
                   share: float = SECOND_HALF_SHARE, game_state: float = GAME_STATE,
                   hint: tuple[float, float] = (1.45, 1.15)) -> np.ndarray:
    """Final-result probabilities given the pre-match forecast and the half-time score."""
    lam, mu = rates_from_probs(p_home, p_away, hint)
    lead = int(ht_home) - int(ht_away)
    lam2, mu2 = lam * share, mu * share
    if lead:                      # the trailing side pushes, the leading side sits back
        s = np.sign(lead)
        lam2 *= 1 - s * game_state
        mu2 *= 1 + s * game_state
    return outcome_probs(lam2, mu2, lead)
