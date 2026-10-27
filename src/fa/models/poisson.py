"""Dixon–Coles goals model.

Each team has an attack and a defence strength. Expected goals:

    log λ_home = μ + home_adv + attack[home] + weakness[away]
    log λ_away = μ            + attack[away] + weakness[home]

(`weakness` > 0 means the team concedes more than average.)

Goals are Poisson, except for a Dixon–Coles correction ρ that fixes how plain
Poisson mis-prices 0-0, 1-0, 0-1 and 1-1. Older matches get less weight
(exponential decay with a half-life), and an L2 penalty shrinks teams with little
data toward average.

The Poisson part is a weighted Poisson GLM, so we let scikit-learn fit it; ρ is then
fitted with the λs held fixed (a standard two-step approximation).

Shots: goals are a noisy signal of team quality, shots much less so. With
goal_weight < 1, the model learns strengths from

    target = goal_weight * goals + (1 - goal_weight) * shot_xg
    shot_xg = c_on * shots_on_target + c_off * shots_off_target

where c_on, c_off are the league's conversion rates, fitted on the same training
window. ρ and all predictions are still about real goals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

MAX_GOALS = 10


@dataclass(frozen=True)
class DCParams:
    half_life_days: float = 180.0   # a match this old counts half as much as today's
    alpha: float = 1e-3             # L2 shrinkage toward an average team
    window_days: int = 3 * 365      # ignore anything older than this
    dixon_coles: bool = True        # False -> plain independent Poisson
    goal_weight: float = 1.0        # 1 = learn from goals only; <1 mixes in shot-based xG


def tau(x, y, lam, mu, rho):
    """Dixon–Coles adjustment factor for low scorelines (1 for everything else)."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


class DixonColes:
    def __init__(self, params: DCParams = DCParams()):
        self.params = params

    # ---------- fitting ----------

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp) -> "DixonColes":
        """Fit on matches strictly before `as_of`, weighting recent ones more."""
        p = self.params
        m = matches[(matches.date < as_of) &
                    (matches.date >= as_of - pd.Timedelta(days=p.window_days))]
        if len(m) < 50:
            raise ValueError(f"only {len(m)} matches before {as_of.date()}")
        teams = sorted(set(m.home_team) | set(m.away_team))
        idx = {t: i for i, t in enumerate(teams)}
        n, t = len(m), len(teams)

        # Two rows per match: home goals, then away goals.
        X = np.zeros((2 * n, 2 * t + 1))
        h = m.home_team.map(idx).to_numpy()
        a = m.away_team.map(idx).to_numpy()
        r = np.arange(n)
        X[r, h] = 1;         X[r, t + a] = 1;         X[r, 2 * t] = 1      # home row
        X[n + r, a] = 1;     X[n + r, t + h] = 1                           # away row
        goals = np.concatenate([m.home_goals.to_numpy(), m.away_goals.to_numpy()]).astype(float)
        y = goals
        self.conversion = None
        if p.goal_weight < 1:
            y = self._blend_with_shots(m, goals, p.goal_weight)
        age = (as_of - m.date).dt.days.to_numpy()
        w = 0.5 ** (age / p.half_life_days)
        w2 = np.concatenate([w, w])

        glm = PoissonRegressor(alpha=p.alpha, max_iter=1000)
        glm.fit(X, y, sample_weight=w2 / w2.mean())
        self.teams, self.idx = teams, idx
        self.mu = glm.intercept_
        self.attack = glm.coef_[:t]
        self.weakness = glm.coef_[t:2 * t]
        self.home_adv = glm.coef_[2 * t]

        # Promoted / unseen teams: assume they're like the weakest current teams,
        # not like an average team (an average-team prior is too optimistic).
        strength = self.attack - self.weakness
        weakest = np.argsort(strength)[:3]
        self.new_attack = float(self.attack[weakest].mean())
        self.new_weakness = float(self.weakness[weakest].mean())

        self.rho = 0.0
        if p.dixon_coles:
            lam, mu = self._lambdas(m.home_team, m.away_team)
            hg, ag = m.home_goals.to_numpy(), m.away_goals.to_numpy()

            def nll(rho):
                tt = tau(hg, ag, lam, mu, rho)
                return np.inf if (tt <= 0).any() else -np.sum(w * np.log(tt))

            self.rho = float(minimize_scalar(nll, bounds=(-0.25, 0.25), method="bounded").x)
        return self

    def _blend_with_shots(self, m: pd.DataFrame, goals: np.ndarray, gw: float) -> np.ndarray:
        if not {"home_shots", "home_sot"} <= set(m.columns):
            raise ValueError("goal_weight < 1 needs shots columns; reload data with shots")
        shots = np.concatenate([m.home_shots.to_numpy(float), m.away_shots.to_numpy(float)])
        sot = np.concatenate([m.home_sot.to_numpy(float), m.away_sot.to_numpy(float)])
        off = np.clip(shots - sot, 0, None)
        ok = ~(np.isnan(shots) | np.isnan(sot))
        if ok.sum() < 100:
            raise ValueError("too few matches with shot data in the training window")
        # Conversion rates by least squares without intercept: goals ≈ c_on*SoT + c_off*off.
        A = np.column_stack([sot[ok], off[ok]])
        c = np.clip(np.linalg.lstsq(A, goals[ok], rcond=None)[0], 0, None)
        self.conversion = {"on_target": float(c[0]), "off_target": float(c[1])}
        y = goals.copy()                          # matches without shots: goals only
        y[ok] = gw * goals[ok] + (1 - gw) * (A @ c)
        return y

    # ---------- prediction ----------

    def _team(self, team: str) -> tuple[float, float]:
        i = self.idx.get(team)
        if i is None:
            return self.new_attack, self.new_weakness
        return self.attack[i], self.weakness[i]

    def _lambdas(self, home_teams, away_teams):
        ha = np.array([self._team(x) for x in home_teams])
        aa = np.array([self._team(x) for x in away_teams])
        lam = np.exp(self.mu + self.home_adv + ha[:, 0] + aa[:, 1])
        mu = np.exp(self.mu + aa[:, 0] + ha[:, 1])
        return lam, mu

    def score_matrix(self, lam: float, mu: float) -> np.ndarray:
        """P(home scores i, away scores j) for i, j in 0..MAX_GOALS."""
        g = np.arange(MAX_GOALS + 1)
        mat = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        x, y = np.meshgrid(g, g, indexing="ij")
        mat *= tau(x, y, np.full_like(mat, lam), np.full_like(mat, mu), self.rho)
        return mat / mat.sum()

    def predict(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Expected goals, H/D/A probabilities and most likely scoreline per match."""
        lam, mu = self._lambdas(matches.home_team, matches.away_team)
        rows = []
        for l, m in zip(lam, mu):
            mat = self.score_matrix(l, m)
            i, j = np.unravel_index(mat.argmax(), mat.shape)
            rows.append({"xg_home": l, "xg_away": m,
                         "p_home": np.tril(mat, -1).sum(), "p_draw": np.trace(mat),
                         "p_away": np.triu(mat, 1).sum(),
                         "top_score": f"{i}-{j}", "top_score_p": mat[i, j]})
        return pd.DataFrame(rows, index=matches.index)


def walk_forward(matches: pd.DataFrame, season: int, params: DCParams = DCParams(),
                 refit_days: int = 7) -> pd.DataFrame:
    """Predict every match of `season`, refitting each league every `refit_days`.

    A match is only ever predicted by a model fitted on matches *before* the
    day the refit happened, which is on or before the match date.
    """
    out = []
    for league, lg in matches.groupby("league"):
        target = lg[lg.season == season].sort_values("date")
        if target.empty:
            continue
        model, fitted_at = None, None
        for date, day in target.groupby("date"):
            if model is None or (date - fitted_at).days >= refit_days:
                model, fitted_at = DixonColes(params).fit(lg, as_of=date), date
            out.append(model.predict(day).assign(match_id=day.match_id.to_numpy()))
    return pd.concat(out).set_index("match_id")
