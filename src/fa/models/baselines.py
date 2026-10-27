"""Reference models. A real model is only interesting if it beats these."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fa.eval.metrics import OUTCOMES


def uniform(df: pd.DataFrame) -> np.ndarray:
    """Knows nothing: 1/3 each."""
    return np.full((len(df), 3), 1 / 3)


class LeagueBaseRates:
    """Knows home advantage and draw rates per league, nothing about the teams."""

    def fit(self, train: pd.DataFrame) -> "LeagueBaseRates":
        counts = pd.crosstab(train["league"], train["result"]).reindex(columns=OUTCOMES, fill_value=0)
        self.rates = counts.div(counts.sum(axis=1), axis=0)
        self.overall = counts.sum() / counts.values.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        rates = self.rates.reindex(df["league"]).fillna(self.overall)
        return rates.to_numpy()


def bookmaker(df: pd.DataFrame) -> np.ndarray:
    """The market: implied probabilities from decimal odds, margin removed proportionally.

    Rows without odds come back as NaN; evaluate this baseline on rows with odds only.
    """
    inv = 1 / df[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    return inv / inv.sum(axis=1, keepdims=True)
