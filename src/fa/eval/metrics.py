"""Scoring rules for 3-way (Home / Draw / Away) probabilistic forecasts.

All functions take
    probs:    array (n, 3) with columns [P(home), P(draw), P(away)]
    outcomes: array (n,) of 'H' / 'D' / 'A'
Lower is better for everything except accuracy.
"""
from __future__ import annotations

import numpy as np

OUTCOMES = np.array(["H", "D", "A"])
EPS = 1e-15


def one_hot(outcomes) -> np.ndarray:
    outcomes = np.asarray(outcomes)
    return (outcomes[:, None] == OUTCOMES[None, :]).astype(float)


def log_loss(probs, outcomes) -> float:
    """Average negative log-probability given to what actually happened."""
    p = np.clip(np.asarray(probs, float), EPS, 1)
    return float(-np.mean(np.log((p * one_hot(outcomes)).sum(axis=1))))


def brier(probs, outcomes) -> float:
    """Multi-class Brier score: mean squared error over the 3 outcomes (range 0..2)."""
    return float(np.mean(((np.asarray(probs, float) - one_hot(outcomes)) ** 2).sum(axis=1)))


def rps(probs, outcomes) -> float:
    """Ranked Probability Score, the standard football metric.

    H < D < A are *ordered*: predicting a draw when the home side wins is less wrong
    than predicting an away win. RPS rewards that; log loss and Brier do not.
    """
    cum_p = np.cumsum(np.asarray(probs, float), axis=1)[:, :2]
    cum_o = np.cumsum(one_hot(outcomes), axis=1)[:, :2]
    return float(np.mean(((cum_p - cum_o) ** 2).sum(axis=1) / 2))


def accuracy(probs, outcomes) -> float:
    return float(np.mean(OUTCOMES[np.argmax(probs, axis=1)] == np.asarray(outcomes)))


def all_metrics(probs, outcomes) -> dict:
    return {"log_loss": log_loss(probs, outcomes), "brier": brier(probs, outcomes),
            "rps": rps(probs, outcomes), "accuracy": accuracy(probs, outcomes),
            "n": len(outcomes)}


def bootstrap_diff(probs_a, probs_b, outcomes, metric=log_loss, n_boot=2000, seed=0):
    """Paired bootstrap of metric(A) - metric(B). Returns (mean, lo95, hi95).

    If the whole interval is below 0, model A is reliably better on this data.
    """
    rng = np.random.default_rng(seed)
    outcomes = np.asarray(outcomes)
    pa, pb = np.asarray(probs_a), np.asarray(probs_b)
    n = len(outcomes)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(pa[idx], outcomes[idx]) - metric(pb[idx], outcomes[idx])
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def calibration_table(p_event, happened, bins=10):
    """For one outcome (e.g. home win): predicted vs observed frequency per bin."""
    p_event, happened = np.asarray(p_event), np.asarray(happened, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_event, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:                      # skip near-empty bins: too noisy
            rows.append((p_event[m].mean(), happened[m].mean(), int(m.sum())))
    return rows
