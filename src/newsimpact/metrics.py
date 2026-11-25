"""Forecast-comparison primitives."""
from __future__ import annotations

import numpy as np


def spearman(p: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation (continuous inputs; ties are negligible)."""
    rp, ry = np.argsort(np.argsort(p)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rp, ry)[0, 1])


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, day: np.ndarray,
                    lag: int) -> tuple[float, float]:
    """Diebold–Mariano test of equal accuracy, A vs B (negative = A better).

    Losses are averaged per decision day first (samples of one day share one
    market), then the day series gets a Newey–West variance with `lag`
    autocovariances: consecutive days' target windows overlap by WINDOW-1
    sessions, so lag = WINDOW-1 is the minimum honest choice.
    Returns (mean daily loss difference, t-statistic).
    """
    _, inv = np.unique(day, return_inverse=True)
    d = np.bincount(inv, loss_a - loss_b) / np.bincount(inv)
    u = d - d.mean()
    T = len(d)
    var = u @ u / T
    for k in range(1, min(lag, T - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / T
    return float(d.mean()), float(d.mean() / np.sqrt(var / T))
