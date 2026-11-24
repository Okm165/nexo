"""Forecast-comparison primitives."""
from __future__ import annotations

import numpy as np


def spearman(p: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation (continuous inputs; ties are negligible)."""
    rp, ry = np.argsort(np.argsort(p)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rp, ry)[0, 1])


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, pos: np.ndarray,
                    group: np.ndarray, lag: int) -> tuple[float, float]:
    """Diebold–Mariano test of equal accuracy, A vs B (negative = A better).

    Losses are averaged per calendar position (samples of one day share one
    market). The Newey–West variance uses autocovariances up to `lag` only
    between days of the same contiguous `group` (test block): consecutive
    targets overlap by WINDOW-1 sessions, so lag = WINDOW-1; days in different
    blocks are separated by excluded data and are not serially paired.
    Returns (mean daily loss difference, t-statistic).
    """
    days, inv = np.unique(pos, return_inverse=True)
    d = np.bincount(inv, loss_a - loss_b) / np.bincount(inv)
    g = np.zeros(len(days), np.int64)
    g[inv] = group
    u = d - d.mean()
    T = len(d)
    var = u @ u / T
    for k in range(1, min(lag, T - 1) + 1):
        same = (g[k:] == g[:-k]) & (days[k:] - days[:-k] == k)
        var += 2 * (1 - k / (lag + 1)) * (u[k:][same] @ u[:-k][same]) / T
    if var <= 0:                       # identical forecasts: equally accurate, t = 0
        return float(d.mean()), 0.0
    return float(d.mean()), float(d.mean() / np.sqrt(var / T))
