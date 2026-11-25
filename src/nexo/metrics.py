"""Forecast-comparison primitives."""
from __future__ import annotations

import math

import numpy as np


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


def t_pvalue(t: float, df: int) -> float:
    """Two-sided p-value of Student's t: I_x(df/2, 1/2) with x = df/(df+t²).

    The incomplete beta is integrated after substituting u = x·v^(1/a), which
    makes the integrand (1 − x·v^(1/a))^(−1/2) bounded on [0, 1] (Simpson).
    """
    if t == 0:
        return 1.0
    a, x = df / 2, df / (df + t * t)
    v = np.linspace(0.0, 1.0, 20001)
    f = (1 - x * v ** (1 / a)) ** -0.5
    integral = (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum()) / (3 * (len(v) - 1))
    log_b = math.lgamma(a) + math.lgamma(0.5) - math.lgamma(a + 0.5)
    return float(min(1.0, x ** a / a * integral / math.exp(log_b)))


def fold_meta(ts: list[float]) -> dict:
    """Combine per-fold DM t-statistics from independent test periods.

    t-test that their mean is zero (heterogeneity across periods enters the
    standard error, so a result carried by one period does not pass), and a
    two-sided sign test on the count of periods in which the first forecast is
    better — distribution-free, but with n periods its p cannot go below
    2^(1-n). With few periods neither is decisive alone; report both.
    """
    ts = np.asarray(ts, dtype=np.float64)
    n, k = len(ts), int((ts < 0).sum())
    tail = sum(math.comb(n, j) for j in range(min(k, n - k) + 1)) / 2 ** n
    out = {"first_better": k, "n": n, "sign_p": min(1.0, 2 * tail)}
    if n < 2 or ts.std(ddof=1) == 0:
        return out | {"t": float("nan"), "p": float("nan")}
    t = float(ts.mean() / (ts.std(ddof=1) / math.sqrt(n)))
    return out | {"t": t, "p": t_pvalue(t, n - 1)}
