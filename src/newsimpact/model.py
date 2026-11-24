"""The model: a linear forecast of log variance, fitted by penalized QLIKE.

    h = w · [1, log-variance lookbacks, news intensity, text PCs]

QLIKE (per sample, h = log forecast variance, y = log proxy variance):
    L = e^{y-h} − (y − h) − 1 ≥ 0,  zero iff the forecast equals the proxy.
It ranks forecasts as the true variance would even though the proxy is noisy
(Patton 2011), and it is convex in h, hence in w: the fit below is exact
(damped Newton, float64), with an L2 penalty on chosen coefficients.
"""
from __future__ import annotations

import numpy as np


def qlike(h: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = y - h
    return np.exp(r) - r - 1


def fit_qlike(A: np.ndarray, y: np.ndarray, penalty: np.ndarray | None = None,
              w0: np.ndarray | None = None, iters: int = 100) -> np.ndarray:
    """argmin_w mean QLIKE(A w, y) + Σ penalty_j w_j²  (convex; damped Newton).

    The optimum does not depend on the start; `w0` only saves iterations.
    """
    A, y = A.astype(np.float64), y.astype(np.float64)
    P = np.diag(np.zeros(A.shape[1]) if penalty is None else penalty)
    w = np.linalg.lstsq(A, y, rcond=None)[0] if w0 is None else w0.astype(np.float64)

    def f(w):
        return qlike(A @ w, y).mean() + w @ P @ w

    for _ in range(iters):
        e = np.exp(y - A @ w)
        g = A.T @ (1 - e) / len(y) + 2 * P @ w
        step = np.linalg.solve(A.T @ (A * e[:, None]) / len(y) + 2 * P, g)
        lr, f0 = 1.0, f(w)
        while f(w - lr * step) > f0 and lr > 1e-10:
            lr /= 2
        w = w - lr * step
        if np.abs(lr * step).max() < 1e-12:
            break
    return w
