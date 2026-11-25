"""The model: a linear forecast of log variance, fitted by penalized QLIKE.

    h = w · [1, log-variance lookbacks, news intensity, text PCs]

QLIKE (per sample, h = log forecast variance, y = log proxy variance):
    L = e^{y-h} − (y − h) − 1 ≥ 0,  zero iff the forecast equals the proxy.
It ranks forecasts as the true variance would even though the proxy is noisy
(Patton 2011), and it is convex in h, hence in w: the fit below is exact
(damped Newton, float64; on the GPU when one is available), with an L2
penalty on chosen coefficients.
"""
from __future__ import annotations

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 262_144                       # rows per Hessian block (bounds device memory)


def qlike(h: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = y - h
    with np.errstate(over="ignore"):     # e^r overflowing means loss = +inf: exact
        return np.exp(r) - r - 1


def fit_qlike(A: np.ndarray, y: np.ndarray, penalty: np.ndarray | None = None,
              w0: np.ndarray | None = None, iters: int = 100) -> np.ndarray:
    """argmin_w mean QLIKE(A w, y) + Σ penalty_j w_j²  (convex; damped Newton).

    The optimum does not depend on the start; `w0` only saves iterations.
    """
    A = torch.as_tensor(np.ascontiguousarray(A), dtype=torch.float64, device=DEVICE)
    y = torch.as_tensor(y, dtype=torch.float64, device=DEVICE)
    n, p = A.shape
    pen = torch.zeros(p, dtype=torch.float64, device=DEVICE) if penalty is None else \
        torch.as_tensor(penalty, dtype=torch.float64, device=DEVICE)
    if w0 is None:
        w = torch.from_numpy(np.linalg.lstsq(A.cpu().numpy(), y.cpu().numpy(),
                                             rcond=None)[0]).to(DEVICE)
    else:
        w = torch.as_tensor(w0, dtype=torch.float64, device=DEVICE)

    def f(w):
        r = y - A @ w
        return float((torch.exp(r) - r - 1).mean() + (pen * w * w).sum())

    for _ in range(iters):
        e = torch.exp(y - A @ w)
        g = A.T @ (1 - e) / n + 2 * pen * w
        H = torch.diag(2 * pen)
        for s in range(0, n, CHUNK):
            a = A[s:s + CHUNK]
            H += a.T @ (a * e[s:s + CHUNK, None]) / n
        # the p×p solve is cheap: do it in numpy float64, which raises on a
        # singular system instead of aborting the process (raises; some GPU
        # solver libraries abort instead)
        step = torch.from_numpy(np.linalg.solve(H.cpu().numpy(), g.cpu().numpy())).to(DEVICE)
        lr, f0 = 1.0, f(w)
        while f(w - lr * step) > f0 and lr > 1e-10:
            lr /= 2
        w = w - lr * step
        if float((lr * step).abs().max()) < 1e-12:
            break
    return w.cpu().numpy()
