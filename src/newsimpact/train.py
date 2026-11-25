"""Fit every arm on every walk-forward fold, and predict each fold's test period once.

    HAR        h = w · [1, log variance of the last 1/5/22/66/132 sessions]
    price      HAR + leverage (the negative part of the 1/5/22-session returns)
    intensity  price + news intensity of the last session
    shuffled   intensity + text PCs of RANDOM train headlines,
               same count per window                         (content control)
    text       intensity + text PCs of the window's headlines

One set of coefficients for all symbols (pooled), and nothing is fitted per
symbol: a level estimated from a symbol's own training targets looked good on
shuffled month blocks, where it interpolates the test month, and lost 11% to
this pooled form when tested forward in time. The 132-session lookback is the
symbol's level, known at the close.

Text features, per fold and from TRAIN only: standardise every embedding
dimension over the training headlines, take the top principal components of
their correlation matrix, score every headline on them, and average the scores
over two parts of each window — its last session, and the sessions before it
(news is short-lived, so the two get separate coefficients). Each part is
taken relative to the same day's mean over all symbols with headlines, so a
feed that changes what it publishes moves no forecast by itself. The number
of components k and the ridge penalty λ on the text coefficients are chosen
on validation — for `shuffled` exactly as for `text`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from newsimpact.dataset import fold_split, load
from newsimpact.model import fit_qlike, qlike

HAR = ("logvar1", "logvar5", "logvar22", "logvar66", "logvar132")
PRICE = (*HAR, "negret1", "negret5", "negret22")
GROUPS = {"HAR": HAR, "price": PRICE, "intensity": (*PRICE, "news_last")}
ARMS = (*GROUPS, "shuffled", "text")
FORECAST = "price"                  # the arm `forecast` uses: the last one test supports
PCS = (16, 64, 256)
RIDGE = (1e-3, 1e-2, 1e-1)
K = max(PCS)
CHUNK = 65536


def n_folds(cfg: dict) -> int:
    return len(cfg["FOLDS"].split(",")) - 1


def window_means(scores: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 out: np.ndarray | None = None) -> np.ndarray:
    """Mean of per-attachment scores over each span [lo, hi); 0 where it is empty.

    Computed in column blocks so the cumulative sums never hold all columns.
    """
    n = (hi - lo)[:, None]
    out = np.empty((len(lo), scores.shape[1])) if out is None else out
    for c in range(0, scores.shape[1], 64):
        S = np.concatenate([np.zeros((1, min(64, scores.shape[1] - c))),
                            np.cumsum(scores[:, c:c + 64], 0)])
        out[:, c:c + 64] = np.where(n > 0, (S[hi] - S[lo]) / np.maximum(n, 1), 0.0)
    return out


def pcs(E: np.ndarray, rows: np.ndarray, npc: int) -> np.ndarray:
    """Every embedding row's scores on the top `npc` principal components of
    the correlation matrix of the rows `rows` (float64, chunked)."""
    s1, s2 = np.zeros(E.shape[1]), np.zeros((E.shape[1], E.shape[1]))
    for s in range(0, len(rows), CHUNK):
        X = E[rows[s:s + CHUNK]].astype(np.float64)
        s1 += X.sum(0)
        s2 += X.T @ X
    mu = s1 / len(rows)
    cov = s2 / len(rows) - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(cov), 0)) + 1e-6
    V = np.linalg.eigh(cov / np.outer(sd, sd))[1][:, ::-1][:, :npc]
    return np.concatenate([((E[s:s + CHUNK].astype(np.float64) - mu) / sd) @ V
                           for s in range(0, len(E), CHUNK)])


def day_relative(T: np.ndarray, day: np.ndarray, full: np.ndarray) -> None:
    """In place: subtract from every row of `full` the mean of its day's `full` rows."""
    d = np.unique(day[full], return_inverse=True)[1]
    n = np.bincount(d)
    for c in range(T.shape[1]):
        T[full, c] -= (np.bincount(d, weights=T[full, c]) / n)[d]


def prepare(ds: dict, E: np.ndarray, tr: np.ndarray, live: bool = False):
    """Design matrices with every statistic taken from the rows `tr`.

    Returns the unpenalised designs per arm, a builder of the text design
    [intensity | last-session PCs | earlier-sessions PCs] for a given
    attachment → headline map, and the headlines inside the train windows.
    With `live`, the rows of the latest close (no target yet) are appended.
    """
    parts = [ds, ds["live"]] if live else [ds]
    ctx, lo, mid, hi, pos = (np.concatenate([p[k].numpy() for p in parts])
                             for k in ("ctx", "lo", "mid", "hi", "pos"))
    names = ds["config"]["ctx"]
    ctx = ctx.astype(np.float64)
    sd = ctx[tr].std(0)
    X = (ctx - ctx[tr].mean(0)) / np.where(sd > 0, sd, 1.0)
    one = np.ones((len(X), 1))
    base = {a: np.c_[one, X[:, [names.index(c) for c in g]]] for a, g in GROUPS.items()}
    q = base["intensity"].shape[1]

    att = ds["att_emb"].numpy()
    cover = np.zeros(len(att) + 1)                  # attachments inside train windows
    np.add.at(cover, lo[tr], 1)
    np.add.at(cover, hi[tr], -1)
    rows = np.unique(att[np.cumsum(cover)[:-1] > 0])
    scores = pcs(E, rows, K)

    def design(src: np.ndarray) -> np.ndarray:
        A = np.empty((len(X), q + 2 * K))
        A[:, :q] = base["intensity"]
        S = scores[src]
        for j, (a, b) in enumerate(((mid, hi), (lo, mid))):     # last session, earlier ones
            T = window_means(S, a, b, A[:, q + j * K:q + (j + 1) * K])
            day_relative(T, pos, b > a)                         # empty parts stay 0
            sd_t = T[tr].std(0)
            T /= np.where(sd_t > 0, sd_t, 1.0)
        return A

    return base, design, rows


def text_cols(q: int, npc: int) -> np.ndarray:
    """Columns of the text design using the first `npc` components of each part."""
    return np.r_[0:q, q:q + npc, q + K:q + K + npc]


def fit_fold(ds: dict, k: int, cfg: dict, E: np.ndarray) -> dict:
    split = fold_split(ds, k, cfg["FOLDS"])
    tr, va, te = (np.where(split == i)[0] for i in range(3))
    y = ds["y"].numpy().astype(np.float64)
    base, design, rows = prepare(ds, E, tr)
    out = {"fold": k, "idx": te, "persistence": ds["prev"].numpy()[te].astype(np.float64)}
    val, w = {}, {}
    for a, A in base.items():
        w[a] = fit_qlike(A[tr], y[tr])
        out[a] = A[te] @ w[a]
        val[a] = float(qlike(A[va] @ w[a], y[va]).mean())

    att = ds["att_emb"].numpy()
    shuf = np.random.default_rng(k).choice(rows, len(att))
    q, sel = base["intensity"].shape[1], {}
    for name, src in (("text", att), ("shuffled", shuf)):
        A = design(src)
        best = None
        for npc in PCS:
            cols = text_cols(q, npc)
            Atr, Ava = A[np.ix_(tr, cols)], A[np.ix_(va, cols)]
            wt = np.r_[w["intensity"], np.zeros(2 * npc)]      # warm start
            for lam in sorted(RIDGE, reverse=True):
                wt = fit_qlike(Atr, y[tr], np.r_[np.zeros(q), np.full(2 * npc, lam)], w0=wt)
                v = float(qlike(Ava @ wt, y[va]).mean())
                if best is None or v < best[0]:
                    best = (v, npc, lam, A[np.ix_(te, cols)] @ wt)
            del Atr, Ava
        del A
        out[name], val[name] = best[3], best[0]
        sel[name] = {"pcs": best[1], "ridge": best[2]}
    out["val"], out["selection"] = val, sel
    return out


def run(cfg: dict) -> None:
    ds = load(cfg["DATASET"])
    E = ds["emb"].numpy()                                # float32; cast per chunk
    out = Path(cfg["STATE_DIR"]) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    for old in (*out.glob("fold*"), *out.glob("split*")):
        old.unlink()
    for k in range(n_folds(cfg)):
        r = fit_fold(ds, k, cfg, E)
        np.savez(out / f"fold{k}.npz", idx=r["idx"],
                 **{a: r[a] for a in (*ARMS, "persistence")})
        meta = {"fold": k, "digest": ds["digest"], "val": r["val"], "selection": r["selection"]}
        (out / f"fold{k}.json").write_text(json.dumps(meta, indent=1))
        s = r["selection"]["text"]
        print(f"fold {k}: validation QLIKE  " + "  ".join(f"{a} {r['val'][a]:.4f}" for a in ARMS)
              + f"   (text k={s['pcs']}, λ={s['ridge']:g})", flush=True)


def forecast(cfg: dict) -> dict:
    """Predict the NEXT `WINDOW` sessions' volatility of every symbol from its
    inputs at the latest close in the data — a window that has not happened yet.

    One fit of the FORECAST arm on every sample whose target is known. Writes
    forecast.csv (symbol, day of the close, predicted annualized vol %, and the
    previous window's realized vol %), highest predicted vol first.
    """
    ds = load(cfg["DATASET"])
    y = ds["y"].numpy().astype(np.float64)
    n = len(y)
    A = prepare(ds, ds["emb"].numpy(), np.arange(n), live=True)[0][FORECAST]
    h = A[n:] @ fit_qlike(A[:n], y)
    live = ds["live"]
    vol = np.sqrt(np.exp(h)) * 100
    prev = np.sqrt(np.exp(live["prev"].numpy().astype(np.float64))) * 100
    day = live["day"].numpy().astype("datetime64[D]")
    names = [ds["symbols"][i] for i in live["sym"].numpy()]
    out = Path(cfg["STATE_DIR"]) / "forecast.csv"
    with out.open("w") as f:
        f.write("symbol,day,predicted_vol_pct,previous_window_vol_pct\n")
        for i in np.argsort(-vol):
            f.write(f"{names[i]},{day[i]},{vol[i]:.2f},{prev[i]:.2f}\n")
    print(f"forecast -> {out}: next {ds['config']['WINDOW']} sessions after {day.max()}, "
          f"{len(names)} symbols, `{FORECAST}` arm, median predicted vol "
          f"{np.median(vol):.1f}%", flush=True)
    return {"symbols": names, "vol": vol.tolist(), "day": [str(d) for d in day]}
