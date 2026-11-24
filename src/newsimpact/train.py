"""Fit every arm on every random split, and predict each split's test samples once.

    HAR        h = w · [1, log variance over 5/22/66/132 sessions]
    intensity  HAR + news intensity                          (how much news)
    shuffled   intensity + text PCs of RANDOM train headlines,
               same count per window                         (content control)
    text       intensity + text PCs of the window's headlines

One set of coefficients for all symbols (pooled), so the text coefficients
are estimated from every stock's history; results are reported for the
whole universe and for TARGET.

Text features, per split and from TRAIN only: standardise every embedding
dimension over the training headlines, take the top principal components of
their covariance, score every headline on them, average the scores over each
window (= projecting the window's mean headline), standardise. The number of
components k and the ridge penalty λ on the text coefficients are chosen on
validation — for `shuffled` exactly as for `text`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from newsimpact.dataset import blocked_split, load
from newsimpact.model import fit_qlike, qlike

ARMS = ("HAR", "intensity", "shuffled", "text")
PCS = (16, 64, 256)
RIDGE = (1e-3, 1e-2, 1e-1)


def split_of(ds: dict, k: int, cfg: dict) -> np.ndarray:
    return blocked_split(ds, k, int(cfg["BLOCK"]), float(cfg["VAL_FRACTION"]),
                         float(cfg["TEST_FRACTION"]))


def window_means(ds: dict, scores: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Mean of per-attachment scores over each sample's window [lo, hi); 0 if empty.

    Computed in column blocks so the cumulative sums never hold all columns.
    """
    lo, hi = ds["lo"].numpy(), ds["hi"].numpy()
    n = (hi - lo)[:, None]
    out = np.empty((len(lo), scores.shape[1])) if out is None else out
    for c in range(0, scores.shape[1], 64):
        S = np.concatenate([np.zeros((1, min(64, scores.shape[1] - c))),
                            np.cumsum(scores[:, c:c + 64], 0)])
        out[:, c:c + 64] = np.where(n > 0, (S[hi] - S[lo]) / np.maximum(n, 1), 0.0)
    return out


def _chunks(idx: np.ndarray, size: int = 65536):
    for s in range(0, len(idx), size):
        yield idx[s:s + size]


def _z(X: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu, sd = X[tr].mean(0), X[tr].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def fit_split(ds: dict, k: int, cfg: dict, E: np.ndarray) -> dict:
    split = split_of(ds, k, cfg)
    tr, va, te = (np.where(split == i)[0] for i in range(3))
    y = ds["y"].numpy().astype(np.float64)
    ctx = ds["ctx"].numpy().astype(np.float64)
    n_price = ds["config"]["n_price"]
    lo, hi, att = ds["lo"].numpy(), ds["hi"].numpy(), ds["att_emb"].numpy()
    empty = hi == lo

    cover = np.zeros(len(att) + 1)                  # attachments inside train windows
    np.add.at(cover, lo[tr], 1)
    np.add.at(cover, hi[tr], -1)
    rows = np.unique(att[np.cumsum(cover)[:-1] > 0])
    # moments and covariance of the train headlines, accumulated in float64 chunks
    s1, s2 = np.zeros(E.shape[1]), np.zeros((E.shape[1], E.shape[1]))
    for c in _chunks(rows):
        X = E[c].astype(np.float64)
        s1 += X.sum(0)
        s2 += X.T @ X
    mu = s1 / len(rows)
    cov = s2 / len(rows) - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(cov), 0)) + 1e-6
    corr = cov / np.outer(sd, sd)                        # covariance of standardised vectors
    V = np.linalg.eigh(corr)[1][:, ::-1][:, : max(PCS)]  # top components, descending
    scores = np.concatenate([((E[c].astype(np.float64) - mu) / sd) @ V
                             for c in _chunks(np.arange(len(E)))])
    shuf = np.random.default_rng(k).choice(rows, len(att))

    one = np.ones((len(y), 1))
    price = np.c_[one, _z(ctx[:, :n_price], tr)]
    inten = np.c_[price, _z(ctx[:, n_price:], tr)]
    q = inten.shape[1]

    def design(src):                     # [intensity | text PCs], filled in place
        A = np.empty((len(y), q + max(PCS)))
        A[:, :q] = inten
        T = window_means(ds, scores[src], A[:, q:])
        mu_t, sd_t = T[tr].mean(0), T[tr].std(0)
        T -= mu_t
        T /= np.where(sd_t > 0, sd_t, 1.0)
        T[empty] = 0.0                                   # an empty window: train mean
        return A
    out = {"split": k, "idx": te}
    w_har = fit_qlike(price[tr], y[tr])
    out["HAR"] = price[te] @ w_har
    w_int = fit_qlike(inten[tr], y[tr])
    out["intensity"] = inten[te] @ w_int
    sel = {}
    for name, src in (("text", att), ("shuffled", shuf)):
        A = design(src)
        best = None
        for npc in PCS:
            cols = slice(0, q + npc)
            Atr = A[tr, cols]
            w = np.r_[w_int, np.zeros(npc)]              # warm start at the intensity fit
            for lam in sorted(RIDGE, reverse=True):
                pen = np.r_[np.zeros(q), np.full(npc, lam)]
                w = fit_qlike(Atr, y[tr], pen, w0=w)
                v = float(qlike(A[va, cols] @ w, y[va]).mean())
                if best is None or v < best[0]:
                    best = (v, npc, lam, A[te, cols] @ w)
            del Atr
        del A
        out[name] = best[3]
        sel[name] = {"val_qlike": best[0], "pcs": best[1], "ridge": best[2]}
    out["persistence"] = ds["prev"].numpy()[te].astype(np.float64)
    out["val_har"] = float(qlike(price[va] @ w_har, y[va]).mean())
    out["val_intensity"] = float(qlike(inten[va] @ w_int, y[va]).mean())
    out["selection"] = sel
    return out


def run(cfg: dict) -> None:
    ds = load(cfg["DATASET"])
    E = ds["emb"].numpy()                                # float32; cast per chunk
    out = Path(cfg["STATE_DIR"]) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("split*"):
        old.unlink()
    for k in range(int(cfg["SPLITS"])):
        r = fit_split(ds, k, cfg, E)
        np.savez(out / f"split{k}.npz", idx=r["idx"], **{a: r[a] for a in (*ARMS, "persistence")})
        meta = {"split": k, "digest": ds["digest"], "val_har": r["val_har"],
                "val_intensity": r["val_intensity"], "selection": r["selection"]}
        (out / f"split{k}.json").write_text(json.dumps(meta, indent=1))
        s = r["selection"]
        print(f"split {k:2d}: val QLIKE HAR {r['val_har']:.4f}  intensity "
              f"{r['val_intensity']:.4f}  text {s['text']['val_qlike']:.4f} "
              f"(k={s['text']['pcs']}, λ={s['text']['ridge']:g})  shuffled "
              f"{s['shuffled']['val_qlike']:.4f} (k={s['shuffled']['pcs']}, "
              f"λ={s['shuffled']['ridge']:g})", flush=True)
