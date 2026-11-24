"""Fit every arm on every random split, and predict each split's test samples once.

    HAR        h = w · [1, log variance over 5/22/66/132 sessions]
    intensity  HAR + news intensity                         (how much news)
    shuffled   intensity + text PCs of RANDOM train headlines,
               same count per window                       (content control)
    text       intensity + text PCs of the window's headlines

Text features, per split and from TRAIN only: standardise every embedding
dimension over the training headlines, take their top principal components,
project each window's mean headline onto them, standardise the scores. The
data supports only a few such numbers (~180 independent weeks per split), so
the number of components k and the ridge penalty λ on the text coefficients
are chosen on validation — for `shuffled` exactly as for `text`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from newsimpact.dataset import blocked_split, load
from newsimpact.model import fit_qlike, qlike

ARMS = ("HAR", "intensity", "shuffled", "text")
PCS = (1, 2, 4, 8, 16)
RIDGE = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def split_of(ds: dict, k: int, cfg: dict) -> np.ndarray:
    return blocked_split(ds, k, int(cfg["BLOCK"]), float(cfg["VAL_FRACTION"]),
                         float(cfg["TEST_FRACTION"]))


def window_sums(ds: dict, src: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Σ of the embeddings attached to each sample's window (rows src[lo:hi])."""
    S = np.concatenate([np.zeros((1, E.shape[1])), np.cumsum(E[src], 0)])
    return S[ds["hi"].numpy()] - S[ds["lo"].numpy()]


def _z(X: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu, sd = X[tr].mean(0), X[tr].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def fit_split(ds: dict, k: int, cfg: dict, E: np.ndarray, sums_real: np.ndarray) -> dict:
    split = split_of(ds, k, cfg)
    tr, va, te = (np.where(split == i)[0] for i in range(3))
    y = ds["y"].numpy().astype(np.float64)
    ctx = ds["ctx"].numpy().astype(np.float64)
    n_price = ds["config"]["n_price"]
    count = (ds["hi"] - ds["lo"]).numpy()
    lo, hi, att = ds["lo"].numpy(), ds["hi"].numpy(), ds["att_emb"].numpy()

    rows = np.unique(np.concatenate([att[a:b] for a, b in zip(lo[tr], hi[tr], strict=True)]))
    mu, sd = E[rows].mean(0), E[rows].std(0) + 1e-6
    Vt = np.linalg.svd((E[rows] - mu) / sd, full_matrices=False)[2][: max(PCS)]
    shuf = np.random.default_rng(k).choice(rows, len(att))
    sums_shuf = window_sums(ds, shuf, E)

    def text(sums):                         # PC scores of the standardized window mean
        has = count > 0
        mean = np.zeros_like(sums)
        mean[has] = sums[has] / count[has, None]
        T = ((mean - mu) / sd) @ Vt.T
        T = _z(T, tr)
        T[~has] = 0.0                       # an empty window sits at the train mean
        return T

    one = np.ones((len(y), 1))
    price = np.c_[one, _z(ctx[:, :n_price], tr)]
    inten = np.c_[price, _z(ctx[:, n_price:], tr)]
    out = {"split": k, "idx": te, "y": y[te]}
    sel = {}
    for name, A in (("HAR", price), ("intensity", inten)):
        out[name] = A[te] @ fit_qlike(A[tr], y[tr])
    for name, T in (("text", text(sums_real)), ("shuffled", text(sums_shuf))):
        best = None
        for npc in PCS:
            A = np.c_[inten, T[:, :npc]]
            for lam in RIDGE:
                pen = np.r_[np.zeros(inten.shape[1]), np.full(npc, lam)]
                w = fit_qlike(A[tr], y[tr], pen)
                v = float(qlike(A[va] @ w, y[va]).mean())
                if best is None or v < best[0]:
                    best = (v, npc, lam, A[te] @ w, w[inten.shape[1]:])
        out[name] = best[3]
        sel[name] = {"val_qlike": best[0], "pcs": best[1], "ridge": best[2],
                     "text_coef": best[4].tolist()}
    out["persistence"] = ds["prev"].numpy()[te].astype(np.float64)
    out["val_har"] = float(qlike(price[va] @ fit_qlike(price[tr], y[tr]), y[va]).mean())
    out["selection"] = sel
    return out


def run(cfg: dict) -> None:
    ds = load(cfg["DATASET"])
    E = ds["emb"].numpy().astype(np.float64)
    sums_real = window_sums(ds, ds["att_emb"].numpy(), E)
    out = Path(cfg["STATE_DIR"]) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("split*"):
        old.unlink()
    for k in range(int(cfg["SPLITS"])):
        r = fit_split(ds, k, cfg, E, sums_real)
        np.savez(out / f"split{k}.npz", idx=r["idx"], y=r["y"],
                 **{a: r[a] for a in (*ARMS, "persistence")})
        meta = {"split": k, "digest": ds["digest"], "val_har": r["val_har"],
                "selection": r["selection"]}
        (out / f"split{k}.json").write_text(json.dumps(meta, indent=1))
        s = r["selection"]
        print(f"split {k:2d}: val QLIKE HAR {r['val_har']:.4f}  text {s['text']['val_qlike']:.4f} "
              f"(k={s['text']['pcs']}, λ={s['text']['ridge']:g})  shuffled "
              f"{s['shuffled']['val_qlike']:.4f} (k={s['shuffled']['pcs']}, "
              f"λ={s['shuffled']['ridge']:g})", flush=True)
