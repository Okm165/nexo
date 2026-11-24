"""Fit every arm on every random split, and predict each split's test samples once.

    HAR        h = w · [1, log variance over 5/22/66/132 sessions]
    intensity  HAR + news intensity                          (how much news)
    shuffled   intensity + text PCs of RANDOM train headlines,
               same count per window                         (content control)
    text       intensity + text PCs of the window's headlines

One slope for all symbols (pooled), with a per-symbol level — a fixed-effects
panel regression:

    y - level[symbol]  =  w · (x - mean[symbol])   fitted for all symbols at once
    h                  =  w · (x - mean[symbol]) + level[symbol]

`level[symbol]` is log of the mean train target variance of that symbol, the
exact QLIKE-optimal location; `mean[symbol]` is the symbol's train mean of a
feature. So the level of each stock is estimated from all of its train days
rather than inferred from four noisy lookbacks, and the pooled slope describes
how a stock moves away from its own norm. QLIKE is unchanged by this
transformation (it depends only on y - h), so the fit is exact, adds no
parameter to the solver and no knob. A symbol absent from a split's train
samples falls back to the pooled level and mean.

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

from newsimpact.dataset import blocked_split, load, time_split
from newsimpact.model import fit_qlike, qlike

ARMS = ("HAR", "intensity", "shuffled", "text")
PCS = (16, 64, 256)
RIDGE = (1e-3, 1e-2, 1e-1)
CHUNK = 65536


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


def _chunks(idx: np.ndarray, size: int = CHUNK):
    for s in range(0, len(idx), size):
        yield idx[s:s + size]


def _z(X: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu, sd = X[tr].mean(0), X[tr].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def levels(y: np.ndarray, sym: np.ndarray, tr: np.ndarray, n_sym: int) -> np.ndarray:
    """log of the mean train target variance of each symbol (QLIKE-optimal location).

    d/dh of mean QLIKE(e^y, h) over a symbol's samples vanishes at
    h = log(mean(e^y)), not at the mean of y, so this is the location the
    loss asks for. Symbols with no train sample take the pooled level.
    """
    e = np.exp(y[tr])
    cnt = np.bincount(sym[tr], minlength=n_sym)
    out = np.full(n_sym, np.log(e.mean()))
    ok = cnt > 0
    out[ok] = np.log(np.bincount(sym[tr], weights=e, minlength=n_sym)[ok] / cnt[ok])
    return out


def means(X: np.ndarray, sym: np.ndarray, n_sym: int) -> np.ndarray:
    """Per-symbol arithmetic mean of X's columns; pooled mean where a symbol is absent."""
    cnt = np.bincount(sym, minlength=n_sym)
    M = np.stack([np.bincount(sym, weights=X[:, c], minlength=n_sym)
                  for c in range(X.shape[1])], 1)
    ok = cnt > 0
    M[ok] /= cnt[ok, None]
    M[~ok] = X.mean(0)
    return M


def centre(X: np.ndarray, M: np.ndarray, sym: np.ndarray, start: int = 0) -> np.ndarray:
    """X[:, start:] -= M[sym] in place, in row chunks (bounds temporaries)."""
    for s in range(0, len(X), CHUNK):
        X[s:s + CHUNK, start:] -= M[sym[s:s + CHUNK]]
    return X


def fit_split(ds: dict, k: int, cfg: dict, E: np.ndarray,
              split: np.ndarray | None = None) -> dict:
    split = split_of(ds, k, cfg) if split is None else split
    tr, va, te = (np.where(split == i)[0] for i in range(3))
    y = ds["y"].numpy().astype(np.float64)
    ctx = ds["ctx"].numpy().astype(np.float64)
    sym = ds["sym"].numpy()
    n_sym = len(ds["symbols"])
    n_price = ds["config"]["n_price"]
    lo, hi, att = ds["lo"].numpy(), ds["hi"].numpy(), ds["att_emb"].numpy()
    empty = hi == lo

    level = levels(y, sym, tr, n_sym)
    yd = y - level[sym]                                 # target deviation from its stock's level

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
    P, N = _z(ctx[:, :n_price], tr), _z(ctx[:, n_price:], tr)
    price = np.c_[one, centre(P, means(P[tr], sym[tr], n_sym), sym)]
    inten = np.c_[price, centre(N, means(N[tr], sym[tr], n_sym), sym)]
    q = inten.shape[1]

    def design(src):                     # [one | price dev | intensity dev | text PCs]
        A = np.empty((len(y), q + max(PCS)))
        A[:, :q] = inten
        T = window_means(ds, scores[src], A[:, q:])
        mu_t, sd_t = T[tr].mean(0), T[tr].std(0)
        T -= mu_t
        T /= np.where(sd_t > 0, sd_t, 1.0)
        T[empty] = 0.0                                   # an empty window: train mean
        return A

    def predict(X, w, rows):
        """Deviations -> log variance, with the symbol's level added back."""
        return X @ w + level[sym[rows]]

    out = {"split": k, "idx": te}
    w_har = fit_qlike(price[tr], yd[tr])
    out["HAR"] = predict(price[te], w_har, te)
    w_int = fit_qlike(inten[tr], yd[tr])
    out["intensity"] = predict(inten[te], w_int, te)
    sel = {}
    for name, src in (("text", att), ("shuffled", shuf)):
        A = design(src)
        best = None
        for npc in PCS:
            cols = slice(0, q + npc)
            Atr = A[tr, cols]
            mT = means(Atr[:, q:], sym[tr], n_sym)       # text deviation per symbol
            centre(Atr, mT, sym[tr], start=q)
            w = np.r_[w_int, np.zeros(npc)]              # warm start at the intensity fit
            for lam in sorted(RIDGE, reverse=True):
                pen = np.r_[np.zeros(q), np.full(npc, lam)]
                w = fit_qlike(Atr, yd[tr], pen, w0=w)
                Ava = centre(A[va, cols], mT, sym[va], start=q)
                v = float(qlike(predict(Ava, w, va), y[va]).mean())
                if best is None or v < best[0]:
                    best = (v, npc, lam, predict(centre(A[te, cols], mT, sym[te], start=q), w, te))
            del Atr
        del A
        out[name] = best[3]
        sel[name] = {"val_qlike": best[0], "pcs": best[1], "ridge": best[2]}
    out["persistence"] = ds["prev"].numpy()[te].astype(np.float64)
    out["val_har"] = float(qlike(predict(price[va], w_har, va), y[va]).mean())
    out["val_intensity"] = float(qlike(predict(inten[va], w_int, va), y[va]).mean())
    out["selection"] = sel
    return out


def _save(r: dict, out: Path, digest: str) -> None:
    k = r["split"]
    np.savez(out / f"split{k}.npz", idx=r["idx"], **{a: r[a] for a in (*ARMS, "persistence")})
    meta = {"split": k, "digest": digest, "val_har": r["val_har"],
            "val_intensity": r["val_intensity"], "selection": r["selection"]}
    (out / f"split{k}.json").write_text(json.dumps(meta, indent=1))
    s = r["selection"]
    print(f"split {k:2d}: val QLIKE HAR {r['val_har']:.4f}  intensity "
          f"{r['val_intensity']:.4f}  text {s['text']['val_qlike']:.4f} "
          f"(k={s['text']['pcs']}, λ={s['text']['ridge']:g})  shuffled "
          f"{s['shuffled']['val_qlike']:.4f} (k={s['shuffled']['pcs']}, "
          f"λ={s['shuffled']['ridge']:g})", flush=True)


def run(cfg: dict) -> None:
    """Every arm on every purged random split (validation chooses k and λ)."""
    ds = load(cfg["DATASET"])
    E = ds["emb"].numpy()                                # float32; cast per chunk
    out = Path(cfg["STATE_DIR"]) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("split*"):
        old.unlink()
    for k in range(int(cfg["SPLITS"])):
        _save(fit_split(ds, k, cfg, E), out, ds["digest"])


def run_holdout(cfg: dict) -> None:
    """One time-ordered split: validation chooses k and λ, the test period is
    scored once. Run with its own STATE_DIR so it never mixes with the random
    splits; `evaluate` then reports this fit on its own."""
    ds = load(cfg["DATASET"])
    out = Path(cfg["STATE_DIR"]) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("split*"):
        old.unlink()
    split = time_split(ds, cfg)
    print(f"holdout: train {int((split == 0).sum()):,}  val {int((split == 1).sum()):,}  "
          f"test {int((split == 2).sum()):,} samples", flush=True)
    _save(fit_split(ds, 0, cfg, ds["emb"].numpy(), split=split), out, ds["digest"])
