"""Score every arm on every walk-forward fold's test period.

Metric: QLIKE (robust to the noisy variance proxy), and skill vs HAR
(1 − QLIKE/QLIKE_HAR). Every comparison gets a Diebold–Mariano test on the
daily mean loss difference, per test period and over all test days together;
the per-period results show whether a pooled result holds in each year or
comes from one of them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nexo.dataset import load
from nexo.metrics import diebold_mariano
from nexo.model import qlike
from nexo.train import ARMS

NAMES = ("persistence", *ARMS)
PAIRS = (("HAR", "persistence"), ("price", "HAR"), ("intensity", "price"),
         ("text", "intensity"), ("text", "shuffled"))


def _score(loss: dict, pos: np.ndarray, group: np.ndarray, lag: int) -> dict:
    q = {n: float(loss[n].mean()) for n in NAMES}
    return {"samples": len(pos), "days": len(np.unique(pos)), "qlike": q,
            "skill_vs_har": {n: 1 - q[n] / q["HAR"] for n in NAMES},
            "dm_t": {f"{a} vs {b}": diebold_mariano(loss[a], loss[b], pos, group, lag)[1]
                     for a, b in PAIRS}}


def main(cfg: dict) -> dict:
    ds = load(cfg["DATASET"])
    W = int(ds["config"]["WINDOW"])
    y_all, pos_all = ds["y"].numpy().astype(np.float64), ds["pos"].numpy()
    day_all = ds["day"].numpy().astype("datetime64[D]")
    files = sorted((Path(cfg["STATE_DIR"]) / "runs").glob("fold*.json"),
                   key=lambda f: int(f.stem[4:]))
    if not files:
        raise SystemExit("no fitted folds — run `nexo train`")
    folds, parts = [], []
    for f in files:
        meta = json.loads(f.read_text())
        if meta["digest"] != ds["digest"]:
            raise SystemExit(f"{f}: fitted on a different dataset — rerun `nexo train`")
        z = np.load(f.with_suffix(".npz"))
        idx = z["idx"]
        loss = {n: qlike(z[n], y_all[idx]) for n in NAMES}
        group = np.full(len(idx), meta["fold"])
        folds.append({"fold": meta["fold"], "from": str(day_all[idx].min()),
                      "to": str(day_all[idx].max()), "selection": meta["selection"],
                      "val_qlike": meta["val"], **_score(loss, pos_all[idx], group, W - 1)})
        parts.append((loss, pos_all[idx], group, z["text"], y_all[idx]))
    loss = {n: np.concatenate([p[0][n] for p in parts]) for n in NAMES}
    pos, group, h, y = (np.concatenate([p[i] for p in parts]) for i in range(1, 5))
    dec = np.clip(np.searchsorted(np.quantile(h, np.linspace(0, 1, 11)), h, "right") - 1, 0, 9)

    def vol(v):                             # mean variance of a decile, shown as vol %
        return float(np.sqrt(np.exp(v).mean()) * 100)

    rep = {"window": W, "digest": ds["digest"], "folds": folds,
           "all": _score(loss, pos, group, W - 1),
           "calibration_text": [{"decile": i + 1, "pred_vol_pct": vol(h[dec == i]),
                                 "realized_vol_pct": vol(y[dec == i])} for i in range(10)]}
    _print(rep)
    out = Path(cfg["STATE_DIR"]) / "eval.json"
    out.write_text(json.dumps(rep, indent=1))
    print(f"-> {out}")
    return rep


def _print(rep: dict) -> None:
    rows = [*((f"{f['from']}..{f['to']}", f) for f in rep["folds"]), ("all test days", rep["all"])]
    print(f"next-{rep['window']}-session Yang–Zhang variance, walk-forward: "
          f"test QLIKE (skill vs HAR)")
    print(f"{'test period':24s}{'days':>5s}" + "".join(f"{n:>17s}" for n in NAMES))
    for lab, r in rows:
        print(f"{lab:24s}{r['days']:5d}" + "".join(
            f"{r['qlike'][n]:9.4f}({r['skill_vs_har'][n]:+6.1%})" for n in NAMES))
    print("Diebold–Mariano t on QLIKE (negative: the first is better)")
    print(f"{'':24s}" + "".join(f"{lab[:10]:>12s}" for lab, _ in rows[:-1]) + f"{'all':>12s}")
    for a, b in PAIRS:
        p = f"{a} vs {b}"
        print(f"{p:24s}" + "".join(f"{r['dm_t'][p]:+12.2f}" for _, r in rows))
    print("text: (k, λ) chosen on validation: "
          + ", ".join(f"({f['selection']['text']['pcs']}, {f['selection']['text']['ridge']:g})"
                      for f in rep["folds"]))
    print("calibration (text, all test days): predicted → realized vol by decile")
    print("  " + "  ".join(f"{r['pred_vol_pct']:.0f}→{r['realized_vol_pct']:.0f}"
                           for r in rep["calibration_text"]))
