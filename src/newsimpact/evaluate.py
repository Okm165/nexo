"""Score every arm on every split's test samples.

Primary metric QLIKE (robust to the noisy variance proxy); also skill vs HAR
(1 − QLIKE/QLIKE_HAR) and rank correlation — for the whole universe and for
TARGET, from the same pooled fits. Each split gets Diebold–Mariano
tests; the summary is mean ± sd across splits and the number of splits in
which the first forecast is better — no conclusion rests on one draw of
test months.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

from newsimpact.dataset import load
from newsimpact.metrics import diebold_mariano, spearman
from newsimpact.model import qlike
from newsimpact.train import ARMS

NAMES = ("persistence", *ARMS)
PAIRS = (("text", "shuffled"), ("text", "intensity"), ("text", "HAR"),
         ("intensity", "HAR"), ("HAR", "persistence"))


def main(cfg: dict) -> dict:
    ds = load(cfg["DATASET"])
    W, pos_all = int(ds["config"]["WINDOW"]), ds["pos"].numpy()
    y_all, sym_all = ds["y"].numpy().astype(np.float64), ds["sym"].numpy()
    target = cfg["TARGET"]
    target_code = ds["symbols"].index(target) if target in ds["symbols"] else -1
    splits: dict[str, list] = {"universe": [], target: []}
    files = glob.glob(os.path.join(cfg["STATE_DIR"], "runs", "split*.json"))
    for f in sorted(files, key=lambda f: int(Path(f).stem[5:])):
        meta = json.loads(Path(f).read_text())
        if meta["digest"] != ds["digest"]:
            raise SystemExit(f"{f}: fitted on a different dataset — rerun `newsimpact train`")
        z = np.load(f[:-5] + ".npz")
        for scope, m in (("universe", np.ones(len(z["idx"]), bool)),
                         (target, sym_all[z["idx"]] == target_code)):
            if not m.any():
                continue
            idx = z["idx"][m]
            y, pos = y_all[idx], pos_all[idx]
            loss = {n: qlike(z[n][m], y) for n in NAMES}
            splits[scope].append({
                "split": meta["split"], "test_samples": len(y), "selection": meta["selection"],
                "qlike": {n: float(loss[n].mean()) for n in NAMES},
                "skill_vs_har": {n: 1 - float(loss[n].mean() / loss["HAR"].mean())
                                 for n in NAMES},
                "spearman": {n: spearman(z[n][m], y) for n in NAMES},
                "dm_t": {f"{a} vs {b}": diebold_mariano(loss[a], loss[b], pos,
                                                        pos // int(cfg["BLOCK"]), W - 1)[1]
                         for a, b in PAIRS},
                "h": z["text"][m], "y": y})
    if not splits["universe"]:
        raise SystemExit("no fitted splits — run `newsimpact train`")
    rep = {"window": W, "digest": ds["digest"],
           **{scope: _report(ss) for scope, ss in splits.items() if ss}}
    for scope in (target, "universe"):
        if scope in rep:
            _print(scope, rep[scope], W)
    out = os.path.join(cfg["STATE_DIR"], "eval.json")
    Path(out).write_text(json.dumps(rep, indent=1))
    print(f"-> {out}")
    return rep


def _report(splits: list) -> dict:
    def agg(key):
        return {n: (float(np.mean([s[key][n] for s in splits])),
                    float(np.std([s[key][n] for s in splits], ddof=1)) if len(splits) > 1
                    else 0.0) for n in splits[0][key]}

    def vol(v):                             # mean variance per decile, shown as vol %
        return float(np.sqrt(np.exp(v).mean()) * 100)

    h = np.concatenate([s["h"] for s in splits])
    y = np.concatenate([s["y"] for s in splits])
    dec = np.clip(np.searchsorted(np.quantile(h, np.linspace(0, 1, 11)), h, "right") - 1, 0, 9)
    return {"n_splits": len(splits), "test_samples": [s["test_samples"] for s in splits],
            "qlike": agg("qlike"), "skill_vs_har": agg("skill_vs_har"),
            "spearman": agg("spearman"), "dm_t": agg("dm_t"),
            "first_better": {p: int(sum(s["dm_t"][p] < 0 for s in splits))
                             for p in splits[0]["dm_t"]},
            "selected_pcs": [s["selection"]["text"]["pcs"] for s in splits],
            "selected_ridge": [s["selection"]["text"]["ridge"] for s in splits],
            "calibration_text": [{"decile": i + 1, "pred_vol_pct": vol(h[dec == i]),
                                  "realized_vol_pct": vol(y[dec == i])} for i in range(10)],
            "splits": [{k: v for k, v in s.items() if k not in ("h", "y")} for s in splits]}


def _print(scope: str, rep: dict, W: int) -> None:
    n = rep["n_splits"]
    print(f"[{scope}] next-{W}-session Yang–Zhang variance, {n} random purged block splits, "
          f"test samples/split ≈ {int(np.mean(rep['test_samples']))}")
    print(f"{'predictor':12s} {'QLIKE':>16s} {'skill vs HAR':>18s} {'Spearman':>16s}")
    for name in rep["qlike"]:
        q, s, r = rep["qlike"][name], rep["skill_vs_har"][name], rep["spearman"][name]
        print(f"{name:12s} {q[0]:8.4f}±{q[1]:.4f} {s[0]:+9.2%}±{s[1]:.2%} {r[0]:8.4f}±{r[1]:.4f}")
    print("Diebold–Mariano on QLIKE (mean t over splits; first better in k of n):")
    for p, t in rep["dm_t"].items():
        print(f"  {p:22s} t={t[0]:+6.2f}±{t[1]:.2f}   first better in {rep['first_better'][p]}/{n}")
    print(f"text model: PCs chosen {rep['selected_pcs']}, ridge {rep['selected_ridge']}")
    print(f"calibration (text, pooled test, {scope}): predicted → realized vol by decile")
    for r in rep["calibration_text"]:
        print(f"  {r['decile']:2d}  {r['pred_vol_pct']:6.1f}% → {r['realized_vol_pct']:6.1f}%")
