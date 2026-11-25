"""Score every trained run on the test split, against two benchmarks.

    persistence  next-window vol = previous-window vol (same length)
    HAR          OLS of log next-window vol on own and market log vol over
                 5/22/66/132 sessions, fitted on TRAIN — the same price inputs
                 the networks see, so it is the bar text has to clear

Per arm the seeds are averaged in log space (an ensemble). Reported per
predictor: MSE and R² of log vol, skill vs HAR (1 − MSE/MSE_HAR), MAE in vol
points, rank correlation, MSE per test year (a real gain holds in every
year), and a decile calibration table (geometric means, matching the
log-space forecast). Pairwise Diebold–Mariano tests answer
"is the difference real". The noise floor is the MSE a perfect forecast
would still have if returns were Gaussian: Var[log sqrt(χ²_W/W)] = ψ'(W/2)/4.
Fat tails make the true floor higher.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from newsimpact.dataset import load
from newsimpact.metrics import diebold_mariano, spearman

PAIRS = (("text", "shuffled"), ("text", "price"), ("text", "HAR"), ("price", "HAR"),
         ("HAR", "persistence"))


def _har(ds: dict) -> np.ndarray:
    split, y = ds["split"].numpy(), ds["y"].numpy()
    X = np.c_[np.ones(len(y)), ds["ctx"].numpy()[:, : ds["config"]["n_price"]]]
    w = np.linalg.lstsq(X[split == 0], y[split == 0], rcond=None)[0]
    return X[split == 2] @ w


def main(cfg: dict) -> dict:
    ds = load(cfg["DATASET"])
    te = ds["split"].numpy() == 2
    y = ds["y"].numpy()[te]
    day = ds["day"].numpy()[te]
    year = day.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    W = int(ds["config"]["WINDOW"])

    preds = {"persistence": ds["prev"].numpy()[te], "HAR": _har(ds)}
    runs: dict[str, list[np.ndarray]] = {}
    for f in sorted(glob.glob(os.path.join(cfg["STATE_DIR"], "runs", "*.json"))):
        row = json.loads(Path(f).read_text())
        if row["digest"] != ds["digest"]:
            raise SystemExit(f"{f}: trained on a different dataset ({row['digest']} != "
                             f"{ds['digest']}) — retrain or move it aside")
        runs.setdefault(row["arm"], []).append(np.load(f[:-5] + ".npy"))
    for arm, ps in runs.items():
        preds[arm] = np.mean(ps, axis=0)

    mse = {k: float(np.mean((p - y) ** 2)) for k, p in preds.items()}
    table = {}
    for name, p in preds.items():
        table[name] = {"mse_log": mse[name], "r2": 1 - mse[name] / float(np.var(y)),
                       "skill_vs_har": 1 - mse[name] / mse["HAR"],
                       "mae_vol_pts": float(np.mean(np.abs(np.exp(p) - np.exp(y)))) * 100,
                       "spearman": spearman(p, y), "bias_log": float(np.mean(p - y)),
                       "mse_by_year": {int(a): float(np.mean((p - y)[year == a] ** 2))
                                       for a in np.unique(year)}}
        if name in runs:
            s = [float(np.mean((q - y) ** 2)) for q in runs[name]]
            table[name] |= {"seeds": len(s), "seed_mse_mean": float(np.mean(s)),
                            "seed_mse_sd": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0}

    tests = {}
    for a, b in PAIRS:
        if a in preds and b in preds:
            diff, t = diebold_mariano((preds[a] - y) ** 2, (preds[b] - y) ** 2, day, W - 1)
            tests[f"{a} vs {b}"] = {"mse_diff": diff, "dm_t": t}

    best = "text" if "text" in preds else "HAR"
    q = np.quantile(preds[best], np.linspace(0, 1, 11))
    k = np.clip(np.searchsorted(q, preds[best], side="right") - 1, 0, 9)
    # the model forecasts log vol, so compare geometric means (exp of mean log):
    # an arithmetic mean of exp(y) would add Jensen's exp(σ²/2) to one side only
    calib = [{"decile": i + 1, "pred_vol_pct": float(np.exp(preds[best][k == i].mean()) * 100),
              "realized_vol_pct": float(np.exp(y[k == i].mean()) * 100)} for i in range(10)]

    floor = float(0.25 * torch.polygamma(1, torch.tensor(W / 2.0)))
    rep = {"test_samples": int(te.sum()), "test_days": len(np.unique(day)), "window": W,
           "digest": ds["digest"], "noise_floor_gaussian": floor,
           "target_variance": float(np.var(y)), "predictors": table, "dm_tests": tests,
           "calibration_" + best: calib}
    _print(rep, best)
    out = os.path.join(cfg["STATE_DIR"], "eval.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"-> {out}")
    return rep


def _print(rep: dict, best: str) -> None:
    years = sorted(next(iter(rep["predictors"].values()))["mse_by_year"])
    print(f"test: {rep['test_samples']:,} samples over {rep['test_days']} days, "
          f"next-{rep['window']}-session realized vol; target variance "
          f"{rep['target_variance']:.3f}, "
          f"Gaussian noise floor {rep['noise_floor_gaussian']:.3f}")
    print(f"{'predictor':12s} {'MSE':>7s} {'R²':>6s} {'skill/HAR':>9s} {'MAE pts':>8s} "
          f"{'Spearman':>8s} {'bias':>6s} " + " ".join(f"{y:>7d}" for y in years)
          + "  seeds MSE mean±sd")
    for name, m in rep["predictors"].items():
        s = (f"  {m['seeds']}: {m['seed_mse_mean']:.4f}±{m['seed_mse_sd']:.4f}"
             if "seeds" in m else "")
        print(f"{name:12s} {m['mse_log']:7.4f} {m['r2']:6.3f} {m['skill_vs_har']:+9.2%} "
              f"{m['mae_vol_pts']:8.2f} {m['spearman']:8.4f} {m['bias_log']:+6.3f} "
              + " ".join(f"{m['mse_by_year'][y]:7.4f}" for y in years) + s)
    print("Diebold–Mariano (MSE diff per day, negative = first is better):")
    for k, v in rep["dm_tests"].items():
        print(f"  {k:22s} {v['mse_diff']:+.5f}  t={v['dm_t']:+.2f}")
    print(f"calibration ({best}): geometric-mean predicted → realized vol, by decile")
    for r in rep["calibration_" + best]:
        print(f"  {r['decile']:2d}  {r['pred_vol_pct']:6.1f}% → {r['realized_vol_pct']:6.1f}%")
