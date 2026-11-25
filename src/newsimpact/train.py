"""Train one (arm, seed). Arms differ only in what the text pathway sees:

    text      the real headlines of the window
    shuffled  random TRAIN-period headlines, same count and ages (content control)
    price     no text pathway and no news-derived inputs at all

Loss: MSE on standardised log realized vol; AdamW with cosine decay. The
epoch is chosen on validation; the test split is predicted once, from the
best-validation weights, and saved for `newsimpact evaluate`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from newsimpact.dataset import load
from newsimpact.model import VolForecaster

ARMS = ("text", "shuffled", "price")


class Batcher:
    """Assemble padded headline sets on the device by slicing shared arrays."""

    def __init__(self, ds: dict, arm: str, max_headlines: int, window: int, device):
        self.text = arm != "price"
        self.W, self.L = window, max_headlines
        self.lo, self.hi, self.t = (ds[k].to(device) for k in ("lo", "hi", "t"))
        if self.text:
            self.emb = ds["emb"].to(device)
            self.src = ds["att_shuf" if arm == "shuffled" else "att_emb"].to(device)
            self.info = ds["att_info"].to(device)

    def __call__(self, idx):
        if not self.text:
            return None, None, None
        hi = self.hi[idx]
        n = (hi - self.lo[idx]).clamp(max=self.L)          # most recent L headlines
        j = torch.arange(int(n.max()), device=idx.device)
        mask = j[None] < n[:, None]
        pos = torch.where(mask, (hi - n)[:, None] + j[None], 0)
        age = (self.t[idx][:, None] - self.info[pos]).clamp(0, self.W - 1)
        return self.emb[self.src[pos]].float(), age, mask


def _predict(model, batch, ctx, idx, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            i = idx[s:s + bs]
            out.append(model(ctx[i], *batch(i)))
    model.train()
    return torch.cat(out)


def run(arm: str, seed: int, cfg: dict, device: str = "cuda") -> dict:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    ds = load(cfg["DATASET"])
    W = int(ds["config"]["WINDOW"])
    split = ds["split"]
    tr, va, te = (torch.where(split == i)[0].to(device) for i in range(3))

    # the price arm sees only price columns: no news information of any kind
    ctx_all = ds["ctx"][:, : ds["config"]["n_price"]] if arm == "price" else ds["ctx"]
    y_all = ds["y"]
    trc = split == 0
    c_mu, c_sd = ctx_all[trc].mean(0), ctx_all[trc].std(0)
    c_sd = torch.where(c_sd > 0, c_sd, torch.ones_like(c_sd))   # a constant column stays 0
    y_mu, y_sd = float(y_all[trc].mean()), float(y_all[trc].std())
    ctx = ((ctx_all - c_mu) / c_sd).to(device)
    y = ((y_all - y_mu) / y_sd).to(device)

    torch.manual_seed(seed)
    model = VolForecaster(ctx.shape[1], W, text=arm != "price",
                          dropout=float(cfg["DROPOUT"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["LR"]),
                            weight_decay=float(cfg["WEIGHT_DECAY"]))
    batch = Batcher(ds, arm, int(cfg["MAX_HEADLINES"]), W, device)
    order = torch.Generator(device="cpu").manual_seed(seed)   # same batch order across arms
    bs, epochs = int(cfg["BATCH"]), int(cfg["EPOCHS"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * -(-len(tr) // bs))

    best = (float("inf"), 0, None)
    curve = []
    for ep in range(1, epochs + 1):
        t0, total = time.time(), 0.0
        perm = tr[torch.randperm(len(tr), generator=order).to(device)]
        for s in range(0, len(perm), bs):
            i = perm[s:s + bs]
            loss = torch.nn.functional.mse_loss(model(ctx[i], *batch(i)), y[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(i)
        val = float(((_predict(model, batch, ctx, va) - y[va]) ** 2).mean())
        if not np.isfinite(val):
            raise FloatingPointError(f"[{arm} s{seed}] non-finite validation loss at epoch {ep}")
        curve.append(val)
        if val < best[0]:
            best = (val, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f"[{arm} s{seed}] ep{ep:02d} train={total / len(tr):.4f} val={val:.4f} "
              f"best={best[0]:.4f}@{best[1]} ({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(best[2])
    pred = (_predict(model, batch, ctx, te) * y_sd + y_mu).cpu().numpy()   # log vol
    out = Path(cfg["STATE_DIR"]) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{arm}_s{seed}.npy", pred)
    row = {"arm": arm, "seed": seed, "digest": ds["digest"], "best_epoch": best[1],
           "val_mse_z": best[0], "val_curve": curve,
           "test_mse_log": float(np.mean((pred - y_all[split == 2].numpy()) ** 2))}
    (out / f"{arm}_s{seed}.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k != "val_curve"}), flush=True)
    return row
