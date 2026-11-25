"""Samples, inputs and targets. One sample per (symbol, trading day t).

The window slides ONE session at a time. For every day t:

    target  annualized realized volatility over the NEXT W sessions
            rv = sqrt(252/W · Σ_{k=1..W} r_{t+k}²),  r = daily close-to-close log return
    input   everything known at the close of t:
            * the symbol's headlines from the last W sessions (the set)
            * own log realized vol over the last 5, 22, 66 and 132 sessions
              (week, month, quarter, half-year: short-term state plus the
              stock's own long-run level that volatility reverts to)
            * market log realized vol: the same four numbers averaged over
              every symbol with a clean history that day (the vol regime)
            * news intensity: log(1 + headline count) minus that day's average
              over symbols — relative, so a vendor covering more news over the
              years does not look like rising volatility

Point in time: a headline counts at the first close at or after its
publication; the close is taken as 20:00 UTC (16:00 New York in summer, an
hour early in winter, so the rule can only be conservative). Market and news
averages use only information available at the close of t.

Data hygiene, exact rules (no tuned thresholds):
  * every session of [t-132, t+W] must be present on the market calendar for
    the symbol (gaps from delisting / ticker reuse make "W sessions" a lie)
  * no |log return| > 1 in that span (a >172% day is an unadjusted split)
  * realized vols must be positive (zero means stale prices)
Splits are purged: a sample is in a split only if t AND t+W fall inside it.
All normalisation statistics (embeddings, inputs, target) come from train.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

CLOSE_UTC_HOUR = 20
HAR = (5, 22, 66, 132)      # price lookbacks, sessions
MAX_ABS_LOGRET = 1.0
SPLITS = ("train", "val", "test")
CTX = (*(f"own_rv{k}" for k in HAR), *(f"mkt_rv{k}" for k in HAR), "news_intensity")
N_PRICE = 2 * len(HAR)      # the price columns: no news information


def realized_vol(c2: np.ndarray, end: np.ndarray, k: int) -> np.ndarray:
    """Annualized realized vol of the k returns ending at index `end`.

    `c2` is the cumulative sum of squared log returns with c2[0] = 0, so the
    returns r_{end-k+1} .. r_end sum to c2[end] - c2[end-k].
    """
    return np.sqrt(252.0 / k * (c2[end] - c2[end - k]))


def _bars(path: str):
    t = pq.read_table(path, columns=["symbol", "day", "close"])
    sym = np.asarray(t["symbol"].to_pylist())
    day = t["day"].cast(pa.date32()).to_numpy().astype("datetime64[D]").astype(np.int64)
    close = t["close"].to_numpy(zero_copy_only=False).astype(np.float64)
    ok = np.isfinite(close) & (close > 0)
    sym, day, close = sym[ok], day[ok], close[ok]
    order = np.lexsort((day, sym))
    return sym[order], day[order], close[order]


def _embeddings(prefix: str):
    E, src, shard = [], [], 0
    while Path(f"{prefix}{shard}.npz").exists():
        d = np.load(f"{prefix}{shard}.npz")
        if d["E"].ndim != 2 or d["E"].shape[1] != 768 or len(d["E"]) != len(d["src_idx"]):
            raise ValueError(f"{prefix}{shard}.npz: expected E[N,768] and src_idx[N]")
        E.append(d["E"].astype(np.float32))
        src.append(d["src_idx"].astype(np.int64))
        shard += 1
    if not E:
        raise FileNotFoundError(f"no embedding shards at {prefix}0.npz — run `newsimpact embed`")
    return np.concatenate(E), np.concatenate(src)


def _attachments(cfg: dict, code: dict):
    """(symbol code, publication second, embedding row) for every tagged headline."""
    E, src = _embeddings(cfg["EMB_PREFIX"])
    news = pq.read_table(cfg["NEWS_PATH"], columns=["symbols", "created_at"])
    if src.max() >= news.num_rows:
        raise ValueError("embedding shards index rows beyond NEWS_PATH — different archives")
    news = news.take(pa.array(src))
    pub = news["created_at"].to_numpy().astype("datetime64[s]").astype(np.int64)
    a_sym, a_pub, a_emb = [], [], []
    for k, tags in enumerate(news["symbols"].to_pylist()):
        for s in set(tags or ()):
            c = code.get(s)
            if c is not None:
                a_sym.append(c)
                a_pub.append(pub[k])
                a_emb.append(k)
    a_sym, a_pub, a_emb = map(np.asarray, (a_sym, a_pub, a_emb))
    order = np.lexsort((a_pub, a_sym))
    return E, a_sym[order], a_pub[order], a_emb[order]


def build(cfg: dict) -> dict:
    W = int(cfg["WINDOW"])
    L = max(HAR)
    train_end = np.datetime64(cfg["TRAIN_END"]).astype(np.int64)
    val_end = np.datetime64(cfg["VAL_END"]).astype(np.int64)

    sym, day, close = _bars(cfg["BARS_PATH"])
    calendar = np.unique(day)
    symbols, sym_start = np.unique(sym, return_index=True)
    bounds = [*sym_start, len(sym)]
    E, a_sym, a_pub, a_emb = _attachments(cfg, {s: i for i, s in enumerate(symbols)})
    a_bounds = np.searchsorted(a_sym, np.arange(len(symbols) + 1))

    # pass 1: per symbol, every day with a clean price history (market inputs),
    # and the subset that is a sample (news in window, clean target window)
    hist_day, hist_own = [], []
    cols = {k: [] for k in ("sym", "t", "day", "lo", "hi", "own", "y", "prev", "split")}
    att_emb, att_info, att_day, n_att = [], [], [], 0
    for c in range(len(symbols)):
        d, px = day[bounds[c]:bounds[c + 1]], close[bounds[c]:bounds[c + 1]]
        n = len(d)
        info = np.searchsorted(d * 86400 + CLOSE_UTC_HOUR * 3600,
                               a_pub[a_bounds[c]:a_bounds[c + 1]], side="left")
        keep = info < n
        info = info[keep]
        att_emb.append(a_emb[a_bounds[c]:a_bounds[c + 1]][keep])
        att_info.append(info)
        att_day.append(d[info])

        r = np.diff(np.log(px))                        # r[i-1] = ln(c_i / c_{i-1})
        c2 = np.concatenate([[0.0], np.cumsum(r ** 2)])
        bad = np.concatenate([[0], np.cumsum(np.abs(r) > MAX_ABS_LOGRET)])
        pos = np.searchsorted(calendar, d)             # position on the market calendar
        t = np.arange(L, n)
        own = np.stack([realized_vol(c2, t, k) for k in HAR], 1)
        hist_ok = ((bad[t] - bad[t - L] == 0) & (pos[t] - pos[t - L] == L)
                   & (own > 0).all(1))
        hist_day.append(d[t][hist_ok])
        hist_own.append(np.log(own[hist_ok]))

        t = t[t + W < n]
        if len(t) == 0:
            n_att += len(info)
            continue
        own = own[: len(t)]
        lo = np.searchsorted(info, t - W + 1, side="left")
        hi = np.searchsorted(info, t, side="right")
        y = realized_vol(c2, t + W, W)
        prev = realized_vol(c2, t, W)
        split = np.full(len(t), -1, np.int8)
        split[d[t + W] <= train_end] = 0
        split[(d[t] > train_end) & (d[t + W] <= val_end)] = 1
        split[d[t] > val_end] = 2
        m = (hist_ok[: len(t)] & (bad[t + W] - bad[t] == 0) & (pos[t + W] - pos[t] == W)
             & (y > 0) & (prev > 0) & (hi > lo) & (split >= 0))
        for k, v in (("sym", np.full(len(t), c, np.int32)), ("t", t), ("day", d[t]),
                     ("lo", lo + n_att), ("hi", hi + n_att), ("split", split)):
            cols[k].append(v[m])
        for k, v in (("own", own), ("y", y), ("prev", prev)):       # positive after masking
            cols[k].append(np.log(v[m]))
        n_att += len(info)

    s = {k: np.concatenate(v) for k, v in cols.items()}
    for i, name in enumerate(SPLITS):
        if not (s["split"] == i).any():
            raise ValueError(f"empty {name} split — check TRAIN_END/VAL_END against the data")

    # market regime: same-day mean over every symbol with a clean history
    hd, ho = np.concatenate(hist_day), np.concatenate(hist_own)
    mdays, inv = np.unique(hd, return_inverse=True)
    mkt = np.stack([np.bincount(inv, ho[:, j]) / np.bincount(inv) for j in range(len(HAR))], 1)
    mkt = mkt[np.searchsorted(mdays, s["day"])]
    # news intensity: count relative to the same day's cross-section of samples
    lc = np.log1p(s["hi"] - s["lo"])
    _, dinv = np.unique(s["day"], return_inverse=True)
    lc = lc - (np.bincount(dinv, lc) / np.bincount(dinv))[dinv]
    s["ctx"] = np.c_[s.pop("own"), mkt, lc].astype(np.float32)
    s["y"], s["prev"] = s["y"].astype(np.float32), s["prev"].astype(np.float32)

    att_emb, att_info, att_day = map(np.concatenate, (att_emb, att_info, att_day))
    used, att_emb = np.unique(att_emb, return_inverse=True)
    E = E[used]
    # embeddings: standardise every dimension with train-period statistics.
    # Raw mean-pooled vectors share one dominant direction (mean cosine ~0.9
    # between random headlines); without this the content is a small residue.
    tr_rows = np.unique(att_emb[att_day <= train_end])
    mu, sd = E[tr_rows].mean(0), E[tr_rows].std(0) + 1e-6
    E = ((E - mu) / sd).astype(np.float16)
    # shuffled-text control: each attachment gets a random train-period
    # headline (fixed seed), keeping count and age — only content changes
    att_shuf = np.random.default_rng(0).choice(att_emb[att_day <= train_end], len(att_emb))

    te = s["split"] == 2
    digest = hashlib.md5(np.concatenate(
        [s["sym"][te], s["t"][te], s["y"][te].view(np.int32)]).tobytes()).hexdigest()[:12]
    ds = {
        "config": {"WINDOW": W, "TRAIN_END": cfg["TRAIN_END"], "VAL_END": cfg["VAL_END"],
                   "HAR": list(HAR), "ctx": list(CTX), "n_price": N_PRICE,
                   "close_utc_hour": CLOSE_UTC_HOUR},
        "digest": digest, "symbols": list(symbols),
        "emb": torch.from_numpy(E),
        "att_emb": torch.from_numpy(att_emb.astype(np.int64)),
        "att_shuf": torch.from_numpy(att_shuf.astype(np.int64)),
        "att_info": torch.from_numpy(att_info.astype(np.int64)),
        **{k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in s.items()},
    }
    out = cfg["DATASET"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ds, out + ".tmp")
    os.replace(out + ".tmp", out)
    _summary(ds)
    print(f"dataset -> {out}")
    return ds


def _summary(ds: dict) -> None:
    n = (ds["hi"] - ds["lo"]).numpy()
    for i, name in enumerate(SPLITS):
        m = ds["split"].numpy() == i
        vol = np.exp(ds["y"].numpy()[m]) * 100
        d = ds["day"].numpy()[m].astype("datetime64[D]")
        print(f"{name:5s} {m.sum():>8,} samples  {d.min()}..{d.max()}  "
              f"headlines/sample p50={np.median(n[m]):.0f} p95={np.percentile(n[m], 95):.0f}  "
              f"next-window vol p50={np.median(vol):.1f}%")
    print(f"{len(ds['emb']):,} headlines, {len(ds['att_emb']):,} symbol attachments, "
          f"digest {ds['digest']}")


def load(path: str) -> dict:
    ds = torch.load(path, weights_only=False, map_location="cpu")
    for k in ("emb", "att_emb", "att_shuf", "att_info", "lo", "hi", "t", "ctx", "y", "split"):
        if k not in ds:
            raise ValueError(f"{path}: missing {k} — rebuild with `newsimpact build`")
    if not torch.isfinite(ds["y"]).all() or not torch.isfinite(ds["ctx"]).all():
        raise ValueError(f"{path}: non-finite inputs or targets")
    return ds
