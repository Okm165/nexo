"""Samples, inputs and targets. One sample per (symbol, trading day t).

The window slides ONE session at a time. For every day t:

    target  log annualized Yang–Zhang variance of the NEXT W sessions
            (t+1 .. t+W), from daily open/high/low/close
    input   everything known at the close of t:
            * the symbol's headlines from the last W sessions, each with its
              age in sessions (possibly none — the model has a "no news" token)
            * own log Yang–Zhang variance over the last 5, 22, 66, 132 sessions
            * news intensity: log(1 + headlines in the window) minus
              log(1 + the window-length share of the last 132 sessions'
              headlines) — attention relative to the symbol's own recent norm,
              so a vendor covering more over the years is not "more news"

Yang–Zhang (2000), for the n days of a window:
    o = ln(O_i / C_{i-1})   overnight      c = ln(C_i / O_i)   open-to-close
    rs = u(u - c) + d(d - c),  u = ln(H_i/O_i), d = ln(L_i/O_i)   (Rogers–Satchell)
    σ² = Var(o) + k Var(c) + (1 - k) mean(rs),  k = 0.34 / (1.34 + (n+1)/(n-1))
Annualized ×252. Unbiased under drift and opening jumps, and several times
more efficient than close-to-close for the same window.

Point in time: a headline counts at the first close at or after its
publication; the close is taken as 20:00 UTC (16:00 New York in summer, an
hour early in winter, so the rule can only be conservative).

Data hygiene, exact rules:
  * a bar is valid only if 0 < low ≤ min(open, close) ≤ max(open, close) ≤ high
  * every session of [t-132, t+W] must be present on the market calendar and
    valid, with no |log close-to-close| > 1 (an unadjusted split)
  * every variance used must be > 0 (zero means the price did not trade)
  * samples start at SAMPLE_START, where the news archive's coverage begins:
    before it an empty window means "not collected", not "no news". Earlier
    bars still serve as price history.

The dataset is split-free; `blocked_split` assigns samples (see there).
Nothing here is fitted to data, so nothing here can leak.
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
LOOKBACKS = (5, 22, 66, 132)        # sessions
MAX_ABS_LOGRET = 1.0
CTX = (*(f"logvar{k}" for k in LOOKBACKS), "news_intensity")
N_PRICE = len(LOOKBACKS)            # the first columns carry no news information


def _windowed(x: np.ndarray, end: np.ndarray, n: int):
    """Sum and sum of squares of x over the n entries ending at `end` (inclusive)."""
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])
    return s1[end + 1] - s1[end + 1 - n], s2[end + 1] - s2[end + 1 - n]


def yang_zhang(o: np.ndarray, c: np.ndarray, rs: np.ndarray, end: np.ndarray,
               n: int) -> np.ndarray:
    """Annualized Yang–Zhang variance of the n days ending at index `end`."""
    so, so2 = _windowed(o, end, n)
    sc, sc2 = _windowed(c, end, n)
    srs, _ = _windowed(rs, end, n)
    # sample variances; cancellation can leave -1e-19 where the truth is 0
    var_o = np.maximum(so2 - so * so / n, 0.0) / (n - 1)
    var_c = np.maximum(sc2 - sc * sc / n, 0.0) / (n - 1)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return 252.0 * (var_o + k * var_c + (1 - k) * srs / n)


def blocked_split(ds: dict, seed: int, block: int, val_frac: float,
                  test_frac: float) -> np.ndarray:
    """Purged blocked random split: 0 train / 1 val / 2 test / -1 dropped.

    The market calendar is cut into consecutive blocks of `block` sessions,
    and blocks — not days — are assigned to train/val/test uniformly at
    random, so every split samples every market regime. A sample is kept only
    if its whole span, from the first session of its headline window to the
    last session of its target window, lies in blocks of ONE split: no target
    session and no headline is shared across splits. (Price lookbacks reach
    further back; they carry no target of another split.)
    """
    W = ds["config"]["WINDOW"]
    pos = ds["pos"].numpy() - int(ds["pos"].min()) + W     # blocks tile the sampled span
    n_blocks = (int(pos.max()) + W) // block + 1
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    n_te, n_va = round(test_frac * n_blocks), round(val_frac * n_blocks)
    assign = np.zeros(n_blocks, np.int8)
    assign[order[:n_te]] = 2
    assign[order[n_te:n_te + n_va]] = 1
    first, last = assign[(pos - W + 1) // block], assign[(pos + W) // block]
    return np.where(first == last, assign[pos // block], -1).astype(np.int8)


def time_split(ds: dict, cfg: dict) -> np.ndarray:
    """Time-ordered split: 0 train < VAL_START ≤ 1 validation < TEST_START ≤ 2 test.

    Purged exactly as `blocked_split` purges: a sample is kept only if its
    whole span, from the first session of its headline window to the last
    session of its target window, lies in one segment. So no train sample's
    target reaches the test period and no test sample's headlines reach the
    training period — at least W sessions of gap on each side. Unlike the
    random splits, this one is a single forward-in-time draw, so it can be
    read once as a clean check on the repeated random splits.
    """
    W = int(ds["config"]["WINDOW"])
    day = pq.read_table(cfg["BARS_PATH"], columns=["day"])["day"]
    cal = np.unique(day.cast(pa.date32()).to_numpy().astype("datetime64[D]").astype(np.int64))
    val_start = np.datetime64(cfg["VAL_START"]).astype(np.int64)
    test_start = np.datetime64(cfg["TEST_START"]).astype(np.int64)
    seg = np.where(cal < val_start, 0, np.where(cal < test_start, 1, 2)).astype(np.int8)
    pos = ds["pos"].numpy()
    first, last = seg[pos - W + 1], seg[pos + W]
    return np.where(first == last, seg[pos], -1).astype(np.int8)


def _bars(path: str):
    t = pq.read_table(path, columns=["symbol", "day", "open", "high", "low", "close"])
    sym = np.asarray(t["symbol"].to_pylist())
    day = t["day"].cast(pa.date32()).to_numpy().astype("datetime64[D]").astype(np.int64)
    ohlc = np.stack([t[k].to_numpy(zero_copy_only=False).astype(np.float64)
                     for k in ("open", "high", "low", "close")], 1)
    order = np.lexsort((day, sym))
    sym, day, ohlc = sym[order], day[order], ohlc[order]
    # sorted by (symbol, day), so a duplicate session is adjacent; a repeated
    # bar would silently corrupt every windowed sum downstream
    if len(sym) > 1 and np.any((sym[1:] == sym[:-1]) & (day[1:] == day[:-1])):
        raise ValueError("bars: duplicate (symbol, day) rows — deduplicate the source")
    return sym, day, ohlc


def _embeddings(prefix: str):
    E, src, shard = [], [], 0
    while Path(f"{prefix}{shard}.npz").exists():
        d = np.load(f"{prefix}{shard}.npz")
        if d["E"].ndim != 2 or len(d["E"]) != len(d["src_idx"]) or \
                (E and d["E"].shape[1] != E[0].shape[1]):
            raise ValueError(f"{prefix}{shard}.npz: expected E[N, width] (same width in "
                             f"every shard) and src_idx[N]")
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


def _digest(arrays: list[np.ndarray]) -> str:
    """md5 over every input array (shape + bytes, hashed in 64 MB chunks): any
    change to any input that training or evaluation reads changes the digest,
    so stale runs are refused instead of silently mixed with new data."""
    h = hashlib.md5()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(np.array(a.shape, dtype=np.int64).tobytes())
        b = a.ravel().view(np.uint8)
        for c in range(0, len(b), 1 << 26):
            h.update(b[c:c + (1 << 26)])
    return h.hexdigest()[:12]


def build(cfg: dict) -> dict:
    W = int(cfg["WINDOW"])
    L = max(LOOKBACKS)
    first_day = np.datetime64(cfg["SAMPLE_START"]).astype(np.int64)
    sym, day, ohlc = _bars(cfg["BARS_PATH"])
    calendar = np.unique(day)
    symbols, sym_start = np.unique(sym, return_index=True)
    bounds = [*sym_start, len(sym)]
    E, a_sym, a_pub, a_emb = _attachments(cfg, {s: i for i, s in enumerate(symbols)})
    a_bounds = np.searchsorted(a_sym, np.arange(len(symbols) + 1))

    cols = {k: [] for k in ("sym", "t", "pos", "day", "lo", "hi", "own", "news", "y", "prev")}
    att_emb, att_info, n_att = [], [], 0
    for c in range(len(symbols)):
        d, bars = day[bounds[c]:bounds[c + 1]], ohlc[bounds[c]:bounds[c + 1]]
        n = len(d)
        info = np.searchsorted(d * 86400 + CLOSE_UTC_HOUR * 3600,
                               a_pub[a_bounds[c]:a_bounds[c + 1]], side="left")
        keep = info < n
        info = info[keep]
        att_emb.append(a_emb[a_bounds[c]:a_bounds[c + 1]][keep])
        att_info.append(info)

        p_open, p_high, p_low, p_close = bars.T
        valid = ((p_low > 0) & (p_low <= np.minimum(p_open, p_close))
                 & (np.maximum(p_open, p_close) <= p_high))
        # logs of valid bars only (invalid ones are excluded below via `bad`)
        lo_, lh, ll, logc = np.log(np.where(valid[:, None], bars, 1.0)).T
        o = np.concatenate([[0.0], lo_[1:] - logc[:-1]])   # overnight; day 0 has none
        cc, u, dn = logc - lo_, lh - lo_, ll - lo_
        rs = u * (u - cc) + dn * (dn - cc)                 # Rogers–Satchell, ≥ 0
        ok_day = valid.copy()
        ok_day[0] = False
        ok_day[1:] &= valid[:-1] & (np.abs(np.diff(logc)) <= MAX_ABS_LOGRET)
        bad = np.concatenate([[0], np.cumsum(~ok_day)])
        pos = np.searchsorted(calendar, d)

        t = np.arange(L, n - W)
        if len(t) == 0:
            n_att += len(info)
            continue
        clean = (bad[t + W + 1] - bad[t - L + 1] == 0) & (pos[t + W] - pos[t - L] == L + W)
        own = np.stack([yang_zhang(o, cc, rs, t, k) for k in LOOKBACKS], 1)
        y = yang_zhang(o, cc, rs, t + W, W)
        prev = yang_zhang(o, cc, rs, t, W)
        daily = np.bincount(info, minlength=n).astype(np.float64)
        cum = np.concatenate([[0.0], np.cumsum(daily)])
        n_win = cum[t + 1] - cum[t + 1 - W]
        base = (cum[t + 1] - cum[t + 1 - L]) * W / L
        m = clean & (own > 0).all(1) & (y > 0) & (prev > 0) & (d[t] >= first_day)
        lo = np.searchsorted(info, t - W + 1, side="left")
        hi = np.searchsorted(info, t, side="right")
        for k, v in (("sym", np.full(len(t), c, np.int32)), ("t", t), ("pos", pos[t]),
                     ("day", d[t]), ("lo", lo + n_att), ("hi", hi + n_att),
                     ("news", np.log1p(n_win) - np.log1p(base))):
            cols[k].append(v[m])
        for k, v in (("own", own), ("y", y), ("prev", prev)):       # positive after masking
            cols[k].append(np.log(v[m]))
        n_att += len(info)

    s = {k: np.concatenate(v) for k, v in cols.items()}
    if not len(s["y"]):
        raise ValueError("no samples — check BARS_PATH coverage against the lookbacks")
    s["ctx"] = np.c_[s.pop("own"), s.pop("news")].astype(np.float32)
    s["y"], s["prev"] = s["y"].astype(np.float32), s["prev"].astype(np.float32)

    att_emb, att_info = map(np.concatenate, (att_emb, att_info))
    used, att_emb = np.unique(att_emb, return_inverse=True)
    digest = _digest([s["sym"], s["t"], s["pos"], s["day"], s["lo"], s["hi"],
                      s["y"], s["prev"], s["ctx"], att_emb, att_info, E])
    ds = {
        "config": {"WINDOW": W, "SAMPLE_START": cfg["SAMPLE_START"],
                   "LOOKBACKS": list(LOOKBACKS), "ctx": list(CTX),
                   "n_price": N_PRICE, "close_utc_hour": CLOSE_UTC_HOUR},
        "digest": digest, "symbols": list(symbols),
        "emb": torch.from_numpy(E[used]),                       # raw float32
        "att_emb": torch.from_numpy(att_emb.astype(np.int64)),
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
    d = ds["day"].numpy().astype("datetime64[D]")
    vol = np.sqrt(np.exp(ds["y"].numpy())) * 100
    print(f"{len(n):,} samples  {d.min()}..{d.max()}  headlines/window "
          f"p50={np.median(n):.0f} p95={np.percentile(n, 95):.0f} none={np.mean(n == 0):.1%}  "
          f"next-window vol p50={np.median(vol):.1f}%")
    print(f"{len(ds['emb']):,} headlines, {len(ds['att_emb']):,} attachments, "
          f"digest {ds['digest']}")


def load(path: str) -> dict:
    ds = torch.load(path, weights_only=False, map_location="cpu")
    for k in ("emb", "att_emb", "att_info", "lo", "hi", "t", "pos", "day", "ctx", "y", "prev"):
        if k not in ds:
            raise ValueError(f"{path}: missing {k} — rebuild with `newsimpact build`")
    if not torch.isfinite(ds["y"]).all() or not torch.isfinite(ds["ctx"]).all():
        raise ValueError(f"{path}: non-finite inputs or targets")
    return ds
