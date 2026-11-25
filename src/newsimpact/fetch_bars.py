"""Fetch daily bars for the news universe (resumable, idempotent).

Optional adapter: any table with columns symbol, day, close works as
BARS_PATH. This one builds it from Alpaca for the news archive's symbols:

    universe = symbols in news.parquet with >= MIN_NEWS articles, in descending
               article order, capped at TOP_N candidates; a symbol enters the
               table only if Alpaca returns daily bars for it.

Every batch is cached as JSONL under data/derived/bars_cache/, so a kill costs
at most one batch. Re-running with the cache present is a no-op.

    newsimpact fetch-bars                 # populate cache, then write bars.parquet
    FORCE_FETCH=1 newsimpact fetch-bars   # ignore the cache
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from newsimpact.env import load_env

ROOT = Path(__file__).resolve().parents[2]
BARS_OUT = ROOT / load_env().get("BARS_PATH", "data/derived/bars.parquet")
BARS_ADJUSTMENT = os.environ.get("BARS_ADJUSTMENT", "split")
# resumable cache, keyed by adjustment so differently adjusted fetches never mix
CACHE = (ROOT / "data" / "derived" /
         f"bars_cache_{os.environ.get('BARS_ADJUSTMENT', 'split')}_oc")
START = "2020-11-01"
TOP_N = int(os.environ.get("UNIVERSE_TOP_N", "1200"))
MIN_NEWS = int(os.environ.get("UNIVERSE_MIN_NEWS", "20"))
BATCH = int(os.environ.get("FETCH_BATCH", "20"))


def _keys() -> tuple[str, str]:
    kid = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not (kid and sec):                     # fall back to the .env file
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("APCA_API_KEY_ID="):
                kid = line.split("=", 1)[1].strip()
            if line.startswith("APCA_API_SECRET_KEY="):
                sec = line.split("=", 1)[1].strip()
    if not (kid and sec):
        raise RuntimeError("no Alpaca credentials in env or .env")
    return kid, sec


def _universe() -> list[str]:
    # Universe = the news archive's own symbols (NEWS_PATH), or the symbols of
    # an existing bars table (BARS_UNIVERSE_FROM) to extend it in place.
    from_existing = os.environ.get("BARS_UNIVERSE_FROM")
    if from_existing:
        t = pq.read_table(from_existing, columns=["symbol"])
        syms = sorted(set(t.column("symbol").to_pylist()))
        print(f"universe source: {from_existing} ({len(syms)} symbols)", flush=True)
        return syms
    src = (os.environ.get("NEWS_UNIVERSE_SRC") or load_env().get("NEWS_PATH")
           or str(BARS_OUT.parent / "news.parquet"))
    src = Path(src)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        raise FileNotFoundError(
            f"no universe source at {src} — set NEWS_PATH, NEWS_UNIVERSE_SRC "
            f"or BARS_UNIVERSE_FROM")
    t = pq.read_table(src, columns=["symbols"])
    print(f"universe source: {src}", flush=True)
    c: Counter[str] = Counter()
    for syms in t["symbols"].to_pylist():
        for s in (syms or []):
            c[s] += 1
    keep = [(s, n) for s, n in c.most_common() if n >= MIN_NEWS][:TOP_N]
    print(f"universe: {len(keep)} candidates (>= {MIN_NEWS} articles, "
          f"top {TOP_N}); most covered {keep[0]}", flush=True)
    return [s for s, _ in keep]


def _get(url: str, kid: str, sec: str) -> dict:
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec,
        "accept": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                raise                        # client error: retrying cannot help
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    return {}


def _fetch_batch(syms: list[str], kid: str, sec: str, out: Path) -> int:
    """Fetch a symbol batch; a 400 means a bad symbol, so bisect instead of abort.

    The API rejects the whole request when one symbol in the list is invalid
    (crypto tickers from the news archive, delisted names). Dropping the batch
    would silently shrink the universe, so the failure is isolated by halving
    until the offending symbol is alone, and only then skipped.
    """
    try:
        return _fetch_pages(syms, kid, sec, out)
    except urllib.error.HTTPError as e:
        if e.code != 400 or len(syms) == 1:
            if len(syms) == 1:
                print(f"  skip {syms[0]}: {e}", flush=True)
                return 0
            raise
        mid = len(syms) // 2
        return (_fetch_batch(syms[:mid], kid, sec, out)
                + _fetch_batch(syms[mid:], kid, sec, out))


def _fetch_pages(syms: list[str], kid: str, sec: str, out: Path) -> int:
    rows = 0
    token = None
    while True:
        q = {"symbols": ",".join(syms), "timeframe": "1Day", "start": START,
             "limit": "10000", "adjustment": BARS_ADJUSTMENT,
             "feed": "sip"}
        if token:
            q["page_token"] = token
        data = _get("https://data.alpaca.markets/v2/stocks/bars?"
                    + urllib.parse.urlencode(q), kid, sec)
        for sym, bars in (data.get("bars") or {}).items():
            for b in bars:
                out.write(json.dumps({"symbol": sym, "day": b["t"][:10],
                                      "open": b["o"], "close": b["c"]}) + "\n")
                rows += 1
        token = data.get("next_page_token")
        if not token:
            break
    return rows


def main() -> None:
    import hashlib

    import pyarrow.compute as pc
    kid, sec = _keys()
    syms = _universe()
    # Batch numbers index INTO a specific symbol list. Resuming a run built
    # from a different universe would mix symbols under the same filenames —
    # the provenance trap, on a JSONL cache. Key the directory by content.
    digest = hashlib.md5(",".join(syms).encode()).hexdigest()[:8]
    cache_dir = CACHE / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"universe digest {digest} -> {cache_dir}", flush=True)
    force = os.environ.get("FORCE_FETCH", "0") == "1"
    for i in range(0, len(syms), BATCH):
        chunk = syms[i:i + BATCH]
        out = cache_dir / f"batch_{i:05d}.jsonl"
        if out.exists() and out.stat().st_size > 0 and not force:
            continue
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as f:
            n = _fetch_batch(chunk, kid, sec, f)
        os.replace(tmp, out)
        print(f"batch {i:5d}: {len(chunk)} symbols, {n} bars", flush=True)

    rows: dict[tuple[str, str], tuple[float | None, float]] = {}
    for f in sorted(cache_dir.glob("batch_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                o = r.get("open")
                rows[(r["symbol"], r["day"])] = (
                    None if o is None or not (o > 0) else float(o),
                    float(r["close"]))
    keys = sorted(rows)
    days = [k[1] for k in keys]          # ISO strings: the pipeline contract
    tbl = pa.table({"symbol": pa.array([k[0] for k in keys], pa.string()),
                    "day": pa.array(days, pa.string()),
                    "open": pa.array([rows[k][0] for k in keys], pa.float64()),
                    "close": pa.array([rows[k][1] for k in keys], pa.float64())})
    BARS_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{BARS_OUT}.tmp"
    pq.write_table(tbl, tmp)
    os.replace(tmp, BARS_OUT)
    nsym = len(pc.unique(tbl["symbol"]))
    print(f"bars: {tbl.num_rows} rows, {nsym} symbols, "
          f"{min(days)}..{max(days)} -> {BARS_OUT}", flush=True)


if __name__ == "__main__":
    main()
