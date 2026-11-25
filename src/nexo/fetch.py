"""Optional Alpaca adapter: daily OHLC bars and news headlines for SYMBOLS.

Writes the two input tables the pipeline needs (any source with the same
columns works just as well):

    BARS_PATH  symbol, day, open, high, low, close   (split-adjusted, SIP feed)
    NEWS_PATH  symbols (list), created_at (UTC), headline

    nexo fetch-bars            # SYMBOLS / START from .env.data
    nexo fetch-news            # replaces NEWS_PATH — point it elsewhere
                                     # if you already hold a longer archive
Data ends at today 00:00 UTC: an unfinished session is never fetched (and
the SIP feed refuses the latest 15 minutes on non-premium plans).
Credentials: APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env or the environment.
"""
from __future__ import annotations

import datetime as dt
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA = "https://data.alpaca.markets"
BARS_SCHEMA = pa.schema([("symbol", pa.string()), ("day", pa.string()),
                         *((k, pa.float64()) for k in ("open", "high", "low", "close"))])


def _get(path: str, params: dict, cfg: dict) -> dict:
    url = f"{DATA}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": cfg["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": cfg["APCA_API_SECRET_KEY"], "accept": "application/json"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            pass                                   # transport failure: retry
        time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {path}")


def _pages(path: str, params: dict, key: str, cfg: dict):
    token = None
    while True:
        data = _get(path, params | ({"page_token": token} if token else {}), cfg)
        yield from data.get(key) or []
        token = data.get("next_page_token")
        if not token:
            return


def _write(table: pa.Table, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path + ".tmp")
    os.replace(path + ".tmp", path)


def symbols(cfg: dict) -> list[str]:
    """SYMBOLS is a comma list, or @path to a file with one symbol per line."""
    v = cfg["SYMBOLS"].strip()
    items = Path(v[1:]).read_text().split() if v.startswith("@") else v.split(",")
    return [s.strip() for s in items if s.strip()]


def _range(cfg: dict) -> tuple[list[str], str, str]:
    syms = symbols(cfg)
    today = dt.datetime.now(dt.UTC).date().isoformat()
    return syms, cfg["START"] + "T00:00:00Z", today + "T00:00:00Z"


def bars(cfg: dict) -> None:
    """One cached parquet per symbol (resumable), assembled into BARS_PATH."""
    syms, start, end = _range(cfg)
    out = Path(cfg["BARS_PATH"])
    cache = out.with_name(out.stem + "_cache") / end[:10]     # a new day is a new fetch
    cache.mkdir(parents=True, exist_ok=True)
    def group(g: list[str]) -> int:
        todo = [s for s in g if not (cache / f"{s}.parquet").exists()]
        if not todo:
            return len(g)
        rows = {s: {k: [] for k in BARS_SCHEMA.names} for s in todo}
        params = {"symbols": ",".join(todo), "timeframe": "1Day", "start": start, "end": end,
                  "adjustment": "split", "feed": "sip", "limit": 10000}
        token = None
        while True:                     # multi-symbol pages: {"bars": {symbol: [...]}}
            data = _get("/v2/stocks/bars", params | ({"page_token": token} if token else {}),
                        cfg)
            for s, bs in (data.get("bars") or {}).items():
                for b in bs:
                    for k, v in (("symbol", s), ("day", b["t"][:10]), ("open", b["o"]),
                                 ("high", b["h"]), ("low", b["l"]), ("close", b["c"])):
                        rows[s][k].append(v)
            token = data.get("next_page_token")
            if not token:
                break
        for s in todo:                  # cached only once the whole group is complete
            _write(pa.table(rows[s], schema=BARS_SCHEMA), str(cache / f"{s}.parquet"))
        return len(g)

    # the connection is latency-bound: few large multi-symbol requests, 8 in flight
    groups = [syms[i:i + 50] for i in range(0, len(syms), 50)]
    done = 0
    with ThreadPoolExecutor(8) as pool:
        for n in pool.map(group, groups):
            done += n
            print(f"bars: {done}/{len(syms)} symbols", flush=True)
    table = pa.concat_tables([pq.read_table(cache / f"{s}.parquet") for s in syms])
    _write(table, cfg["BARS_PATH"])
    print(f"bars: {table.num_rows} rows, {len(syms)} symbols -> {cfg['BARS_PATH']}", flush=True)


def news(cfg: dict) -> None:
    syms, start, end = _range(cfg)
    seen, news = set(), {"symbols": [], "created_at": [], "headline": []}
    params = {"symbols": ",".join(syms), "start": start, "end": end, "limit": 50,
              "sort": "asc", "include_content": "false"}
    for i, n in enumerate(_pages("/v1beta1/news", params, "news", cfg)):
        if n["id"] in seen or not (n.get("headline") or "").strip():
            continue
        seen.add(n["id"])
        news["symbols"].append(n.get("symbols") or [])
        news["created_at"].append(dt.datetime.fromisoformat(n["created_at"].replace("Z", "+00:00")))
        news["headline"].append(n["headline"].strip())
        if i % 5000 == 0:
            print(f"news: {len(seen)} items, at {n['created_at'][:10]}", flush=True)
    table = pa.table({"symbols": pa.array(news["symbols"], pa.list_(pa.string())),
                      "created_at": pa.array(news["created_at"], pa.timestamp("us", tz="UTC")),
                      "headline": pa.array(news["headline"], pa.string())})
    _write(table, cfg["NEWS_PATH"])
    print(f"news: {table.num_rows} headlines -> {cfg['NEWS_PATH']}", flush=True)
