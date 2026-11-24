"""Optional Alpaca adapter: daily OHLC bars and news headlines for SYMBOLS.

Writes the two input tables the pipeline needs (any source with the same
columns works just as well):

    BARS_PATH  symbol, day, open, high, low, close   (split-adjusted, SIP feed)
    NEWS_PATH  symbols (list), created_at (UTC), headline

    newsimpact fetch-bars            # SYMBOLS / START from .env.data
    newsimpact fetch-news            # replaces NEWS_PATH — point it elsewhere
                                     # if you already hold a longer archive
Data ends at today 00:00 UTC: an unfinished session is never fetched (and
the SIP feed refuses the latest 15 minutes on non-premium plans).
Credentials: APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env or the environment.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA = "https://data.alpaca.markets"


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
        except (urllib.error.URLError, TimeoutError):
            pass
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


def _range(cfg: dict) -> tuple[list[str], str, str]:
    symbols = [s.strip() for s in cfg["SYMBOLS"].split(",") if s.strip()]
    today = dt.datetime.now(dt.UTC).date().isoformat()
    return symbols, cfg["START"] + "T00:00:00Z", today + "T00:00:00Z"


def bars(cfg: dict) -> None:
    symbols, start, end = _range(cfg)
    rows = {k: [] for k in ("symbol", "day", "open", "high", "low", "close")}
    for s in symbols:
        for b in _pages(f"/v2/stocks/{s}/bars", {"timeframe": "1Day", "start": start,
                                                "end": end, "adjustment": "split",
                                                "feed": "sip", "limit": 10000}, "bars", cfg):
            for k, v in (("symbol", s), ("day", b["t"][:10]), ("open", b["o"]),
                         ("high", b["h"]), ("low", b["l"]), ("close", b["c"])):
                rows[k].append(v)
        print(f"bars {s}: {sum(1 for x in rows['symbol'] if x == s)} days", flush=True)
    _write(pa.table(rows), cfg["BARS_PATH"])


def news(cfg: dict) -> None:
    symbols, start, end = _range(cfg)
    seen, news = set(), {"symbols": [], "created_at": [], "headline": []}
    params = {"symbols": ",".join(symbols), "start": start, "end": end, "limit": 50,
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
