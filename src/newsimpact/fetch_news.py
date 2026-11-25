"""Fetch the news archive from Alpaca (resumable, idempotent).

The news table is the pipeline's ground-truth input. Until now it existed only
as a snapshot file; this stage makes its provenance code, like fetch-bars.

    newsimpact fetch-news                 # -> NEWS_PATH

Optional adapter: any table with columns symbols (list of str), created_at
(UTC timestamp) and headline works as NEWS_PATH. This one fetches everything
in the date range from Alpaca.

Resume: progress is the provider's page token, checkpointed every page batch
to data/derived/news_fetch/state.json — a kill costs one batch, a rerun
continues. FORCE_FETCH=1 restarts from scratch.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from newsimpact.env import load_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / load_env().get("NEWS_PATH", "data/derived/news.parquet")
STATE = ROOT / "data" / "derived" / "news_fetch"
START = os.environ.get("NEWS_START", "2020-11-01")
END = os.environ.get("NEWS_END", "")          # empty = up to now
PAGE_LIMIT = 50                                # provider max for /news


def _keys() -> tuple[str, str]:
    kid = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not (kid and sec):
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("APCA_API_KEY_ID="):
                kid = line.split("=", 1)[1].strip()
            if line.startswith("APCA_API_SECRET_KEY="):
                sec = line.split("=", 1)[1].strip()
    if not (kid and sec):
        raise RuntimeError("no Alpaca credentials in env or .env")
    return kid, sec


def _get(url: str, kid: str, sec: str) -> dict:
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec,
        "accept": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                raise
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def _load_state() -> dict:
    f = STATE / "state.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"page_token": None, "items": 0}


def _save_state(token: str | None, items: int, out) -> None:
    # Atomic: two crashed runs once left a stale (token, items) pairing — a
    # restart then re-downloaded 9 months of duplicates. tmp+rename guarantees
    # a reader sees either the old state or the new one, never a mix.
    tmp = STATE / "state.json.tmp"
    tmp.write_text(json.dumps({"page_token": token, "items": items}))
    os.replace(tmp, STATE / "state.json")
    out.flush()
    os.fsync(out.fileno())


def main() -> None:
    kid, sec = _keys()
    STATE.mkdir(parents=True, exist_ok=True)
    st = _load_state()
    if os.environ.get("FORCE_FETCH") == "1":
        st = {"page_token": None, "items": 0}
        for old in STATE.glob("part_*.jsonl"):
            old.unlink()

    token = st["page_token"]
    items = st["items"]
    out_path = STATE / f"part_{items:09d}.jsonl"
    fresh = not out_path.exists()
    if not fresh:
        # Append-resume guard: a killed run can die mid-line, leaving a
        # partial record. Appending after it would glue two records into one
        # corrupt line (this exact defect was caught by the live audit, line
        # 129180). Truncate to the last complete line before resuming.
        with open(out_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 1))
            if f.read(1) != b"\n":
                with open(out_path, "rb+") as f2:
                    f2.seek(0, 2)
                    end = f2.tell()
                    f2.seek(0)
                    data = f2.read(end)
                    cut = data.rfind(b"\n") + 1
                    f2.truncate(cut)
                print(f"resume: truncated {end - cut} bytes of partial record",
                      flush=True)
    out = open(out_path, "a" if not fresh else "w")  # noqa: SIM115 - closed after the fetch loop (state is fsync'd through _save_state)

    q = {"limit": str(PAGE_LIMIT), "sort": "asc"}
    if START:
        q["start"] = START
    if END:
        q["end"] = END
    base = "https://data.alpaca.markets/v1beta1/news?"
    n_pages = 0
    t0 = time.time()
    while True:
        qq = dict(q)
        if token:
            qq["page_token"] = token
        data = _get(base + urllib.parse.urlencode(qq), kid, sec)
        for it in data.get("news", []):
            out.write(json.dumps({
                "id": it.get("id"),
                "symbols": it.get("symbols") or [],
                "created_at": it.get("created_at"),
                "headline": (it.get("headline") or "").strip(),
            }) + "\n")
            items += 1
        n_pages += 1
        token = data.get("next_page_token")
        if n_pages % 200 == 0:                 # checkpoint every 200 pages
            _save_state(token, items, out)
            rate = items / max(time.time() - t0, 1)
            print(f"fetch-news: {items} items ({rate:.0f}/s), "
                  f"token={'…' if token else 'DONE'}", flush=True)
        if not token:
            break
    _save_state(None, items, out)
    out.close()

    consolidate()


def consolidate() -> None:
    """Parts -> deduped parquet. Safe to run any time; idempotent."""
    # Dedupe: provider page-token pagination overlaps across resumes (measured
    # 1.9% of items), so ID is the primary key and (timestamp, headline) the
    # fallback. Corrupt lines from a killed write are skipped, not fatal.
    rows = {"id": [], "symbols": [], "created_at": [], "headline": []}
    seen: set = set()
    skipped_dup = skipped_corrupt = 0
    for part in sorted(STATE.glob("part_*.jsonl")):
        with open(part) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    skipped_corrupt += 1
                    continue
                key = (r["id"], r["created_at"], r["headline"])
                if key in seen:
                    skipped_dup += 1
                    continue
                seen.add(key)
                rows["id"].append(r["id"])
                rows["symbols"].append(r["symbols"])
                rows["created_at"].append(r["created_at"])
                rows["headline"].append(r["headline"])
    if skipped_dup or skipped_corrupt:
        print(f"consolidate: skipped {skipped_dup} duplicates, "
              f"{skipped_corrupt} corrupt lines", flush=True)
    order = sorted(range(len(rows["created_at"])),
                   key=lambda i: rows["created_at"][i])
    tbl = pa.table({
        "id": pa.array([rows["id"][i] for i in order], pa.int64()),
        "symbols": pa.array([rows["symbols"][i] for i in order],
                            pa.list_(pa.string())),
        "created_at": pa.array([rows["created_at"][i] for i in order],
                               pa.string()),
        "headline": pa.array([rows["headline"][i] for i in order],
                             pa.string()),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{OUT}.tmp"
    pq.write_table(tbl, tmp)
    os.replace(tmp, OUT)
    print(f"fetch-news: {tbl.num_rows} items, "
          f"{rows['created_at'][order[0]][:10]}..{rows['created_at'][order[-1]][:10]}"
          f" -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
