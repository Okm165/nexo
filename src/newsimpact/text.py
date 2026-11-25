"""Headlines → frozen sentence embeddings.

Every headline tagged with at least one symbol present in the bars table is
encoded by a frozen ModernBERT-base (≤64 tokens, masked mean pool). Shards
split rows round-robin so they can run in parallel, one per device.

Writes $EMB_PREFIX<shard>.npz with E [N,768] float16 and src_idx [N] (row in
NEWS_PATH); which symbols and timestamps a headline belongs to is read back
from the news table at build time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

MODEL_ID = "answerdotai/ModernBERT-base"
MAX_TOKENS = 64
BATCH = 256


def embed(cfg: dict, shard: int = 0, shards: int = 1, limit: int = 0,
          device: str = "cuda") -> Path:
    from transformers import AutoModel, AutoTokenizer

    universe = set(pq.read_table(cfg["BARS_PATH"], columns=["symbol"])["symbol"].to_pylist())
    news = pq.read_table(cfg["NEWS_PATH"], columns=["symbols", "headline"])
    tags, heads = news["symbols"].to_pylist(), news["headline"].to_pylist()
    rows = [i for i in range(shard, len(heads), shards)
            if (heads[i] or "").strip() and universe.intersection(tags[i] or ())]
    if limit:
        rows = rows[:limit]
    texts = [heads[i].strip() for i in rows]

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    order = np.argsort([len(t) for t in texts])          # similar lengths per batch
    E = np.zeros((len(texts), 768), np.float16)
    with torch.no_grad():
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            enc = tok([texts[j] for j in b], padding=True, truncation=True,
                      max_length=MAX_TOKENS, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            E[b] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
            if (s // BATCH) % 200 == 0:
                print(f"shard {shard}: {s + len(b)}/{len(texts)}", flush=True)

    out = Path(f"{cfg['EMB_PREFIX']}{shard}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, E=E, src_idx=np.asarray(rows, np.int64))
    print(f"shard {shard}: {len(texts)} headlines -> {out}", flush=True)
    return out
