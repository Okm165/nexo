"""Next-window volatility forecaster.

    ŷ = linear(ctx)  +  head( ctx_embed + text_pool )

`ctx` holds own and market log realized vol (5/22/66/132 sessions) and, for
the text arms, news intensity.
The linear term is a learned HAR regression; the head is zero-initialised,
so training starts exactly at that baseline and the text pathway can only
earn its way in by reducing error.

Text pathway (absent in the price-only arm):
    each headline  → Linear(768→d)  + age embedding (0..W-1 sessions)
                                                + ctx embedding
    one pre-LN transformer layer over the set  (headlines read each other)
    attention pool with a ctx-conditioned query (padding masked everywhere)

Order-free over headlines, invariant to padding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VolForecaster(nn.Module):
    def __init__(self, n_ctx: int, window: int, text: bool = True, d: int = 128,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.text = text
        self.base = nn.Linear(n_ctx, 1)
        self.ctx = nn.Sequential(nn.Linear(n_ctx, d), nn.GELU(), nn.Linear(d, d))
        if text:
            self.tok = nn.Linear(768, d)           # inputs are standardised per dimension
            self.age = nn.Embedding(window, d)
            self.mix = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, activation="gelu",
                                                  batch_first=True, norm_first=True)
            self.query = nn.Parameter(torch.zeros(1, 1, d))
            self.pool = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, ctx, emb=None, age=None, mask=None):
        """ctx [B,C]; emb [B,L,768]; age [B,L] int; mask [B,L] True = real headline."""
        c = self.ctx(ctx)
        h = c
        if self.text:
            pad = ~mask
            t = self.tok(emb) + self.age(age) + c[:, None]
            t = self.mix(t, src_key_padding_mask=pad)
            q = self.query.expand(len(c), -1, -1) + c[:, None]
            h = h + self.pool(q, t, t, key_padding_mask=pad, need_weights=False)[0][:, 0]
        return self.base(ctx).squeeze(-1) + self.head(h).squeeze(-1)
