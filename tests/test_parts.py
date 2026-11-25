"""Model, metrics and config invariants."""
import re
from pathlib import Path

import numpy as np
import torch

from newsimpact.metrics import diebold_mariano, spearman
from newsimpact.model import VolForecaster

ROOT = Path(__file__).resolve().parents[1]


def _inputs(b=3, L=6):
    torch.manual_seed(0)
    mask = torch.arange(L)[None] < torch.tensor([[2], [6], [4]])
    return torch.randn(b, 4), torch.randn(b, L, 768), torch.randint(0, 5, (b, L)), mask


def test_starts_at_the_linear_baseline():
    m = VolForecaster(4, 5).eval()
    ctx, emb, age, mask = _inputs()
    assert torch.allclose(m(ctx, emb, age, mask), m.base(ctx).squeeze(-1))


def test_padding_carries_no_information():
    m = VolForecaster(4, 5).eval()
    torch.nn.init.normal_(m.head[-1].weight)
    ctx, emb, age, mask = _inputs()
    emb2, age2 = emb.clone(), age.clone()
    emb2[~mask] = torch.randn_like(emb2[~mask]) * 100
    age2[~mask] = 4
    with torch.no_grad():
        assert torch.allclose(m(ctx, emb, age, mask), m(ctx, emb2, age2, mask), atol=1e-5)


def test_headline_order_does_not_matter():
    m = VolForecaster(4, 5).eval()
    torch.nn.init.normal_(m.head[-1].weight)
    ctx, emb, age, _ = _inputs()
    mask = torch.ones(3, 6, dtype=torch.bool)
    p = torch.randperm(6)
    with torch.no_grad():
        assert torch.allclose(m(ctx, emb, age, mask), m(ctx, emb[:, p], age[:, p], mask),
                              atol=1e-5)


def test_price_arm_has_no_text_pathway():
    m = VolForecaster(4, 5, text=False)
    assert not hasattr(m, "tok")
    assert m(torch.randn(3, 4)).shape == (3,)


def test_metrics():
    y = np.arange(100.0)
    assert abs(spearman(y ** 3, y) - 1.0) < 1e-12
    day = np.repeat(np.arange(20), 5)
    rng = np.random.default_rng(0)
    la = rng.random(100)
    diff, t = diebold_mariano(la, la + 0.1 + 0.01 * rng.random(100), day, 4)
    assert diff < 0 and t < -10


def test_every_config_key_is_read_and_paths_are_relative():
    code = "\n".join(p.read_text() for p in (ROOT / "src" / "newsimpact").glob("*.py"))
    for f in (".env.data", ".env.train"):
        for line in (ROOT / f).read_text().splitlines():
            if re.match(r"^[A-Z_]+=", line):
                k, v = line.split("=", 1)
                assert f'"{k}"' in code, f"{f}: {k} is declared but never read"
                assert not v.startswith(("/", "~")), f"{f}: {k} is host-absolute"


def test_readme_states_the_configured_splits():
    readme = (ROOT / "README.md").read_text()
    for line in (ROOT / ".env.train").read_text().splitlines():
        if line.startswith(("TRAIN_END=", "VAL_END=", "WINDOW=")):
            assert line.split("=")[1] in readme, f"README out of sync with {line}"
