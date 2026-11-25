"""Loss, fit, metrics and config invariants."""
import re
from pathlib import Path

import numpy as np

from nexo import dataset
from nexo.metrics import diebold_mariano, fold_meta, t_pvalue
from nexo.model import fit_qlike, qlike

ROOT = Path(__file__).resolve().parents[1]


def test_qlike_is_zero_only_at_the_truth():
    y = np.array([0.0, 1.0, -2.0])
    assert np.allclose(qlike(y, y), 0.0)
    assert (qlike(y + 0.3, y) > 0).all() and (qlike(y - 0.3, y) > 0).all()


def test_fit_qlike_reaches_the_optimum():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = X @ [0.5, -0.2, 0.1] + 0.3 + rng.normal(0, 0.8, 2000)
    A = np.c_[np.ones(2000), X]
    for pen in (None, np.array([0.0, 0.1, 0.1, 0.1])):
        w = fit_qlike(A, y, pen)
        P = np.zeros(4) if pen is None else pen
        grad = A.T @ (1 - np.exp(y - A @ w)) / 2000 + 2 * P * w
        assert np.abs(grad).max() < 1e-8
    ols = np.linalg.lstsq(A, y, rcond=None)[0]
    assert qlike(A @ fit_qlike(A, y), y).mean() < qlike(A @ ols, y).mean()


def test_diebold_mariano():
    pos = np.arange(100)
    rng = np.random.default_rng(0)
    la = rng.random(100)
    diff, t = diebold_mariano(la, la + 0.1 + 0.01 * rng.random(100), pos, pos // 20, 4)
    assert diff < 0 and t < -10


def test_every_config_key_is_read_and_paths_are_relative():
    code = "\n".join(p.read_text() for p in (ROOT / "src" / "nexo").glob("*.py"))
    code += (ROOT / "justfile").read_text()
    for f in (".env.data", ".env.train"):
        for line in (ROOT / f).read_text().splitlines():
            if re.match(r"^[A-Z_]+=", line):
                k, v = line.split("=", 1)
                assert f'"{k}"' in code or f"'{k}'" in code or f"[{k}]" in code, \
                    f"{f}: {k} is declared but never read"
                assert not v.startswith(("/", "~")), f"{f}: {k} is host-absolute"


def test_readme_states_the_configured_design():
    readme = (ROOT / "README.md").read_text()
    for line in (ROOT / ".env.train").read_text().splitlines():
        if line.startswith(("WINDOW=", "SAMPLE_START=", "FOLDS=")):
            for value in line.split("=")[1].split(","):
                assert value in readme, f"README out of sync with {line}"


def test_digest_covers_every_input():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert dataset._digest([a]) == dataset._digest([a.copy()])     # stable
    b = a.copy()
    b[1, 1] += np.float32(1)
    assert dataset._digest([b]) != dataset._digest([a])            # value change
    assert dataset._digest([a, a]) != dataset._digest([a])         # count
    assert dataset._digest([a.reshape(4, 3)]) != dataset._digest([a])  # shape


def test_t_pvalue_matches_tables_and_fold_meta():
    for t, df in ((12.7062, 1), (4.3027, 2), (3.1824, 3), (2.7764, 4), (2.2281, 10)):
        assert abs(t_pvalue(t, df) - 0.05) < 2e-4         # two-sided 5% critical values
    assert t_pvalue(0.0, 3) == 1.0 and t_pvalue(-3.1824, 3) == t_pvalue(3.1824, 3)
    m = fold_meta([-2.0, -2.2, -1.8, -2.0])
    assert m["first_better"] == 4 and m["t"] < -10 and m["p"] < 0.01
    assert m["sign_p"] == 0.125                               # 2 · (1/2)^4: the floor at n = 4
    assert fold_meta([-1.0, 1.0, -1.0, 1.0])["sign_p"] == 1.0
    one = fold_meta([-8.0, 0.1, 0.2, -0.1])                # carried by one period
    assert one["p"] > 0.3
