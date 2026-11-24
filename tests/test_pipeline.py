"""End-to-end on a synthetic market, plus the properties the results rest on:
point in time, one-day slide, the Yang–Zhang target by hand, leak-free
blocked splits, exact data-hygiene rules, train-only control content."""
import datetime as dt

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from newsimpact import dataset, evaluate, train

W = 5


def _days(start="2022-06-01", end="2025-06-30"):
    d, out = dt.date.fromisoformat(start), []
    while d <= dt.date.fromisoformat(end):
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("world")
    rng = np.random.default_rng(0)
    days = _days()
    n = len(days)
    bars = {k: [] for k in ("symbol", "day", "open", "high", "low", "close")}
    for s in ("AAA", "BBB"):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.005, n))
        hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.01, n)))
        lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
        if s == "BBB":
            hi[400] = lo[400] * 0.5                # an inconsistent bar (high < low)
        bars["symbol"] += [s] * n
        bars["day"] += [d.isoformat() for d in days]
        for k, v in (("open", o), ("high", hi), ("low", lo), ("close", c)):
            bars[k] += v.tolist()
    pq.write_table(pa.table(bars), tmp / "bars.parquet")

    news = {"symbols": [], "created_at": [], "headline": []}
    for i, d in enumerate(days):                   # AAA: news on 2 of every 3 days
        if i % 3:
            news["symbols"].append(["AAA"])
            news["created_at"].append(dt.datetime(d.year, d.month, d.day, 14, tzinfo=dt.UTC))
            news["headline"].append(f"AAA {d}")
    for ts in (dt.datetime(2024, 3, 6, 21, tzinfo=dt.UTC),    # Wed after close -> Thu
               dt.datetime(2024, 3, 9, 12, tzinfo=dt.UTC)):   # Saturday -> Monday
        news["symbols"].append(["AAA", "ZZZ"])                 # ZZZ has no bars
        news["created_at"].append(ts)
        news["headline"].append("edge")
    pq.write_table(pa.table(news), tmp / "news.parquet")
    m = len(news["headline"])
    np.savez(tmp / "emb_s0.npz", E=rng.normal(size=(m, 768)).astype(np.float32),
             src_idx=np.arange(m))
    cfg = {"BARS_PATH": str(tmp / "bars.parquet"), "NEWS_PATH": str(tmp / "news.parquet"),
           "EMB_PREFIX": str(tmp / "emb_s"), "DATASET": str(tmp / "ds.pt"),
           "STATE_DIR": str(tmp), "WINDOW": str(W), "SAMPLE_START": "2023-01-02",
           "BLOCK": "21", "VAL_FRACTION": "0.15", "TEST_FRACTION": "0.15",
           "SPLITS": "3", "TARGET": "AAA"}
    return cfg, dataset.build(cfg), bars, news, days


def _row(ds, sym, day):
    s = ds["symbols"].index(sym)
    m = (ds["sym"].numpy() == s) & (ds["day"].numpy() == np.datetime64(day).astype(np.int64))
    return int(np.where(m)[0][0])


def test_headlines_are_known_at_the_close(world):
    _, ds, _, news, _ = world
    pub = np.array([t.timestamp() for t in news["created_at"]])
    assert len(ds["emb"]) == len(pub)              # every synthetic headline is attached
    for i in range(0, len(ds["y"]), 13):
        day = dt.date(1970, 1, 1) + dt.timedelta(days=int(ds["day"][i]))
        close = dt.datetime(day.year, day.month, day.day, 20, tzinfo=dt.UTC).timestamp()
        lo, hi = int(ds["lo"][i]), int(ds["hi"][i])
        assert (pub[ds["att_emb"][lo:hi].numpy()] <= close).all()
        assert ((int(ds["t"][i]) - ds["att_info"][lo:hi].numpy()) < W).all()


def test_publication_timing(world):
    _, ds, _, news, days = world
    m = len(news["headline"])
    got = []
    for row in (m - 2, m - 1):
        a = np.where(ds["att_emb"].numpy() == row)[0]
        assert len(a) == 1, "the co-tag without bars must not create an attachment"
        got.append(days[int(ds["att_info"][a[0]])])
    assert got == [dt.date(2024, 3, 7), dt.date(2024, 3, 11)]


def test_window_slides_one_day_and_quiet_windows_exist(world):
    _, ds, _, _, _ = world
    a = ds["symbols"].index("AAA")
    t = np.sort(ds["t"].numpy()[ds["sym"].numpy() == a])
    assert (np.diff(t) == 1).all()
    assert ((ds["hi"] - ds["lo"]).numpy() == 0).any()     # BBB never has news


def test_target_is_yang_zhang_by_hand(world):
    _, ds, bars, _, days = world
    i = _row(ds, "AAA", "2024-06-03")
    t = int(ds["t"][i])
    po, ph, pl, pc = (np.array(bars[k][:len(days)]) for k in ("open", "high", "low", "close"))

    def yz(a, b):                                   # days a..b inclusive
        o = np.log(po[a:b + 1] / pc[a - 1:b])
        c = np.log(pc[a:b + 1] / po[a:b + 1])
        u, d = np.log(ph[a:b + 1] / po[a:b + 1]), np.log(pl[a:b + 1] / po[a:b + 1])
        n = b - a + 1
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        rs = np.mean(u * (u - c) + d * (d - c))
        return 252 * (np.var(o, ddof=1) + k * np.var(c, ddof=1) + (1 - k) * rs)

    assert np.exp(float(ds["y"][i])) == pytest.approx(yz(t + 1, t + W), rel=1e-5)
    assert np.exp(float(ds["prev"][i])) == pytest.approx(yz(t - W + 1, t), rel=1e-5)
    assert np.exp(float(ds["ctx"][i, 1])) == pytest.approx(yz(t - 21, t), rel=1e-5)


def test_sample_start_and_bad_bars(world):
    _, ds, _, _, days = world
    assert ds["day"].numpy().min() >= np.datetime64("2023-01-02").astype(np.int64)
    b = ds["symbols"].index("BBB")
    d = ds["day"].numpy()[ds["sym"].numpy() == b]
    bad = np.datetime64(days[400].isoformat()).astype(np.int64)
    assert not ((d > bad - 7) & (d < bad + 150)).any(), "inconsistent bar inside a span"


def test_blocked_split_shares_nothing_across_splits(world):
    _, ds, _, _, _ = world
    pos = ds["pos"].numpy()
    a = dataset.blocked_split(ds, 0, 21, 0.15, 0.15)
    b = dataset.blocked_split(ds, 1, 21, 0.15, 0.15)
    assert {0, 1, 2} <= set(a) and not np.array_equal(a, b)
    by_pos = {}
    for p, s in zip(pos, a, strict=True):
        if s >= 0:
            by_pos[p] = s
    for p, s in zip(pos, a, strict=True):                # whole span in one split
        if s >= 0:
            for q in range(p - W + 1, p + W + 1):
                assert by_pos.get(q, s) == s


def test_window_means_equal_projected_window_mean(world):
    """Averaging per-headline scores equals scoring the window's mean headline."""
    _, ds, _, _, _ = world
    E = ds["emb"].numpy().astype(np.float64)
    V = np.random.default_rng(0).normal(size=(768, 3))
    got = train.window_means(ds, E[ds["att_emb"].numpy()] @ V)
    i = int(np.argmax((ds["hi"] - ds["lo"]).numpy()))
    lo, hi = int(ds["lo"][i]), int(ds["hi"][i])
    assert np.allclose(got[i], E[ds["att_emb"][lo:hi].numpy()].mean(0) @ V)
    assert (got[(ds["hi"] == ds["lo"]).numpy()] == 0).all()


def test_fit_and_evaluate_end_to_end(world):
    cfg, *_ = world
    cfg = cfg | {"SPLITS": "3"}
    train.run(cfg)
    rep = evaluate.main(cfg)
    for scope in ("universe", "AAA"):
        r = rep[scope]
        assert set(r["qlike"]) == {"persistence", "HAR", "intensity", "shuffled", "text"}
        assert r["n_splits"] == 3
        assert all(np.isfinite(list(s["qlike"].values())).all() for s in r["splits"])


def test_fixed_effects_remove_the_symbol_level():
    """Shift y and every feature of a symbol by a constant: the fitted deviations
    must not move, because that constant is exactly what the level absorbs."""
    rng = np.random.default_rng(0)
    n_sym, per = 4, 60
    sym = np.repeat(np.arange(n_sym), per)
    tr = np.arange(n_sym * per)
    X = rng.normal(size=(len(sym), 2)) + rng.normal(0, 2, n_sym)[sym][:, None]
    y = rng.normal(3, 1, n_sym)[sym] + X @ np.array([0.5, -0.3]) + rng.normal(0, 0.4, len(sym))

    def fit(X, y):
        Xd = train.centre(X.copy(), train.means(X, sym, n_sym), sym)
        yd = y - train.levels(y, sym, tr, n_sym)[sym]
        return train.fit_qlike(np.c_[np.ones(len(sym)), Xd], yd)

    c = rng.normal(0, 3, n_sym)                     # a different constant per stock
    assert np.allclose(fit(X, y), fit(X + c[sym][:, None], y + c[sym]), atol=1e-8)


def test_level_is_the_qlike_optimal_location():
    """The per-symbol level is the minimiser of the loss, not the mean of y."""
    y = np.random.default_rng(1).normal(3, 1, 200)
    lev = train.levels(y, np.zeros(200, int), np.arange(200), 1)[0]
    grid = lev + np.linspace(-2, 2, 401)
    losses = [train.qlike(np.full(200, g), y).mean() for g in grid]
    assert np.argmin(losses) == 200                 # lev is the centre of the grid
    assert abs(lev - y.mean()) > 0.05               # not the mean of log variance


def test_levels_and_means_fall_back_when_a_symbol_is_absent():
    sym = np.array([0, 0, 0, 1])
    y = np.log(np.array([1.0, 4.0, 9.0, 100.0]))
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    tr = np.array([0, 1, 2])                        # symbol 1 is missing from train
    lev = train.levels(y, sym, tr, 2)
    M = train.means(X[tr], sym[tr], 2)
    assert np.isclose(lev[0], np.log(np.mean(np.exp(y[[0, 1, 2]]))))
    assert np.isclose(lev[1], np.log(np.mean(np.exp(y[[0, 1, 2]]))))
    assert np.allclose(M[0], X[[0, 1, 2]].mean(0))
    assert np.allclose(M[1], X[[0, 1, 2]].mean(0))
