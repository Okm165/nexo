"""End-to-end on a synthetic market: build → train (all arms, CPU) → evaluate,
plus the properties the design rests on (point in time, one-day slide,
target formula, purged splits, data-error guard)."""
import datetime as dt

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from newsimpact import dataset, evaluate, train

W = 5


def _days(start="2023-01-02", end="2025-06-30"):
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
    rows = {"symbol": [], "day": [], "close": []}
    for s in ("AAA", "BBB"):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
        if s == "BBB":
            px[300:] *= 3.0                  # an unadjusted split: |log r| > 1
        rows["symbol"] += [s] * len(days)
        rows["day"] += [d.isoformat() for d in days]
        rows["close"] += px.tolist()
    pq.write_table(pa.table(rows), tmp / "bars.parquet")

    news = {"symbols": [], "created_at": [], "headline": []}
    for d in days:                           # one intraday headline per symbol per day
        for s in ("AAA", "BBB"):
            news["symbols"].append([s])
            news["created_at"].append(dt.datetime(d.year, d.month, d.day, 14, tzinfo=dt.UTC))
            news["headline"].append(f"{s} news {d}")
    special = {                              # timing edge cases, all for AAA
        "after_close": dt.datetime(2023, 3, 1, 21, tzinfo=dt.UTC),   # Wed 21:00 -> Thu
        "weekend": dt.datetime(2023, 3, 4, 12, tzinfo=dt.UTC),       # Sat -> Mon
    }
    for ts in special.values():
        news["symbols"].append(["AAA", "ZZZ"])          # ZZZ has no bars: ignored
        news["created_at"].append(ts)
        news["headline"].append("edge")
    pq.write_table(pa.table(news), tmp / "news.parquet")
    n = len(news["headline"])
    np.savez(tmp / "emb_s0.npz", E=rng.normal(size=(n, 768)).astype(np.float16),
             src_idx=np.arange(n))

    cfg = {"BARS_PATH": str(tmp / "bars.parquet"), "NEWS_PATH": str(tmp / "news.parquet"),
           "EMB_PREFIX": str(tmp / "emb_s"), "DATASET": str(tmp / "ds.pt"),
           "STATE_DIR": str(tmp), "WINDOW": str(W), "TRAIN_END": "2023-12-31",
           "VAL_END": "2024-12-31", "MAX_HEADLINES": "8", "EPOCHS": "2", "BATCH": "64",
           "LR": "1e-3", "WEIGHT_DECAY": "1e-2", "DROPOUT": "0.1"}
    ds = dataset.build(cfg)
    return cfg, ds, rows, news, special, days


def _sample(ds, sym, day):
    s = ds["symbols"].index(sym)
    m = (ds["sym"].numpy() == s) & (ds["day"].numpy() == np.datetime64(day).astype(np.int64))
    return int(np.where(m)[0][0]) if m.any() else None


def test_headlines_are_known_at_the_close(world):
    """Every headline in sample t was published before the close of t, and is
    at most WINDOW-1 sessions old. (All synthetic headlines are attached, so
    dataset embedding row == news row.)"""
    _, ds, _, news, _, _ = world
    pub = np.array([t.timestamp() for t in news["created_at"]])
    assert len(ds["emb"]) == len(pub)
    for i in range(0, len(ds["y"]), 37):
        t_day = dt.date(1970, 1, 1) + dt.timedelta(days=int(ds["day"][i]))
        close = dt.datetime(t_day.year, t_day.month, t_day.day, 20, tzinfo=dt.UTC).timestamp()
        lo, hi = int(ds["lo"][i]), int(ds["hi"][i])
        assert hi > lo
        assert (pub[ds["att_emb"][lo:hi].numpy()] <= close).all()
        assert ((int(ds["t"][i]) - ds["att_info"][lo:hi].numpy()) < W).all()


def test_publication_timing(world):
    _, ds, _, news, _, days = world
    n = len(news["headline"])
    info = {}
    for name, row in (("after_close", n - 2), ("weekend", n - 1)):
        a = np.where(ds["att_emb"].numpy() == row)[0]
        assert len(a) == 1, "the unknown co-tag ZZZ must not create an attachment"
        info[name] = days[int(ds["att_info"][a[0]])]
    assert info["after_close"] == dt.date(2023, 3, 2)     # Wed 21:00 UTC -> Thu close
    assert info["weekend"] == dt.date(2023, 3, 6)         # Saturday -> Monday close


def test_window_slides_one_day(world):
    _, ds, _, _, _, _ = world
    a = ds["symbols"].index("AAA")
    t = np.sort(ds["t"].numpy()[ds["sym"].numpy() == a])
    assert (np.diff(t) == 1).mean() > 0.99


def test_target_is_next_window_realized_vol(world):
    _, ds, rows, _, _, days = world
    i = _sample(ds, "AAA", "2024-06-03")
    px = np.array(rows["close"][:len(days)])
    t = int(ds["t"][i])
    r = np.diff(np.log(px[t:t + W + 1]))
    assert np.exp(float(ds["y"][i])) == pytest.approx(np.sqrt(252 / W * (r ** 2).sum()),
                                                      rel=1e-5)
    prev = np.diff(np.log(px[t - W:t + 1]))
    assert np.exp(float(ds["prev"][i])) == pytest.approx(
        np.sqrt(252 / W * (prev ** 2).sum()), rel=1e-5)


def test_splits_are_purged(world):
    _, ds, _, _, _, _ = world
    split, day = ds["split"].numpy(), ds["day"].numpy().astype("datetime64[D]")
    assert day[split == 0].max() < np.datetime64("2023-12-31") - np.timedelta64(5, "D")
    assert day[split == 1].min() > np.datetime64("2023-12-31")
    assert day[split == 2].min() > np.datetime64("2024-12-31")


def test_bad_return_windows_are_dropped(world):
    _, ds, _, _, _, days = world
    b = ds["symbols"].index("BBB")
    d = ds["day"].numpy()[ds["sym"].numpy() == b].astype("datetime64[D]")
    jump = np.datetime64(days[300].isoformat())
    assert not ((d >= jump - np.timedelta64(30, 'D')) & (d < jump)).all()
    assert not ((d > jump - np.timedelta64(7, 'D')) & (d < jump + np.timedelta64(22, 'D'))).any()


def test_train_and_evaluate_end_to_end(world):
    cfg, *_ = world
    rows = [train.run(arm, 0, cfg, "cpu") for arm in train.ARMS]
    assert all(np.isfinite(r["test_mse_log"]) for r in rows)
    rep = evaluate.main(cfg)
    assert {"text", "shuffled", "price", "HAR", "persistence"} <= set(rep["predictors"])
    assert "text vs shuffled" in rep["dm_tests"]


def test_shuffled_control_draws_train_headlines(world):
    _, ds, _, news, _, _ = world
    train_end = dt.datetime(2023, 12, 31, 23, tzinfo=dt.UTC)
    shuf = ds["att_shuf"].numpy()
    assert shuf.shape == ds["att_emb"].shape
    assert (shuf != ds["att_emb"].numpy()).mean() > 0.9
    assert all(news["created_at"][k] <= train_end for k in np.unique(shuf))
