"""End-to-end on a synthetic market, plus the properties the results rest on:
point in time, one-day slide, target and inputs by hand, forward-in-time
purged folds, exact data-hygiene rules, a forecast of a window not yet seen."""
import datetime as dt
import pathlib

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
           "FOLDS": "2024-01-01,2024-07-01,2025-01-01"}
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


def test_target_and_inputs_by_hand(world):
    _, ds, bars, news, days = world
    n = len(days)
    po, ph, pl, pc = (np.array(bars[k][:n]) for k in ("open", "high", "low", "close"))
    col = ds["config"]["ctx"].index

    def yz(a, b):                                   # days a..b inclusive
        o = np.log(po[a:b + 1] / pc[a - 1:b])
        c = np.log(pc[a:b + 1] / po[a:b + 1])
        u, d = np.log(ph[a:b + 1] / po[a:b + 1]), np.log(pl[a:b + 1] / po[a:b + 1])
        m = b - a + 1
        k = 0.34 / (1.34 + (m + 1) / (m - 1))
        rs = np.mean(u * (u - c) + d * (d - c))
        return 252 * (np.var(o, ddof=1) + k * np.var(c, ddof=1) + (1 - k) * rs)

    i = _row(ds, "AAA", "2024-06-03")
    t, x = int(ds["t"][i]), ds["ctx"][i].numpy().astype(np.float64)
    assert np.exp(float(ds["y"][i])) == pytest.approx(yz(t + 1, t + W), rel=1e-5)
    assert np.exp(float(ds["prev"][i])) == pytest.approx(yz(t - W + 1, t), rel=1e-5)
    assert np.exp(x[col("logvar22")]) == pytest.approx(yz(t - 21, t), rel=1e-5)
    o, c = np.log(po[t] / pc[t - 1]), np.log(pc[t] / po[t])
    u, d = np.log(ph[t] / po[t]), np.log(pl[t] / po[t])
    assert np.exp(x[col("logvar1")]) == pytest.approx(
        252 * (o * o + u * (u - c) + d * (d - c)), rel=1e-5)
    assert x[col("negret5")] == pytest.approx(min(np.log(pc[t] / pc[t - 5]), 0.0), abs=1e-6)
    # news of session t, relative to the symbol's own 132-session norm
    per_day = np.bincount(ds["att_info"].numpy()[:len(news["headline"])], minlength=n)
    assert x[col("news_last")] == pytest.approx(
        np.log1p(per_day[t]) - np.log1p(per_day[t - 131:t + 1].sum() / 132), abs=1e-6)


def test_live_rows_are_the_latest_close(world):
    """The forecast inputs: one row per symbol at the last session, built from
    the same formulas, with no target (its window has not happened)."""
    _, ds, _, _, days = world
    live, n = ds["live"], len(days)
    assert set(live["sym"].tolist()) == {0, 1}
    assert (live["t"].numpy() == n - 1).all()
    assert (live["day"].numpy() == np.datetime64(days[-1].isoformat()).astype(np.int64)).all()
    assert "y" not in live and int(ds["t"].max()) == n - 1 - W
    last = _row(ds, "AAA", days[-1 - W].isoformat())             # the newest sample with a target
    a = int(np.where(live["sym"].numpy() == 0)[0][0])
    assert int(live["hi"][a]) >= int(ds["hi"][last])              # windows keep sliding forward
    assert int(live["hi"][a]) - int(live["lo"][a]) >= 0


def test_sample_start_and_bad_bars(world):
    _, ds, _, _, days = world
    assert ds["day"].numpy().min() >= np.datetime64("2023-01-02").astype(np.int64)
    b = ds["symbols"].index("BBB")
    d = ds["day"].numpy()[ds["sym"].numpy() == b]
    bad = np.datetime64(days[400].isoformat()).astype(np.int64)
    assert not ((d > bad - 7) & (d < bad + 150)).any(), "inconsistent bar inside a span"


def test_folds_run_forward_in_time_and_are_purged(world):
    """Train < validation < test by date in every fold, and no sample's span
    (headline window .. target window) crosses out of its part."""
    cfg, ds, _, _, _ = world
    pos, day = ds["pos"].numpy(), ds["day"].numpy()
    cuts = [np.datetime64(d).astype(np.int64) for d in cfg["FOLDS"].split(",")]
    cal = ds["calendar"].numpy()
    for k in range(train.n_folds(cfg)):
        split = dataset.fold_split(ds, k, cfg["FOLDS"])
        tr, va, te = (np.where(split == i)[0] for i in range(3))
        assert len(tr) and len(va) and len(te)
        assert day[tr].max() < day[va].min() <= day[va].max() < day[te].min()
        assert cal[pos[tr] + W].max() < cuts[k]                    # train targets end first
        assert cal[pos[va] - W + 1].min() >= cuts[k] and cal[pos[va] + W].max() < cuts[k + 1]
        assert cal[pos[te] - W + 1].min() >= cuts[k + 1]
        if k + 2 < len(cuts):
            assert cal[pos[te] + W].max() < cuts[k + 2]
    with pytest.raises(ValueError):
        dataset.fold_split(ds, 0, "2025-01-01,2024-01-01")


def test_window_means_equal_projected_window_mean(world):
    """Averaging per-headline scores equals scoring the window's mean headline."""
    _, ds, _, _, _ = world
    E = ds["emb"].numpy().astype(np.float64)
    V = np.random.default_rng(0).normal(size=(768, 3))
    lo_, mid_, hi_ = (ds[k].numpy() for k in ("lo", "mid", "hi"))
    got = train.window_means(E[ds["att_emb"].numpy()] @ V, lo_, hi_)
    i = int(np.argmax(hi_ - lo_))
    lo, hi = int(lo_[i]), int(hi_[i])
    assert np.allclose(got[i], E[ds["att_emb"][lo:hi].numpy()].mean(0) @ V)
    assert (got[hi_ == lo_] == 0).all()
    # the window splits into its last session and the sessions before it
    assert ((lo_ <= mid_) & (mid_ <= hi_)).all()
    info, t = ds["att_info"].numpy(), ds["t"].numpy()
    for i in range(0, len(t), 17):
        assert (info[mid_[i]:hi_[i]] == t[i]).all() and (info[lo_[i]:mid_[i]] < t[i]).all()


def test_day_relative_removes_what_a_day_shares():
    """Add one vector to every headline-bearing row of a day: nothing changes."""
    rng = np.random.default_rng(0)
    day, full = np.repeat(np.arange(6), 5), rng.random(30) < 0.7
    T = np.where(full[:, None], rng.normal(size=(30, 3)), 0.0)
    shifted = T + np.where(full[:, None], rng.normal(size=(6, 3))[day], 0.0)
    train.day_relative(T, day, full)
    train.day_relative(shifted, day, full)
    assert np.allclose(T, shifted) and (T[~full] == 0).all()
    assert np.allclose([T[full & (day == d)].sum(0) for d in range(6)], 0.0)


def test_fit_and_evaluate_end_to_end(world):
    cfg, *_ = world
    train.run(cfg)
    rep = evaluate.main(cfg)
    arms = {"persistence", "HAR", "price", "intensity", "shuffled", "text"}
    assert len(rep["folds"]) == 2 and rep["folds"][0]["to"] < rep["folds"][1]["from"]
    for r in (*rep["folds"], rep["all"]):
        assert set(r["qlike"]) == arms and np.isfinite(list(r["qlike"].values())).all()
        assert np.isfinite(list(r["dm_t"].values())).all()
    assert rep["all"]["samples"] == sum(f["samples"] for f in rep["folds"])


def test_build_refuses_duplicate_bars(world):
    """A repeated (symbol, day) row would corrupt every windowed sum: refused."""
    cfg, _, _, _, _ = world
    t = pq.read_table(cfg["BARS_PATH"])
    pq.write_table(pa.concat_tables([t, t.slice(0, 1)]), cfg["BARS_PATH"] + ".dup")
    with pytest.raises(ValueError, match="duplicate"):
        dataset.build(cfg | {"BARS_PATH": cfg["BARS_PATH"] + ".dup"})


def test_forecast_is_for_a_window_not_yet_seen(world):
    """One predicted vol per symbol from its inputs at the latest close."""
    cfg, _, _, _, days = world
    out = train.forecast(cfg)
    assert set(out["symbols"]) == {"AAA", "BBB"}
    assert set(out["day"]) == {days[-1].isoformat()}              # the last bar, not W before
    assert all(np.isfinite(out["vol"])) and all(v > 0 for v in out["vol"])
    lines = (pathlib.Path(cfg["STATE_DIR"]) / "forecast.csv").read_text().splitlines()
    assert lines[0] == "symbol,day,predicted_vol_pct,previous_window_vol_pct"
    vols = [float(row.split(",")[2]) for row in lines[1:]]
    assert len(vols) == 2 and vols == sorted(vols, reverse=True)
