# nexo

Forecast each stock's **realized volatility over the next `WINDOW` sessions**,
from what is known at today's close: its volatility history, its recent
headlines, and how much news it got. Output: annualized vol per symbol, the
unit an implied-vol quote uses. One linear model for the whole universe.

```
            ┌─ headlines ──┐┌─────── target ────────┐
 ───────────[t-W+1  ...   t][t+1       ...      t+W]───► sessions
  lookbacks 1…132 end at t ▲
                           forecast made at the close of t
 the window slides one session at a time: every day is a sample
```

The method, and the reason for each choice, are in
**[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## Pipeline

```
 news ─► embed ─► embeddings ─┐
 news ────────────────────────┤
 bars ────────────────────────┴─► build ─► dataset ─┬─► train ─► runs ─► evaluate ─► eval.json
                                                    └─► forecast ──────────────────► forecast.csv
```

| stage | does |
|---|---|
| `embed` | headline → frozen sentence embedding (any Hugging Face encoder) |
| `build` | samples, Yang–Zhang target, inputs; exact data-hygiene rules; nothing fitted |
| `train` | every model on every walk-forward fold; statistics from train data only |
| `evaluate` | QLIKE per test year, Diebold–Mariano tests, fold-level test, calibration |
| `forecast` | next-window vol per symbol from the **latest** close (not yet realized) |

## Models compared

```
 persistence   previous window's variance                    (naive bar)
 HAR           + log variance over 1/5/22/66/132 sessions    (standard bar)
 price         + leverage: negative 1/5/22-session returns
 intensity     + last session's headline count vs its norm
 ├ shuffled    + text features of RANDOM headlines           (control)
 └ text        + text features of the window's headlines
```

## Results (one example run)

About 1.2k US-listed stocks, daily bars and a vendor headline archive,
2021–2026, `WINDOW=22` (one trading month), walk-forward test years
2023–2026 (805 test days). Test QLIKE, skill vs HAR:

```
 year   persistence   price    text
 2023      −225%      +2.1%    +3.2%
 2024       −24%      +2.0%    +6.4%
 2025       −18%      +3.6%    +8.2%
 2026       −17%      +1.2%    +2.3%
 all        −87%      +2.3%    +5.2%
```

```
 claim               per-year DM t            fold p   sign p   verdict
 HAR > persistence   −1.6 −1.8 −1.6 −2.0      <.001    .125     holds
 price > HAR         −3.9 −2.0 −2.7 −0.9       .034    .125     holds (not Bonferroni)
 intensity > price   +0.8 −1.0 +0.8 +0.5       .548    .625     no
 text > shuffled     −0.9 −2.3 −1.2 −0.5       .049    .125     not established
```

Tested in a fixed order at α = 0.05, stopping at the first failure; the
chain stops at intensity, so text is not formally supported. Its design was
also chosen on validation years that overlap the test years, and the news
feed changed in 2026. The shipped forecast therefore uses **price**.
Calibration (predicted → realized vol by decile): 24→21, 30→26, …, 79→90,
136→145. Full tables and caveats: [METHODOLOGY §8](docs/METHODOLOGY.md#8-results-one-run).

These are numbers from one run; with your own universe, feed or horizon,
rerun the same protocol and let the same tests decide.

## Usage

```bash
uv sync                          # see "Torch build" for non-default hardware
uv run pytest -q                 # 19 tests, incl. a synthetic end-to-end run

# data: two parquet tables (any source), or the optional adapter
uv run nexo fetch-bars           # credentials in .env (see .env.example)
uv run nexo fetch-news
uv run nexo embed --shard 0 --shards 1 --device cuda

uv run nexo build                # → state/dataset.pt
uv run nexo train                # → state/runs/fold*.{npz,json}
uv run nexo evaluate             # → state/eval.json
uv run nexo forecast             # → state/forecast.csv
scripts/run.sh                   # all of the above
```

Input schemas:

```
 bars  symbol, day, open, high, low, close         (daily)
 news  symbols (list), created_at (UTC), headline
```

## Configuration

| file | keys |
|---|---|
| `.env.data` | `SYMBOLS` (list or `@file`), `START`, `BARS_PATH`, `NEWS_PATH`, `EMB_PREFIX`, `DATASET`, `STATE_DIR`, `ENCODER` |
| `.env.train` | `WINDOW`, `SAMPLE_START`, `FOLDS` |
| `.env` | provider credentials (fetchers only; never committed) |

The example run: `WINDOW=22`, `SAMPLE_START=2021-01-01`,
`FOLDS=2022-01-01,2023-01-01,2024-01-01,2025-01-01,2026-01-01`.
OS environment overrides any key (`WINDOW=10 STATE_DIR=state/w10 uv run nexo build`).
Paths are repo-relative. Everything under `data/` and `state/` is
regenerated and gitignored.

**Torch build.** The lockfile pins torch to a GPU-specific wheel index. On
other hardware, point `[[tool.uv.index]]` in `pyproject.toml` at your build
(`…/whl/cpu`, `…/whl/cu121`, …) and `uv lock`. CI does exactly this for CPU.

## Code

```
src/nexo/
  dataset.py   samples, target, inputs, walk-forward folds, hygiene rules
  text.py      headline embeddings (frozen encoder)
  model.py     QLIKE and its exact penalized fit (damped Newton, float64)
  train.py     text features, k/λ selection, every model, predictions, forecast
  evaluate.py  QLIKE, skill, Diebold–Mariano, fold-level test, calibration
  metrics.py   Diebold–Mariano, Student-t p-value, fold-level test
  fetch.py     optional data adapter
  cli.py env.py
```

## Extending

`WINDOW` (horizon), `SYMBOLS` (universe), `ENCODER` (text model) are config.
The biggest untried levers are data, not model: intraday bars for a cleaner
target, an earnings calendar, a news archive with stable coverage.
