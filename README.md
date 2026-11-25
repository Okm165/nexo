# nexo

Forecast each stock's **realized volatility over the next `WINDOW` trading
sessions** from what is known at today's close: its volatility history, its
recent headlines, and how much news it got. The output is annualized vol per
symbol, in the units an implied-vol quote uses. One linear model is shared by
the whole universe.

```
            ┌─ headlines ──┐┌─────── target ────────┐
 ───────────[t-W+1  ...   t][t+1       ...      t+W]───► sessions
  lookbacks 1…132 end at t ▲
                           forecast made at the close of t
 the window slides one session at a time: every day is a sample
```

Why each design choice was made, the statistics, and the threats to
validity: **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## Requirements

| need | detail |
|---|---|
| [uv](https://docs.astral.sh/uv/) | installs Python 3.12 and the locked dependencies |
| [just](https://just.systems/) | optional; every recipe is one `uv run nexo …` call |
| GPU | optional. Fitting and embedding use one if present; CPU works (CI runs on CPU) |
| RAM | training peaks at about 20 GB on the example run (1.57M samples); it scales with samples × text components |
| disk | example run: dataset 2.7 GB, embeddings 2.6 GB, environment ~13 GB (GPU torch) |
| time | example run, with a GPU: build under a minute, train 12–14 min, evaluate and forecast ~1 min each; embedding 860k headlines is the slowest step |

## Quick start

```bash
just setup            # uv sync  (see "Torch build" if you have no AMD GPU)
just check            # lint + 19 tests, incl. a synthetic end-to-end run
just fetch            # optional: download bars + headlines (needs .env, below)
just all              # embed → build → train → evaluate → forecast
just show             # top rows of state/forecast.csv
just                  # list every command
```

## Inputs

Two parquet tables. Any source works; the bundled adapter (`just fetch`) is
one way to make them.

```
 data/input/bars.parquet   symbol, day, open, high, low, close   daily, split-adjusted
 data/input/news.parquet   symbols (list), created_at (UTC), headline
```

- **Bars must be split-adjusted.** A daily log return above 1 in absolute value is
  treated as an unadjusted split, and every sample whose span touches it is dropped.
- **A headline counts at the first close (20:00 UTC) at or after its
  publication.**
- **Samples start at `SAMPLE_START`**, which should be the first date the news archive
  covers. Before that, an empty window would mean "not collected", not "no news".
- **The adapter** uses the Alpaca market-data API (daily bars from the
  consolidated SIP feed, plus its news API). It needs `APCA_API_KEY_ID` and
  `APCA_API_SECRET_KEY` in `.env`; copy `.env.example` to start.

## Outputs (`state/`, regenerated, gitignored)

| file | content |
|---|---|
| `dataset.pt` | samples, targets, inputs, and the rows for the latest close |
| `runs/fold<k>.{npz,json}` | test predictions of every model; validation scores; chosen k, λ |
| `eval.json` | every number in the results below |
| `forecast.csv` | `symbol, day, predicted_vol_pct, previous_window_vol_pct` |

In `forecast.csv`, `day` is the latest close, and `predicted_vol_pct` is the
annualized vol forecast for the next `WINDOW` sessions after it.
`previous_window_vol_pct` is the vol that actually happened over the `WINDOW`
sessions ending at `day`.

## Pipeline

```
 news ─► embed ─► embeddings ─┐
 news ────────────────────────┤
 bars ────────────────────────┴─► build ─► dataset ─┬─► train ─► runs ─► evaluate ─► eval.json
                                                    └─► forecast ──────────────────► forecast.csv
```

| stage | does |
|---|---|
| `embed` | headline → frozen sentence embedding (`ENCODER`, any Hugging Face model) |
| `build` | samples, Yang–Zhang target, inputs; exact data-hygiene rules; nothing fitted |
| `train` | every model on every walk-forward fold; every statistic from train data only |
| `evaluate` | QLIKE per test year, Diebold–Mariano tests, fold-level tests, calibration |
| `forecast` | fits `price` on all data; predicts the window after the latest close |

## Models compared

Each row adds its inputs to the row above.

```
 persistence   next window's variance = the previous window's   (naive bar)
 HAR           log variance over 1/5/22/66/132 sessions         (standard bar)
 price         + leverage: negative part of 1/5/22-session returns
 intensity     + headlines on the last session vs the stock's own norm
 ├ shuffled    + text features of RANDOM headlines              (control)
 └ text        + text features of the window's own headlines
```

## Results of the example run

The run uses 1,154 US-listed stocks with samples, 2021–2026, with
`WINDOW=22` (one trading month).

- **Test years:** walk-forward, 2023–2026, 805 test days in total.
- **Year boundaries:** each test year runs from about 22 sessions after
  1 January to about 22 sessions before 31 December, so no window crosses
  into the next year.
- **Skill vs HAR:** `1 − QLIKE / QLIKE_HAR`; +2% means 2% less loss than HAR.

```
 year   persistence   price    text
 2023      −225%      +2.1%    +3.2%
 2024       −24%      +2.0%    +6.4%
 2025       −18%      +3.6%    +8.2%
 2026       −17%      +1.2%    +2.3%
 all        −87%      +2.3%    +5.2%
```

```
 claim               DM t per year            fold p   sign p   verdict
 HAR > persistence   −1.6 −1.8 −1.6 −2.0      <.001    .125     holds
 price > HAR         −3.9 −2.0 −2.7 −0.9       .034    .125     holds (not Bonferroni)
 intensity > price   +0.8 −1.0 +0.8 +0.5       .548    .625     no
 text > shuffled     −0.9 −2.3 −1.2 −0.5       .049    .125     not established
```

- **DM t:** the year's Diebold–Mariano statistic; negative means the first
  model has lower loss.
- **fold p:** a t-test over the four years' DM t.
- **sign p:** a sign test over the four years; with 4 years it cannot go
  below .125.
- **Order:** claims are tested in this order at α = .05, stopping at the
  first failure. The chain stops at intensity, so text is not formally
  supported. Text's design was also chosen on validation years that overlap
  the test years, and the news feed more than doubled its volume from April
  2026.
- **The shipped forecast therefore uses `price`.**
- **Calibration** (mean predicted → realized vol by decile): 24→21, 30→26,
  33→31, 37→35, 42→40, 47→48, 54→56, 63→70, 79→90, 136→145.
- **Persistence's swing** (−225% in 2023, −17% in 2026) comes from its tails:
  its worst 1% of forecasts carry 83% of its 2023 loss.

These numbers belong to this universe, feed and period. With your own data,
rerun the same protocol and let the same tests decide. Caveats and full
tables: [METHODOLOGY §8–9](docs/METHODOLOGY.md#8-results-one-run).

## Configuration

| file | keys |
|---|---|
| `.env.data` | `SYMBOLS` (comma list or `@file`), `START` (first bar date to fetch), `BARS_PATH`, `NEWS_PATH`, `EMB_PREFIX`, `DATASET`, `STATE_DIR`, `ENCODER` |
| `.env.train` | `WINDOW` (sessions), `SAMPLE_START` (first sample date), `FOLDS` (walk-forward cut dates) |
| `.env` | provider credentials, used by `just fetch` only; gitignored |

- **The example run:** `WINDOW=22`, `SAMPLE_START=2021-01-01`,
  `FOLDS=2022-01-01,2023-01-01,2024-01-01,2025-01-01,2026-01-01`.
- **Fold k** trains on everything before cut k, validates on the year that
  starts at cut k, and tests on the year after that.
- **Overrides:** the OS environment overrides any key, e.g.
  `WINDOW=10 just build`. `just window 10` runs build, train and evaluate at
  another horizon, in `state/w10/`.

**Torch build.** `pyproject.toml` pins torch to the ROCm (AMD GPU) wheel
index. On other hardware, switch the index and relock, as CI does for CPU:

```bash
sed -i 's#whl/rocm[0-9.]*#whl/cpu#; s/pytorch-rocm/pytorch-cpu/g' pyproject.toml   # or whl/cu126, …
uv lock && uv sync
```

## Code

```
src/nexo/
  dataset.py   samples, target, inputs, walk-forward folds, hygiene rules
  text.py      headline embeddings (frozen encoder)
  model.py     QLIKE and its exact penalized fit (damped Newton, float64)
  train.py     text features, k/λ selection, every model, predictions, forecast
  evaluate.py  QLIKE, skill, Diebold–Mariano, fold-level tests, calibration
  metrics.py   Diebold–Mariano, Student-t p-value, fold-level tests
  fetch.py     optional Alpaca adapter
  cli.py env.py
justfile       every command, with a one-line explainer
```

## Extending

`WINDOW` (horizon), `SYMBOLS` (universe) and `ENCODER` (text model) are
config. The biggest untried levers are data, not model: intraday bars for a
cleaner target, an earnings calendar, and a news archive with stable
coverage. To avoid choosing the universe with hindsight, build `SYMBOLS` from
data that predates your first test year.

## License

MIT, see [LICENSE](LICENSE).
