# newsimpact

A research pipeline that forecasts, for every symbol in a universe, the
**realized volatility of the next `WINDOW` trading sessions** — as a number,
in the units an implied-volatility quote uses — from the information available
at the close of day `t`: the symbol's volatility history, the headlines written
about it during the previous `WINDOW` sessions, and how much news it received.
One linear model is shared by the whole universe; the window slides one session
at a time, so every trading day is a sample.

The pipeline is deliberately narrow: one target, one loss, one evaluation
protocol, and a set of controls that make every claim falsifiable. It is
data-agnostic (plain parquet in, parquet out), horizon-agnostic (`WINDOW`),
and encoder-agnostic (any Hugging Face model).

## Architecture, and why

```
bars parquet ─┐
              ├─► fetch ─► build ─► dataset.pt ─► train ─► runs/fold* ─► evaluate ─► eval.json
news parquet ─┘        (embed)                    (fit)                       └► forecast ─► forecast.csv
```

Each stage is a pure function of its inputs, separated so that nothing fitted
can leak backwards into data construction.

| decision | justification |
|---|---|
| **target: log variance of the next W sessions, Yang–Zhang from daily OHLC** | range-based estimators use the day's high/low, not just closes, and are unbiased under drift and opening jumps — several times more efficient than close-to-close for the same number of bars |
| **loss: QLIKE on variance** | with a noisy volatility proxy, MSE on log vol mis-ranks forecasts; QLIKE keeps the ranking a true-variance loss would give (Patton 2011) and is convex, so the fit is exact by damped Newton in float64 — no SGD, no seeds, no epochs |
| **one pooled model, no per-symbol terms** | a per-stock level fitted on training targets looks better under shuffled cross-validation, where it interpolates the test month, and loses when tested forward in time; the longest lookback carries the symbol's level instead |
| **pooled features standardized within the training fold only** | any statistic computed over the full sample is a leak |
| **text: frozen encoder → principal components → window means, split by recency** | headlines are sparse and short-lived; averaging all days equally dilutes the most recent session, and day-relative centering stops a changing news feed from moving forecasts on its own |
| **evaluation: walk-forward folds** | shuffled month blocks reward anything that interpolates slow state; forward-in-time folds, purged so no target session or headline is shared across parts, are the standard in the volatility-forecasting literature |
| **statistics: per-fold Diebold–Mariano plus a fold-level meta-test** | a pooled daily t-statistic can be significant while the effect is unstable across years; the meta-test over independent test years is what a reader should rely on |
| **controls: persistence, HAR, and a shuffled-headline arm tuned identically** | every claim about content must beat both a no-content arm and a pure-flexibility arm, or it is not content |

## Data

Two parquet tables, whatever their origin:

- **bars**: `symbol, day, open, high, low, close` — daily bars.
- **news**: `symbols, created_at, headline` — headlines with publication
  timestamps and the symbols they are about.

An adapter for one market-data provider is included (`fetch-bars`,
`fetch-news`; credentials in `.env`), but any source with these columns works:
the pipeline never sees the provider. Embeddings are produced by any Hugging
Face encoder (`ENCODER` in `.env.data`).

## Method

**Inputs at the close of day t.** Log variance over the last 1, 5, 22, 66, 132
sessions (Yang–Zhang; `o² + rs` for the single session); leverage, the negative
part of the 1/5/22-session log return; news intensity, the last session's
headline count relative to the symbol's own recent norm; and the window's
headlines as frozen sentence embeddings, averaged separately over the last
session and the sessions before it, each part taken relative to the same day's
cross-symbol mean.

**Arms.** Each adds to the previous: `persistence` (the previous window's
variance) → `HAR` (the variance lookbacks) → `price` (+ leverage) →
`intensity` (+ news intensity) → `shuffled` (+ text PCs of random train
headlines, tuned identically — the flexibility control) → `text` (+ text PCs
of the window's headlines). The ridge penalty and the number of components are
chosen on validation, identically for `text` and `shuffled`.

**Selection discipline.** Feature and design choices are made by ablations on
validation periods only, under a rule fixed before running them. A period used
for any choice cannot afterwards be claimed as test evidence — recorded in
`AGENTS.md` and honoured in the results below.

## Results achieved (one full run, as an example)

A universe of about 1.2k US-listed symbols, daily bars and a vendor headline
archive, 2021–2026, `WINDOW = 22` (one trading month), four walk-forward folds
with test years 2023–2026. All numbers are from the fitted runs in `state/`.

Test QLIKE, skill vs the HAR arm:

| test year | persistence | price | text |
|---|---|---|---|
| 2023 | −225% | +2.1% | +3.2% |
| 2024 | −24% | +2.0% | +6.4% |
| 2025 | −18% | +3.6% | +8.2% |
| 2026 | −17% | +1.2% | +2.3% |
| all (805 days) | −87% | **+2.3%** | **+5.2%** |

- `price` vs `HAR`: positive in all four test years; fold-level meta-test
  p = 0.033 — **established**.
- `text` vs its shuffled control: positive in all four years, p = 0.048 —
  **borderline**, because the design was selected on overlapping validation
  periods and the news feed changed mid-test. It is therefore excluded from
  the shipped forecast until confirmed on later data.
- Calibration of predicted vs realized vol by decile: 24→21 … 136→145 —
  the numbers read as volatility forecasts.

The result generalizes as a *method*, not as these numbers: with a different
universe, feed, or horizon, rerun the same protocol and let the same tests
decide.

## Usage

```bash
uv sync
uv run pytest -q                     # includes a synthetic end-to-end run

# data (optional adapters; or drop in your own parquet)
uv run newsimpact fetch-bars         # credentials in .env
uv run newsimpact fetch-news
uv run newsimpact embed --shard s --shards N --device cuda:0

# the pipeline
uv run newsimpact build              # samples, inputs, targets
uv run newsimpact train              # every arm on every walk-forward fold
uv run newsimpact evaluate           # test-period table, DM tests, calibration
uv run newsimpact forecast           # the deliverable: forecast.csv
scripts/run.sh                       # all of the above in order
```

Artifacts go to `STATE_DIR`: `dataset.pt`, `runs/fold<k>.{npz,json}`,
`eval.json`, `forecast.csv`.

## Configuration

| file | keys |
|---|---|
| `.env.data` | `SYMBOLS` (list or `@file`), `START`, `BARS_PATH`, `NEWS_PATH`, `EMB_PREFIX`, `DATASET`, `STATE_DIR`, `ENCODER` |
| `.env.train` | `WINDOW`, `SAMPLE_START`, `FOLDS` (the walk-forward cut dates) |
| `.env` | provider credentials, needed only by the fetchers |

The example run referenced above used: `WINDOW=22`, `SAMPLE_START=2021-01-01`,
`FOLDS=2022-01-01,2023-01-01,2024-01-01,2025-01-01,2026-01-01`.

The OS environment overrides any key, so variants (`WINDOW=10`,
`STATE_DIR=state/w10`, …) run without editing files. Paths are repo-relative;
inputs are plain parquet.

## Code

```
src/newsimpact/
  fetch.py      optional provider adapter (bars, news)
  text.py       headline → frozen encoder embedding (≤64 tokens, mean pool)
  dataset.py    samples, target, inputs, walk-forward folds; exact hygiene rules
  model.py      QLIKE and its exact penalized fit (damped Newton, float64)
  train.py      per fold: text features, k/λ selection, every arm, predictions
  evaluate.py   QLIKE, skill, Diebold–Mariano, fold-level meta-test, calibration
  metrics.py    Diebold–Mariano
scripts/run.sh  the whole pipeline
```

## Extending

The narrow scope is a choice, and the seams are where you would widen it:

- **horizon**: `WINDOW` (anything from 1 session up, as long as the news
  archive covers the sample start);
- **universe**: `SYMBOLS` (any list; the model is pooled, so adding symbols
  only adds samples);
- **text representation**: `ENCODER` (any Hugging Face model; a domain-trained
  encoder may do better than a general one — test, do not assume);
- **target quality**: intraday bars would replace the daily Yang–Zhang proxy
  and remove most of the measurement noise;
- **scheduled events**: an earnings calendar is the most predictable
  volatility feature the current input set lacks.

## Limits

- The target from daily bars is noisy; the worst 1% of samples carry about a
  third of the loss, which is why the horizon decision matters.
- News archives change what they publish; check volume and template stability
  across the sample before trusting any text result.
- A frozen encoder pretrained on web text is not point-in-time: its embeddings
  may encode knowledge of events after a headline's date, so text results are
  upper bounds.
- Every design choice spends the data it used; the README of each run should
  say which periods informed which choices.
