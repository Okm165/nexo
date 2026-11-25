# newsimpact

Forecast how volatile a stock will be over the **next window** of trading
sessions, from what is known **at today's close**: its price history, the
market's volatility regime, and the headlines written about it during the
**previous window**.

The output is a number: annualized realized volatility over the next
`WINDOW` sessions (default 5, one trading week), in vol points, the unit an
implied-volatility quote uses. It is trained and scored against the
volatility that actually realized. No options data is used; the model
predicts realized vol, which an implied-vol quote is the market's forecast
of.

## The question, exactly

For symbol *s* and trading day *t*:

```
target  rv(t) = sqrt( 252/W · Σ_{k=1..W} r²_{t+k} )        r = daily log return
                                                           (the next window)
input   headlines about s published at or before the close of t,
        in sessions t-W+1 .. t                             (the previous window)
        own log realized vol over 5, 22, 66, 132 sessions  (state + long-run level)
        market log realized vol over the same lookbacks    (vol regime, all symbols)
        news intensity = log(1+count) − same-day mean      (relative attention)
```

The window **slides one session at a time**: every trading day is a new
sample, not every W days. Consecutive samples share W−1 sessions of target.
This matters for both training and testing; see Methodology.

```
sessions   …  t-4  t-3  t-2  t-1   t  │ t+1  t+2  t+3  t+4  t+5
headlines  [====== previous window ====]│
target                                 │[======= next window ======]
next day        [===== previous ========]│[======== next ========]
```

## Pipeline

```
bars.parquet  (symbol, day, close)            ─┐
news.parquet  (symbols, created_at, headline) ─┼─ embed ─► emb_s*.npz  (768-d per headline)
                                                └─ build ─► dataset.pt
                                     train  text / shuffled / price × seeds
                                                    evaluate ─► eval.json
```

**embed**: every headline tagged with a symbol that has bars is encoded once
by a frozen ModernBERT-base (≤64 tokens, mean-pooled).

**build** turns the two tables into samples. Every rule is exact; there are
no tuned thresholds.

| step | rule | why |
|---|---|---|
| point in time | a headline belongs to the first close (20:00 UTC) at or after its publication | 20:00 UTC is the New York close in summer and an hour early in winter, so the rule is never late. Weekend and after-hours news goes to the next session |
| multi-ticker headlines | attach to every tagged symbol that has bars | the vendor says it is about each of them |
| calendar gaps | every session of [t−132, t+W] must exist for the symbol | delistings and reused tickers otherwise turn "5 sessions" into months |
| data errors | no \|log return\| > 1 in that span | a >172% day is an unadjusted split, not volatility |
| stale prices | every realized vol must be > 0 | zero means the price did not trade |
| market regime | same-day mean over every symbol with a clean **history** | uses no future information, not even a symbol's future data quality |
| news intensity | count relative to the same day's cross-section | headlines per sample rise ~35% in 2026 because the vendor covers more, not because markets are busier |
| purged splits | a sample is in a split only if t **and** t+W fall inside it | no target window straddles a boundary |
| embeddings | standardized per dimension with train-period mean and sd | raw mean-pooled vectors share one dominant direction: two random headlines have a mean cosine of 0.91. After standardizing it is 0.00 |
| inputs and target | standardized with train statistics inside `train` | nothing about val or test leaks into scaling |

Each headline is stored once. A sample is an index range into the symbol's
time-sorted headlines, so overlapping windows cost no memory. Return
arithmetic runs in float64; inputs and targets are stored as float32, and
the standardized embeddings as float16 (they lie within ±10).

**train**: one run per (arm, seed). The loss is MSE on standardized log
realized vol, with AdamW and cosine decay. The epoch is chosen on
validation, and test is predicted once from the best-validation weights.

**evaluate** scores every arm (seeds averaged in log space) against two
benchmarks, with significance tests.

## Model

```
ctx = [own rv5,22,66,132 | market rv5,22,66,132 | news intensity]   (log, standardized)

ŷ  = Linear(ctx)                                     ← a learned HAR regression
   + Head( MLP(ctx) + Pool(headlines | ctx) )        ← correction (Head starts at zero)

Pool:  headline → Linear 768→128 + age embedding (0..W−1 sessions) + MLP(ctx)
       → one pre-LN transformer layer over the set   (headlines read each other)
       → attention pooling, query = learned vector + MLP(ctx)
```

- **Residual on HAR.** HAR (Corsi 2009) is the standard realized-vol
  forecast. Because the head starts at zero, training begins exactly at a
  linear HAR and moves away from it only when that lowers error.
- **A set, not a sequence.** Headlines are order-free. Order enters only
  through each headline's age in sessions. Padding is masked, and both
  properties are tested.
- **Small.** 0.33M trainable parameters over a frozen encoder.

## Controls and benchmarks

| predictor | sees | answers |
|---|---|---|
| `text` | price + market + news intensity + the window's real headlines | the model |
| `shuffled` | same, but headlines replaced by random train-period ones (same count and ages) | does headline *content* matter? |
| `price` | price + market only; no news input of any kind | does news matter at all? |
| HAR | least squares of log next-window vol on the same price + market inputs, fitted on train | the standard bar to clear |
| persistence | next-window vol = previous-window vol | the naive bar |

The three arms use identical seeds and batch order.

## Methodology (what keeps the numbers honest)

- **Splits:** train ≤ 2023-12-31, validation 2024 (through 2024-12-31),
  test 2025 onward. All of them are purged. The few hyperparameters
  (learning rate, dropout, weight decay) and every input choice were decided
  on validation, and test was scored once at the end.
- **Overlapping targets.** Neighbouring samples share W−1 target sessions,
  so a network can lower *train* loss by interpolating between neighbours.
  On the default settings every arm peaked at epoch 1–2 for this reason.
  Validation is purged and a year apart, so it is the judge. A lower
  learning rate with stronger regularization lets the curve settle before
  selection.
- **Significance:** Diebold–Mariano tests on per-day mean squared error,
  with Newey–West variance at lag W−1 because consecutive targets overlap.
  Every arm reports its seed mean ± sd as well as the ensemble.
- **Stability:** MSE is reported per test year. A real improvement holds in
  each one.
- **Noise floor:** five squared returns measure a week's volatility only
  roughly. With Gaussian returns even a perfect forecast would have
  log-MSE = ψ′(W/2)/4 ≈ 0.12 at W=5, and fat tails push the floor higher.
  R² is reported against that context, not against 1.
- **Units:** predictions are in log space, so exp(prediction) is a
  median-type forecast. MAE in vol points is reported on that basis.

## Results

Next-5-session realized vol, 10 seeds per arm, test = 377,839 samples over
470 days (2025-01 .. 2026-11). The variance of log vol is 0.521, and the
Gaussian noise floor is 0.123.

| predictor | MSE (log) | R² | vs HAR | MAE (vol pts) | Spearman | 2025 | 2026 | seed MSE |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.4371 | 0.162 | −62.5% | 24.5 | 0.582 | 0.4784 | 0.3950 | — |
| HAR | 0.2690 | 0.484 | 0 | 18.0 | 0.698 | 0.2934 | 0.2443 | — |
| price | 0.2756 | 0.471 | −2.5% | 18.1 | 0.680 | 0.3080 | 0.2427 | 0.2797 ± 0.0080 |
| shuffled | 0.2583 | 0.505 | +4.0% | 17.7 | 0.709 | 0.2789 | 0.2374 | 0.2624 ± 0.0037 |
| **text** | **0.2557** | **0.510** | **+5.0%** | **17.6** | **0.713** | **0.2737** | **0.2373** | 0.2599 ± 0.0020 |

Diebold–Mariano t (negative = first is better): text vs HAR **−5.94**,
text vs price −4.81, text vs shuffled −2.42, price vs HAR +1.73.

Validation (2024, mean over seeds): text 0.2566, price 0.2591, shuffled
0.2604, HAR 0.2695.

What this supports, and what it does not:

- **News improves next-week volatility forecasts beyond HAR**: −5% MSE,
  t = −5.9, better in both 2025 and 2026, with the tightest seed spread.
- **Most of that gain is news *intensity and timing*, not wording.** The
  shuffled arm keeps the headline counts and ages with random content, and it
  already gets +4.0% over HAR. Real content adds about 1% more (t = −2.4).
  That increment is clear on validation and in 2025, but zero in 2026
  (0.2373 vs 0.2374), so it is real but small and not yet stable.
- **A price-only network is not robust.** It beats HAR on 2024 validation
  (0.2591 vs 0.2695) but loses on 2025–26 and has the widest seed spread.
  The nonlinear price fit does not transfer across regimes; linear HAR does.
  This was found after testing and is reported, not tuned away.
- **Calibration** (text, geometric means per predicted decile): predicted
  14.5 / 29.3 / 48.7 / 90.7% against realized 15.5 / 30.5 / 49.3 / 90.1%
  for deciles 1 / 5 / 8 / 10. The model runs about 3.5% low on average
  (log bias −0.036), with the level anchored to the calmer validation year.
- **Headroom.** R² of 0.51 against an irreducible floor of at least 0.123
  (out of 0.521): the best model removes about 67% of the reducible error
  under the Gaussian bound, and more under fat tails.

## Quick start

```bash
uv sync
uv run pytest -q                  # includes a synthetic end-to-end run on CPU

# inputs: your own tables at BARS_PATH and NEWS_PATH (schemas above), or
uv run newsimpact fetch-news      # optional Alpaca adapters (keys in .env)
uv run newsimpact fetch-bars

scripts/run.sh                    # embed → build → train 3 arms × SEEDS on all GPUs → evaluate
```

Stage by stage:

```bash
uv run newsimpact embed --shard 0 --shards 1 --device cuda   # one shard per GPU
uv run newsimpact build
uv run newsimpact train --arm text --seed 0                  # text | shuffled | price
uv run newsimpact evaluate
```

Artifacts go to `STATE_DIR` (default `./state`): `dataset.pt`,
`runs/<arm>_s<seed>.{json,npy}` (validation curve, best epoch, test
predictions in log vol) and `eval.json`.

## Configuration

`.env.data` holds paths and `.env.train` the sample definition and
hyperparameters. `.env` holds credentials and is only needed by the Alpaca
fetchers. The OS environment overrides any key (`WINDOW=10 uv run
newsimpact build`). Paths are repo-relative, inputs are plain parquet, and
the device is whatever `torch` device you pass (CUDA or GPU).

| key | default | meaning |
|---|---|---|
| `WINDOW` | 5 | sessions in both the headline window and the target window |
| `TRAIN_END` / `VAL_END` | 2023-12-31 / 2024-12-31 | purged split boundaries |
| `MAX_HEADLINES` | 64 | most recent headlines kept per window |
| `EPOCHS`, `BATCH`, `LR`, `WEIGHT_DECAY`, `DROPOUT` | 16, 256, 1e-4, 0.1, 0.3 | training (chosen on validation) |

## Code

```
src/newsimpact/
  text.py       embed: headline → vector
  dataset.py    build: sliding samples, point-in-time attachment, inputs, targets
  model.py      VolForecaster
  train.py      one (arm, seed) run
  evaluate.py   benchmarks, metrics, Diebold–Mariano, per-year, calibration
  metrics.py    rank correlation, Diebold–Mariano
  fetch_*.py    optional Alpaca adapters
scripts/run.sh  the whole pipeline over all GPUs
```

## Limits

- The target is realized volatility. Comparing against implied vol needs an
  options feed.
- Samples are (symbol, day) pairs with at least one headline in the window,
  and the benchmarks are scored on the same samples. This is not the full
  cross-section.
- Close-to-close returns over 5 sessions are a noisy target. A range-based
  estimator (high/low) would cut that noise but needs OHLC bars.
- The fixed 20:00 UTC close ignores half-day sessions.
