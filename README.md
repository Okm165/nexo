# newsimpact

Forecast how volatile a stock will be over the **next window** of trading
sessions, from what is known **at today's close**: its price history and the
headlines written about it during the **previous window**. The window slides
one session at a time, so every trading day is a sample. One model is fitted
across a universe of stocks (1,159 by default), and results are reported for
the universe and for a target stock (`TARGET`, default **TSLA**).

The output is a number: annualized variance (shown as vol %) of the next
`WINDOW` sessions (default 5), the unit an implied-volatility quote uses. It
is trained and scored against the volatility that actually realized,
measured with the Yang–Zhang range estimator. No options data is used.

## The question, exactly

```
target  log annualized Yang–Zhang variance of sessions t+1 .. t+W   (next window)
input   own log Yang–Zhang variance over 5, 22, 66, 132 sessions     (price state)
        news intensity: log(1+headlines in t-W+1..t) − log(1+its 132-session norm)
        the window's headlines, as frozen sentence embeddings        (previous window)
```

Yang–Zhang (2000), over the n days of a window, from daily open, high, low
and close:

```
o = ln(Oᵢ/Cᵢ₋₁)   c = ln(Cᵢ/Oᵢ)   rs = u(u−c) + d(d−c),  u = ln(Hᵢ/Oᵢ), d = ln(Lᵢ/Oᵢ)
σ² = Var(o) + k·Var(c) + (1−k)·mean(rs),   k = 0.34 / (1.34 + (n+1)/(n−1)),   ×252
```

It is unbiased under drift and opening jumps, and several times more
efficient than close-to-close returns for the same window.

## Pipeline

```
fetch-bars ─► bars.parquet  (symbol, day, open, high, low, close)
news archive (symbols, created_at, headline) ─► embed ─► emb_s*.npz (768-d, float32)
                                       build ─► dataset.pt   (samples, inputs, targets)
                                       train ─► runs/split*  (every arm, every split)
                                    evaluate ─► eval.json
```

**build**: exact rules, nothing fitted, so nothing here can leak.

| rule | why |
|---|---|
| a headline counts at the first close (20:00 UTC) at or after its publication | point in time. 20:00 UTC is the summer New York close and an hour early in winter, so the rule is never late. Weekend and after-hours news goes to the next session |
| a bar is valid only if 0 < low ≤ min(open, close) ≤ max(open, close) ≤ high | inconsistent OHLC is a data error |
| every session of [t−132, t+W] present, valid, with \|log close-to-close\| ≤ 1 | gaps and unadjusted splits are not volatility |
| every variance > 0 | zero means the price did not trade |
| samples start at `SAMPLE_START` (2021-01-01) | where the news archive's coverage begins. Earlier, an empty window would mean "not collected", not "no news". Earlier bars still serve as price history |

Return arithmetic runs in float64, with windowed sums taken from cumulative
sums. Embeddings are stored as float32: in float16 the small
headline-specific part riding on the shared direction loses a few percent
of its precision.

## Model

A linear forecast of log variance, fitted by **QLIKE**:

```
h = w · [1, log-variance lookbacks, news intensity, text PCs]
QLIKE(h, y) = e^(y−h) − (y−h) − 1        y = log proxy variance
```

QLIKE ranks forecasts as the true variance would even though a week of
returns measures it noisily (Patton 2011). MSE on log vol does not have that
property. QLIKE is convex, so the fit is exact (damped Newton, float64, Hessian on the GPU when present),
with no SGD, no seeds and no epochs.

**Text features**, per split, from train data only:
1. Standardize every embedding dimension over the training headlines. Raw
   mean-pooled vectors share one dominant direction: two random headlines
   have a mean cosine of 0.91, and 0.00 after standardizing.
2. Take the training headlines' top principal components.
3. Project each window's mean headline onto them, and standardize the
   scores.

A ridge penalty λ applies to the text coefficients only. The number of
components k ∈ {1, 2, 4, 8, 16} and λ are chosen on validation.

**One model for all stocks.** One stock gives about 180 independent weeks
per split. On TSLA alone, a neural attention model memorized its training
data (train QLIKE 0.02) and never beat its HAR starting point on validation,
and a linear text model's validation gains vanished on test. Pooling 1,159
stocks gives about 1.1M training samples per split. The same coefficients
apply to every stock, and each stock's own variance lookbacks carry its
level.

**Components.** k was chosen on validation from {4, 16, 64}, then
{16, 64, 256} once 64 won all 10 splits; 256 then won 8 of 10. The full
embedding was tried as well, with grid {64, 256, 768}, and validation chose
768 in 6/10 splits (+0.2% on validation). On test it was worse: universe
+0.46% ± 1.10% against +0.76% ± 1.11%, and the shuffled control fell from
−0.52% to −0.78%, which is overfitting. The shipped grid is back to
{16, 64, 256}. That revert was informed by test, so treat the +0.76% below
as slightly optimistic; the protocol-clean figure for the larger grid is
+0.46%. Both say the same thing: a content effect of about +0.5–0.8%, and
extra capacity does not raise it.

## Controls and benchmarks

| predictor | inputs | answers |
|---|---|---|
| persistence | next-window variance = previous window's | the naive bar |
| HAR | log-variance lookbacks | the standard bar (Corsi 2009) |
| intensity | HAR + news intensity | does the *amount* of news help? |
| shuffled | intensity + text PCs of random train headlines, same counts; k and λ chosen exactly as for text | does headline *content* help? |
| text | intensity + text PCs of the window's headlines | the model |

## Evaluation (what keeps the numbers honest)

- **Purged random month blocks.** The calendar is cut into blocks of
  `BLOCK` = 21 sessions, and blocks, not days, go to train / validation /
  test (70/15/15) uniformly at random, so every split samples every market
  regime. A sample is kept only if its whole span, from the first session of
  its headline window to the last session of its target window, lies in one
  split. No target session and no headline is shared across splits; this
  purge costs about 20% of samples.
- **Repeated.** `SPLITS` = 10 independent random assignments. Results are
  mean ± sd across splits plus the number of splits each comparison wins, so
  no conclusion rests on one draw of test months.
- **Train-only statistics.** Feature scaling, embedding standardization,
  principal components, the shuffled-headline pool and every coefficient
  come from the split's train data. Validation picks k and λ; test is
  predicted once.
- **Diebold–Mariano tests on QLIKE** for every split. The Newey–West
  variance uses lag W−1, counted only within contiguous test blocks.

## Results — pooled fit, next-5-session vol, 10 random purged block splits

Universe (about 155k test samples per split):

| predictor | QLIKE | skill vs HAR | text better in | Spearman |
|---|---|---|---|---|
| persistence | 0.936 ± 0.717 | −155% | — | 0.824 |
| HAR | 0.363 ± 0.023 | 0 | 8/10 | 0.866 |
| intensity | 0.364 ± 0.023 | −0.11% ± 0.20% | 8/10 | 0.865 |
| shuffled | 0.365 ± 0.023 | −0.52% ± 0.40% | **10/10** | 0.864 |
| **text** | **0.360 ± 0.023** | **+0.76% ± 1.11%** | — | 0.861 |

TSLA (about 144 test days per split):

| predictor | QLIKE | skill vs HAR | text better in |
|---|---|---|---|
| HAR | 0.210 ± 0.042 | 0 | 4/10 |
| intensity | 0.210 ± 0.042 | −0.19% ± 0.60% | 4/10 |
| shuffled | 0.209 ± 0.042 | +0.11% ± 1.17% | 4/10 |
| text | 0.209 ± 0.043 | +0.31% ± 4.02% | — |

What this supports, and what it does not:
- **Across stocks, headline content improves next-week volatility
  forecasts, by a small amount**: +0.76% QLIKE over HAR. It beats the
  shuffled-headline control in 10/10 splits and HAR in 8/10. The controls
  gain nothing: news volume alone is −0.11%, and shuffled content −0.52%,
  so the gain is content, not added flexibility.
- **It is not significant within any single split** (mean DM t −1.2 to
  −1.7). The splits reuse the same data, so 10/10 is consistent evidence,
  not ten independent tests.
- **Validation overstated it.** Validation gains were 1–4%; test gives
  0.76%, and only test is reported as the result.
- **TSLA alone: no detectable effect** (+0.31% ± 4.02%, 4/10). An effect
  of about 1% cannot be resolved with about 144 test days per split. The
  TSLA-only fit, before pooling, was −6.7% ± 25.6%.
- **Calibration** (pooled test, universe, predicted → realized vol by
  decile): 21.0 → 19.8%, 41.2 → 39.0%, 67.6 → 69.3%, 136.5 → 149.3%.
- **HAR beats persistence in every split**; the benchmark is strong.

Diagnosis of the result (test predictions, no refitting; so the test set
has been looked at for this analysis):
- **QLIKE is tail-dominated.** The worst 1% of samples hold 29–44% of each
  split's HAR loss, so a split's result depends on which extreme weeks
  land in it. This is why validation gains (1–4%) overstated test (0.76%).
  The effective sample is far smaller than 155k.
- **The gain is concentrated in time.** Text lowers the loss in 33 of 58
  test months; the best 5 months give 84% of the net gain, and the worst 5
  cancel 63%.
- **The gain grows with the number of headlines** (ex ante, pooled test):
  0 headlines −1.3%, 1 −2.5%, 2–3 +1.2%, 4–7 +3.3%, 8–15 +4.1%, 16+
  +5.9%. Information comes from aggregating headlines. Windows with 0–1
  headlines (45% of samples) are worse than HAR.
- **Tried and rejected on validation:** empirical-Bayes shrinkage of each
  window mean toward neutral by n/(n + σ²_w/σ²_b). It was better in only
  3/10 splits on validation, so it was reverted without being scored on
  test.

## Quick start

```bash
uv sync
uv run pytest -q                    # includes a synthetic end-to-end run
uv run newsimpact fetch-bars        # optional Alpaca adapter (keys in .env)
uv run newsimpact fetch-news        # only if you have no longer archive
scripts/run.sh                      # embed → build → train → evaluate
```

Artifacts go to `STATE_DIR` (default `./state`): `dataset.pt`,
`runs/split<k>.{npz,json}` (test predictions per arm, and the selected k
and λ) and `eval.json`.

## Configuration

`.env.data` holds `SYMBOLS` (a comma list, or `@file` with one symbol per
line, e.g. `@data/input/symbols.txt`), `START` and the paths; `.env.train` holds
`WINDOW` (5), `SAMPLE_START` (2021-01-01), `BLOCK` (21), `VAL_FRACTION` /
`TEST_FRACTION` (0.15), `SPLITS` (10) and `TARGET` (TSLA). `.env` holds credentials, needed
only by the fetchers. The OS environment overrides any key. Paths are
repo-relative and inputs are plain parquet, so any source with the same
columns works.

## Code

```
src/newsimpact/
  fetch.py      optional Alpaca adapter (bars, news)
  text.py       headline → frozen ModernBERT-base embedding (≤64 tokens, mean pool)
  dataset.py    samples, Yang–Zhang target and inputs, blocked random split
  model.py      QLIKE and its exact penalized fit
  train.py      per split: text features, k/λ selection, every arm, test predictions
  evaluate.py   QLIKE, skill, Diebold–Mariano, calibration, across splits
  metrics.py    rank correlation, Diebold–Mariano
scripts/run.sh  the whole pipeline
```

**Encoder tried and rejected:** all-MiniLM-L6-v2, a model trained for
sentence similarity (random-headline cosine 0.14 vs ModernBERT's 0.91).
Ran the identical fit on identical splits: it lost to ModernBERT on
validation in 7/10 splits (mean text QLIKE 0.3701 vs 0.3688). Cleaner
geometry did not help; the representation axis is closed.

## Limits

- A single stock's test period is too small to detect a ~1% effect. The TSLA
  numbers are reported, but they cannot confirm or rule out the effect.
- The Alpaca news API on this account serves only recent news (from
  2026-04). The 2021–2026 archive comes from earlier collection.
- Price lookbacks reach across split blocks. They contain no target of
  another split, but volatility persistence makes neighbouring blocks
  similar.
