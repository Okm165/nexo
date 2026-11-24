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

A linear forecast of log variance, fitted by **QLIKE**, with a per-stock
level (a fixed-effects panel regression):

```
y − level[s] = w · (x − mean[s])         fitted on all stocks at once
h            = w · (x − mean[s]) + level[s]
x            = [1, log-variance lookbacks, news intensity, text PCs]
QLIKE(h, y)  = e^(y−h) − (y−h) − 1       y = log proxy variance
```

`level[s]` is `log mean(e^y)` over that stock's **train** targets — the exact
QLIKE-optimal location, which is not the mean of log variance. `mean[s]` is
the stock's train mean of a feature. Each stock's permanent volatility level
is therefore estimated from all of its train days, and the pooled slope
describes how a stock moves away from its own norm. The transformation is
exact — QLIKE depends only on `y − h` — so it adds no parameter to the solver
and no knob; a stock absent from a split's train samples takes the pooled
level. Without it, the four noisy lookbacks were the only route by which the
model could see a stock's level; fixing that improved the benchmark by 6.9%
on validation and 5.4% on test.

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
components k ∈ {16, 64, 256} and λ ∈ {1e−3, 1e−2, 1e−1} are chosen on
validation.

**One model for all stocks.** One stock gives about 180 independent weeks
per split. On TSLA alone, a neural attention model memorized its training
data (train QLIKE 0.02) and never beat its HAR starting point on validation,
and a linear text model's validation gains vanished on test. Pooling 1,159
stocks gives about 1.1M training samples per split. One slope applies to
every stock; the per-stock level is estimated separately (above).

**Components.** k was chosen on validation from {4, 16, 64}, then
{16, 64, 256} once 64 won all 10 splits; 256 then won 8 of 10. The full
embedding was tried as well, with grid {64, 256, 768}, and validation chose
768 in 6/10 splits (+0.2% on validation). On test it was worse: universe
+0.46% ± 1.10% against +0.76% ± 1.11%, and the shuffled control fell from
−0.52% to −0.78%, which is overfitting. The shipped grid is back to
{16, 64, 256}. That revert was informed by test, so treat the +0.76% below
as slightly optimistic; the protocol-clean figure for the larger grid is
+0.46%. On the random splits both say the same thing: a content effect of
about +0.5–0.8%, and extra capacity does not raise it. The forward holdout
shows that effect does not replicate out of sample.

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
- **A forward holdout.** Because the random splits reuse the same history
  (and their test months were read during development), the honest check is
  a single time-ordered split: train < 2025, validation 2025, test 2026,
  scored once with the design frozen (`newsimpact holdout`). It is the
  reason the content claim below is stated as unsupported.

## Results — pooled fit, next-5-session vol, 10 random purged block splits

Universe (about 155k test samples per split):

| predictor | QLIKE | skill vs HAR | text better in | Spearman |
|---|---|---|---|---|
| persistence | 0.501 ± 0.102 | −48.0% | — | 0.818 |
| HAR | 0.343 ± 0.033 | 0 | 9/10 | 0.851 |
| intensity | 0.344 ± 0.033 | −0.06% ± 0.17% | 9/10 | 0.851 |
| shuffled | 0.344 ± 0.033 | −0.12% ± 0.22% | 9/10 | 0.851 |
| **text** | **0.340 ± 0.032** | **+0.96% ± 0.69%** | — | 0.851 |

TSLA (about 144 test days per split):

| predictor | QLIKE | skill vs HAR | text better in |
|---|---|---|---|
| persistence | 0.280 ± 0.065 | −39.6% | — |
| HAR | 0.200 ± 0.039 | 0 | 3/10 |
| intensity | 0.200 ± 0.039 | −0.40% ± 0.65% | 6/10 |
| shuffled | 0.200 ± 0.039 | −0.29% ± 0.58% | 6/10 |
| text | 0.201 ± 0.040 | −0.56% ± 2.64% | — |

What this supports, and what it does not:
- **On these random splits, headline content improves next-week volatility
  forecasts by a small amount**: +0.96% QLIKE over HAR, better in 9/10
  splits, mean DM t −2.33 (below −2 in 6/10 splits; the splits reuse the
  same history, so that is consistent evidence, not ten independent tests).
  **This is the number the forward holdout below contradicts** — read that
  section before believing this one.
- **The controls gain nothing**: news volume alone is −0.06% and shuffled
  content −0.12% against HAR, while text beats both in 9/10 splits (mean t
  −2.85 and −2.99). The gain is content, not added flexibility.
- **Validation overstated it.** Validation gains were 1.5% on average
  (10/10 splits); test gives 0.96%, and only test is reported as the result.
- **TSLA alone: no detectable effect** (−0.56% ± 2.64%, 3/10). An effect of
  about 1% cannot be resolved with about 144 test days per split. The
  TSLA-only fit, before pooling, was −6.7% ± 25.6%.
- **Calibration is close**: mean realized vol by predicted decile (pooled
  test, universe) 20, 26, 30, 34, 39, 47, 56, 69, 92, 149% against
  predicted 19, 25, 29, 34, 39, 46, 55, 68, 89, 154%. TSLA: predicted 45–76%
  against realized 42–75%.
- **HAR beats persistence in every split** (10/10, mean t −3.22 on the
  random splits, −5.83 on the holdout); the benchmark is strong. Persistence
  is calibrated like every fitted arm — the previous window's deviation from
  the stock's norm, plus the stock's level — so the comparison is fair
  (uncalibrated, it lost by 181%, an artefact of the loss, not of the
  forecast).

The per-stock level mattered for the benchmark itself: fixing it took
universe test QLIKE from 0.363 to 0.343 (−5.4%), TSLA from 0.210 to 0.200,
and it raised text over HAR from +0.76% (8/10) to +0.96% (9/10). TSLA is the
one place the change hurt the text arm (+0.31% → −0.56%): with a better
benchmark there is less room for a noisy content effect at 144 days.

Diagnosis of the result (test predictions, no refitting; so the test set
has been looked at for this analysis):
- **QLIKE is tail-dominated.** The worst 1% of samples hold 26–39% of each
  split's HAR loss, so a split's result depends on which extreme weeks land
  in it. The effective sample is far smaller than 155k.
- **The gain is spread over time**, unlike the previous fit: text lowers the
  loss in 42 of 58 test months; the best 5 months give 55% of the net gain
  and the worst 5 cancel 22% (previously 84% and 63%).
- **The gain grows with the number of headlines** (ex ante, pooled test):
  0 headlines −0.1%, 1 −0.1%, 2–3 +1.1%, 4–7 +1.8%, 8–15 +2.9%, 16+ +2.7%.
  Information comes from aggregating headlines. Quiet windows (46% of
  samples) are now neutral rather than losing, because the per-stock level
  no longer has to be inferred from headline counts.
- **Tried and rejected on validation:** empirical-Bayes shrinkage of each
  window mean toward neutral by n/(n + σ²_w/σ²_b) (better in 3/10 splits);
  the full 768-dimension embedding (worse on test); all-MiniLM-L6-v2 as the
  encoder (worse in 7/10 splits).
- **None of this survives the forward holdout.** In 2026 the same model is
  worse than HAR across the universe and the headline-count pattern reverses,
  so these regularities describe the random splits, not the signal.

## Forward holdout — 2026, never used for any decision

The random splits reuse the same 2021–2026 history, and their test months had
been read several times during the work above. A single forward-in-time split
is the one clean check available: train on everything before 2025, validation
on 2025, and the test period is **2026-01-01 to 2026-11-16**, predicted once.
The purge rule is the same as for the random splits (each sample's whole span
lies in one segment, giving at least `WINDOW` sessions of gap at each
boundary):

```
holdout: train 1,080,382   val 262,770   test 228,893 samples
```

Universe (228,893 test samples) and TSLA (216 test days):

| predictor | universe QLIKE | vs HAR | TSLA QLIKE | vs HAR |
|---|---|---|---|---|
| persistence | 0.413 | −16.2% | 0.212 | −13.2% |
| HAR | 0.356 | 0 | 0.187 | 0 |
| intensity | 0.355 | +0.07% | 0.188 | −0.34% |
| shuffled | 0.355 | +0.03% | 0.188 | −0.46% |
| text | 0.358 | **−0.75%** (t = +1.41) | 0.172 | **+8.32%** (t = −4.57) |

**The universe gain does not replicate.** The random splits said +
0.96%; the untouched 2026 period says −0.75%, and text beats HAR in only
2 of its 11 months. Worse, the pattern reverses: in 2026 the text model loses
most where headlines are dense (8–15 headlines −2.0%, 16+ −1.1%), while
0–1-headline windows are neutral — the opposite of the random-split pattern.
So the +0.96% was an artefact of months that had been seen during
development, and **headline content is not established to improve universe
volatility forecasts**. The model still forecasts well: HAR beats persistence
by 36% with t = −7.3, and the per-stock level is what makes it competitive.

**TSLA's +8.32% is not evidence of an effect either.** It is consistent
inside that year (better in 10 of 11 months, only 3% of the loss in the worst
1% of days) but it contradicts the random-split estimate for the same stock
(−0.56%, 3/10 splits), and 97% of TSLA's 2026 windows carry 16+ headlines, so
it is one dense-news year of one stock — a different regime from the rest of
the sample, not a replicated finding.

**Why it fails: the 2026 text inputs are out of distribution.** Window-mean
PC scores are standardized with train moments, so a drift shows up as a mean
shift. On the random splits the test means sit within 0.1–0.3 window-sd of
the train mean; in 2026 they are 2–3× further (PC6 at −0.61 window-sd), while
the scale is unchanged (sd ratio ≈ 1) and 99.4% of test symbols have train
samples. The archive's headline coverage grew about 35% over the sample, so
the 2026 headlines themselves are a different distribution from the ones the
text coefficients were estimated on. Measured, not corrected: any correction
fitted on 2026 would consume the holdout.

What this means for the project: the pipeline, the per-stock level and the
HAR/QLIKE evaluation stand, and the forward split is now part of the method.
The claim that headline text adds forecasting power does not stand.

## Quick start

```bash
uv sync
uv run pytest -q                    # includes a synthetic end-to-end run
uv run newsimpact fetch-bars        # optional Alpaca adapter (keys in .env)
uv run newsimpact fetch-news        # only if you have no longer archive
scripts/run.sh                      # embed → build → train → evaluate
STATE_DIR=state/holdout uv run newsimpact holdout   # one forward-in-time split
STATE_DIR=state/holdout uv run newsimpact evaluate  # score it (test used once)
```

Artifacts go to `STATE_DIR` (default `./state`): `dataset.pt`,
`runs/split<k>.{npz,json}` (test predictions per arm, and the selected k
and λ) and `eval.json`.

## Configuration

`.env.data` holds `SYMBOLS` (a comma list, or `@file` with one symbol per
line, e.g. `@data/input/symbols.txt`), `START` and the paths; `.env.train` holds
`WINDOW` (5), `SAMPLE_START` (2021-01-01), `BLOCK` (21), `VAL_FRACTION` /
`TEST_FRACTION` (0.15), `SPLITS` (10), `TARGET` (TSLA) and the holdout
boundaries `VAL_START` (2025-01-01) / `TEST_START` (2026-01-01). `.env` holds credentials, needed
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
