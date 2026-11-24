# newsimpact

Forecast how volatile a stock (default **TSLA**) will be over the **next
window** of trading sessions, from what is known **at today's close**: its
price history and the headlines written about it during the **previous
window**. The window slides one session at a time, so every trading day is a
sample.

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
property. QLIKE is convex, so the fit is exact (damped Newton, float64),
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

**Why linear.** One stock gives about 880 training days per split. Because
target windows overlap, that is only about 180 independent weeks. A neural
attention model memorized it completely (train QLIKE 0.02) and never beat
its HAR starting point on validation, and a shuffled-headline network
memorized just as well. The data supports a few coefficients, not a
network.

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
- **Repeated.** `SPLITS` = 20 independent random assignments. Results are
  mean ± sd across splits plus the number of splits each comparison wins, so
  no conclusion rests on one draw of test months.
- **Train-only statistics.** Feature scaling, embedding standardization,
  principal components, the shuffled-headline pool and every coefficient
  come from the split's train data. Validation picks k and λ; test is
  predicted once.
- **Diebold–Mariano tests on QLIKE** for every split. The Newey–West
  variance uses lag W−1, counted only within contiguous test blocks.

## Results — TSLA, next-5-session vol, 20 splits (~150 test days each)

| predictor | QLIKE | skill vs HAR | better than HAR in | Spearman |
|---|---|---|---|---|
| persistence | 0.323 ± 0.128 | −59.4% | 2/20 | 0.42 |
| HAR | 0.200 ± 0.046 | 0 | — | 0.41 |
| intensity | 0.195 ± 0.044 | +2.1% ± 5.8% | 12/20 | 0.43 |
| shuffled | 0.200 ± 0.042 | −1.0% ± 7.5% | — | 0.41 |
| text | 0.212 ± 0.063 | −6.7% ± 25.6% | 6/20 | 0.40 |

Text vs shuffled: text is better in only 5/20 splits (mean DM t = +0.38).

What this shows:
- **On TSLA alone, headline content does not improve next-week volatility
  forecasts out of sample.** On validation the text model beat HAR in 18/20
  splits; on test it does not. Choosing among 30 (k, λ) settings on about
  150 validation days fits noise, and the test set exposes it.
- **News intensity may help slightly** (+2%, 12/20 splits), but the effect
  is not significant.
- **HAR beats persistence robustly** (18/20). The evaluation separates real
  differences from noise.
- **Calibration** of the text model on pooled test: predicted 40.0 / 53.6 /
  76.5% against realized 42.7 / 52.6 / 73.3% for deciles 1 / 5 / 10.
- **Consistent with the 1,159-stock run** of the earlier design (380k test
  samples): there, content added about 1% beyond news volume and timing.
  An effect that small needs far more than one stock's 180 independent
  weeks to detect.

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

`.env.data` holds `SYMBOLS`, `START` and the paths; `.env.train` holds
`WINDOW` (5), `SAMPLE_START` (2021-01-01), `BLOCK` (21), `VAL_FRACTION` /
`TEST_FRACTION` (0.15) and `SPLITS` (20). `.env` holds credentials, needed
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

## Limits

- One stock gives about 180 independent weeks per split, which is too few to
  detect a content effect of the ~1% size seen across 1,159 stocks. Pooling
  stocks for training and scoring TSLA is the way to gain power.
- The Alpaca news API on this account serves only recent news (from
  2026-04). The 2021–2026 archive comes from earlier collection.
- Price lookbacks reach across split blocks. They contain no target of
  another split, but volatility persistence makes neighbouring blocks
  similar.
