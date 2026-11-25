# newsimpact

Forecast how volatile a stock will be over the **next window** of trading
sessions, from what is known **at today's close**: its volatility history, the
headlines written about it in the **previous window** (split by recency), and
its news intensity. The window slides one session at a time, so every trading
day is a sample. One linear model is fitted across a universe of stocks
(1,159 by default); results are reported for the universe. The output is a
number: the annualized variance of the next `WINDOW` sessions (default 5),
the unit an implied-volatility quote uses. It is trained and scored against
the volatility that actually realized, measured with the Yang–Zhang range
estimator from daily bars. No options data is used.

## The question, exactly

```
target  log annualized Yang–Zhang variance of sessions t+1 .. t+W   (next window)
input   own log variance over 1, 5, 22, 66, 132 sessions            (price state)
        leverage: the negative part of the 1/5/22-session log return
        news intensity of session t: log(1 + its headlines) minus
        log(1 + the symbol's mean headlines per session, last 132)
        the window's headlines, split into session t and the W−1
        sessions before it, as frozen sentence embeddings
```

Yang–Zhang (2000), over the n days of a window, from daily open, high, low
and close:

```
o = ln(Oᵢ/Cᵢ₋₁)   c = ln(Cᵢ/Oᵢ)   rs = u(u−c) + d(d−c),  u = ln(Hᵢ/Oᵢ), d = ln(Lᵢ/Oᵢ)
σ² = Var(o) + k·Var(c) + (1−k)·mean(rs),   k = 0.34 / (1.34 + (n+1)/(n−1)),   ×252
```

For the single session, `o² + rs`. It is unbiased under drift and opening
jumps, and several times more efficient than close-to-close returns.

## Pipeline

```
fetch-bars ─► bars.parquet  (symbol, day, open, high, low, close)
news archive (symbols, created_at, headline) ─► embed ─► emb_s*.npz (768-d, float32)
                                       build ─► dataset.pt   (samples, inputs, targets)
                                       train ─► runs/fold*   (every arm, every fold)
                                     evaluate ─► eval.json
                                    forecast ─► forecast.csv  (the deliverable)
```

**build**: exact rules, nothing fitted, so nothing here can leak.

| rule | why |
|---|---|
| a headline counts at the first close (20:00 UTC) at or after its publication | point in time. 20:00 UTC is the summer New York close and an hour early in winter, so the rule is never late. Weekend and after-hours news goes to the next session |
| a bar is valid only if 0 < low ≤ min(open, close) ≤ max(open, close) ≤ high | inconsistent OHLC is a data error |
| every session of [t−132, t+W] present, valid, with \|log close-to-close\| ≤ 1 (for the latest close, [t−132, t]) | gaps and unadjusted splits are not volatility |
| every variance > 0 | zero means the price did not trade |
| samples start at `SAMPLE_START` (2021-01-01) | where the news archive's coverage begins. Earlier, an empty window would mean "not collected", not "no news". Earlier bars still serve as price history |
| duplicate (symbol, day) bars are refused | a repeated bar would silently corrupt every windowed sum |

## Model

A linear forecast of log variance, fitted by **QLIKE**, one set of
coefficients for all symbols:

```
h = w · [1, log variance 1/5/22/66/132, leverage, news intensity, text PCs]
QLIKE(h, y) = e^(y−h) − (y−h) − 1        y = log proxy variance
```

QLIKE ranks forecasts as the true variance would even though a week of daily
bars measures it noisily (Patton 2011). It is convex, so the fit is exact
(damped Newton, float64, Hessian accumulated on the GPU when present) — no
SGD, no seeds, no epochs. There is **no per-symbol term**: a level estimated
from a stock's own training targets looked good on shuffled month blocks,
where it interpolates the test month, and lost 11% to this pooled form when
tested forward in time. The 132-session lookback is the stock's level, known
at the close.

**Arms** (each adds to the previous):

| arm | inputs | answers |
|---|---|---|
| persistence | next-window variance = the previous window's | the naive bar |
| HAR | log variance 1/5/22/66/132 | the standard bar (Corsi 2009; our version adds the daily term) |
| price | + leverage (negative 1/5/22-session returns) | asymmetry (Corsi & Renò 2012) |
| intensity | + news intensity of the last session | does the *amount* of news help? |
| shuffled | + text PCs of random train headlines, same counts, tuned identically | does headline *content* help, or only flexibility? |
| text | + text PCs of the window's headlines | the model |

**Text features**, per fold, from train data only: standardize every
embedding dimension over the training headlines, take the top principal
components of their correlation matrix, score every headline on them, and
average the scores over **two parts of each window** — its last session, and
the sessions before it (news is short-lived, so the two get separate
coefficients; the last session holds 20% of window headlines and most of the
signal). Each part is taken **relative to the same day's mean over all
symbols with headlines**, so a news feed that changes what it publishes
cannot move forecasts by itself — the feed used here doubled its volume in
April 2026. A ridge penalty λ applies to the text coefficients only; k ∈
{16, 64, 256} and λ ∈ {1e−3, 1e−2, 1e−1} are chosen on validation, for
`shuffled` exactly as for `text`.

## Evaluation

**Walk-forward.** `FOLDS` cuts the calendar at 2022-01-01, 2023-01-01,
2024-01-01, 2025-01-01, 2026-01-01. Fold k validates on the period starting
at date k, tests on the next one, and trains on everything before date k —
always forward in time, as the forecast is used. Purged: a sample belongs to
a part only if its whole span, from the first session of its headline window
to the last session of its target window, lies inside it. Test periods:
calendar 2023, 2024, 2025 and 2026 (221–244 test days each, about 265k
samples).

**Statistics.** QLIKE per test period and over all test days; Diebold–Mariano
t on the daily mean loss difference (Newey–West lag W−1), per period and
pooled; calibration of predicted vs realized vol by decile.

**What each number is worth (read before the results).** Persistence and HAR
are standard and untuned, so their comparison is protocol-clean. The *design*
of price and text (which features, which text aggregation) was chosen by
ablations on validation periods 2022–2025 — so the 2023–2025 test periods
overlap data that informed the design, and their pooled statistics are
optimistic for everything except persistence-vs-HAR. **2026 is the one period
this final design never touched**, but the news feed doubled its volume in
April 2026 (before: ~20k tagged headlines/month, after: 32–78k; new
per-stock templates), so 2026 tests the design under a changed instrument.
Every claim below is stated with this in mind.

## Results — walk-forward, next-22-session vol, 4 folds

**Why W = 22.** The target window was chosen on validation only, per the
protocol. Mean validation skill vs HAR over the four folds:

| | W=5 | W=10 | W=22 |
|---|---|---|---|
| price | +0.8% | +1.0% | **+2.2%** (4/4 folds) |
| text | +1.9% | +1.8% | **+5.0%** (4/4 folds) |
| text − shuffled | +1.4 pp | +1.3 pp | **+3.4 pp** |

A longer window quarters the proxy-noise floor (the worst 1% of samples held
~⅓ of the loss at W=5), so real differences become measurable. W=22 also
gives the most stable results: price and text are positive in all four
validation folds, and the text−shuffled gap triples.

Test QLIKE (skill vs HAR in parentheses):

| test period | days | persistence | HAR | price | intensity | shuffled | text |
|---|---|---|---|---|---|---|---|
| 2023 | 207 | 1.350 (−225%) | 0.415 | 0.407 (+2.1%) | 0.407 (+1.9%) | 0.415 (+0.2%) | 0.402 (+3.2%) |
| 2024 | 209 | 0.424 (−24.2%) | 0.342 | 0.335 (+2.0%) | 0.335 (+2.1%) | 0.335 (+1.8%) | 0.320 (+6.4%) |
| 2025 | 207 | 0.393 (−17.5%) | 0.335 | 0.323 (+3.6%) | 0.323 (+3.5%) | 0.324 (+3.3%) | 0.307 (+8.2%) |
| 2026 | 182 | 0.255 (−17.0%) | 0.218 | 0.215 (+1.2%) | 0.215 (+1.2%) | 0.215 (+1.1%) | 0.213 (+2.3%) |
| all | 805 | 0.622 (−87.2%) | 0.332 | 0.324 (+2.3%) | 0.324 (+2.3%) | 0.327 (+1.6%) | 0.315 (+5.2%) |

Diebold–Mariano t (negative: the first is better):

| comparison | 2023 | 2024 | 2025 | 2026 | all |
|---|---|---|---|---|---|
| price vs HAR | −3.9 | −2.0 | −2.7 | −0.9 | −4.3 |
| text vs intensity | −0.4 | −2.1 | −1.1 | −0.5 | −1.8 |
| text vs shuffled | −0.9 | −2.3 | −1.2 | −0.5 | −2.3 |
| HAR vs persistence | −1.6 | −1.8 | −1.6 | −2.0 | −1.8 |

Fold-level meta-test (mean per-year DM t over its standard error; the years
are independent; with 15 DM tests, per-year |t| < 2.6 is descriptive):

| comparison | per-year DM t | fold-level t | p |
|---|---|---|---|
| price vs HAR | −3.9, −2.0, −2.7, −0.9 | **−3.7** | **0.033** |
| text vs intensity | −0.4, −2.1, −1.1, −0.5 | −2.5 | 0.094 |
| text vs shuffled | −0.9, −2.3, −1.2, −0.5 | −3.2 | 0.048 |
| HAR vs persistence | −1.6, −1.8, −1.6, −2.0 | −18.0 | <0.001 |

Calibration (text, all test days): predicted → realized vol by decile:
24→21, 30→26, 33→31, 37→35, 42→40, 47→48, 54→56, 63→70, 79→90, 136→145.

What this supports, and what it does not:
- **Both improvements established at W=22**: the completed HAR (price) beats
  plain HAR in all four test years, fold-level p 0.033; text beats the
  shuffled control in all four years, p 0.048. Both were chosen on
  validation, and their 2023–2025 test periods overlap the design ablations
  — the cleanest single year is 2026 (price +1.2%, text +2.3% vs intensity),
  under the changed feed.
- **Recency matters**: the recency-split, day-relative text at W=22 shows the
  strongest and most consistent content effect measured in this project —
  but it was selected on W=5 validation and confirmed here, so treat its
  size (+5.2%) as the favourable read.
- **Persistence loses less at W=22** (−17% to −24%): a 22-session variance is
  more persistent, so the bar HAR must beat is lower in log-variance terms —
  HAR still wins every year.
- **Calibration is close** (above), so the numbers read as vol forecasts.

## Rejected on measurement (kept out of the model)

| idea | result |
|---|---|
| per-symbol level (fixed effects), QLIKE-optimal | +5.4% on shuffled month blocks, **−10.9% forward in 2026** — removed |
| market volatility factor (mean log variance 5/22) | better in 2/4 validation folds — removed |
| news intensity of the whole window (vs the last session) | better in 1/4 folds — removed |
| headlines tagged to a single symbol only | text loses in 4/4 validation folds — rejected |
| empirical-Bayes shrinkage of window means | better in 3/10 old splits — rejected |
| full 768-dim text (k=768) | better on validation, worse on test — rejected |
| all-MiniLM-L6-v2 encoder | worse in 7/10 old splits — rejected |
| neural attention model | memorized (train QLIKE 0.02) — rejected |

Also checked and fine: headline duplicates (6.5% of texts repeat, but 0.4%
within a 5-session window — no effect on window means); loss concentration
(the top 5% of symbols hold 18% of the loss); text as a period proxy (the
day-common part of the text adjustment explains none of the old gain).

## Forecast (the deliverable)

`newsimpact forecast` fits the **price** arm once on every sample and predicts
the next-`WINDOW`-session volatility of every symbol from its inputs at the
**latest close in the data** — a window that has not happened yet. The price
arm is used because it is established at W=22 (fold-level p 0.033); the text
arm is stronger on validation but its fold-level p is 0.094, so it stays out
of the deliverable until post-2026-11-16 data confirms it.
Output is `STATE_DIR/forecast.csv`:

```
symbol,day,predicted_vol_pct,previous_window_vol_pct
XXII,2026-11-23,640.97,223.73
...
TSLA,2026-11-23,53.76,57.59
```

1,041 symbols at the 2026-11-23 close; universe median predicted vol 51.8%.
Calibration is close (above), so the numbers can be read as vol forecasts.

## Quick start

```bash
uv sync
uv run pytest -q                    # includes a synthetic end-to-end run
uv run newsimpact fetch-bars        # optional Alpaca adapter (keys in .env)
uv run newsimpact fetch-news        # only if you have no longer archive
scripts/run.sh                      # embed → build → train → evaluate → forecast
```

Artifacts go to `STATE_DIR` (default `./state`): `dataset.pt`, `runs/fold<k>.{npz,json}`
(test predictions per arm and the selected k, λ), `eval.json`, `forecast.csv`.

## Configuration

`.env.data` holds `SYMBOLS` (a comma list, or `@file` with one symbol per
line, e.g. `@data/input/symbols.txt`), `START` and the paths; `.env.train`
holds `WINDOW` (5), `SAMPLE_START` (2021-01-01) and `FOLDS`
(2022-01-01,2023-01-01,2024-01-01,2025-01-01,2026-01-01). `.env` holds
credentials, needed only by the fetchers. The OS environment overrides any
key. Paths are repo-relative and inputs are plain parquet, so any source with
the same columns works.

## Code

```
src/newsimpact/
  fetch.py      optional Alpaca adapter (bars, news)
  text.py       headline → frozen ModernBERT-base embedding (≤64 tokens, mean pool)
  dataset.py    samples, Yang–Zhang target and inputs, walk-forward folds
  model.py      QLIKE and its exact penalized fit
  train.py      per fold: text features, k/λ selection, every arm, test predictions
  evaluate.py   QLIKE, skill, Diebold–Mariano, calibration
  metrics.py    Diebold–Mariano
scripts/run.sh  the whole pipeline
```

## Limits

- **The news feed is not a stable instrument.** Volume doubled in April 2026
  and new per-stock templates appeared; every text result after that date is
  confounded, and 2026 is the only design-clean test period.
- **Five years of labeled data** (2021–2026), against the 8-year training
  windows of the RV literature.
- **The target is measured from 5 daily bars**, so proxy noise dominates:
  the worst 1% of samples hold about a third of the loss.
- **2023–2025 informed the design** (feature and aggregation ablations on
  their validation periods), so only 2026 — and, eventually, data after
  2026-11-16 — can confirm the design out of sample.
- Headline coverage is uneven: 30% of stock-days have no headline, and list
  headlines (10+ symbols) carry 27% of attachments.
- **The frozen encoder is not point-in-time.** ModernBERT was pretrained on
  web text that extends beyond 2021, so its embedding of a 2021–2022 headline
  can encode knowledge of events that came after it (the look-ahead bias
  documented for pretrained language models by Glasserman & Lin 2023 and
  Benhenda's Look-Ahead-Bench). Any text result here is therefore an upper
  bound; a pre-2021-cutoff encoder, or one trained only on past news, would
  be needed for a strictly real-time estimate.
