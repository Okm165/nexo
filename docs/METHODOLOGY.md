# Methodology

What nexo estimates, how, and why each choice was made, plus what the
evidence does and does not support. Every claim below is produced by the
code in `src/nexo/`, or recorded as a measured experiment.

```
 1 question   2 target   3 loss   4 inputs   5 model   6 evaluation   7 statistics
 8 results    9 threats  10 rejected   11 reproduce
```

## 1. The question

> At the close of day t, how volatile will stock s be over sessions
> t+1 … t+W?

```
            ┌─ headlines ──┐┌─────── target ────────┐
 ───────────[t-W+1  ...   t][t+1       ...      t+W]───► sessions
  lookbacks 1…132 end at t ▲ forecast
```

The answer is a number: annualized variance, read as vol %. One sample per
(symbol, day), and the window slides one session at a time. Neighbouring
samples share W−1 target sessions, so they are strongly dependent. Sections 6
and 7 exist because of that dependence.

## 2. Target: Yang–Zhang variance from daily bars

For the n sessions of a window, from open O, high H, low L, close C:

```
 o  = ln(Oᵢ/Cᵢ₋₁)                 overnight
 c  = ln(Cᵢ/Oᵢ)                   open-to-close
 rs = u(u−c) + d(d−c),  u = ln(Hᵢ/Oᵢ), d = ln(Lᵢ/Oᵢ)   (Rogers–Satchell)
 σ² = 252 · [ Var(o) + k·Var(c) + (1−k)·mean(rs) ],   k = 0.34 / (1.34 + (n+1)/(n−1))
```

**Why.** Close-to-close returns use one price a day. Range estimators also
use the high and low, so they extract more information from the same bars.
Yang–Zhang (2000) has the lowest variance in its class and stays unbiased
under drift and opening jumps. For one session, `252·(o² + rs)` is used.

**Why log.** Variance is right-skewed over orders of magnitude, while log
variance is close to symmetric. Linear models in logs are standard
(HAR-log).

**The limit.** With W daily bars this is a *noisy proxy* of the true
variance: at W=5 the worst 1% of samples carried about a third of the loss.
Longer windows average the noise out. This is why `WINDOW` mattered more than
any model change (§8).

## 3. Loss: QLIKE

```
 QLIKE(h, y) = e^(y−h) − (y−h) − 1       y = log proxy variance, h = log forecast
```

- **Robust to the noisy proxy.** Under an unbiased but noisy proxy, QLIKE
  and MSE on variance rank forecasts as the true variance would; MSE on log
  vol does not (Patton 2011). QLIKE also penalizes under-prediction of
  variance more, which is the costly error for risk.
- **The right location.** The first-order condition `E[e^(y−h)] = 1` gives
  `h = log E[σ² | x]`: the fit targets the conditional *variance*. OLS on
  logs targets `E[log σ²]` and needs a Jensen correction.
- **Exact fit.** QLIKE is convex in h, hence in the weights of a linear h.
  `model.fit_qlike` solves it by damped Newton in float64 to machine
  precision: no SGD, no seeds, no epochs. Results are reproducible bit for
  bit. A ridge penalty applies to the text coefficients only.

## 4. Inputs, all known at the close of t

| input | definition | why |
|---|---|---|
| log variance 1, 5, 22, 66, 132 | Yang–Zhang over the last n sessions | the HAR cascade (Corsi 2009): volatility is driven by traders at daily, weekly and monthly horizons. The 1-day term is part of standard HAR, and the 132-session term carries the stock's own level (§5) |
| leverage | `min(ln Cₜ/Cₜ₋ₙ, 0)`, n = 1, 5, 22 | volatility rises more after falls than after rises (Corsi & Renò 2012) |
| news intensity | `log(1+headlines on t) − log(1+mean per session over the last 132)` | attention relative to the stock's *own* norm, so a vendor covering more stocks over the years is not "more news" |
| text | frozen embeddings → principal components → two window means (below) | headline content |

**Text features**, per fold, every statistic from train data only:

```
 headline ─► frozen encoder ─► standardize each dim ─► top-k PCs of the ─► score
             (≤64 tokens,       (train headlines)       train correlation
              mean pool)
 window   ─► mean of scores over session t      ┐
          ─► mean over sessions t−W+1 … t−1     ┴─► minus same-day mean ─► ÷ train sd
                                                     over all symbols
```

- **Standardize before PCA.** Raw mean-pooled vectors share one dominant
  direction: two random headlines had cosine 0.91. Without standardizing,
  that direction would become the first component.
- **Two parts, separate weights.** News decays fast. The last session holds
  ~20% of window headlines but most of the signal: validation favoured the
  split in 4/4 folds.
- **Day-relative.** Subtracting the day's cross-symbol mean removes anything
  common to the whole feed that day. A feed that changes what it publishes
  (this one doubled its volume mid-sample) cannot move forecasts by itself.
- **Empty parts are 0**, the neutral value after centering. A window with no
  headline gets the intensity model's forecast.
- **k ∈ {16, 64, 256}, λ ∈ {1e−3, 1e−2, 1e−1}** chosen on validation,
  identically for `text` and its `shuffled` control.

## 5. Model: linear, pooled, no per-symbol terms

```
 h = w · [1, log var 1/5/22/66/132, leverage, intensity, text PCs]    one w for all stocks
```

- **Linear.** A year of one stock holds about 50 non-overlapping weeks, and
  only about 12 non-overlapping 22-session windows: far too few for a flexible
  model. An attention network trained on one stock memorized it (train QLIKE
  0.02) and never beat HAR on validation. Pooling 1,154 stocks gives 1.57
  million samples for at most 522 coefficients (10 price/news + 2 × 256 text).
- **No per-symbol level.** A level estimated from each stock's own training
  targets improved shuffled cross-validation by 5.4%, and lost 10.9% when
  tested forward in time. Under shuffled folds the level is fitted on months
  *around* the test month, so it interpolates. Forward, it is stale. The
  132-session lookback is a level known at the close, so the model uses it
  instead.

## 6. Evaluation: walk-forward, purged

```
              2021   2022   2023   2024   2025   2026
 fold 0       train  val    TEST
 fold 1       train  train  val    TEST
 fold 2       train  train  train  val    TEST
 fold 3       train  train  train  train  val    TEST
              ─────────────────────────────────────────►  `FOLDS` cut dates
```

- **Forward in time**, as the forecast is used. Shuffled k-fold is valid
  only for purely autoregressive models with uncorrelated errors (Bergmeir,
  Hyndman & Koo 2018). Slow per-stock state breaks that, and shuffled folds
  approved the per-symbol level above.
- **Purged.** A sample enters a part only if its whole span, from the first
  headline session to the last target session, lies inside it. No target
  session and no headline is shared between train, validation and test.
  This is tested in `tests/test_pipeline.py`.
- **Train-only statistics**: scaling, embedding moments, PCs, the shuffled
  headline pool, coefficients. Validation chooses k and λ. Each test period
  is predicted once.
- **Controls.** `persistence` (the naive bar) and `HAR` (the standard bar)
  anchor every comparison. `shuffled` replaces each headline with a random
  training headline, keeping the counts, and is tuned identically. Text
  must beat it, or the gain is flexibility, not content.
- **Selection rule, fixed before running.** A feature or design is kept
  only if it lowers validation QLIKE in ≥3 of 4 folds *and* on average.
  These ablations were run with a base model that still contained the
  market factor, which was dropped afterwards.

## 7. Statistics

**Diebold–Mariano per test period.** Loss differences are first averaged
over symbols per day, which absorbs cross-sectional correlation: all stocks
share one market. The daily series gets a Newey–West variance with Bartlett
weights and lag W−1, because consecutive targets overlap by W−1 sessions.
Autocovariances are taken only within a test period.

**Fold-level test.** A pooled daily t can be significant while the effect
lives in one year. The four test years are treated as independent draws, and
two tests are applied to their per-period DM t-statistics (`metrics.fold_meta`):

- a **t-test that their mean is zero**: heterogeneity across years enters its
  standard error;
- a **sign test**: distribution-free, but with 4 periods its p cannot go
  below 0.125.

Neither is decisive alone with n = 4. If the per-year statistics cluster
tightly, the t-test can be extreme even when no single year is significant;
HAR vs persistence is the case here.

**Multiplicity.** Five comparisons are reported. Two corrections apply:

- **Bonferroni:** α = 0.01.
- **Fixed sequence:** test the comparisons in their pre-set order (the ladder
  persistence → HAR → price → intensity → text, as coded in
  `evaluate.PAIRS`) at α = 0.05, and stop at the first failure.

## 8. Results (one run)

**Data of this run.**

| item | value |
|---|---|
| source | the bundled Alpaca adapter (`fetch.py`): daily bars, split-adjusted, consolidated (SIP) feed; headlines from the same provider's news API |
| universe | 1,159 US-listed symbols in `SYMBOLS`; 1,154 produce samples; 1,041 have valid inputs at the latest close (2026-11-23) |
| how the universe was chosen | consistent with "the most-covered names in the headline archive": 1,117 of the 1,159 are among its 1,159 most-tagged symbols, and every member has ≥465 headlines. Delisted names are included (77 symbols' bars end before June 2026) |
| samples | 1,569,899 (symbol, day) pairs, 2021-01-04 … 2026-10-22 |
| headlines | 860,207 headlines, 1,371,087 (headline, symbol) attachments |
| encoder | ModernBERT-base, frozen |

**Reading the numbers.**

- **QLIKE** is a mean loss, so lower is better. **Skill vs HAR** is `1 − QLIKE / QLIKE_HAR`: +2% means 2% less loss than HAR.
- **A test year** holds only the samples whose whole span lies inside that calendar year. With W=22, each year therefore runs from about 22 sessions after 1 January to about 22 sessions before 31 December (for example 2023-02-02 … 2023-11-28). 2026 ends on 2026-10-22, the last day whose 22-session target had fully happened by the data end.

**Horizon.** Mean validation skill vs HAR (validation years 2022–2025):

```
                    W=5      W=10     W=22
 price             +1.7%    +2.1%    +3.1%
 text              +3.6%    +3.9%   +10.6%
 text − shuffled   +1.9pp   +2.1pp   +7.6pp
```

W=22 wins every row. Note: the first comparison was mistakenly computed on
*test* QLIKE, including 2026, and was redone on validation, with the same
answer. 2026 has therefore been *seen*, though the choice does not depend on
it.

**Test QLIKE, skill vs HAR, W=22** (805 test days):

```
 year   days   persistence   price   intensity   shuffled   text
 2023    207      −225%      +2.1%     +1.9%       +0.2%    +3.2%
 2024    209       −24%      +2.0%     +2.1%       +1.8%    +6.4%
 2025    207       −18%      +3.6%     +3.5%       +3.3%    +8.2%
 2026    182       −17%      +1.2%     +1.2%       +1.1%    +2.3%
 all     805       −87%      +2.3%     +2.3%       +1.6%    +5.2%
```

**Tests** (DM t per year; fold-level t-test p; sign test p):

```
 comparison           2023   2024   2025   2026   fold t   p      sign p   verdict
 HAR > persistence    −1.6   −1.8   −1.6   −2.0   −17.8   <.001   .125    holds (sequence step 1)
 price > HAR          −3.9   −2.0   −2.7   −0.9    −3.7   .034    .125    holds (sequence step 2)
 intensity > price    +0.8   −1.0   +0.8   +0.5    +0.7   .548    .625    fails: sequence stops
 text > intensity     −0.4   −2.1   −1.1   −0.5    −2.5   .085    .125    not tested formally
 text > shuffled      −0.9   −2.3   −1.2   −0.5    −3.2   .049    .125    not tested formally
```

**What this supports.**

- **The forecast beats the naive bar every year**, and it is calibrated
  (predicted → realized vol by decile: 24→21, 30→26, 33→31, 37→35, 42→40,
  47→48, 54→56, 63→70, 79→90, 136→145). This is the claim to rely on.
  Persistence loses by a very different amount each year (−225% in 2023,
  −17% in 2026): its worst 1% of forecasts carry 83% of its 2023 loss,
  because QLIKE punishes under-prediction exponentially, and a calm month
  is often followed by a violent one.
- **Leverage and the daily term add ~2%**, positive in all four years. They
  pass the fixed sequence but not Bonferroni. Treat this as likely, not
  proven.
- **Headline content: not established.** Text is positive in all four
  years, but the sequence stops at intensity. Text vs shuffled has
  p = 0.049 uncorrected. Its design was chosen on validation years that
  overlap test years 2023–2025, and the feed changed in 2026. Validation
  gains (+10.6%) are about twice the test gains (+5.2%), which is the
  expected optimism of selecting k and λ on validation.
- **Intensity adds nothing** on test (1/4 years), even though it passed the
  validation rule.

The shipped forecast therefore uses the **price** model.

## 9. Threats to validity

| threat | effect | status |
|---|---|---|
| design choices spend data | ablations used validation years 2022–2025, which are test years of other folds | only sessions after the data end are fully clean |
| news feed not stationary | tagged headlines per month were ~20k until March 2026, then 32k (April) and 78k (May), with new templates such as "X Adjusts Price Target" | day-relative features mitigate it; text claims stay open |
| encoder not point-in-time | a pretrained encoder may know events after a headline's date (look-ahead documented for language models in finance) | text results are an upper bound |
| proxy noise | daily-bar target; heavy-tailed loss | W=22; intraday bars would be better |
| universe chosen with hindsight | membership matches news coverage counted over the whole 2021–2026 archive, test years included; delisted names stay in | biases *which* stocks are studied (heavily covered ones), not any forecast input; build `SYMBOLS` from data before your first test year to remove it |
| short sample | 5 labeled years, 4 test periods | low power; fold-level tests are coarse |
| many related tests | 5 comparisons × 5 periods | fixed-sequence and Bonferroni reported |

## 10. Rejected, with measurements

Two protocols were used over the project's life: first 10 shuffled
month-block splits (abandoned, see §5–6), then walk-forward folds. The
protocol behind each measurement is shown.

| idea | outcome | protocol |
|---|---|---|
| per-symbol level (fixed effects) | +5.4% on shuffled blocks, −10.9% forward: removed | both, W=5 |
| market volatility factor | better in 2/4 validation folds: removed | walk-forward, W=5 |
| news count over the whole window | 1/4 folds: replaced by the last session's | walk-forward, W=5 |
| single-symbol headlines only | text worse in 4/4 folds | walk-forward, W=5 |
| one equal-weight text mean over the window | lost to the recency split in 4/4 folds | walk-forward, W=5 |
| all 768 embedding dims | better on validation, worse on test (overfit) | shuffled, W=5 |
| sentence-similarity encoder (MiniLM) | lost on validation in 7/10 splits | shuffled, W=5 |
| empirical-Bayes shrinkage of window means | better in 3/10 splits | shuffled, W=5 |
| attention network, single stock | memorized; never beat HAR on validation | time split, W=5 |
| W = 5, 10 | lower validation skill than W = 22 on every row (§8) | walk-forward |

The shuffled-split results may not hold under walk-forward evaluation; they
were not rerun.

## 11. Reproduce

```bash
just all                        # embed → build → train → evaluate → forecast
just check                      # point in time, purge, target by hand, live rows, p-values
```

`eval.json` holds every number in §8. `build` stamps an md5 over every
stored array, and `evaluate` refuses runs fitted on a different dataset.

**References.** Corsi (2009) *J. Financial Econometrics*, HAR · Corsi &
Renò (2012) *JBES*, leverage in HAR · Yang & Zhang (2000) *J. Business*,
range estimator · Patton (2011) *J. Econometrics*, robust loss functions ·
Diebold & Mariano (1995) *JBES* · Newey & West (1987) *Econometrica* ·
Bergmeir, Hyndman & Koo (2018) *CSDA*, CV for time series · Rahimikia,
Zohren & Poon, *Realised volatility forecasting: machine learning via
financial word embedding*, arXiv 2108.00480 · Bollerslev, Hood, Huss &
Pedersen (2018) *RFS*, panel volatility models.
