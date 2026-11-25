# AGENTS.md — rules for every session on this repo

Read `README.md` first; it is the only documentation.

## The task (narrow on purpose)

Forecast the log annualized Yang–Zhang variance of the NEXT `WINDOW`
sessions per symbol — one pooled model over the universe, reported for the
universe (a single stock's ~200 test days cannot resolve a ~1% effect) — from
what is known at the close of day t:
its log-variance lookbacks (5/22/66/132), its news intensity, and its
headlines from the last `WINDOW` sessions. One sample per day; the window
slides one session at a time. Nothing else is in scope.

## Rules (each one prevents a mistake this project has made)

- **Point in time.** A headline counts at the first close (20:00 UTC) at or
  after its publication. Tested in `tests/test_pipeline.py`.
- **Splits.** Purged random month blocks, repeated (`SPLITS`). A sample's
  whole span (headline window + target window) lies in one split. Never
  split individual days at random: neighbours share WINDOW−1 target sessions.
- **Fitting.** Every statistic (scaling, embedding standardisation, PCs,
  shuffled pool, coefficients) comes from the split's train data;
  validation chooses k and λ; test is predicted once. Never tune on test.
- **Loss and metric.** QLIKE on variance. MSE on log vol can mis-rank
  forecasts under a noisy proxy.
- **Controls.** Every claim about text compares `text` with `shuffled`
  (same tuning) and `intensity`, and every model with HAR and persistence.
  Report mean ± sd across splits and how many splits each comparison wins.
- **Capacity.** One stock is ~180 independent weeks per split; a network
  memorized it and a TSLA-only linear model's val gains vanished on test.
  Fit pooled; report the universe, not one stock, as the result.
- **Config.** `.env` credentials, `.env.data` paths, `.env.train` design;
  OS environment wins; no absolute paths; every key is read (tested).
- **The holdout.** A full re-tune on the same history re-reads every test
  month; the 2026 holdout already showed such gains do not replicate. Its
  result stands: no headline-content effect is established. Only sessions
  after the last held-out date are clean evidence, and `newsimpact holdout`
  scores them once.
- **Scope.** Prefer deleting code to adding it.
