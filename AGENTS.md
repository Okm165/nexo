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

- **Point in time.** A headline counts at the first close (20:00 UTC) at or
  after its publication. Tested in `tests/test_pipeline.py`.
- **Walk-forward, not shuffled blocks.** Fold k of `FOLDS` validates on the
  period starting at date k, tests on the next one, trains on everything
  before date k; a sample's whole span (headline window + target window)
  lies in one part. Shuffled month blocks approved a per-stock level that
  lost 11% forward: never go back to them.
- **A choice spends its data.** Whatever a design decision used — even as
  validation — cannot be claimed as test evidence afterwards. The 2023–2025
  test periods overlap the design ablations; only 2026 (and future sessions)
  are clean for the shipped design, and the news feed doubled in April 2026,
  so 2026 text results are confounded by the instrument.
- **Fitting.** Every statistic (scaling, embedding standardisation, PCs,
  shuffled pool, coefficients) comes from the fold's train data;
  validation chooses k and λ; the test period is predicted once. Never tune
  on a test period. No per-symbol fitted terms.
- **Loss and metric.** QLIKE on variance. MSE on log vol can mis-rank
  forecasts under a noisy proxy. Diebold–Mariano per test period and pooled.
- **Controls.** Every claim about text compares `text` with `shuffled`
  (same tuning) and `intensity`, and every model with HAR and persistence.
- **Capacity.** One stock is ~180 independent weeks; a network memorized it.
  Fit pooled; report the universe, not one stock.
- **Config.** `.env` credentials, `.env.data` paths, `.env.train` design;
  OS environment wins; no absolute paths; every key is read (tested).
- **Scope.** Prefer deleting code to adding it.
