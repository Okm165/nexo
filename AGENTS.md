# AGENTS.md — rules for every session on this repo

Read `README.md` first; it is the only documentation.

## The task (narrow on purpose)

Forecast the annualized realized volatility of the NEXT `WINDOW` sessions
for one symbol, from what is known at the close of day t: that symbol's
headlines from the last `WINDOW` sessions, its own and the market's log
realized vol over 5/22/66/132 sessions, and its relative news intensity. One sample per (symbol, day); the window slides
one session at a time. Nothing else is in scope.

## Rules (each one prevents a mistake this project has made)

- **Point in time.** A headline counts at the first close (20:00 UTC) at or
  after its publication. Never let a headline or price after the close of t
  into sample t. Tested in `tests/test_pipeline.py`.
- **Purged splits.** A sample is in a split only if its decision day and the
  end of its target window are both inside it. Statistics (embedding,
  feature and target normalisation, HAR fit, shuffled-control pool) come
  from train only.
- **Overlap.** Neighbouring samples share WINDOW−1 target sessions; train
  loss is optimistic by construction. Judge only on purged validation.
- **Selection.** Epoch, hyperparameters and input choices are decided on
  validation; test is predicted once from the best-validation weights.
  Never tune on test.
- **Controls.** Every claim about text compares `text` against `shuffled`
  (same count and ages, random train-period content) and `price` (no text),
  trained with identical seeds, and against the HAR and persistence
  benchmarks.
- **Statistics.** Overlapping target windows: compare forecasts with the
  Diebold–Mariano test (daily-averaged losses, Newey–West lag WINDOW−1).
  Multi-seed mean ± sd for every arm. No single-seed claims.
- **Config.** `.env` (credentials), `.env.data` (paths), `.env.train`
  (hyperparameters); OS environment wins. No absolute or host-specific
  paths in the repo. Every declared key is read by code (tested).
- **Scope.** No heuristic features, no extra heads, no new knobs without a
  measured reason. Prefer deleting code to adding it.
