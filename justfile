# nexo — every command you need.  `just` lists them.
#
#   setup ─► fetch ─► embed ─► build ─► train ─► evaluate ─► forecast
#                              └──────────── all ─────────────────┘
#
# Config lives in .env.data / .env.train; any key can be overridden inline:
#   WINDOW=10 just build

set shell := ["bash", "-euo", "pipefail", "-c"]

nexo := "uv run nexo"

# list the commands
default:
    @just --list --unsorted

# install the locked environment
setup:
    uv sync

# lint and run the tests (synthetic end-to-end included)
check:
    uv run ruff check src tests
    uv run pytest -q

# download daily bars and headlines (optional; needs .env credentials)
fetch:
    {{nexo}} fetch-bars
    {{nexo}} fetch-news

# embed headlines, one shard per GPU (CPU if none); skipped if already done
embed:
    #!/usr/bin/env bash
    set -euo pipefail
    emb=$(uv run python -c "from nexo.env import load_env; print(load_env()['EMB_PREFIX'])")
    if [ -f "${emb}0.npz" ]; then echo "embeddings exist at ${emb}*.npz — skipping"; exit 0; fi
    n=$(uv run python -c "import torch; print(torch.cuda.device_count())")
    if [ "$n" -eq 0 ]; then {{nexo}} embed --device cpu; exit 0; fi
    for g in $(seq 0 $((n - 1))); do {{nexo}} embed --shard "$g" --shards "$n" --device "cuda:$g" & done
    wait

# samples, targets and inputs → state/dataset.pt (nothing fitted here)
build:
    {{nexo}} build

# fit every model on every walk-forward fold → state/runs/
train:
    {{nexo}} train

# test-year QLIKE, Diebold–Mariano and fold-level tests → state/eval.json
evaluate:
    {{nexo}} evaluate

# next-window vol per symbol from the latest close → state/forecast.csv
forecast:
    {{nexo}} forecast

# the whole lifecycle: embed → build → train → evaluate → forecast
all: embed build train evaluate forecast

# same protocol at another horizon, in its own state dir: just window 10
window W:
    WINDOW={{W}} DATASET=state/w{{W}}/dataset.pt STATE_DIR=state/w{{W}} {{nexo}} build
    WINDOW={{W}} DATASET=state/w{{W}}/dataset.pt STATE_DIR=state/w{{W}} {{nexo}} train
    WINDOW={{W}} DATASET=state/w{{W}}/dataset.pt STATE_DIR=state/w{{W}} {{nexo}} evaluate

# show the top of the latest forecast
show n="10":
    head -n $(({{n}} + 1)) state/forecast.csv | column -t -s,

# delete the dataset and fitted artifacts (input data and embeddings stay)
clean:
    rm -rf state/runs state/eval.json state/forecast.csv state/dataset.pt
