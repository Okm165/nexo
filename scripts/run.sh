#!/usr/bin/env bash
# End to end: embed (if needed, one shard per GPU) -> build -> train -> evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
NI="uv run newsimpact"
EMB=$(uv run python -c "from newsimpact.env import load_env; print(load_env()['EMB_PREFIX'])")
if [ ! -f "${EMB}0.npz" ]; then
  G=$(uv run python -c "import torch; print(max(torch.cuda.device_count(), 1))")
  for g in $(seq 0 $((G - 1))); do $NI embed --shard "$g" --shards "$G" --device "cuda:$g" & done
  wait
fi
$NI build
$NI train
$NI evaluate
$NI forecast
