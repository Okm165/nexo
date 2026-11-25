#!/usr/bin/env bash
# End to end: embed (if needed) -> build -> train ARMS x SEEDS on every GPU -> evaluate.
#   SEEDS=10 PER_GPU=2 scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-10}
PER_GPU=${PER_GPU:-2}
ARMS=${ARMS:-"text shuffled price"}
NI="uv run newsimpact"
G=$(uv run python -c "import torch; print(max(torch.cuda.device_count(), 1))")
EMB=$(uv run python -c "from newsimpact.env import load_env; print(load_env()['EMB_PREFIX'])")
LOGS=$(uv run python -c "from newsimpact.env import load_env; print(load_env()['STATE_DIR'])")/logs
mkdir -p "$LOGS"

if [ ! -f "${EMB}0.npz" ]; then
  for g in $(seq 0 $((G - 1))); do
    $NI embed --shard "$g" --shards "$G" --device "cuda:$g" > "$LOGS/embed_$g.log" 2>&1 &
  done
  wait
fi
$NI build

jobs=()
for s in $(seq 0 $((SEEDS - 1))); do for a in $ARMS; do jobs+=("$a $s"); done; done
slots=$((G * PER_GPU))
for k in $(seq 0 $((slots - 1))); do
  (
    for ((j = k; j < ${#jobs[@]}; j += slots)); do
      set -- ${jobs[$j]}
      $NI train --arm "$1" --seed "$2" --device "cuda:$((k % G))" > "$LOGS/train_$1_s$2.log" 2>&1
    done
  ) &
done
wait
$NI evaluate
