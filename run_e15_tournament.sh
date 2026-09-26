#!/usr/bin/env bash
# Wait for the E15 control run (gess_e15_run.pid) to exit, then play its final
# checkpoint, E14 and E13 round robin in tdgauntlet
# (examples/gess_e15_three_way.toml), all under Gumbel MCTS (128 sims) on the GPU.
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
TRAIN_PID=$(cat "$PGX/gess_e15_run.pid")
E15=$PGX/checkpoints/gess_20260926222338/000024.ckpt
E14=$PGX/checkpoints/gess_20260926190427/000024.ckpt
E13=$PGX/checkpoints/gess_20260916200542/000072.ckpt
RESULTS=$TDG/results
NAME=gess_e15_three_way

echo "$(date) waiting for training pid $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "$(date) training exited"
if [ ! -f "$E15" ]; then
  echo "$(date) $E15 missing: training did not finish, not starting the tournament"
  exit 1
fi
sleep 30  # let the GPU memory free up

start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$3") \
    > "$RESULTS/$NAME.$1.log" 2>&1 &
  echo $!
}
P1=$(start e15 9441 "$E15")
P2=$(start e14 9442 "$E14")
P3=$(start e13 9443 "$E13")
trap 'kill $P1 $P2 $P3 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:9441/health >/dev/null && curl -sf localhost:9442/health >/dev/null \
    && curl -sf localhost:9443/health >/dev/null && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/$NAME.toml) > "$RESULTS/$NAME.log" 2>&1
echo "$(date) tournament exited with $?"
