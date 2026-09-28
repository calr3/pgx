#!/usr/bin/env bash
# Wait for the E17 continuation (PID in gess_e17_run.pid) to exit, then play its final
# checkpoint against E16 in tdgauntlet (examples/gess_e17_v_e16.toml), both
# models under Gumbel MCTS (128 sims) on the GPU.
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
TRAIN_PID=$(cat "$PGX/gess_e17_run.pid")
E17=$PGX/checkpoints/gess_20260926222338/000266.ckpt
E16=$PGX/checkpoints/gess_20260926222338/000184.ckpt
RESULTS=$TDG/results

echo "$(date) waiting for training pid $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "$(date) training exited"
if [ ! -f "$E17" ]; then
  echo "$(date) $E17 missing: training did not finish, not starting the tournament"
  exit 1
fi
sleep 30  # let the GPU memory free up

start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$3") \
    > "$RESULTS/gess_e17_v_e16.$1.log" 2>&1 &
  echo $!
}
P1=$(start e17 9431 "$E17")
P2=$(start e16 9432 "$E16")
trap 'kill $P1 $P2 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:9431/health >/dev/null && curl -sf localhost:9432/health >/dev/null && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e17_v_e16.toml) \
  > "$RESULTS/gess_e17_v_e16.log" 2>&1
echo "$(date) tournament exited with $?"
