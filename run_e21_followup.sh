#!/usr/bin/env bash
# After Gess E21 (E19 + the joint policy term; PID in gess_e21_run.pid) finishes:
# 1. the long-slide capture benchmark on its final checkpoint, on the CPU
#    (gess_slide_diag.py; E19's result is gess_slide_e19.json), and
# 2. a 400-game match against E19 in tdgauntlet (examples/gess_e21_v_e19.toml),
#    both under Gumbel MCTS (128 sims) on the GPU.
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
E19=checkpoints/gess_20260929163401/000332.ckpt
RESULTS=$TDG/results
cd "$PGX"

TRAIN_PID=$(cat gess_e21_run.pid)
echo "$(date) waiting for training pid $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "$(date) training exited"
E21=$(grep -o "checkpoints/gess_[0-9]*/000366.ckpt" gess_e21_run.log | tail -1)
if [ -z "$E21" ]; then
  echo "$(date) E21 did not finish, stopping"
  exit 1
fi

JAX_PLATFORMS=cpu python -u examples/alphazero/gess_slide_diag.py \
  "$RESULTS/gess_e18_v_e17.json" "$E21" 2000 64 gess_slide_e21.json > gess_slide_e21.log 2>&1
echo "$(date) benchmark exited with $?"
for f in gess_slide_e19.json gess_slide_e21.json; do
  echo "== $f"; python examples/alphazero/gess_slide_report.py "$f"
done

start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
    > "$RESULTS/gess_e21_v_e19.$1.log" 2>&1 &
  echo $!
}
P1=$(start e21 9431 "$E21")
P2=$(start e19 9432 "$E19")
trap 'kill $P1 $P2 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:9431/health >/dev/null && curl -sf localhost:9432/health >/dev/null && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e21_v_e19.toml) \
  > "$RESULTS/gess_e21_v_e19.log" 2>&1
echo "$(date) tournament exited with $?"
