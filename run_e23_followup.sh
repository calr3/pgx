#!/usr/bin/env bash
# After Gess E23 (E22 + the auxiliary immediate-win target; PID in gess_e23_run.pid) finishes:
# E23 and E22 against the app's negamax at 2 s, and against each other
# (tdgauntlet examples/gess_e23_negamax.toml), all at 128 simulations.
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
E22=checkpoints/gess_20260930235206/000400.ckpt
cd "$PGX"
TRAIN_PID=$(cat gess_e23_run.pid)
echo "$(date) waiting for training pid $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "$(date) training exited"
E23=$(grep -o "checkpoints/gess_[0-9]*/000434.ckpt" gess_e23_run.log | tail -1)
if [ -z "$E23" ]; then echo "$(date) E23 did not finish, stopping"; exit 1; fi
sleep 30
start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
    > "$RESULTS/gess_e23_negamax.$1.log" 2>&1 &
  echo $!
}
P1=$(start e23 9431 "$E23")
P2=$(start e22 9432 "$E22")
(cd "$TDG" && exec python3 clients/gess_negamax/gess_negamax.py --port 9441 --workers 12 --name negamax2s) \
  > "$RESULTS/gess_e23_negamax.negamax.log" 2>&1 &
P3=$!
trap 'kill $P1 $P2 $P3 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:9431/health >/dev/null && curl -sf localhost:9432/health >/dev/null \
    && curl -sf localhost:9441/health >/dev/null && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e23_negamax.toml) > "$RESULTS/gess_e23_negamax.log" 2>&1
echo "$(date) tournament exited with $?"
