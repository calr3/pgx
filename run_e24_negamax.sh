#!/usr/bin/env bash
# E24 (whole-move gess_joint) and E23 (two-step) against the app's negamax at 2 s, and against
# each other (tdgauntlet examples/gess_e24_negamax.toml), all at 128 simulations.
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
E24=checkpoints/gess_joint_20261002072552/000028.ckpt
E23=checkpoints/gess_20261001235654/000434.ckpt
cd "$PGX"
start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
    > "$RESULTS/gess_e24_negamax.$1.log" 2>&1 &
  echo $!
}
P1=$(start e24 9431 "$E24")
P2=$(start e23 9432 "$E23")
(cd "$TDG" && exec python3 clients/gess_negamax/gess_negamax.py --port 9441 --workers 12 --name negamax2s) \
  > "$RESULTS/gess_e24_negamax.negamax.log" 2>&1 &
P3=$!
trap 'kill $P1 $P2 $P3 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:9431/health >/dev/null && curl -sf localhost:9432/health >/dev/null \
    && curl -sf localhost:9441/health >/dev/null && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e24_negamax.toml) > "$RESULTS/gess_e24_negamax.log" 2>&1
echo "$(date) tournament exited with $?"
