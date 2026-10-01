#!/usr/bin/env bash
# Start E22, E18 and the negamax client, then play examples/gess_negamax_pilot.toml in tdgauntlet.
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
    > "$RESULTS/gess_negamax_pilot.$1.log" 2>&1 &
  echo $!
}
P1=$(start e22 9431 checkpoints/gess_20260930235206/000400.ckpt)
P2=$(start e18 9432 checkpoints/gess_20260926222338/000298.ckpt)
(cd "$TDG" && exec python3 clients/gess_negamax/gess_negamax.py --port 9441 --workers 12 --name negamax2s) \
  > "$RESULTS/gess_negamax_pilot.negamax.log" 2>&1 &
P3=$!
trap 'kill $P1 $P2 $P3 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:9431/health >/dev/null && curl -sf localhost:9432/health >/dev/null \
    && curl -sf localhost:9441/health >/dev/null && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_negamax_pilot.toml) \
  > "$RESULTS/gess_negamax_pilot.log" 2>&1
echo "$(date) tournament exited with $?"
