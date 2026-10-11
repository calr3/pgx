#!/usr/bin/env bash
# E30 (24 h from scratch) against E28 at 128 sims (tdgauntlet examples/gess_e30_v_e28.toml).
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
cd "$PGX"
start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
    > "$RESULTS/gess_e30_v_e28.$1.log" 2>&1 &
  echo $!
}
P1=$(start e30 9431 checkpoints/gess_joint_20261009075914/000180.ckpt)
P2=$(start e28 9432 checkpoints/gess_joint_20261008042104/000449.ckpt)
trap 'kill $P1 $P2 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:9431/health >/dev/null && curl -sf localhost:9432/health >/dev/null && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e30_v_e28.toml) > "$RESULTS/gess_e30_v_e28.log" 2>&1
echo "$(date) tournament exited with $?"
