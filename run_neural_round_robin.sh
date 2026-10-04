#!/usr/bin/env bash
# E25 (it 183, 120), E24, E23 and E22 head to head at 128 simulations
# (tdgauntlet examples/gess_neural_round_robin.toml).
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
cd "$PGX"
start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
    > "$RESULTS/gess_neural_round_robin.$1.log" 2>&1 &
  echo $!
}
P1=$(start e25_183 9431 checkpoints/gess_joint_20261002213102/000183.ckpt)
P2=$(start e25_120 9432 checkpoints/gess_joint_20261002213102/000120.ckpt)
P3=$(start e24 9433 checkpoints/gess_joint_20261002072552/000028.ckpt)
P4=$(start e23 9434 checkpoints/gess_20261001235654/000434.ckpt)
P5=$(start e22 9435 checkpoints/gess_20260930235206/000400.ckpt)
trap 'kill $P1 $P2 $P3 $P4 $P5 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  ok=1
  for p in 9431 9432 9433 9434 9435; do curl -sf localhost:$p/health >/dev/null || ok=0; done
  [ $ok = 1 ] && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_neural_round_robin.toml) > "$RESULTS/gess_neural_round_robin.log" 2>&1
echo "$(date) tournament exited with $?"
