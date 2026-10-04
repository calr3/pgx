#!/usr/bin/env bash
# E25 checkpoints 60/120/183 and E24 against negamax v0 (the app's .wasm) and v1 (native, refitted
# weights) at 2 s a move, all models at 128 simulations (tdgauntlet examples/gess_e25_negamax.toml).
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
E25=checkpoints/gess_joint_20261002213102
cd "$PGX"
start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
    > "$RESULTS/gess_e25_negamax.$1.log" 2>&1 &
  echo $!
}
P1=$(start e25_183 9431 "$E25/000183.ckpt")
P2=$(start e25_120 9432 "$E25/000120.ckpt")
P3=$(start e25_060 9433 "$E25/000060.ckpt")
P4=$(start e24 9434 checkpoints/gess_joint_20261002072552/000028.ckpt)
(cd "$TDG" && exec python3 clients/gess_negamax/gess_negamax.py --port 9441 --workers 10 --name v0) \
  > "$RESULTS/gess_e25_negamax.v0.log" 2>&1 &
P5=$!
(cd "$TDG" && exec ./target/release/tdg-gess-negamax --port 9451 --threads 10 --name v1 \
  --weights clients/gess_negamax_rs/weights/v1.json) > "$RESULTS/gess_e25_negamax.v1.log" 2>&1 &
P6=$!
trap 'kill $P1 $P2 $P3 $P4 $P5 $P6 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  ok=1
  for p in 9431 9432 9433 9434 9441 9451; do curl -sf localhost:$p/health >/dev/null || ok=0; done
  [ $ok = 1 ] && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e25_negamax.toml) > "$RESULTS/gess_e25_negamax.log" 2>&1
echo "$(date) tournament exited with $?"
