#!/usr/bin/env bash
# E25 (it 183) at 128 / 512 / 2048 simulations vs. negamax v0 at 2 s
# (tdgauntlet examples/gess_e25_sims_v_negamax.toml).
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
E25=checkpoints/gess_joint_20261002213102/000183.ckpt
start() {  # name port sims
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 exec python -m tdg_jax \
    --port "$2" --sims "$3" --max-batch 32 --name "$1" --checkpoint "$PGX/$E25") \
    > "$RESULTS/gess_e25_sims_v_negamax.$1.log" 2>&1 &
  echo $!
}
P1=$(start e25_128 9431 128)
P2=$(start e25_512 9432 512)
P3=$(start e25_2048 9433 2048)
(cd "$TDG" && exec python3 clients/gess_negamax/gess_negamax.py --port 9441 --workers 12 --name v0) \
  > "$RESULTS/gess_e25_sims_v_negamax.v0.log" 2>&1 &
P4=$!
trap 'kill $P1 $P2 $P3 $P4 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  ok=1
  for p in 9431 9432 9433 9441; do curl -sf localhost:$p/health >/dev/null || ok=0; done
  [ $ok = 1 ] && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e25_sims_v_negamax.toml) \
  > "$RESULTS/gess_e25_sims_v_negamax.log" 2>&1
echo "$(date) scaling match exited with $?"
