#!/usr/bin/env bash
# E26 (final) and E25 (it 183) against negamax v0 at 2 s and head to head, at 128 simulations
# (tdgauntlet examples/gess_e26_negamax.toml). Waits for E26's training to exit first.
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
E26=checkpoints/gess_joint_20261005001759/000279.ckpt
E25=checkpoints/gess_joint_20261002213102/000183.ckpt
cd "$PGX"
while pgrep -f "examples/alphazero/train.py" >/dev/null; do sleep 30; done
[ -f "$E26" ] || { echo "$(date) no $E26"; exit 1; }
echo "$(date) training finished"
start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
    > "$RESULTS/gess_e26_negamax.$1.log" 2>&1 &
  echo $!
}
P1=$(start e26 9431 "$E26")
P2=$(start e25 9432 "$E25")
(cd "$TDG" && exec python3 clients/gess_negamax/gess_negamax.py --port 9441 --workers 12 --name v0) \
  > "$RESULTS/gess_e26_negamax.v0.log" 2>&1 &
P3=$!
trap 'kill $P1 $P2 $P3 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  ok=1
  for p in 9431 9432 9441; do curl -sf localhost:$p/health >/dev/null || ok=0; done
  [ $ok = 1 ] && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e26_negamax.toml) > "$RESULTS/gess_e26_negamax.log" 2>&1
echo "$(date) tournament exited with $?"
