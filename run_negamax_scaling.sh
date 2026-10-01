#!/usr/bin/env bash
# 1. E22 at 128 / 512 / 2048 simulations vs. negamax at 2 s (examples/gess_e22_sims_v_negamax.toml).
# 2. Negamax referees E22's losses from the pilot (results/gess_negamax_pilot.json): which of its
#    moves were blunders, did it see them, and do the 512- and 2048-simulation players avoid them?
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
E22=checkpoints/gess_20260930235206/000400.ckpt
start() {  # name port sims
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 exec python -m tdg_jax \
    --port "$2" --sims "$3" --max-batch 32 --name "$1" --checkpoint "$PGX/$E22") \
    > "$RESULTS/gess_e22_sims_v_negamax.$1.log" 2>&1 &
  echo $!
}
P1=$(start e22_128 9431 128)
P2=$(start e22_512 9432 512)
P3=$(start e22_2048 9433 2048)
(cd "$TDG" && exec python3 clients/gess_negamax/gess_negamax.py --port 9441 --workers 12 --name negamax2s) \
  > "$RESULTS/gess_e22_sims_v_negamax.negamax.log" 2>&1 &
P4=$!
trap 'kill $P1 $P2 $P3 $P4 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:9431/health >/dev/null && curl -sf localhost:9432/health >/dev/null \
    && curl -sf localhost:9433/health >/dev/null && curl -sf localhost:9441/health >/dev/null && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e22_sims_v_negamax.toml) \
  > "$RESULTS/gess_e22_sims_v_negamax.log" 2>&1
echo "$(date) scaling match exited with $?"
# The negamax client's workers are busy no longer; the referee starts its own.
(cd "$TDG" && python3 clients/gess_negamax/blunders.py results/gess_negamax_pilot.json \
  --player e22 --opponent negamax2s --workers 12 \
  --recheck http://127.0.0.1:9432 --recheck http://127.0.0.1:9433 \
  --out results/gess_negamax_pilot.blunders.json) > "$RESULTS/gess_negamax_pilot.blunders.log" 2>&1
echo "$(date) blunder analysis exited with $?"
