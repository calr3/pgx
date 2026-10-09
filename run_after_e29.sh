#!/usr/bin/env bash
# After E29 (run_e29_after.sh, PID $1) has finished: E29 v E30 head to head and each against
# negamax v0 at 2 s (tdgauntlet examples/gess_e29_e30.toml, 128 sims), then the one-hour
# opening analysis with E28 (analyse_opening.py -> results/gess_opening_e28.json).
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
cd "$PGX"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
final() {  # log, iteration -> that iteration's checkpoint
  dir=$(grep -oE "Saving checkpoint: checkpoints/gess_joint_[0-9]+" "$1" | head -1 | sed 's/Saving checkpoint: //')
  printf '%s/%06d.ckpt' "$dir" "$2"
}
E29=$(final gess_e29_run.log 125)
E30=$(final gess_e30_run.log 180)
echo "$(date) E29 $E29, E30 $E30"
if [ -f "$E29" ] && [ -f "$E30" ]; then
  start() {  # name port checkpoint
    (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec python -m tdg_jax \
      --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
      > "$RESULTS/gess_e29_e30.$1.log" 2>&1 &
    echo $!
  }
  P1=$(start e29 9431 "$E29")
  P2=$(start e30 9432 "$E30")
  (cd "$TDG" && exec python3 clients/gess_negamax/gess_negamax.py --port 9441 --workers 12 --name v0) \
    > "$RESULTS/gess_e29_e30.v0.log" 2>&1 &
  P3=$!
  for _ in $(seq 1 180); do
    ok=1
    for p in 9431 9432 9441; do curl -sf localhost:$p/health >/dev/null || ok=0; done
    [ $ok = 1 ] && break
    sleep 10
  done
  echo "$(date) clients ready"
  (cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e29_e30.toml) > "$RESULTS/gess_e29_e30.log" 2>&1
  echo "$(date) tournament exited with $?"
  kill $P1 $P2 $P3 2>/dev/null
  sleep 10
else
  echo "$(date) a final checkpoint is missing; matches skipped"
fi
echo "$(date) opening analysis start"
python -u examples/alphazero/analyse_opening.py checkpoints/gess_joint_20261008042104/000449.ckpt \
  "$RESULTS/gess_opening_e28.json" > "$RESULTS/gess_opening_e28.log" 2>&1
echo "$(date) opening analysis exited with $?"
