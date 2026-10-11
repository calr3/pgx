#!/usr/bin/env bash
# After E31 and E32 (run_e31_e32.sh, PID $1): E31 v E32 v E28 from random 2-ply openings
# (deterministic), then from the starting position with root Gumbel noise (tdgauntlet
# examples/gess_e31_e32{,_start}.toml), then the opening analysis of E31 and of E32.
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
RESULTS=$TDG/results
cd "$PGX"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
final() {  # log -> its run's checkpoint at the last iteration (e31_e32.maxit)
  dir=$(grep -oE "Saving checkpoint: checkpoints/gess_joint_[0-9]+" "$1" | head -1 | sed 's/Saving checkpoint: //')
  printf '%s/%06d.ckpt' "$dir" "$(cat e31_e32.maxit)"
}
E31=$(final gess_e31_run.log)
E32=$(final gess_e32_run.log)
E28=checkpoints/gess_joint_20261008042104/000449.ckpt
echo "$(date) E31 $E31, E32 $E32"
[ -f "$E31" ] && [ -f "$E32" ] || { echo "$(date) a final checkpoint is missing"; exit 1; }
match() {  # config name, extra client flags
  local cfg=$1; shift
  local pids=()
  for spec in "e31 9431 $E31" "e32 9432 $E32" "e28 9433 $E28"; do
    set -- $spec "$@"
    (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 exec python -m tdg_jax \
      --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3" "${@:4}") \
      > "$RESULTS/$cfg.$1.log" 2>&1 &
    pids+=($!)
    shift 3
  done
  for _ in $(seq 1 180); do
    ok=1
    for p in 9431 9432 9433; do curl -sf localhost:$p/health >/dev/null || ok=0; done
    [ $ok = 1 ] && break
    sleep 10
  done
  echo "$(date) $cfg: clients ready"
  (cd "$TDG" && ./target/release/tdgauntlet run "examples/$cfg.toml") > "$RESULTS/$cfg.log" 2>&1
  echo "$(date) $cfg: tournament exited with $?"
  kill "${pids[@]}" 2>/dev/null
  sleep 15
}
match gess_e31_e32
match gess_e31_e32_start --gumbel-scale 1.0
for run in e31 e32; do
  ck=$E31; [ $run = e32 ] && ck=$E32
  echo "$(date) opening analysis $run"
  python -u examples/alphazero/analyse_opening.py "$ck" "$RESULTS/gess_opening_$run.json" \
    --root-sims 1000,4000,16000 --play-budget-min 25 --reply-budget-min 8 \
    > "$RESULTS/gess_opening_$run.log" 2>&1
  echo "$(date) opening analysis $run exited with $?"
done
