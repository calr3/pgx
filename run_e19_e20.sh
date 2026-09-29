#!/usr/bin/env bash
# Gess E19/E20: a matched pair of ~5 h continuations of E18 (it 298), differing
# only in the observation: E19 reads v2's clock planes (obs_planes=6, stem widened
# with zero weights on resume), E20 does not (obs_planes=4). Each is a new run
# (resume_as_new_run) that restores E18's buffer and games, so E18's directory is
# left untouched. They run one after the other, then play each other in
# tdgauntlet (examples/gess_e19_v_e20.toml), both under Gumbel MCTS (128 sims).
set -u
PGX=$(cd "$(dirname "$0")" && pwd)
TDG=$(cd "$PGX/../tdgauntlet" && pwd)
E18=checkpoints/gess_20260926222338/000298.ckpt
RESULTS=$TDG/results
cd "$PGX"

train() {  # name obs_planes
  python3 -u examples/alphazero/train.py env_id=gess architecture=gessformer seed=1 \
    resume_from=$E18 resume_as_new_run=true max_num_iters=332 obs_planes="$2" \
    selfplay_bf16=true num_simulations=64 playout_cap_prob=0.25 \
    fast_num_simulations=8 symmetry_augmentation=true continue_games=true \
    selfplay_batch_size=1024 max_num_steps=256 training_batch_size=4096 train_micro_batches=2 \
    num_updates_per_iter=256 replay_buffer_iters=4 \
    learning_rate=0.0073815 lr_final_ratio=0.001355 weight_decay=1e-4 warmup_steps=200 grad_clip_norm=1.0 \
    lr_schedule=cosine save_data_state=true eval_interval=4 \
    mcts_eval_opponent=$E18 mcts_eval_interval_hours=1.0 \
    mcts_eval_games=128 mcts_eval_batch_size=128 mcts_eval_simulations=32 \
    > "gess_$1_run.log" 2>&1 < /dev/null &
  echo $! > "gess_$1_run.pid"
  echo "$(date) $1 started, pid $(cat "gess_$1_run.pid")"
  wait "$(cat "gess_$1_run.pid")"
  echo "$(date) $1 exited with $?"
}

final_ckpt() {  # name -> path of the run's iteration-332 checkpoint, or nothing
  grep -o "checkpoints/gess_[0-9]*/000332.ckpt" "gess_$1_run.log" | tail -1
}

# MATCH_ONLY=1 skips training and replays just the match (e.g. after a crash).
if [ -z "${MATCH_ONLY:-}" ]; then
  train e19 6
  sleep 30  # let the GPU memory free up
  train e20 4
fi
E19=$(final_ckpt e19)
E20=$(final_ckpt e20)
if [ -z "$E19" ] || [ -z "$E20" ]; then
  echo "$(date) a run did not finish (e19: '$E19', e20: '$E20'), not starting the tournament"
  exit 1
fi
sleep 30

start() {  # name port checkpoint
  (cd "$TDG/clients/jax" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 exec python -m tdg_jax \
    --port "$2" --sims 128 --max-batch 32 --name "$1" --checkpoint "$PGX/$3") \
    > "$RESULTS/gess_e19_v_e20.$1.log" 2>&1 &
  echo $!
}
P1=$(start e19 9431 "$E19")
P2=$(start e20 9432 "$E20")
trap 'kill $P1 $P2 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  curl -sf localhost:9431/health >/dev/null && curl -sf localhost:9432/health >/dev/null && break
  sleep 10
done
echo "$(date) clients ready"
(cd "$TDG" && ./target/release/tdgauntlet run examples/gess_e19_v_e20.toml) \
  > "$RESULTS/gess_e19_v_e20.log" 2>&1
echo "$(date) tournament exited with $?"
