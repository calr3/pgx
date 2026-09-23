#!/bin/sh
# After E20 finishes: preserve its checkpoints, run the four-way ladder, then
# launch E21 — the control arm that isolates the simulation count.
#
# E20 and E21 both resume from E18's 000200.ckpt, and train.py puts a resumed
# run's checkpoints in the SAME directory as the checkpoint it resumed from
# (train.py:483). Their iteration numbers therefore collide, so E20's have to
# be moved out of that directory before E21 starts, or E21 overwrites them.
set -e
cd /home/charlie/calr3gh/pgx
log() { echo "$(date '+%H:%M:%S') $*" >> e20_chain.log; }

while kill -0 1729 2>/dev/null; do sleep 60; done
log "E20 finished"

RUN=checkpoints/epaminondas_20260922180057
SAVE=checkpoints/epaminondas_e20_64sims
mkdir -p "$SAVE"
for f in "$RUN"/0*.ckpt; do
  n=$(basename "$f" .ckpt)
  if [ "$(echo "$n" | sed 's/^0*//')" -gt 200 ] 2>/dev/null; then
    mv "$f" "$SAVE/"
  fi
done
log "moved E20 checkpoints to $SAVE: $(ls "$SAVE" | tr '\n' ' ')"

E20_CKPT=$(ls "$SAVE"/0*.ckpt | sort | tail -1)
log "E20 final = $E20_CKPT"

python -u examples/alphazero/elo_ladder.py env_id=epaminondas \
  models=e17=checkpoints/epaminondas_20260922033505/000252.ckpt,e19ctl=checkpoints/epaminondas_20260923005823/000200.ckpt,e18=$RUN/000200.ckpt,e20=$E20_CKPT \
  games_per_pair=256 batch_size=256 num_simulations=32 random_opening_plies=3 \
  max_num_steps=3000 results_file=elo_epaminondas_sims.json > e20_ladder.log 2>&1
log "ladder done"

# E21: the control. Same start, same wall clock as E20 (~6.9 h), 32 sims not 64.
python -u examples/alphazero/train.py env_id=epaminondas architecture=boardformer \
  seed=0 selfplay_bf16=true num_simulations=32 playout_cap_prob=0.25 \
  fast_num_simulations=8 continue_games=true symmetry_augmentation=true \
  selfplay_batch_size=1024 max_num_steps=384 training_batch_size=4096 \
  num_updates_per_iter=64 replay_buffer_iters=2 max_pending_steps=512 \
  learning_rate=1e-3 weight_decay=1e-4 warmup_steps=100 grad_clip_norm=1.0 \
  lr_schedule=cosine eval_interval=12 max_num_iters=405 \
  resume_from=$RUN/000200.ckpt > e21_run.log 2>&1
log "E21 finished"
