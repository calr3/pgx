#!/bin/sh
# E18 (symmetry augmentation on) then E19 (the control), one after the other on
# the single GPU. Same seed, same schedule; the flag is the only difference.
cd /home/charlie/calr3gh/pgx

common="env_id=epaminondas architecture=boardformer seed=0 selfplay_bf16=true \
num_simulations=32 playout_cap_prob=0.25 fast_num_simulations=8 continue_games=true \
selfplay_batch_size=1024 max_num_steps=384 training_batch_size=4096 \
num_updates_per_iter=64 replay_buffer_iters=2 max_pending_steps=512 \
learning_rate=1e-3 weight_decay=1e-4 warmup_steps=100 grad_clip_norm=1.0 \
lr_schedule=cosine eval_interval=12 max_num_iters=200"

echo "E18 starting $(date)" >> e18_e19_runner.log
python -u examples/alphazero/train.py $common symmetry_augmentation=true > e18_run.log 2>&1
echo "E18 exited $? at $(date)" >> e18_e19_runner.log

echo "E19 starting $(date)" >> e18_e19_runner.log
python -u examples/alphazero/train.py $common symmetry_augmentation=false > e19_run.log 2>&1
echo "E19 exited $? at $(date)" >> e18_e19_runner.log
