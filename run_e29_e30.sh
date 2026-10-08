#!/usr/bin/env bash
# E29 (one token per cell) then E30 (2x2 patches, the control), each from scratch for ~24 h on
# the same recipe; see GESS_EXPERIMENTS.md "E29/E30". Run detached from pgx's root:
#   nohup setsid ./run_e29_e30.sh > e29_e30.log 2>&1 &
set -u
cd "$(dirname "$0")"
BASE="env_id=gess_joint seed=1 reseed_on_resume=False obs_planes=0 num_channels=128 num_layers=6 resnet_v2=True num_heads=-1 num_attention_layers=1 cell_policy_head=False architecture=gessformer mlp_width=256 mlp_layers=3 mlp_onehot_bins=0 gf_stem_channels=64 gf_stem_blocks=2 gf_ffn_mult=2.0 gf_gab=True gf_joint_head=0 rf_embed_dim=96 rf_num_layers=4 rf_num_heads=4 rf_ffn_mult=2.0 rf_stem_blocks=1 rf_remat=True rf_attn_bf16=True rf_symmetric=False bf_stem_channels=64 bf_stem_blocks=2 bf_embed_dim=192 bf_num_layers=6 bf_num_heads=8 bf_ffn_mult=2.0 bf_gab=True bf_remat=True selfplay_batch_size=1024 num_simulations=64 chance_samples=1 chance_exact=False qtransform=completed_by_mix_value aux_win_weight=0.0 selfplay_bf16=True playout_cap_prob=0.25 fast_num_simulations=8 max_num_steps=256 continue_games=True selfplay_sample_plies=0 max_pending_steps=1024 training_batch_size=4096 symmetry_augmentation=True replay_buffer_iters=4 num_updates_per_iter=256 weight_decay=0.0001 grad_clip_norm=1.0 lr_schedule=cosine eval_interval=4 mcts_eval_interval_hours=1.0 mcts_eval_games=128 mcts_eval_batch_size=128"
COMMON="$BASE learning_rate=5e-4 warmup_steps=500 lr_final_ratio=0.02 train_bf16=True \
  train_micro_batches=4 save_data_state=True \
  mcts_eval_opponent=checkpoints/gess_joint_e23/000000.ckpt mcts_eval_simulations=32"
# E29's first attempt (8 Oct) crashed in self-play before the train.py fix for cuDNN attention
# under pmap, and E30 ran first; E29 then followed E30 (run_e29_after.sh). The order does not
# matter for an equal-time pair.
run_e29() {
  echo "$(date) E29 start"
  # shellcheck disable=SC2086
  python -u examples/alphazero/train.py $COMMON max_num_iters=125 gf_remat=True \
    gf_patch=1 gf_attn_bias=rel gf_fused_attention=True gf_num_heads=4 gf_embed_dim=128 gf_num_layers=6 \
    > gess_e29_run.log 2>&1
  echo "$(date) E29 exited with $?"
}
run_e30() {
  echo "$(date) E30 start"
  # shellcheck disable=SC2086
  python -u examples/alphazero/train.py $COMMON max_num_iters=180 gf_remat=False \
    gf_num_heads=8 gf_embed_dim=192 gf_num_layers=8 \
    > gess_e30_run.log 2>&1
  echo "$(date) E30 exited with $?"
}
if [ "${1:-both}" = "e29" ]; then
  run_e29
else
  run_e29
  run_e30
fi
