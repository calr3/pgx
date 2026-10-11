#!/usr/bin/env bash
# E31 (opening fix: sampled openings and a larger opening value_scale) and E32 (the control),
# matched branches from E28 it 449, ~12 h each, one after the other; see GESS_EXPERIMENTS.md
# "E31/E32". Both read their last iteration from e31_e32.maxit (written here if absent), so a
# mid-run resize of E31 carries over to E32. Run detached from pgx's root:
#   nohup setsid ./run_e31_e32.sh > e31_e32.log 2>&1 &
set -u
cd "$(dirname "$0")"
[ -f e31_e32.maxit ] || echo 565 > e31_e32.maxit
E28=checkpoints/gess_joint_20261008042104/000449.ckpt
# The LR re-warms to 2e-4 at E28's last optimizer step (114,523) and decays to 1e-5 at the
# last iteration; learning_rate/lr_final_ratio solved for that (here for 565 iterations).
BASE="env_id=gess_joint seed=1 reseed_on_resume=False obs_planes=0 num_channels=128 num_layers=6 resnet_v2=True num_heads=-1 num_attention_layers=1 cell_policy_head=False architecture=gessformer mlp_width=256 mlp_layers=3 mlp_onehot_bins=0 gf_stem_channels=64 gf_stem_blocks=2 gf_embed_dim=192 gf_num_heads=8 gf_ffn_mult=2.0 gf_gab=True gf_joint_head=0 rf_embed_dim=96 rf_num_layers=4 rf_num_heads=4 rf_ffn_mult=2.0 rf_stem_blocks=1 rf_remat=True rf_attn_bf16=True rf_symmetric=False bf_stem_channels=64 bf_stem_blocks=2 bf_embed_dim=192 bf_num_layers=6 bf_num_heads=8 bf_ffn_mult=2.0 bf_gab=True bf_remat=True selfplay_batch_size=1024 num_simulations=64 chance_samples=1 chance_exact=False qtransform=completed_by_mix_value aux_win_weight=0.0 selfplay_bf16=True playout_cap_prob=0.25 fast_num_simulations=8 max_num_steps=256 continue_games=True max_pending_steps=1024 save_data_state=True training_batch_size=4096 symmetry_augmentation=True replay_buffer_iters=4 num_updates_per_iter=256 weight_decay=0.0001 warmup_steps=200 grad_clip_norm=1.0 lr_schedule=cosine eval_interval=4 mcts_eval_interval_hours=1.0 mcts_eval_games=128 mcts_eval_batch_size=128 mcts_eval_simulations=32 gf_num_layers=8 train_bf16=True gf_remat=False train_micro_batches=4"
COMMON="$BASE resume_from=$E28 resume_as_new_run=True mcts_eval_opponent=$E28 \
  learning_rate=1.8459e-3 lr_final_ratio=0.0054174"
run() {  # name, extra overrides...
  local name=$1; shift
  local maxit; maxit=$(cat e31_e32.maxit)
  echo "$(date) $name start, max_num_iters=$maxit"
  # shellcheck disable=SC2086
  python -u examples/alphazero/train.py $COMMON max_num_iters=$maxit "$@" > gess_${name}_run.log 2>&1
  echo "$(date) $name exited with $?"
}
run e31 selfplay_sample_plies=6 opening_value_scale_plies=6 opening_value_scale=0.2
run e32
