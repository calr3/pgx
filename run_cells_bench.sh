#!/usr/bin/env bash
# Cost of one-token-per-cell GessFormers (gf_patch=1) against today's patched network, on the GPU:
# forward pass, env step and training update (profile_gess_parts.py), then self-play steps
# (bench_gess_selfplay.py), all with bfloat16 training. See GESS_PLAN.md, "no 2x2 patching".
set -u
cd "$(dirname "$0")"
# Positions: boards/colors of real games (an .npz; see bench_gess_selfplay.py).
POS=${POS:-$HOME/.cache/gess_positions.npz}
# E28's settings, from the first line of its log, less what each variant sets itself.
ARGS="$(head -1 gess_e28_run.log | tr ' ' '\n' | sed "s/'//g" | grep "=" \
  | grep -v "^resume_from=\|^max_num_iters=\|^learning_rate=\|^lr_final_ratio=\|^mcts_eval_opponent=\|^wandb_run_id=\|^init_from=\|^value_scale_by_stage=\|^sim_schedule=\|^gf_remat=\|^train_micro_batches=\|^resume_as_new_run=\|^gf_patch=\|^gf_attn_bias=\|^gf_line_attention=\|^gf_fused_attention=\|^gf_embed_dim=" \
  | tr '\n' ' ') resume_as_new_run=False"
CELLS="gf_patch=1 gf_attn_bias=rel gf_fused_attention=True"
run() {  # label, overrides...
  local label=$1; shift
  echo "=== $label"
  for mb in 4 8; do
    out=$(timeout 1800 python -u examples/alphazero/profile_gess_parts.py init $ARGS "$@" train_micro_batches=$mb 2>&1)
    echo "$out" | grep -E "ms$|min$"
    echo "$out" | grep -q "RESOURCE_EXHAUSTED" && { echo "(out of memory at $mb micro-batches)"; continue; }
    break
  done
  timeout 1800 python -u examples/alphazero/bench_gess_selfplay.py new $POS init $ARGS "$@" max_num_steps=16 playout_cap_prob=1.0 2>&1     | grep -E "a step at|RESOURCE_EXHAUSTED: Out"
}
run "patched (E26/E28), remat off" gf_remat=False
run "cells, full attention, width 192" $CELLS gf_remat=True
run "cells, line attention, width 192" $CELLS gf_line_attention=True gf_remat=True
run "cells, full attention, width 128" $CELLS gf_embed_dim=128 gf_remat=True
run "cells, line attention, width 128" $CELLS gf_line_attention=True gf_embed_dim=128 gf_remat=True
echo "$(date) cells benchmark done"
