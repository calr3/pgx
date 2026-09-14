# Gess AlphaZero: plan for the TPU run

Goal: have the strongest possible Gess model at the end of a ~3-day rented
Google Cloud TPU run. Local pilots (RTX 5070 Ti, 16 GB) choose the recipe;
the TPU run executes it.

Experiment details, results and conclusions: [`GESS_EXPERIMENTS.md`](GESS_EXPERIMENTS.md).

## Where we are

### Recipe so far

GessFormer (hybrid conv stem + transformer with Geometric Attention Bias) with:

- `continue_games=true` – games carry over between iterations, so every
  position gets a real value target
- `replay_buffer_iters=4`, ~2x sample reuse
- `lr_schedule=cosine`, AdamW (`weight_decay=1e-4`), warmup, `grad_clip_norm=1.0`
- `symmetry_augmentation=true` – random one of the 8 board symmetries per sample
- `selfplay_bf16=true` – bfloat16 self-play inference (~2x faster network)
- `save_data_state=true` on preemptible machines – exact resume

### Pilot ladder (`elo_ladder.py`, 128 games/pair, 32 sims, anchor = old ResNet baseline)

Pilots: 256 games x 128 steps per iteration, training batch 2048. Ratings are
relative to the pool and shift as models are added; compare gaps.

| Model | Train time | Elo |
|---|---|---|
| old ResNet baseline (`gess_v0`) | 14.6 h | 0 |
| **GessFormer + augmentation + bf16 self-play (it 96)** | 1.29 h | **-100** |
| GessFormer + augmentation + bf16 self-play (it 90) | 1.21 h | -127 |
| GessFormer + augmentation | 1.23 h | -248 |
| RayFormer (60 it) | 1.88 h | -313 |
| GessFormer | 1.25 h | -394 |
| RayFormer (40 it) | 1.22 h | -481 |
| RayFormer + augmentation, symmetry-tied | 1.93 h | -538 |
| RayFormer + augmentation | 1.82 h | -584 |
| ResNet pilot | 0.99 h | -737 |

Cached results: `elo_gess_pilots.json` (repo root). RayFormer is more
sample-efficient but ~50% slower per iteration and does not benefit from
augmentation at its current size; parked.

### Full-size measurements (GessFormer recipe, local GPU)

1024 games x 256 steps per iteration, training batch 4096, 128 updates/iteration.

| | |
|---|---|
| Iteration time | 10.6-10.9 min (~1.46M positions/hour) |
| Share of time in self-play | ~95% (training ~0.2 s/update) |
| Host RAM (training process) | 14 GB with a full 4-iteration buffer (6.6 GB) |
| `data_state.pkl` | 9.1 GB, saved in ~14 s |

Self-play throughput is the bottleneck, so steps 1 and 2 target it.

## Step 1: faster self-play

1. ~~**Profile**~~ (done) a full-size self-play move (batch 1024, GessFormer):
   one forward pass is 63 ms and a 32-simulation move 1.8 s, i.e. network
   inference is essentially all of it; env steps (1.4 ms) and MCTS tree
   operations are negligible. Cost scales linearly with `num_simulations`.
2. **bfloat16 inference** (`selfplay_bf16=true`; weights stay float32 for
   training and evaluation). Implemented: forward pass 1.95x faster (61 -> 31 ms),
   pilot-size iterations 1.6x faster (72 -> 45 s). On the trained GessFormer +
   augmentation checkpoint, 32-sim search picks the same move 96-97% of the
   time as float32 (policy-target TV distance 0.04). **Adopted**: the pilot
   (96 iterations, 1.29 h) at equal time (iteration 90, 1.21 h) is +121 Elo
   over the float32 GessFormer + augmentation pilot and won 101.5-26.5
   head-to-head; at iteration 96 it is the best pilot so far.
3. **Fewer simulations / playout cap randomization** (KataGo): most moves use a
   cheap search and are excluded from the policy loss; a fraction use the full
   search and provide policy targets. Try plain `num_simulations=16` first as
   the simplest variant (pilot running: bfloat16, 140 iterations). Playout cap
   randomization is implemented (`playout_cap_prob`, `fast_num_simulations`)
   for the follow-up pilot.

Acceptance: pilot-scale runs at equal wall-clock time, compared on the Elo
ladder against the current best recipe (now: bf16 pilot, iteration 90 at
1.21 h). Adopt whatever is stronger at equal time.

## Step 2: more sample reuse

Training is nearly free relative to self-play, so try ~4x reuse
(`num_updates_per_iter=64` at pilot scale, vs. 32) on top of the Step 1 recipe.
Watch for overfitting (train loss falling while ladder strength stalls).

Acceptance: equal-time pilot, ladder comparison as above.

## Step 3: record the TPU command and checklist

- Final `train.py` command with full-size settings, scaled to the TPU's device
  count (batch sizes divisible by devices) and a `max_num_iters` that fits the
  rental so the cosine schedule completes.
- `save_data_state=true`; checkpoints (and `data_state.pkl`) on durable storage.
- Copy the baseline checkpoint that `pgx/_src/baseline.py` loads for evaluation.
- Short multi-device smoke test on the TPU before the long run: throughput,
  memory, no recompiles, resume from a checkpoint.
- Periodic `elo_ladder.py` against earlier checkpoints to track progress.

## Log

- 2026-09-14: pilots and ladder above; full-size measurement run.
- 2026-09-15: profiled self-play (inference dominates); bfloat16 self-play
  adopted (ladder, 10 models, anchor = old baseline): bf16 pilot iteration 96
  -100, iteration 90 -127, float32 GessFormer + augmentation -248, RayFormer
  -313, GessFormer -394 (ratings shift as models are added; compare gaps).
