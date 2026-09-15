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
- `replay_buffer_iters=4`, ~4x sample reuse at full size (E10; ~2x in earlier
  pilots)
- `lr_schedule=cosine`, AdamW (`weight_decay=1e-4`), warmup, `grad_clip_norm=1.0`
- `symmetry_augmentation=true` – random one of the 8 board symmetries per sample
- `selfplay_bf16=true` – bfloat16 self-play inference (~2x faster network)
- `playout_cap_prob=0.25 fast_num_simulations=8` (with `num_simulations=32`) –
  playout cap randomization; policy trained on full-search moves only
- `save_data_state=true` on preemptible machines – exact resume

### Pilot ladder

See the current ladder in [`GESS_EXPERIMENTS.md`](GESS_EXPERIMENTS.md). Best at
equal time (~1.22 h): + 4x reuse (E10) -90; + playout cap randomization (E9)
-98; bf16 self-play (E7) -142; 16 simulations (E8) -266; float32 (E5) -310;
RayFormer -370 (at 1.88 h); ResNet pilot -717 (anchor: old ResNet baseline,
14.6 h, 0).

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
   the simplest variant. **E8: rejected**, -103 Elo vs. 32 simulations at equal
   time. **E9, playout cap randomization** (`playout_cap_prob=0.25`,
   `fast_num_simulations=8`): **adopted, tentatively**, +45 Elo vs. E7 at equal
   time in the ladder, 52-56% head-to-head.

Acceptance: pilot-scale runs at equal wall-clock time, compared on the Elo
ladder against the current best recipe (now: E9, iteration 147 at 1.22 h).
Adopt whatever is stronger at equal time.

## Step 2: more sample reuse

Training is nearly free relative to self-play, so try ~4x reuse
(`num_updates_per_iter=64` at pilot scale, vs. 32) on top of the Step 1 recipe.
Watch for overfitting (train loss falling while ladder strength stalls).

Acceptance: equal-time pilot, ladder comparison as above.

**E10 result:** tie at pilot scale (+8 Elo, 52% head-to-head vs. E9) despite
~1/3 fewer games, because extra updates cost ~0.44 s each at pilot size. At
full size updates are ~4% of iteration time, so **adopt ~4x reuse for the
full-size run** (e.g. `num_updates_per_iter=256` with 1024 x 256 samples and
batch 4096).

## Step 3: record the TPU command and checklist

- Final `train.py` command with full-size settings, scaled to the TPU's device
  count (batch sizes divisible by devices) and a `max_num_iters` that fits the
  rental so the cosine schedule completes.
- `save_data_state=true`; checkpoints (and `data_state.pkl`) on durable storage.
- Resume memory (E11): on the 16 GB GPU, a resumed full-size run OOMed on the
  batch-4096 train step's single ~10.3 GiB allocation that the original process
  had fit; `train_micro_batches=2` fixed it. On TPU each chip only sees
  `training_batch_size / devices`, so this allocation is much smaller, but if a
  resume OOMs, add `train_micro_batches=2` (or more).
- Copy the baseline checkpoint that `pgx/_src/baseline.py` loads for evaluation.
- Short multi-device smoke test on the TPU before the long run: throughput,
  memory, no recompiles, resume from a checkpoint. (Multi-device logic and
  cross-device-count resume already pass on 8 simulated CPU devices.)
- Track progress during the run with `mcts_eval_opponent=<fixed checkpoint>`
  (hourly MCTS match, `eval/mcts/score` and `elo` in wandb; ~1 min per 128
  games locally). Good opponents: the best pilot (E10 it 100), later a copy of
  an early TPU checkpoint once the model outgrows it.
- Periodic `elo_ladder.py` against earlier checkpoints for a fuller picture.

## Maybe later

Ideas not on the current path; revisit if there is time or a need.

- **Bootstrapping a larger model once the current one plateaus.** Options, most
  practical first: (1) grow GessFormer in place by adding transformer blocks
  whose output projections start at zero (identity at first), keeping weights,
  optimizer state and `data_state.pkl`; (2) train a larger model on the smaller
  model's self-play games/replay buffer and switch self-play to it once it wins
  on the ladder (KataGo-style); (3) distill the small model's policy/value (or
  search results) into the larger one, then continue RL; (4) generate targets
  with much deeper search. Check first that a plateau is really a capacity
  limit (not search budget, replay window or LR).
- **Measure the original ResNet with the original training loop** as a pilot,
  to replace the estimated part of the total gain over the original setup
  (roughly +600 to +1,000 Elo at equal pilot time).
- **RayFormer**: close its ~50% iteration-time gap, or try a larger variant; it
  learns more per game than GessFormer (E4) but loses at equal time.
- **Optimizer tuning and alternatives** (training is ~5% of iteration time, so
  expect modest gains; each is an equal-time pilot + ladder). Most valuable
  first: (1) tune the current AdamW (LR 2.5e-4 / 1e-3 vs. 5e-4, warmup, weight
  decay, beta2) – never tuned for this setup; (2) EMA/SWA weights for the
  self-play/evaluation network (KataGo-style); (3) Muon (`optax.contrib.muon`,
  AdamW for norms/biases/embeddings) – promising for transformers, unproven
  for AlphaZero RL; (4) schedule-free AdamW (`optax.contrib.schedule_free_adamw`)
  – no fixed run length, handy for a rental that may stop or extend. Less
  promising: Lion, Shampoo/SOAP/Sophia, SGD+momentum.

## Status

2026-09-15: Steps 1 and 2 done; multi-device validation passed; wall-clock MCTS
evaluation added. Running E11, an ~8 h full-size run of the final recipe
(started 15:27, with a Ctrl+C/resume rehearsal). Step 3 waits for the rental
details (TPU type/chips, duration, storage).

## Log

- 2026-09-14: pilots and ladder above; full-size measurement run.
- 2026-09-15: profiled self-play (inference dominates); bfloat16 self-play
  adopted (ladder, 10 models, anchor = old baseline): bf16 pilot iteration 96
  -100, iteration 90 -127, float32 GessFormer + augmentation -248, RayFormer
  -313, GessFormer -394 (ratings shift as models are added; compare gaps).
- 2026-09-15: E8 (16 simulations, bf16) at equal time -232 vs. E7 -129 (12-model
  ladder); rejected.
- 2026-09-15: E9 (playout cap randomization) -90 vs. E7 -135 at equal time
  (14-model ladder), 52-56% head-to-head; adopted tentatively.
- 2026-09-15: E10 (4x reuse) -90 vs. E9 -98 at equal time (15-model ladder),
  52% head-to-head; tie at pilot scale, adopted for full size where updates
  are cheap.
