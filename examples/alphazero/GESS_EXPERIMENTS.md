# Gess AlphaZero experiments

A log of experiments on the Gess AlphaZero setup: what each tested, how, the
result, and the conclusion. The forward-looking plan lives in
[`GESS_TPU_PLAN.md`](GESS_TPU_PLAN.md).

Wandb runs are under `https://wandb.ai/aptometry/pgx-az-gess/runs/<id>`.
Checkpoints are under `checkpoints/` (not in git).

## Conventions

**Pilot settings.** Unless stated otherwise, pilots use:

```
env_id=gess selfplay_batch_size=256 max_num_steps=128 training_batch_size=2048
num_simulations=32 learning_rate=5e-4 weight_decay=1e-4 warmup_steps=100
grad_clip_norm=1.0 continue_games=true replay_buffer_iters=4
num_updates_per_iter=32 lr_schedule=cosine eval_interval=5
```

(~2x sample reuse). Training time ("hours") excludes evaluation. Local GPU:
RTX 5070 Ti, 16 GB.

**Measuring strength.** `eval/vs_baseline/*` in wandb (raw policy sampling vs.
the ResNet baseline, 256 games) proved too noisy to rank runs. Runs are
compared with MCTS games: `model_tournament.py` (two checkpoints) or
`elo_ladder.py` (round robin, Elo fit; 128 games/pair, 32 sims, seat-swapped
pairs with a 2-ply random opening). Ladder results are cached in
`elo_gess_pilots.json` at the repo root. Reported ± are optimistic (paired
games are correlated), and ladder ratings shift as models are added, so
compare gaps within one ladder fit.

**Equal time.** Recipes are compared at equal training time (the checkpoint
nearest the reference run's hours), since that is what matters for a
fixed-length rental.

## Current ladder

10 models, anchor = old ResNet baseline (`checkpoints/gess_20260604081951/000125.ckpt`,
32.8M positions, 14.6 h).

| Model | Exp. | Train time | Elo |
|---|---|---|---|
| old ResNet baseline | – | 14.6 h | 0 |
| GessFormer + aug + bf16 (it 96) | E7 | 1.29 h | -100 ± 19 |
| GessFormer + aug + bf16 (it 90) | E7 | 1.21 h | -127 ± 19 |
| GessFormer + aug | E5 | 1.23 h | -248 ± 19 |
| RayFormer (it 60) | E4 | 1.88 h | -313 ± 19 |
| GessFormer | E2 | 1.25 h | -394 ± 20 |
| RayFormer (40-iteration run) | E4 | 1.22 h | -481 ± 20 |
| RayFormer + aug, symmetry-tied | E6 | 1.93 h | -538 ± 21 |
| RayFormer + aug | E6 | 1.82 h | -584 ± 21 |
| ResNet pilot | E3 | 0.99 h | -737 ± 24 |

---

## E1. GessFormer, original training loop

**Question.** Does the new GessFormer architecture train at all, and how does
the original training loop (fresh games each iteration, train once on each
iteration's data, constant LR) do?

**Setup.** `architecture=gessformer`, pilot sizes but *without*
`continue_games`, replay buffer or LR schedule; `warmup_steps=100`,
`max_num_iters=60`. Run `thu9wxkn`, `checkpoints/gess_20260914010326`.

**Result.** 60 iterations, 1.16 h. Raw-policy win rate vs. baseline peaked at
~8% and was noisy. Only 10-20% of game slots finished within the 128-step
window, so most positions had no value target (and `train/value_loss` looked
small because masked positions still counted in the mean).

**Conclusion.** Trains, but the data pipeline wastes most positions and biases
toward openings. Motivated E2.

## E2. Continued games + replay buffer + cosine LR

**Question.** Do continuing games across iterations (so every position gets a
real value target), a replay buffer with ~2x reuse and cosine LR decay help?

**Setup.** Pilot settings (the conventions above). Run `rkblsgw2`,
`checkpoints/gess_20260914031715`.

**Result.** Every emitted sample has a value target. MCTS tournament vs. E1 at
iteration 60: **92.8%, +443 Elo** (512 games). Vs. the old baseline: 7.6%
(the baseline had 17x the data).

**Conclusion.** Large improvement from the data pipeline alone; adopted as the
standard training setup.

## E3. ResNet with the E2 training setup

**Question.** Is GessFormer better than the original ResNet under the same
training setup?

**Setup.** Pilot settings with `architecture=resnet` (128 channels, 6 blocks).
Run `5jva80ke`, `checkpoints/gess_20260914051848`. Also added self-play
draw-rate / game-length logging.

**Result.** 0.99 h for 60 iterations (~25% cheaper than GessFormer). GessFormer
(E2) vs. this ResNet: **91.8%, +420 Elo**, both at iteration 60 and with
GessFormer at iteration 50 (1.04 h, ~equal time). Rising `train/value_loss`
(0.09 -> 0.40) tracked self-play draws vanishing (30-47% draws mid-run -> 0%):
draws are easy value targets.

**Conclusion.** GessFormer clearly beats the ResNet at equal time. Rising value
loss is not by itself a warning sign.

## E4. RayFormer (full-resolution ray attention)

**Question.** Does attention restricted to rows, columns and diagonals (the
Gess move directions), one token per cell, beat GessFormer?

**Setup.** `architecture=rayformer` (embed 96, 4 layers, 4 heads, bfloat16
attention, per-family remat; sized to match GessFormer's inference cost),
`train_micro_batches=2`. Two runs: 60 iterations (`d7w1qpfb`,
`checkpoints/gess_20260914071958`, 1.88 h) and 40 iterations so its LR
schedule completes in ~equal time (`gay4d22q`,
`checkpoints/gess_20260914161042`, 1.22 h).

**Result.** Iterations ~113 s vs. ~75 s for GessFormer (benchmarks predicted
~20% overhead; measured ~50%). Vs. GessFormer (E2, it 60): RayFormer it 60
(equal games) **+185 Elo**; RayFormer it 40 of the 60-iteration run (equal time,
mid-schedule) -105; 40-iteration run (equal time, full schedule) **-189**.

**Conclusion.** RayFormer learns more per game (with 13x fewer parameters) but
loses at equal time because its iterations are slower. Not adopted; worth
revisiting only if its self-play speed gap is closed.

## E5. Symmetry augmentation for GessFormer

**Question.** Does training on a random one of the 8 board symmetries per sample
help? (Gess rules were verified invariant under all 8 on 1,176 env
state/symmetry pairs.)

**Setup.** Pilot settings + `symmetry_augmentation=true`. Run `9ijbqmik`,
`checkpoints/gess_20260914190157`.

**Result.** Same iteration time (1.23 h). Vs. GessFormer (E2): **80.5%, +246 Elo**
(384 games). Vs. RayFormer it 60: 79.7%, +240.

**Conclusion.** Clear gain at no cost; adopted.

## E6. RayFormer + symmetry augmentation, untied and symmetry-tied

**Question.** Does augmentation stack with RayFormer's sample efficiency? If
not, does tying its position parameters across symmetries (`rf_symmetric`:
shared row/column and diagonal/anti-diagonal biases depending on |offset|,
position embedding shared over the 55 symmetry orbits) fix it?

**Setup.** E4's RayFormer settings (60 iterations) + `symmetry_augmentation=true`:
untied (`35xg0okm`, `checkpoints/gess_20260914210805`, 1.82 h) and
`rf_symmetric=true` (`b8yd78ko`, `checkpoints/gess_20260914235054`, 1.93 h).
Equivariance of the tied encoder was verified to float precision.

**Result.** Untied: -584 in the ladder, ~270 Elo below RayFormer without
augmentation (lost 114-14 head-to-head); value loss stayed flat at ~0.47 and
policy loss ended at 1.19 vs. 0.76 — underfitting. Tied: -538, beat untied
76.5-51.5 but still far below RayFormer without augmentation.

**Conclusion.** Augmentation hurts the small RayFormer (likely too little
capacity to learn all orientations; the conv stem, FFNs and heads still must).
Tying helps modestly. RayFormer parked.

## E7. bfloat16 self-play inference

**Question.** Self-play is ~95% of iteration time and network inference is
essentially all of self-play (profile: 63 ms forward pass vs. 1.4 ms env step at
batch 1024; 32-sim move 1.8 s). Does running the self-play network in
bfloat16 (training stays float32) speed things up without hurting quality?

**Setup.** E5 settings + `selfplay_bf16=true`, 96 iterations to fill ~equal
time. Run `zzd7cr98`, `checkpoints/gess_20260915035816`.

**Result.** Forward pass 1.95x faster; pilot iterations 72 s -> ~48 s. On the
E5 checkpoint, bf16 search picks the same move as float32 96-97% of the time
(policy-target TV distance 0.04). Ladder at equal time (iteration 90, 1.21 h):
**-127 vs. -248 for E5, +121 Elo**; won 101.5-26.5 head-to-head. Iteration 96:
-100, best so far. Raw-policy win rate vs. baseline 23-34% late in the run
(previous pilots <= 13%) and value loss kept falling (0.29).

**Conclusion.** Adopted. More games per hour outweigh the small numerical
difference.

## E8. 16 simulations per move (running)

**Question.** Does halving search to 16 simulations (roughly halving self-play
cost again) beat 32 at equal time?

**Setup.** E7 settings + `num_simulations=16`, 140 iterations,
`eval_interval=10`. Run `q8z0tred`, `checkpoints/gess_20260915055020`.
Compare the checkpoint nearest 1.21 h with E7 iteration 90.

**Result.** Pending.

---

## Infrastructure checks

- **Full-size measurement** (E5 recipe, float32 self-play; 1024 games x 256
  steps, batch 4096, 128 updates/iteration): 10.6-10.9 min/iteration
  (~1.46M positions/hour); training ~0.2 s/update, so self-play ~95% of time;
  host RAM 14 GB with a full 4-iteration buffer (6.6 GB); `data_state.pkl`
  9.1 GB saved in ~14 s.
- **GPU OOM at full size** (first full run): the training loop kept two copies
  of self-play samples on the GPU; fixed by freeing them and shuffling on the
  host. Replay minibatches are now also gathered one at a time on the host.
- **Exact resume** (`save_data_state=true`): Ctrl+C mid-run and resume reproduces
  the uninterrupted run bit-for-bit (params, optimizer state, RNG), both at and
  between evaluation iterations.
- **Refactors verified bit-identical** on the default path: replay buffer,
  trajectory value targets (vs. the original `compute_loss_input`), GessFormer
  head refactor, lazy minibatch gathering, bfloat16 plumbing, playout cap
  randomization.
