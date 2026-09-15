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

15 models, anchor = old ResNet baseline (`checkpoints/gess_20260604081951/000125.ckpt`,
32.8M positions, 14.6 h).

| Model | Exp. | Train time | Elo |
|---|---|---|---|
| old ResNet baseline | – | 14.6 h | 0 |
| GessFormer + aug + bf16 + playout cap, 4x reuse (it 100) | E10 | 1.22 h | -90 ± 14 |
| GessFormer + aug + bf16 + playout cap (it 147) | E9 | 1.22 h | -98 ± 14 |
| GessFormer + aug + bf16 + playout cap (it 140) | E9 | 1.16 h | -104 ± 14 |
| GessFormer + aug + bf16 (it 96) | E7 | 1.29 h | -119 ± 14 |
| GessFormer + aug + bf16 (it 90) | E7 | 1.21 h | -142 ± 14 |
| GessFormer + aug + bf16, 16 sims (it 130) | E8 | 1.13 h | -262 ± 14 |
| GessFormer + aug + bf16, 16 sims (it 140) | E8 | 1.21 h | -266 ± 14 |
| GessFormer + aug | E5 | 1.23 h | -310 ± 15 |
| RayFormer (it 60) | E4 | 1.88 h | -370 ± 15 |
| GessFormer | E2 | 1.25 h | -433 ± 15 |
| RayFormer (40-iteration run) | E4 | 1.22 h | -509 ± 16 |
| RayFormer + aug, symmetry-tied | E6 | 1.93 h | -559 ± 16 |
| RayFormer + aug | E6 | 1.82 h | -595 ± 16 |
| ResNet pilot | E3 | 0.99 h | -717 ± 18 |

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

## E8. 16 simulations per move

**Question.** Does halving search to 16 simulations (roughly halving self-play
cost again) beat 32 at equal time?

**Setup.** E7 settings + `num_simulations=16`, 140 iterations,
`eval_interval=10`. Run `q8z0tred`, `checkpoints/gess_20260915055020`.

**Result.** 140 iterations in 1.21 h (~31 s/iteration vs. ~48 s for E7), so
iteration 140 is exactly equal-time with E7 iteration 90. Policy loss fell much
lower (0.40 vs. 0.94 for E7), as expected with sharper 16-simulation targets;
value loss stayed ~0.43 (E7: 0.29-0.31). Ladder: iteration 140 **-232 vs. -129
for E7 iteration 90, -103 Elo**; E7 iteration 90 won 92.5-35.5 head-to-head
(85-43 vs. iteration 130). Iterations 130 and 140 are level (-226/-232; 66-62
head-to-head). Still +40 over float32 32-simulation GessFormer + aug (E5).

**Conclusion.** Not adopted: at equal time, 50% more games with half the search
lose clearly to 32 simulations. Search quality of the training targets matters
more here than game count. Playout cap randomization (full search for policy
targets on a fraction of moves) remains worth testing, since it keeps
full-quality policy targets.

## E9. Playout cap randomization

**Question.** Can cheap searches on most moves buy more games without E8's
loss of policy-target quality, by training the policy only on full-search
moves (KataGo's playout cap randomization)?

**Setup.** E7 settings + `playout_cap_prob=0.25 fast_num_simulations=8`
(`num_simulations=32`; ~14 simulations per move on average), 147 iterations
(~29 s each, 1.22 h), `eval_interval=7`. Run `6ymiv09q`,
`checkpoints/gess_20260915173622`.

**Result.** Full searches on ~25% of steps as intended. Raw-policy win rate vs.
baseline reached 41-48% at the end (E7: 23-34%); value loss ~0.38. Ladder at
equal time: iteration 147 **-90 vs. -135 for E7 iteration 90, +45 Elo**
(iteration 140, 1.16 h: -92). Head-to-head is closer: vs. E7 iteration 90,
iteration 147 scored 66.5/128 (52%) and iteration 140 71.5/128 (56%); vs. E7
iteration 96 (1.29 h), 60.5 and 57/128 (47%, 45%). E9 beats the weaker models
more decisively than E7 does (e.g. 105-108/128 vs. E8, against E7's 85-92),
which drives the fitted gap. Iterations 140 and 147 are level (65-63).

**Conclusion.** Adopted, tentatively: at least as strong as E7 at equal time and
probably modestly stronger (+45 in the fit, 52-56% head-to-head), with a much
stronger raw policy. The gain is small relative to the noise of single pilots;
unlike E8, keeping full-search policy targets avoids the loss from cheap search.

## E10. 4x sample reuse

**Question.** Training is cheap relative to self-play; does doubling gradient
updates per iteration (~4x reuse of each position instead of ~2x) help at equal
time?

**Setup.** E9 settings + `num_updates_per_iter=64`. Iterations ~43 s (vs. ~29 s
for E9: each extra update costs ~0.44 s at pilot size), so 100 iterations in
1.22 h — about a third fewer self-play games than E9. Run `hws697pp`,
`checkpoints/gess_20260915200817`.

**Result.** Raw-policy win rate vs. baseline 44.5% at the end (E9: 48%); value
loss 0.34 (E9: 0.38). Ladder at equal time: **-90 vs. -98 for E9 iteration
147, +8 Elo** — a tie within noise. Head-to-head vs. E9 iteration 147: 66.5/128
(52%); vs. E9 iteration 140: 73.5 (57%); vs. E7 iterations 90/96: 71.5 and
70.5 (56%, 55%).

**Conclusion.** At pilot scale, 4x reuse matches 2x at equal time while playing
~1/3 fewer games, i.e. each game is worth more. At pilot size the extra updates
are expensive (~50% longer iterations); at full size they are cheap (training
~0.2 s/update with batch 4096, ~4% of an iteration at 128 updates), so doubling
updates there costs only ~4% more time. **Adopt ~4x reuse for the full-size
run**, where it is nearly free; no evidence of overfitting at this scale.

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
