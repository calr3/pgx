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

**Opening bug (fixed 2026-09-15, `eb39f31`).** Until then ~13% of ladder
games (and 22/128 training MCTS-eval games) ended during the 2-ply random
opening: a Gess piece whose every move destroys its owner's last ring could be
chosen. Mirrored seats gave both models the same free points, so rankings were
unaffected, but **all ladder Elo gaps below are compressed by roughly that
proportion**. The ladder cache is versioned, so new ladders replay games.

**Equal time.** Recipes are compared at equal training time (the checkpoint
nearest the reference run's hours), since that is what matters for a
fixed-length rental.

## Current ladder

10 models, 128 games/pair, 32 sims, fixed openings (`game_version=2`), anchor =
old ResNet baseline (`checkpoints/gess_20260604081951/000125.ckpt`, 32.8M
positions, 14.6 h). Trimmed to one model per conclusion; earlier ladders (with
the opening bug) had compressed gaps.

(A separate 5-model ladder played under **v1 rules** is in E12.)

| Model | Exp. | Train time | Elo |
|---|---|---|---|
| **full-size recipe (it 72)** | E11 | 7.2 h | **+99 ± 22** |
| old ResNet baseline | – | 14.6 h | 0 |
| + 4x sample reuse (it 100) | E10 | 1.22 h | -231 ± 20 |
| + playout cap randomization (it 147) | E9 | 1.22 h | -252 ± 20 |
| + bf16 self-play (it 90) | E7 | 1.21 h | -281 ± 20 |
| full-size recipe (it 36) | E11 | 3.6 h | -289 ± 21 |
| + symmetry augmentation | E5 | 1.23 h | -559 ± 24 |
| RayFormer (it 60) | E4 | 1.88 h | -731 ± 27 |
| GessFormer | E2 | 1.25 h | -866 ± 30 |
| ResNet pilot | E3 | 0.99 h | -1422 ± 63 |

Note the fit is not fully transitive: head-to-head the baseline beat E11 it 72
**75-53 (59%)**, but E11 beats the weak pilots far more decisively than the
baseline does (128-0 vs. the ResNet pilot and RayFormer, where the baseline
scores 117/128), which lifts its fitted rating. Read "E11 is close to the
baseline at half the training time", not "clearly ahead".

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

## E11. Full-size run of the final recipe, ~8 h

**Question.** Does the adopted recipe behave well at full size over a long run
(memory, held-back steps, game length and draws, late-schedule stability),
how fast does it improve, and when does it pass the old baseline? Also a
rehearsal of Ctrl+C + resume at full scale.

**Setup.** Full size: `selfplay_batch_size=1024 max_num_steps=256
training_batch_size=4096 num_updates_per_iter=256` (~4x reuse),
`replay_buffer_iters=4`, `architecture=gessformer selfplay_bf16=true
num_simulations=32 playout_cap_prob=0.25 fast_num_simulations=8
symmetry_augmentation=true continue_games=true`, AdamW `learning_rate=5e-4
weight_decay=1e-4 warmup_steps=500 grad_clip_norm=1.0 lr_schedule=cosine`,
`save_data_state=true`, `max_num_iters=72 eval_interval=4`, hourly
`mcts_eval_opponent=checkpoints/gess_20260915200817/000100.ckpt` (E10 it 100).
A timing run measured ~5-6 min/iteration (training ~0.5 s/update, about a
third of the iteration at 256 updates). Started 15:27; planned Ctrl+C after
iteration 5 and resume. Run `txe8dngl`, `checkpoints/gess_20260915232736`.

**Result.** Resume rehearsal succeeded: Ctrl+C at 15:49 (after iteration 4),
the run finished iteration 5, saved the checkpoint and `data_state.pkl`, and
was relaunched at 15:55 with `resume_from`; it restored the full replay buffer
(1,048,576 samples), 51,292 held-back steps and the in-progress games, and
continued the same wandb run. However, **the resumed process ran out of GPU
memory at its first training step** (a single 10.29 GiB allocation for the
batch-4096 train step, which the uninterrupted process had handled), twice —
the second time with the GPU confirmed free, so not a driver-release delay.
Likely allocator fragmentation: the resumed process allocates in a different
order, leaving no contiguous 10.3 GiB block in JAX's ~12 GiB preallocated pool
(unconfirmed). Resumed successfully at 16:07 with `train_micro_batches=2`
(halves that allocation; same gradients). ~40 min lost in total. Starting MCTS
score vs. E10: 0.086. That score stayed at exactly 11/128 at iterations 5 and 15;
debugging showed all 11 "wins" were opening-bug games (E11 lost every real game
to the fully annealed E10 so far), which led to the opening fix. The running
process still uses the old openings, so its `eval/mcts/score` has a floor of
11/128 but stays comparable within the run.

Run completed: 72 iterations, 7.21 h of training time (18.9M positions), no
errors after the resume. Training curve:

| Iteration | Hours | MCTS score vs. E10 | Raw policy vs. old baseline | Policy loss | Value loss | Self-play draws | Steps/game |
|---|---|---|---|---|---|---|---|
| 16 | 1.6 | 0.086 (it 15; floor) | 3.7% | 2.63 | 0.45 | 1% | 97 |
| 24 | 2.4 | 0.44 (it 25) | 5.1% | 2.48 | 0.40 | 2% | 100 |
| 32 | 3.2 | 0.39 (it 35) | 8.0% | 1.88 | 0.37 | 4% | 120 |
| 40 | 4.0 | 0.69 (it 45) | 18.8% | 1.62 | 0.31 | 8% | 154 |
| 56 | 5.6 | 0.81 (it 55) | 43.8% | 1.12 | 0.27 | 20% | 187 |
| 64 | 6.4 | 0.88 (it 65) | 49.3% | 0.95 | 0.26 | 21% | 195 |
| 72 | 7.2 | 0.88 | 48.6% | 0.82 | 0.26 | 24% | 186 |

(MCTS scores include ~11/128 free opening-bug wins; excluding them E11 won
~87% of real games vs. E10 at the end. Raw policy vs. the old baseline without
search reached ~49%, where pilots never exceeded ~13%.) Most of the gain came
in the second half as the cosine schedule decayed; strength vs. E10 flattened
between iterations 65 and 72. Self-play draws rose steadily to 24% and games
lengthened to ~190 steps as play strengthened — watch in longer runs. Memory
stayed healthy (host RAM ~14 GB).

Ladder (fixed openings, table above): **E11 it 72 +99, it 36 -289** — a ~390 Elo
swing in the second half as the cosine schedule decayed. It beats every pilot
decisively (128-0 vs. the ResNet pilot and RayFormer, 126.5-1.5 vs. E5) but
loses to the old baseline head-to-head 53-75.

**Conclusion.** The full recipe works at full scale: in 7.2 h it reaches roughly
the strength of a 14.6 h baseline trained with the original setup, and its
ranking of the adopted steps matches the pilots. The ~8 h shape of the curve
(most gains late, flattening over the last ~7 iterations) suggests sizing
`max_num_iters` so the schedule completes just within the rental. Watch the
rising draw rate (24%) and game length (~190 steps) in longer runs.

## E12. Gess v1 rules: captureless stalemate decided on stone count

**Question.** E11's self-play draw rate rose to 24% (games ~190 steps) as play
strengthened. Does deciding a captureless stalemate on material (the player
with more stones wins; only an exact tie draws) remove the draws without
costing strength?

**Setup.** Rule implemented in `pgx/_src/games/gess.py` (env version v1,
`f190112`). Pilot with E9's settings (bf16, playout cap 0.25/8, augmentation,
replay 4, 32 updates/iteration, cosine), 147 iterations, 1.18 h. Run
`husyrrt4`, `checkpoints/gess_20260916165614`. Ladder of 5 models played
**entirely under v1 rules** (`elo_gess_v1rules.json`), so no model has a rules
handicap.

**Result.** Draws fell from 15-20% (v0 pilots) and 24% (E11) to **~1%**, and
games shortened (~164 vs. ~190 steps). But E12 is **weaker**: ladder -348 vs.
E9 -259 and E10 -249 (E9 beat it 84-44, 66%); raw policy vs. the old baseline
41% vs. E9's 48%. Value loss stayed higher all run (0.41 vs. 0.38): a material
verdict is harder to predict than a draw, and "be ahead on material, then avoid
captures for 20 turns" is an extra strategy to learn.

**Conclusion.** The rule does what it was meant to do about draws, but at pilot
scale it costs ~90 Elo. One seed at one scale; a longer run might close the gap.
Not adopted for now (see plan).

## E13. Full-size run under v1 rules, ~8 h

**Question.** E12 showed the v1 stalemate rule removes draws but cost ~90 Elo
at pilot scale. Adopted anyway (user decision: the rule is the game we want).
How does the full recipe behave under v1 at full size, and how does it compare
with E11 (same recipe, v0 rules, 7.2 h)?

**Setup.** As E11 (1024 games x 256 steps, batch 4096, 256 updates/iteration,
replay 4, bf16, playout cap 0.25/8, augmentation, cosine, warmup 500,
`save_data_state=true`, 72 iterations, hourly MCTS eval vs. E10) plus
`train_micro_batches=2` from the start (E11 needed it to resume). Started
12:05. Run `zfnfbzor`.

**Result.** 72 iterations, 7.16 h, no errors. Final MCTS score vs. E10 **0.816**
(E11, v0 rules: 0.879) and raw-policy win rate vs. the old baseline **52.1%**
(E11: 48.6%). Draws stayed low all run (0.4-7.5%, ending 6.4%) where E11 reached
24%; games still lengthened to ~185 steps. Value loss settled ~0.31 (E11: 0.26),
as expected when outcomes hinge on material rather than an easy draw. Progress
was slower than E11 until the last third (0.0 at iteration 21, 0.19 at 31, 0.39
at 51, 0.68 at 61, 0.82 at the end). Checkpoint
`checkpoints/gess_20260916200542`.

**Conclusion.** Under v1 the recipe reaches roughly E11's level at full size
(slightly behind on the MCTS metric, slightly ahead on raw policy), while nearly
eliminating draws. The pilot-scale ~90 Elo cost of the rule change (E12) does
not obviously persist at full size; a v1-rules ladder of E11 vs. E13 would settle
it.

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
- **Multi-device** (8 simulated CPU devices, tiny GessFormer, every recipe
  feature incl. micro-batches, bf16, playout cap, augmentation, data state and
  MCTS eval): runs cleanly; Ctrl+C and resume on 8 devices is bit-identical to
  an uninterrupted 8-device run; resuming 1 -> 8 and 8 -> 1 devices restores the
  buffer, held-back steps and games. Added a check that `selfplay_batch_size`
  divides by the device count (it was silently floored).
- **Opening bug**: see Conventions. Found via the constant E11 MCTS-eval score;
  the old code ended 22/128 (eval key) and 66/512 (ladder seed) games in the
  opening, the fixed code 0/640.
- **Wall-clock MCTS evaluation during training** (`mcts_eval_opponent`):
  training verified bit-identical with and without it; ~40-65 s per 128-game
  match against a pilot checkpoint on the local GPU.
- **Refactors verified bit-identical** on the default path: replay buffer,
  trajectory value targets (vs. the original `compute_loss_input`), GessFormer
  head refactor, lazy minibatch gathering, bfloat16 plumbing, playout cap
  randomization.
