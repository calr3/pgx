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

## E14. Clock planes (v2), warm-started from E13

**Question.** The network cannot see the captureless-move clock, yet under v1 the
clock decides games on stone count after only 20 turns (10 each). In the web app
E13's search walked into short-horizon losses it had no way to anticipate. Does
showing the network the clock help?

**Setup.** pgx gess **v2** appends two constant planes to the observation (rules
unchanged; the tdgauntlet conformance vectors regenerate identically):
plane 4 is `no_capture_turns / 20` for both colours, plane 5 is the same value
signed by who is ahead on stones (+ mover ahead, - behind, 0 level). Plane 5 is
Epaminondas's v8 signed clock, but without its colour leak: in Gess a stone tie
is a draw, so a symmetric position reads 0 for both sides. v1 checkpoints play
on the leading four planes (model_tournament, the training MCTS eval, pgx's
ResNet baseline and tdgauntlet's JAX client all narrow the observation).

Warm start with the new `init_from`: E13's weights, with the stem convolution
widened by zero weights for the two planes, so iteration 0 computes exactly E13
(checked: identical logits and values; MCTS eval vs. E13 0.500). Then E13's
full-size recipe for 24 iterations (~2.4 h), fresh optimizer, `seed=1`, cosine
from a lower peak for a fine-tune (`learning_rate=2e-4 warmup_steps=200`), hourly
MCTS eval vs. E13. Run `ajau3ng7`, `checkpoints/gess_20260926190427`.

Afterwards: tdgauntlet, E14 it 24 vs. E13, both at 128 Gumbel MCTS sims, 200
random 2-ply openings played both ways (`examples/gess_e14_v_e13.toml`).

**Caveat.** E14 also has 2.4 h more training than E13. A gain could be training
rather than the planes; a matched control (E13 warm-started for 24 iterations
without the planes) would separate them.

**Result.** 24 iterations in 2.33 h of training time. The training-time MCTS
eval vs. E13 (128 games, 32 sims, fixed openings) went 0.50 (it 0), **0.25** (~1 h),
0.59 (it 21), 0.75 (it 24): the fine-tune first knocked the network back, then
passed E13.

tdgauntlet, 128 sims, 400 games (`results/gess_e14_v_e13.json`), ~36 min:

```
player   score    Elo   counted     excluded      W-D-L
e14      90.5%   +391       118     82 (41%)  283-25-92
e13       9.5%   -391       118     82 (41%)  92-25-283
```

Per game that is 73.9% (+181 Elo), in line with the training eval's 0.75. The
tdgauntlet figure is higher because it scores minimatches and drops the 41%
where each side won one game. No seat effect: E14 scored 0.752 as black and 0.725
as white. By how games ended: E14 went 207-57 in games ended by a broken ring
(0.78) and 76-25-35 in games decided by the 20-move rule (0.65). So most of the
gain is in ordinary tactical play, not only the clock endings.

**Conclusion.** E14 is clearly stronger than E13 (+~180 Elo per game) and shows
no colour leak. How much of that is the planes and how much is 2.3 h more
training is not settled: the gain being largest in games that never reach the
clock points at least partly to training. The matched control (E13 warm-started
for 24 iterations under v1, same settings) would separate the two. E14 is now the
web app's model.

## E15. Control for E14: the same fine-tune without the clock planes

**Question.** How much of E14's +180 Elo over E13 is the v2 planes, and how much
is 2.3 h more training?

**Setup.** E14's exact command (same seed, schedule and eval) with `obs_planes=4`:
the new Config field makes the network read only the leading four planes of v2's
observation, so the architecture is E13's and `init_from` loads it unchanged
(iteration 0 reproduces E13 exactly). Started 14:23. `checkpoints/gess_20260926222338`.

Afterwards: tdgauntlet round robin of E15, E14 and E13 at 128 sims, 200 random
2-ply openings played both ways (`examples/gess_e15_three_way.toml`, started by
`run_e15_tournament.sh`).

**Result.** Training eval vs. E13 went 0.50, 0.21, 0.41, 0.68 (E14: 0.50, 0.25,
0.59, 0.75): the same early dip, so that is the fine-tune, not the planes. The
round robin, 400 games per pairing, ~2.3 h (`results/gess_e15_three_way.json`),
per-game scores with 95% intervals:

| pairing | W-D-L | score | Elo |
|---|---|---|---|
| E14 vs. E13 | 275-19-106 | 0.711 ± 0.044 | +157 |
| E15 vs. E13 | 276-20-104 | 0.715 ± 0.044 | +160 |
| E14 vs. E15 | 192-40-168 | 0.530 ± 0.049 | +21 |

By ending, E14 vs. E15 was level in games decided by the 20-move rule (73-40-79)
and E14's small edge came from ring-breaking games (119-89). Against E13, E15
did better than E14 in the 20-move endings (0.775 vs. 0.665), the opposite of what
the planes were meant to do.

**Conclusion.** The clock planes are worth about +20 Elo at most, not significant
here; E14's +160 over E13 is the extra 2.3 h of training, which the control
reproduces exactly. Showing the network the clock did not help it play the
clock endings. v2 stays (it is harmless), but the planes are not a lever worth
pursuing further at this strength. E15 replaced E14 as the web app's model.

## E16. E15 continued for ~16 h

**Question.** How far does more training take the E13 -> E15 line? E13's curve
was still climbing at 72 iterations, and 24 more (E15) gained ~160 Elo.

**Setup.** `resume_from` E15 it 24 with `max_num_iters=184` (160 more iterations,
~16 h), E15's settings otherwise (`obs_planes=4`, peak LR 2e-4 cosine to 2e-5 over
the whole 184). Resuming keeps the Adam state, so this should avoid the ~1 h dip
both fine-tunes took from a fresh optimizer and empty buffer;
`save_data_state=true` so a later continuation keeps the buffer too. Hourly MCTS
eval vs. E15 it 24. Same run and directory as E15 (`jkdwf4m9`,
`checkpoints/gess_20260926222338`). Started 19:52 (a 19:44 start, not detached,
was stopped after one iteration and relaunched with `setsid`).

**Result.** 160 iterations, 16.2 h of training time, no errors; finished 13:11.
Hourly eval vs. E15: 0.15 after an hour (the dip came back despite the resume:
the LR jumped from E15's 2e-5 floor to ~1.95e-4 with an empty buffer), 0.22-0.34
while the LR stayed high, then 0.47 (it 105), 0.53, 0.57, 0.58, 0.67, 0.66, 0.82,
0.79 and 0.75 (it 184) as it decayed.

tdgauntlet, E16 it 184 vs. E15 it 24, 128 sims, 400 games (`results/gess_e16_v_e15.json`):
**308-16-76, 0.790 +/- 0.040 per game (+230 Elo)**; 93.3% of counted minimatches.
0.805 as black, 0.775 as white; 0.81 in ring-breaking games and 0.75 in games
decided by the 20-move rule.

**Conclusion.** More training still pays heavily: E13 -> E15 -> E16 is roughly
+160 then +230 Elo. The climb came almost entirely once the LR fell below ~1e-4,
so the next extension resumes at 8e-5 rather than re-warming to the peak.

## E17. E16 continued, resuming at a lower LR

**Question.** Does the line keep improving, and does resuming at the LR where
E16's climb began (instead of re-warming to the peak) avoid the dip?

**Setup.** `resume_from` E16 it 184 with `save_data_state` (buffer, held-back steps
and in-progress games all restored), 82 more iterations to `max_num_iters=266`,
sized to finish ~23:30 from a 14:36 start. The cosine schedule is kept, but its
parameters are solved so that at the resumed optimizer step (46,125) it reads
8e-5 and it ends at 1e-5: `learning_rate=3.0551e-4 lr_final_ratio=0.0327` (the
nominal peak is never reached). Hourly MCTS eval vs. E16 it 184. Same run and
directory as E15/E16.

**Result.** 82 iterations, 8.6 h of training time, finished 23:58. The resume
restored the buffer and state and the dip was mild: hourly eval vs. E16 read
0.40, 0.46, 0.50, 0.54, 0.59 (it 229), then 0.56, 0.52, 0.54, 0.49 and 0.51 (it 266).

tdgauntlet, E17 it 266 vs. E16 it 184, 128 sims, 400 games (`results/gess_e17_v_e16.json`):
**175-66-159, 0.520 +/- 0.049 per game (+14 Elo)**, a tie. 0.545 as black, 0.495 as
white; 0.57 in ring-breaking games and 0.47 in 20-move endings. 208 of the 400
games went to the 20-move rule (E16 vs. E15: 134), and draws rose to 16.5%.

**Conclusion.** The line stalled. Two explanations fit and this run cannot tell
them apart: E16's gain may need its high-LR phase (it re-warmed to ~2e-4 and ran
16 h; E17 started at 8e-5 and ran 8.6 h), or this network at 32 sims is near its
ceiling. A 16 h extension that re-warms to ~2e-4 would decide it; if that also
ties, the next lever is capacity, not time. E16 stays the web app's model.

## E18. E17 continued with 64-simulation searches and the LR re-warmed

**Question.** Is the E17 plateau a ceiling of 32-simulation training targets?
Each Gess move is two search decisions, so 32 simulations look at only about a
dozen full moves; deeper searches should give targets that see refutations the
network's prior misses. A 5 h validation run.

**Setup.** `resume_from` E17 it 266 with `save_data_state` (buffer, held-back steps
and in-progress games restored), 32 more iterations to `max_num_iters=298`.
Full searches (a quarter of moves, `playout_cap_prob=0.25`) at 48 simulations for
iterations 267-270, then 64: `num_simulations=64 sim_schedule=48@0,64@271`; fast
searches stay at 8. The LR jumps back to 2e-4 at the resumed step (68,096) and
decays to 1e-5 by the end, via solved cosine parameters
`learning_rate=6.7167e-3 lr_final_ratio=1.489e-3` (nominal peak never reached).
Hourly MCTS eval vs. E17 it 266 (32 sims). Otherwise E17's settings. Same run and
directory as E15-E17. Started 18:53.

**Result.** 32 iterations, 4.45 h of training time, no errors; finished 23:42.
Iterations took ~7.5 min at 48 simulations and ~8.3 min at 64 (E17: ~6.2 min at
32); GPU memory stayed at 14.8 of 16 GB. Hourly eval vs. E17: 0.36 (it 273, the
dip after the LR jump), 0.49, 0.48, 0.59 and 0.59 (it 298). Policy loss rose
from 0.64 to 0.95 while the LR was high (sharper targets) and ended at 0.75.

tdgauntlet, E18 it 298 vs. E17 it 266, 128 sims, 400 games (`results/gess_e18_v_e17.json`):
**213-71-116, 0.621 +/- 0.043 per game (+86 Elo)**. 0.645 as first player, 0.598
as second; 0.72 in the 211 ring-breaking games and 0.51 in the 189 20-move endings.
Draws 17.8%.

**Conclusion.** The plateau was not the network's ceiling: 4.45 h of this recipe
gained +86 Elo where E17's 8.6 h gained +14. The run changed two things at once,
so it does not say whether the deeper search or the re-warmed LR did it. The gain
is almost entirely in games decided by breaking a ring, i.e. tactics, which is
what deeper targets should improve; 20-move endings stayed level. E18 replaces
E16 as the web app's model. Next: a longer run on the same settings, or a
re-warm at 32 simulations to separate the two causes.

## E19/E20. Clock planes again, as a matched pair from E18

**Question.** With ~half of match games now ending by the 20-move rule and E18's
gain confined to ring-breaking games, do the v2 clock planes help at this
strength? (E14 vs. E15 tested them at E13's level: level.)

**Setup.** Two new runs from E18 it 298 (`resume_as_new_run=true`, which restores
E18's buffer and games but writes elsewhere), each 34 iterations to 332 on E18's
settings (64-simulation full searches, LR re-warmed to 2e-4, cosine to 1e-5:
`learning_rate=7.3815e-3 lr_final_ratio=1.355e-3`). E19: `obs_planes=6`, the stem
widened on resume with zero weights and zero Adam moments for the two new planes
(checked on CPU: outputs identical to E18's). E20: `obs_planes=4`. Hourly eval vs.
E18. New metric `selfplay/clock_end_rate` (share of finished games ending by the
20-move rule). E19 `checkpoints/gess_20260929163401`, E20
`checkpoints/gess_20260929214955` (wandb `jwkrka8y`). `run_e19_e20.sh`.

**Result.** Both finished without errors, 4.81 h (E19) and 4.78 h (E20) of training
time, so the planes cost almost nothing per iteration. Hourly eval vs. E18 at
the end: E19 0.72, E20 0.54 (128 games each at 32 sims; noisy). E19's clock-plane
weights stayed tiny all run (RMS ~0.003-0.004 against 0.12-0.20 for the other
inputs). Self-play clock endings fell from ~28-31% to 21% (E19) and 27% (E20).

tdgauntlet E19 vs. E20 (`examples/gess_e19_v_e20.toml`, 128 sims): the first
attempt crashed WSL after 262 of 400 games, at **119-26-117 (0.504)**; a rerun was
stopped at 123 games, 59-16-48 (0.545). No full result.

**Conclusion.** Any benefit from the planes is small at this strength; the
network barely uses them. Kept anyway (user decision): they cost under 1% of
training time and may pay off in a stronger network, so the line continues from
E19 with `obs_planes=6`; E18 stays the web app's model until a successor is
matched against it. Replaying
E18 vs. E17's 20-move endings showed why they are hard to exploit: the loser's
last move usually had no capture that kept their own ring, and a self-capture only
widens the deficit, so the search rightly sees these positions as lost; in 4 of
118 games a levelling capture was missed.

## The long-capture benchmark

`gess_slide_diag.py` (CPU, ~20 min): 2000 positions sampled with a fixed seed from
`tdgauntlet/results/gess_e18_v_e17.json`; every capturing move (opponent stones
taken, own ring kept) and the move played are scored by a 64-simulation search of
the resulting position with the checkpoint's own network, and the raw network's
ranks of each capture's piece (stage 0) and destination (stage 1) are reported by
slide distance (`gess_slide_report.py`). E19 (`gess_slide_e19.json`): for captures
at least as good as the move played, the capturing destination was the network's
first choice for its piece 92-94% of the time at distances 1-3, **13% at 4-6 and 0%
at 7+** (median rank 7th of 8, 12th of 14). In 323 of 2000 positions a capture was
clearly better (>= 0.2) than the move E18/E17 played; 285 of those were slides of 4+.
A 512-simulation recheck of 160 of them kept 92% at >= 0.2 better.

Supervised fine-tuning from E19 on these captures (CPU, 300 updates) taught the
network to rank held-out long captures first 100% of the time with or without a
joint policy head, so the architecture could represent them: the self-play
training signal was the problem (E22).

## E21. Joint piece->destination policy head (version 1), from E19

**Question.** Does a policy term scoring (source, destination) pairs (features of
both cells plus a learned bias per relative offset; stage 0 takes a log-sum-exp
over a source's destinations, stage 1 its row) fix the long-capture blind spot?

**Setup.** `resume_as_new_run` from E19 it 332, `gf_joint_head=1` (the term scaled
by two gates starting at zero), 34 iterations to 366 on E19's settings,
`train_micro_batches=4`. Interrupted at 346 for a reboot and resumed. Run
`wg0172pa`, `checkpoints/gess_20260930055738`.

**Result.** The gates never moved (~0.001 after 34 iterations): gate and score were
both ~0 at the start, so neither had a useful gradient. The benchmark was unchanged
(4-6: 12.8% first). E21 vs. E19, 400 games: **198-51-151, 0.559 per game (+41 Elo)**
- in effect a plain 5 h continuation, the control for later runs. Version 2 (query
projection at zero, key at a normal init, no gates, log-mean-exp; LoRA-style)
starts function-preserving and trains, but the supervised test above showed it is
not needed. Both stay as options (`gf_joint_head`), off by default.

## E22. Destination targets follow the search: per-stage value_scale

**Question.** Why does self-play never teach long captures? Probe (64-sim searches
from E21 on 256 positions with a confirmed good long capture, piece already
chosen): the training target ranked the capture first in 2% at mctx's default
`value_scale=0.1`, 6% at 0.5, 36% at 1.0, 58% at 2.0. At 0.1 a move can gain at most
~7 logits over its prior, and these captures start further down. But a large
value_scale at the piece step makes those targets near-one-hot on ~3 noisy visits
(median KL from the 0.1 target 5.5 at 2.0); at the destination step, where every
destination is searched, it moved ordinary targets little (KL 0.15).

**Setup.** `value_scale_by_stage=0.1,2.0` (self-play only; the node's stage is read
from the tree's embeddings): 2.0 at destination nodes, 0.1 at piece nodes; checked
on CPU (destination targets rank the capture first 56%, piece targets KL 0.01).
`resume_as_new_run` from E21 it 366 (joint head dropped), 34 iterations to 400,
E19's settings. Run `kibrn1oc`, `checkpoints/gess_20260930235206`.

**Result.** No dip: hourly eval vs. E21 0.66, 0.73, 0.82, 0.92, 0.81. Self-play games
ending on the 20-move rule fell from ~16% to ~4%, draws to 1-2%. Benchmark: capturing
destination first for its piece at 4-6 **83%** (E21 13%), 7+ **76%** (0%); pieces in the
top 16 98-100%. E22 vs. E21, 400 games: **380-0-20**, median game 30 moves (E22 plays
long slides from the opening; E21's eval stays ~0 until it collapses). E22 vs. E18:
79-1. Deployed to gesstest (replacing E18).

**Against an outside opponent.** The web app's negamax engine at 2 s a move
(single-threaded; `tdgauntlet/clients/gess_negamax`, depth ~5): E22 at 128 sims
0.375 (80 games) and 0.56 (32) on other openings, E18 0.125 (80) - so E22 is ~+250
Elo over E18 measured through negamax, far less than 79-1 suggests, and still below
negamax. More search did not help: 512 sims 0.50, 2048 sims 0.38 (32 games each,
+/-0.17). Refereeing E22's 50 losses with negamax (comparing its move with
negamax's best at the same depth, opponent to move in both): 44 contained a move
into a forced loss, 24 of them a mate-in-one, with E22's own eval median +0.56 at
the blunder. E22's network ranked the opponent's killing reply well (piece top 16
84%, destination first 68%), but its 64-sim search playing that side found it only
32% (46% for immediate wins): a candidate piece gets ~1 visit, which only reaches
the "piece selected" node and is judged by the value head. Two levels per move puts
a ring-breaking win at depth 2 and its refutation at depth 4.

## E23. Auxiliary immediate-win target

**Question.** Can a rules-labelled training target teach the network (and through
the shared trunk, the value head) to see ring-breaking wins?

**Setup.** `aux_win_weight=1.0`: pgx `gess.winning_actions` labels, for every
self-play position, the actions that win at once by breaking the opponent's last
ring (exact against brute force on 800 positions; 9 ms per 1024 positions on the
GPU); an extra GessFormer head (training only) predicts them per action plus "any",
with output biases at the base rates. `resume_as_new_run` from E22 it 400, 34
iterations to 434, otherwise E22's settings. Run `p5ggab6z`,
`checkpoints/gess_20261001235654`.

**Result.** Aux loss 0.138 -> 0.012. Self-play's missed immediate wins
(`selfplay/win_missed`) fell from 6.8% to 2.1% of positions with one. Hourly eval vs.
E22: 0.39, 0.50, 0.72, 0.68. tdgauntlet, 128 sims, 60 openings x both colours
(`results/gess_e23_negamax.json`): E23 v E22 **88-1-31 (0.74)**; E23 v negamax
**51-69 (0.43)**; E22 v negamax 55-65 (0.46) - level against negamax (+/-0.09).

**Conclusion.** The target did its job inside self-play but not against negamax.
Since E22, gains within the line (head to head) have not transferred to the outside
yardstick; judge future runs against negamax first. Next: make one simulation cover
a whole move (piece + destination), so the search sees as far in moves as its depth
suggests.

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
