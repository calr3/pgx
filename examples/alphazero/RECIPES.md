# Per-game recipes

What is actually known to work, per game, and why. The detailed evidence lives
in `GESS_EXPERIMENTS.md` and `PIG_EXPERIMENTS.md`; this file is the summary you
should start a new run from.

**Evidence standard.** A setting is listed as adopted only if it won a
comparison at **equal wall-clock time** (not equal iterations: recipes differ in
cost per iteration, and most of the gain arrives late as the cosine schedule
decays). Gess comparisons are Elo ladders from `elo_ladder.py`; pig is scored
directly against exact optimal play by `pig_optimal.py`.

**A head-to-head, never the training curves.** Self-play statistics describe the
data a run generates, not how well it plays. Two Epaminondas arms here finished
with lower policy loss, shorter games and fewer draws than the arm they were
measured against, and lost by 575 and 95 Elo respectively. If a change is only
supported by its curves, it is not supported.

Sections whose heading says **PROPOSAL** are untested design sketches. Don't
quote them as if they were results.

## Defaults that transfer between games

These won on Gess and have no game-specific content, so start any new game with
them. Elo figures are Gess pilots at ~1.2 h each.

| Setting | Why |
|---|---|
| `continue_games=true`, `replay_buffer_iters=4`, `lr_schedule=cosine` | The data pipeline: every position gets a real value target instead of a truncated one. **+443 Elo**, the single biggest win. |
| `selfplay_bf16=true` | ~2x faster search inference; picks the same move 96-97% of the time. **+121 Elo**. |
| `playout_cap_prob=0.25`, `fast_num_simulations=8` | Full search on a quarter of moves for policy targets, cheap search elsewhere. Small gain. |
| `num_updates_per_iter` for ~4x sample reuse | Free at full size, where training is ~4% of an iteration. |
| AdamW: `weight_decay=1e-4 warmup_steps=100-500 grad_clip_norm=1.0` | Needed for stable transformer training; harmless elsewhere. |
| `num_simulations=32` | Halving it to 16 lost **-103 Elo** at equal time: target quality beats game count. |

Two things are **not** general defaults:

- `symmetry_augmentation=true` is Gess-only in the code, and requires the game to
  actually be symmetric - verify with a check across all 8 symmetries (legal
  masks, observations and resulting boards) before trusting it. **+246 Elo** on
  Gess.
- `qtransform=completed_unscaled` matters when a node offers **few actions whose
  Q difference is itself the quantity of interest** - see pig below. It is not a
  rule about small action spaces as such: on Epaminondas, whose stage-1 and
  stage-2 nodes have a median of 3 and 2 legal actions, it lost a head-to-head
  **119-0 (Elo -575)** and drove self-play to drawing 96% of its games. Pig's two
  actions are hold and roll, and the unscaled gap between them is a real
  difference in win probability; Epaminondas's are continuations of one move,
  where the spread is noise and only the rank carries signal. Check which case
  you are in before changing it.

## Gess

```sh
python3 -u examples/alphazero/train.py env_id=gess architecture=gessformer \
  selfplay_bf16=true num_simulations=32 playout_cap_prob=0.25 \
  fast_num_simulations=8 symmetry_augmentation=true continue_games=true \
  selfplay_batch_size=1024 max_num_steps=256 training_batch_size=4096 \
  num_updates_per_iter=256 replay_buffer_iters=4 \
  learning_rate=5e-4 weight_decay=1e-4 warmup_steps=500 grad_clip_norm=1.0 \
  lr_schedule=cosine save_data_state=true max_num_iters=72 eval_interval=4
```

**Architecture: `gessformer`** (+420 Elo over the ResNet). Gess needs both 3x3
locality (a piece is a 3x3 footprint) and long-range interaction (pieces slide up
to 17 cells), so it pairs a full-resolution conv stem with a transformer over
2x2-merged tokens, and decodes back through a skip connection to a
source->destination policy head over the 20x20 action grid.

`rayformer` (attention along rows/columns/diagonals) learns more per game but is
~50% slower per iteration and loses at equal time. Not adopted.

Estimated total gain over the original ResNet recipe: **~+600 to +1,000 Elo**
(summing the adopted steps; not measured end-to-end in a single ladder).

Size `max_num_iters` so the cosine schedule finishes inside the time budget -
strength climbs steeply in the last third, so a schedule that does not complete
wastes most of the run. On the 16 GB GPU an interrupted-and-resumed full-size
run needs `train_micro_batches=2`.

## Epaminondas

```sh
python3 -u examples/alphazero/train.py env_id=epaminondas architecture=boardformer \
  selfplay_bf16=true num_simulations=32 playout_cap_prob=0.25 \
  fast_num_simulations=8 continue_games=true \
  selfplay_batch_size=256 max_num_steps=384 training_batch_size=2048 \
  num_updates_per_iter=32 replay_buffer_iters=4 \
  learning_rate=5e-4 weight_decay=1e-4 warmup_steps=100 grad_clip_norm=1.0 \
  lr_schedule=cosine eval_interval=40 max_num_iters=160
```

**Architecture: `boardformer`**, GessFormer generalised to any even-sized board
with one action per cell. It transfers to Epaminondas's 14x12 board and 168
actions without change.

Status: nine runs (`EPAMINONDAS_EXPERIMENTS.md`). The block above is still the
winning arm, and it is a **pilot-scale** recipe - keep `selfplay_batch_size=256`.

**Compare v6 runs against a v6 baseline, not against E7.** E7 is the only strong
Epaminondas checkpoint trained under **v5** rules, and every v6 run measured so
far lands 120-140 Elo below it - including E13, which is E7's exact recipe with
a different seed and nothing else changed. That gap is the capture clock, now
measured three times (95 and 112 Elo in earlier experiments, ~140 here), not a
fault in the runs. **The correct baseline for a v6 run is about -130 against
E7.**

**Scale the learning rate with the batch.** This is the one setting that decides
whether more compute buys anything. At `selfplay_batch_size=1024` /
`training_batch_size=4096`, use **`learning_rate=1e-3`**, not the pilot's 5e-4.
Direct head-to-heads, all v6 and 7 planes:

| comparison | Elo |
|---|---|
| E9 (batch 1024, lr 5e-4) vs E13 (pilot batch 256) | **-33** |
| E14 (batch 1024, lr 1e-3) vs E13 (pilot batch 256) | **+38** |
| E14 vs E9 (learning rate isolated, same seed) | **+63** |

Keeping the pilot's learning rate at 4x the batch is an under-stepping
optimizer, and it cost the entire benefit of scaling. Fixing it is worth ~70
Elo. 1e-3 is not known to be optimal - a linear rule would suggest 2e-3, untested.

Temper expectations on the size of the win: E14 spent **2.5x E13's wall clock
for +38 Elo**, at ~1.8 sigma. Scaling works, but the return on compute is
moderate.

**Compare runs head to head, never by subtracting their scores against a third
model.** Two claims in this file were wrong in opposite directions because of
that - "scaling is broken" (a v6 run measured against v5 E7) and "scaling is
fine" (E9 and E13 compared through E7).

**Measure against `tdgauntlet`'s alpha-beta, not only against our own
checkpoints.** Every Elo in this file is relative to another of our runs. Against
the Rust alpha-beta in `~/calr3gh/tdgauntlet`, E7 and E14 score **-311 and -257
Elo** and lost every counted minimatch (77-0-3 over 80 games), while being given
5-6x its thinking time. The family is a long way below a competent classical
engine, and no 2x-scale run closes that. It is also a fixed external opponent, so
it is the right yardstick for future runs.

**v6 gives black a large advantage in pgx at 32 simulations, and not in
tdgauntlet at 256.** In pgx self-mirror
matches at 32 sims: E7 (v5) 0.531, E13 (v6) 0.625, E14 (v6, strongest) 0.727 -
independent of the observation planes and not explained by game length. But in
tdgauntlet at 256 sims white won 0.475 of the strength-balanced pairing, showing
no such edge. Search size, self-mirror versus distinct opponents, and the harness
all differ; which regime is representative is untested. It contaminates no Elo
either way, since every pairing is played seat-swapped.

**Do not add an observation plane derived from the tiebreak.** Measured against
E13 (the matched v6 baseline), the v8 signed-clock plane costs **210-270 Elo**:
E11 -351 and E12 -411 where E13 is -140, on identical settings at matched wall
clock. Two seeds, so not seed luck. Any such plane carries the "black wins an
exact mirror" default, which is a colour signal in every near-symmetric
position. v9 unwires it, and a test asserts both sides see an identical
observation in a mirrored position.

Note the motivation was also measured in the wrong regime: "69% of games are
decided by the clock" came from *random* play.

**A mirror seat asymmetry is not by itself evidence about the observation.**
E13 has no plane and still shows black at 0.625, because its mirror games run
498 plies and reach the tiebreak, which favours black. The plane's own leak is
visible only where games are too short for that - E11's mirror averaged 76
plies and still showed 0.773. Check game length before attributing an asymmetry.

**Measure seat effects with a model against itself, never with a head-to-head.**
A weak model loses from both seats, which hides its asymmetry: E11's head-to-head
showed a 0.016 gap while its mirror showed 0.546.

**Scaling detail.** E9 ran these settings at**Scaling detail.** E9 ran these settings at
`selfplay_batch_size=1024` / `training_batch_size=4096` /
`num_updates_per_iter=64` for 3.29 h under v6 rules, and lost to the 1.3 h E7
pilot **118 Elo** (0.336 +/- 0.030 over 256 games) despite 4x the batch, 2.5x
the wall clock and 4x the data. Two candidate causes are confounded in that run
- v6's capture clock, and a 4x batch left at `learning_rate=5e-4` - and neither
has been isolated. Do not run full-size Epaminondas expecting the pilot's
quality until one of them is.

Sizing figures from E9, at batch 1024 on a 16 GB GPU: ~95-104 s/iteration
(self-play ~90 s of it), 7.88 GiB replay buffer at `replay_buffer_iters=4`,
~14.5 GiB of GPU (no `train_micro_batches` needed), ~30 GB peak host RAM, and a
14.1 GB `data_state.pkl`. The last two are inflated by v6's long games.

The decisions behind the pilot settings:

- **`num_simulations=32` is tested here, not just inherited.** 64 lost at equal
  time - -49 Elo even against the E7 checkpoint it out-spent by 5%, and -184
  against the one at matched time (E8). Gess separately found 16 costs 103 Elo.
  Both games are worse either side of 32, so treat it as a real optimum and
  resist raising it when given more hardware: doubling the search halves the
  games, and the games win.

- **Keep the default `qtransform`.** `completed_unscaled` lost 119-0. See the
  note above - this is the one place a plausible reading of the pig result would
  send you wrong.
- **The game rules mattered more than any hyperparameter.** Reaching the move cap
  used to be a draw, which made stalling safe; deciding it on material (v1) took
  draws from 83% at iteration 40 to zero by iteration 20, and raised completed
  games per iteration from ~120-300 to ~1,400.
- **A 60-move capture clock cost ~100 Elo (v2, v3).** Ending the game 60 quiet
  moves after the last capture lost three times - -95 Elo against a material
  tiebreak, -112 against the advancement one, +95 the other way once an absolute
  cap replaced it. Epaminondas's manoeuvring phases are longer than 60 moves, so
  it cut games off before a full strategic arc.
- **v6 is a capture clock again, at 100 moves - and it is now under suspicion.**
  It was adopted as a design decision, on the view that an absolute cap pays
  whoever is ahead to run the clock out and that 60 was too short a horizon
  rather than the idea being wrong. The only run under it (E9) lost by 118 Elo,
  which would make capture clocks 0 for 4 (-95, -112, +95 when one was removed,
  -118). That result is confounded with a scale-up, so it is not yet a verdict,
  but if you need a rules choice today, **v5 is the one with a winning record**.
  The mechanism to watch is `selfplay/games_finished`: under v6 it fell 1,616 ->
  ~500 per iteration as the net strengthened, because stronger play keeps
  resetting the clock with occasional captures, so self-play spends its budget
  on games that never resolve.
  For tournaments, use `max_num_steps=3000` and check the truncated count is
  zero. v6 has no absolute bound on game length (worst case ~5600 moves) and 28%
  of *random-play* games exceed 900 actions, but trained play is far shorter -
  E9 vs E7 averaged 334 plies with 0 truncations.
- **The tiebreak costs nothing.** v5 against v1 - the same horizon, advancement
  scoring and no draws instead of material - is +16 Elo, half a standard error
  from parity. The rules were adopted for being right, not for strength, and
  they are free.
- **v3 decides the cap on advancement instead of material**, comparing piece
  counts rank by rank from the opponent's home rank inwards. Material was the
  wrong quantity - the game is won by getting up the board, not by hoarding
  pieces - and, since equal material is common and equal advancement is not, it
  also made draws common. This removed v2's cost entirely: no degenerate early
  phase, draws at 0.000 from iteration 5, where v2 drew 99% of games for 40
  iterations.
- **v4 makes draws impossible**: an exact rank-for-rank mirror falls back to
  whoever captured last, then to black for moving second. A draw is the one
  outcome a stalling player can aim at without being better, so removing it
  closes the loop that v0 opened.
- Do not compare checkpoints trained under different rule versions as if the
  rules were fixed; `env.version` records which is which.

Sizing notes that cost time to learn:

- `max_num_steps` is in **steps, not moves**: a move costs three here (lead,
  rear, destination). 192 covered a fifth of a game. Watch
  `train/value_target_fraction` - it should sit at 1.00.
- Estimate iteration cost from **steady state**: iterations 1-3 took 55/122/49 s
  against 25.4 s steady, and sizing off them made a planned 1-hour pilot run 15
  minutes.
- `eval_interval` gates **checkpointing**, not just evaluation. Keep it a divisor
  of `max_num_iters` if you want intermediate checkpoints at round numbers.
- For tournaments here, pass `max_num_steps=900` (games run long) and
  `random_opening_plies` in multiples of 3, so a random opening ends on a move
  boundary instead of mid-move.

Symmetry augmentation is not wired up for this env (the board is not square, so
only the horizontal reflection applies).

## Pig

```sh
python3 -u examples/alphazero/train.py env_id=pig architecture=mlp \
  mlp_onehot_bins=101 mlp_width=256 mlp_layers=3 \
  qtransform=completed_unscaled \
  selfplay_batch_size=1024 num_simulations=32 max_num_steps=256 \
  training_batch_size=4096 replay_buffer_iters=4 \
  learning_rate=1e-3 weight_decay=1e-4 warmup_steps=200 grad_clip_norm=1.0 \
  lr_schedule=cosine max_num_iters=60
```

**Architecture: `mlp`.** Pig's observation is six counters, not a board, so
nothing convolutional applies. `mlp_onehot_bins=101` one-hot encodes each counter
next to its scaled raw value: the value function turns sharply on thresholds
("does banking now reach 100?"), which a near-tabular input represents easily,
while the raw values still give it something smooth to generalise along.

**`qtransform=completed_unscaled` is the critical setting** - it tripled the
score against optimal play (0.100 -> 0.298) at equal wall clock. With two
actions, mctx's default rescaling of Q to the range across a node's actions
leaves only the *sign* of the advantage in the policy target. Expect this to
matter for any game where few actions are legal.

`chance_samples` (averaging each search edge over several chance outcomes) was
tried and **hurt** while targets were sign-quantised; it has not been retried
since the q-transform fix. Not currently recommended.

Scoring: `pig_optimal.py` measures against the exact solution. Reference points
for its seat-balanced win rate: optimal 0.500, hold-at-20 0.461, always-roll
0.000; the best model so far is 0.298 from its policy head and 0.448 when its
value head is played by one-ply expectimax.

## Backgammon - PROPOSAL, NOTHING RUN YET

Nothing below has been trained or measured. It is a design sketch from reading
`pgx/backgammon.py` and from what pig taught us about stochastic games; treat
every claim as a hypothesis to test, not a recommendation.

What the env gives you: a 156-wide action space (`6 * 26`, encoded
`src * 6 + die`, `pgx/backgammon.py:37`), a 34-int observation (28 signed board
entries - 24 points, bar and off for each side - plus a 6-vector of playable dice
counts), **one die per step**, so a turn is 2-4 stages like Gess and
Epaminondas, and rewards of **±1 / ±2 / ±3** for normal, gammon and backgammon
wins (`_calc_win_score`, `pgx/backgammon.py:469`).

### Two blockers to clear before any architecture work

1. **The value head cannot represent a gammon.** Every architecture here ends in
   `tanh` (range ±1, `network.py:160,236,350`) while `value_tgt` would be up to
   ±3, and the l2 loss (`train.py:294`) would hold it saturated on exactly the
   positions that decide matches. Either scale the head (`3 * tanh`) or give it
   the outcome distribution strong bots use - win / gammon / backgammon for each
   side, trained with cross-entropy, with the expectation passed to the search.
   This is a real limitation for **any** env whose rewards leave [-1, 1], not
   just backgammon.
2. **Chance handling, which is pig's problem with 21 outcomes per turn instead
   of 6.** The search keeps one sampled successor per edge forever. The classical
   fix is to evaluate **afterstates** - the position after your move but before
   the opponent's roll, which is how TD-Gammon got away with a tiny network - so
   the node the network scores should be the end-of-turn position, with the roll
   as an explicit chance node. Also expect to need
   `qtransform=completed_unscaled`: few actions are legal per die, which is the
   regime where the default rescaling destroys the signal.

### Sketch: a TD-Gammon-style MLP first

`architecture=mlp` plus a thermometer encoding (a signed-count sibling of
`mlp_onehot_bins`: per point, own checkers as 1/2/3/4+, the same for the
opponent, and blot / made-point / spare flags), two hidden layers of 256. Signed
integer counts are a poor network input; thermometers make "blot vs point vs
prime" linearly available. This is roughly the 1992 design that first played
backgammon well, it is an hour's work, and it gives the token model below
something to beat.

### Sketch: "GammonFormer", the Gess machinery on a 1-D track

- ~30 tokens: 24 points, bar and off per side, and a dice token. Full attention
  over 30 tokens is free, so none of the 2x2 patch merging GessFormer needed;
  a plain pre-LN transformer, 4-6 layers x 128-192.
- **Relative-distance attention bias** bucketed by signed pip distance: ±1-6 is a
  direct shot, 7-12 indirect. This is the Geometric Attention Bias idea reduced
  to a cheap learned 1-D table, and it is where backgammon's structure lives.
- A 1-D conv stem, kernel 13 (±6), so priming and shot patterns are available
  without attention having to discover them.
- **Policy head scored on (source, destination) pairs** rather than src x die:
  the destination is determined by `src + die`, and what matters about a move is
  where the checker lands - hitting a blot, making a point, breaking an anchor.
  This is GessFormer's source->destination head, which should transfer directly.

Later refinement, if it matters: GNU Backgammon uses **separate networks per game
phase** (contact / race / crashed), a pure race being a different function
altogether. The cheap version is a phase feature and a phase-conditioned head
rather than three networks.

Expectation to test, stated up front so it can be wrong: the value-head fix and
the afterstate handling should be worth more than any amount of trunk capacity.

## Choosing settings for a new game

1. **Architecture.** A board with one action per cell -> `boardformer`. A flat
   vector of counters -> `mlp` (add `mlp_onehot_bins` if the values are small
   integers with sharp thresholds). Anything else -> `resnet` first, as a
   baseline to beat.
2. **Action space.** A small action space is *not* on its own a reason for
   `qtransform=completed_unscaled` - it helped pig and lost 119-0 on
   Epaminondas. Ask whether the Q difference between a node's actions is the
   real quantity being decided (pig: hold vs roll) or an artefact of how one
   move resolves (Epaminondas: which rear, which destination). Test it; do not
   assume it.
3. **Step budget.** `max_num_steps` must comfortably exceed a full game in
   *steps*, which is moves x stages for multi-stage turns. Check
   `selfplay/steps_per_game` and `train/value_target_fraction` in the first
   iterations: a low fraction means truncation is starving the value targets.
4. **Randomness.** Stochastic envs work without special flags, but a search that
   samples one successor per edge keeps that sample's variance forever. See the
   pig log before trusting search quality in a stochastic game.
5. **Reward scale.** Every value head ends in `tanh`, so an env whose rewards
   leave [-1, 1] (backgammon's gammons, for instance) needs the head rescaled or
   replaced by an outcome distribution, or it trains saturated.
6. **A baseline to evaluate against.** `train.py` needs
   `pgx.make_baseline_model("<env>_v0")`. A scripted opponent of known strength
   (pig's hold-at-20) is far more informative than an untrained network.
7. Then run equal-time pilots and record them in a `<GAME>_EXPERIMENTS.md`.
