# Per-game recipes

What is actually known to work, per game, and why. The detailed evidence lives
in `GESS_EXPERIMENTS.md` and `PIG_EXPERIMENTS.md`; this file is the summary you
should start a new run from.

**Evidence standard.** A setting is listed as adopted only if it won a
comparison at **equal wall-clock time** (not equal iterations: recipes differ in
cost per iteration, and most of the gain arrives late as the cosine schedule
decays). Gess comparisons are Elo ladders from `elo_ladder.py`; pig is scored
directly against exact optimal play by `pig_optimal.py`.

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

Status: four runs of ~1.3 h each (`EPAMINONDAS_EXPERIMENTS.md`). These settings
are the winning arm; the two decisions behind them:

- **Keep the default `qtransform`.** `completed_unscaled` lost 119-0. See the
  note above - this is the one place a plausible reading of the pig result would
  send you wrong.
- **The game rules mattered more than any hyperparameter.** Reaching the move cap
  used to be a draw, which made stalling safe; deciding it on material (v1) took
  draws from 83% at iteration 40 to zero by iteration 20, and raised completed
  games per iteration from ~120-300 to ~1,400. v2 counts that cap from the last
  capture rather than the start of the game.

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
