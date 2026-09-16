# Per-game recipes

What is actually known to work, per game, and why. The detailed evidence lives
in `GESS_EXPERIMENTS.md` and `PIG_EXPERIMENTS.md`; this file is the summary you
should start a new run from.

**Evidence standard.** A setting is listed as adopted only if it won a
comparison at **equal wall-clock time** (not equal iterations: recipes differ in
cost per iteration, and most of the gain arrives late as the cosine schedule
decays). Gess comparisons are Elo ladders from `elo_ladder.py`; pig is scored
directly against exact optimal play by `pig_optimal.py`.

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
- `qtransform=completed_unscaled` matters when **few actions are legal** - see
  pig below. The default rescales Q to the range across a node's actions, which
  is fine with hundreds of actions and destructive with two.

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
  selfplay_batch_size=256 max_num_steps=192 training_batch_size=2048 \
  num_updates_per_iter=32 replay_buffer_iters=4 \
  learning_rate=5e-4 weight_decay=1e-4 warmup_steps=100 grad_clip_norm=1.0 \
  lr_schedule=cosine max_num_iters=35
```

**Architecture: `boardformer`**, GessFormer generalised to any even-sized board
with one action per cell. It transfers to Epaminondas's 14x12 board and 168
actions without change.

Status: one ~15 min pilot only, so these are **starting settings, not a
validated recipe**. Two things to fix before a real run:

- `max_num_steps=192` was too short - games rarely finished inside the window, so
  value targets were starved. A move costs three steps here (lead, rear,
  destination), so the step budget must be ~3x the intended move count.
- Estimate iteration cost from **steady state**, not from the first iterations:
  iteration 3 took 94 s against 26 s steady, and sizing off it made a planned
  1-hour pilot run 15 minutes.

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

## Choosing settings for a new game

1. **Architecture.** A board with one action per cell -> `boardformer`. A flat
   vector of counters -> `mlp` (add `mlp_onehot_bins` if the values are small
   integers with sharp thresholds). Anything else -> `resnet` first, as a
   baseline to beat.
2. **Action space.** Fewer than roughly a dozen legal actions ->
   `qtransform=completed_unscaled`.
3. **Step budget.** `max_num_steps` must comfortably exceed a full game in
   *steps*, which is moves x stages for multi-stage turns. Check
   `selfplay/steps_per_game` and `train/value_target_fraction` in the first
   iterations: a low fraction means truncation is starving the value targets.
4. **Randomness.** Stochastic envs work without special flags, but a search that
   samples one successor per edge keeps that sample's variance forever. See the
   pig log before trusting search quality in a stochastic game.
5. **A baseline to evaluate against.** `train.py` needs
   `pgx.make_baseline_model("<env>_v0")`. A scripted opponent of known strength
   (pig's hold-at-20) is far more informative than an untrained network.
6. Then run equal-time pilots and record them in a `<GAME>_EXPERIMENTS.md`.
