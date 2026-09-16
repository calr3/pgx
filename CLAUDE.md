# Working notes for this repo

Pgx fork used to train AlphaZero agents (Gess, plus Epaminondas and Pig envs
added here). What follows is what was learned the hard way.

- **Starting a run for a game: `examples/alphazero/RECIPES.md`** - the
  architecture and settings known to work per game, and the evidence for them.
  Keep it current: when an experiment changes what you would recommend, update
  that file, not just the experiment log.
- Experiment history: `examples/alphazero/GESS_EXPERIMENTS.md`,
  `examples/alphazero/PIG_EXPERIMENTS.md`. The forward plan for the TPU rental
  is `examples/alphazero/GESS_TPU_PLAN.md`.

## Environment

- `import pgx` resolves to **this** checkout (editable install). It used to point
  at `~/githubit/pgx`, so edits here had no effect at runtime — check
  `python -c "import pgx; print(pgx.__file__)"` if behaviour contradicts the code.
- **JAX and its CUDA plugin must stay in lockstep** (`jax`, `jaxlib`,
  `jax-cuda12-*` all 0.8.2 today). Installing a package that upgrades JAX
  silently drops the GPU. Install extra tooling (e.g. `jax2onnx`) into a
  throwaway venv instead, and verify `jax.devices()` afterwards.
- Angular work in the sibling repo `~/calr3gh/gess` needs Node >= 20.19; the
  default `node` is 20.18, so put `~/node22/bin` on PATH.

## Running training

- Run from the repo root: `python -u examples/alphazero/train.py env_id=... k=v`.
  Without `-u`, output is block-buffered and progress looks stalled.
- Config lives in `examples/alphazero/config.py`; every field is a CLI key.
  Checkpoints land in `checkpoints/<env>_<timestamp>/`.
- Long runs: launch detached (`nohup setsid ... &`), record the PID, and watch
  the log. Harness background tasks get killed by a low-memory guard even with
  tens of GB free; detached processes survive. Monitors also expire after ~30
  minutes, so re-arm them for anything longer.
- **Watch processes by PID, never by `pgrep -f <pattern>`.** A watcher's own
  command line contains the pattern, so it matches itself. Both failure modes
  happened here:
  - `until ! pgrep -f "train.py env_id=gess..."; do sleep 30; done` never
    exited, because `pgrep` kept finding the watcher. A tournament queued behind
    such a loop never started and the GPU sat idle for 35 minutes.
  - `pkill -f epam_equiv` killed the calling shell (and the running monitors)
    before the rest of the command ran, since the pattern appeared in its own
    argument list.

  Instead: write the PID when launching (`echo $! > run.pid`) and poll with
  `kill -0 $PID`. Note `nohup setsid cmd &` gives the PID of `setsid`, which
  exits immediately — capture the real one with `pgrep -af` once, or have the
  launcher script record it. To stop something, `kill` that PID.
- Resume with `resume_from=<ckpt>`; add `save_data_state=true` to also restore
  the replay buffer, held-back steps and in-progress games (verified
  bit-identical to an uninterrupted run). On the 16 GB GPU a resumed full-size
  run needs `train_micro_batches=2`, or the batch-4096 training step fails to
  find a ~10 GB block.
- JAX preallocates ~75% of the GPU; a single large allocation can fail even when
  totals look fine. Keep sample data on the host (the replay buffer and
  minibatch gathering already do).

## Measuring strength

- `eval/vs_baseline/*` in wandb (raw policy sampling) is far too noisy to rank
  runs; use it only as a coarse trend.
- Real comparisons: `model_tournament.py` for two checkpoints,
  `elo_ladder.py` for a round robin with an Elo fit (results cached in
  `elo_gess_*.json`, keyed by checkpoint paths, tournament settings and
  `game_version`).
- **Bump `game_version` in `elo_ladder.py` whenever game generation or the rules
  change**, so old cached games are replayed rather than mixed in.
- Compare at **equal wall-clock time**, not equal iterations: recipes differ in
  cost per iteration and most gains arrive late, as the cosine schedule decays.
- Elo from these ladders is not fully transitive: a model that crushes weak
  opponents can rate above one that beats it head-to-head. Quote the head-to-head
  when the direct question is "is A stronger than B?".
- `train.py` can also play a fixed opponent during training on a wall-clock
  cadence (`mcts_eval_opponent=<ckpt>`, `mcts_eval_interval_hours`), which is the
  cheapest honest progress signal. Pick an opponent near the model's level: a far
  stronger one pins the score at zero for hours.

## Adding a game

Follow `pgx/_src/games/epaminondas.py` and `pgx/epaminondas.py`:

1. **Rules module** `pgx/_src/games/<game>.py`: a `GameState` NamedTuple and a
   `Game` class with `init`, `step`, `observe`, `legal_action_mask`,
   `is_terminal`, `rewards`.
2. **Env wrapper** `pgx/<game>.py`: a `State` dataclass plus `_init`, `_step`,
   `_observe`, `id`, `version`, `num_players`.
3. **Register** in `pgx/core.py`: add to `EnvId` and to `make()`.
4. **Visualizer**: add a branch in `pgx/_src/visualizer.py` and a drawing module
   under `pgx/_src/dwg/`. Without it `pgx.api_test` asserts.
5. **Baseline**: add an untrained entry in `pgx/_src/baseline.py`
   (`<env>_v0`), or `train.py`'s evaluation cannot start.
6. **Tests** `tests/test_<game>.py`: rules, terminal conditions, and
   `pgx.api_test(env, 3, use_key=False)`.

Design notes that mattered:

- **Multi-stage turns** (Gess: piece then destination; Epaminondas: lead, rear,
  destination) keep the action space tiny — one action per board cell — at the
  cost of several search steps per move. `current_player` flips only on the last
  stage; the observation carries stage planes and selection markers.
- Encode the whole move in the observation the network sees, so one per-cell
  policy head can serve every stage.
- Document deviations from the published rules in the module docstring (e.g. a
  move cap for a draw, or what happens when a player has no legal move).

JAX pitfalls hit while writing envs:

- Indexing a **NumPy** constant table with a traced value raises
  `TracerArrayConversionError`; keep a `jnp` copy for those lookups.
- `lax.switch` under `vmap` evaluates **every** branch; index a stacked array
  instead.
- Precompute geometry (ray tables) as constants at import time.
- Don't build per-square 4-D legality arrays: compute "can this square move at
  all?" from line lengths, and full legality only for the square actually
  chosen. That was a ~20x cost difference in Epaminondas.

## Architectures (`examples/alphazero/network.py`)

`make_forward(num_actions, config)` dispatches on `config.architecture`:

- `resnet` — the original AZNet.
- `gessformer` — conv stem, 2x2 patch merge, transformer with Chessformer's
  Geometric Attention Bias, upsample with stem skip, source->destination policy
  head. Gess-specific.
- `rayformer` — one token per cell, attention along rows/columns/diagonals.
  Learns more per game but is ~50% slower per iteration, so it loses at equal
  time. `rf_symmetric` ties its position parameters across board symmetries.
- `boardformer` — GessFormer generalised to any even-sized board with one action
  per cell (used for Epaminondas).

Only LayerNorm, so `model_state` stays empty and checkpoints are just params.
`selfplay_bf16=true` runs search inference in bfloat16 (training stays float32).

## Getting a strong model

Ranked by what actually helped (equal-time pilots, ~1.2 h each):

1. **Data pipeline** (+443 Elo): `continue_games=true` so every position gets a
   real value target, a replay buffer, and cosine LR decay. Biggest single win.
2. **Architecture** (+420): GessFormer over the ResNet.
3. **Symmetry augmentation** (+246 for Gess): train on a random one of the 8
   board symmetries. Verify the game really is invariant first — the Gess check
   compared legal masks, observations and resulting boards across all 8.
4. **bfloat16 self-play** (+121): ~2x faster inference, and search picks the same
   move 96-97% of the time.
5. **Playout cap randomization** (small): full search on 25% of moves for policy
   targets, cheap search elsewhere.
6. **Sample reuse**: ~4x is free at full size (training is ~4% of an iteration
   there, ~35% at pilot size).

Rejected: halving simulations to 16 (-103 Elo at equal time — target quality
beats game count).

Practicalities:

- Self-play is ~95% of iteration time at full size, and network inference is
  essentially all of self-play. Optimise throughput there, not the optimiser.
- Size `max_num_iters` so the cosine schedule finishes inside the time budget;
  strength climbs steeply in the last third.
- Rising `train/value_loss` is not necessarily bad: it tracked draws
  disappearing from self-play.
- Verify every refactor of the data path is **bit-identical** on a small run
  (params, optimizer state and RNG) before trusting it.
- Watch the evaluation itself: a bug that let random openings end the game gave
  both sides free wins, pinned one metric at a constant, and compressed every
  Elo gap by ~13%.

## Shipping a model elsewhere

`examples/alphazero/export_onnx.py` (`dump` / `convert` / `verify`) exports a
checkpoint's network to ONNX — opset 20, standard ops, ~29 MB for GessFormer,
matching JAX to ~0.01 on logits of scale 35. The browser app in `~/calr3gh/gess`
consumes it with onnxruntime-web (WebGPU, WASM fallback); its `verify/` scripts
check the TypeScript observation encoder and the exported model against
positions exported from here. Rules and search still have to exist on the target
platform — the export is the network only.
