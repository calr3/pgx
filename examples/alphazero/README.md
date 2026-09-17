# AlphaZero example

A simple (Gumbel) AlphaZero [[Silver+18](https://www.science.org/doi/10.1126/science.aar6404), [Danihelka+22](https://openreview.net/forum?id=bERaNdoegnO)] example using [Mctx](https://github.com/deepmind/mctx) library. See [Pgx paper](https://openreview.net/forum?id=UvX8QfhfUx) for more details.

![](assets/pgx-az-training.png)

> [!NOTE]
> This implementation of AlphaZero demonstrates sufficient learning performance in environments including 9x9 Go, but it has some slight differences in learning details compared to the original AlphaZero and Gumbel AlphaZero. An implementation that addresses these differences and focuses on enhanced efficiency is currently under development and is expected to be released shortly.

> **Start here for a specific game:** [`RECIPES.md`](RECIPES.md) lists the
> architecture and settings known to work for each game, and what evidence backs
> them. The experiment logs behind it are [`GESS_EXPERIMENTS.md`](GESS_EXPERIMENTS.md)
> and [`PIG_EXPERIMENTS.md`](PIG_EXPERIMENTS.md).

## Usage

Note that you need to install `jax` and `jaxlib` in addition to the packages written in `requirements.txt` according to your execution environment.

```sh
$ pip install -U pip && pip install -r requirements.txt
$ python3 train.py env_id=go_9x9 seed=0
```

### GessFormer (Gess only)

`architecture=gessformer` swaps the ResNet for `network.GessFormer`, a hybrid
network adapted from Chessformer ([arXiv 2605.19091](https://arxiv.org/html/2605.19091v1)):

- a full-resolution 3x3 conv stem, for Gess's 3x3 piece footprints;
- a 2x2 patch merge to 10x10 tokens and a pre-LN transformer with Geometric
  Attention Bias, for long-range sliding moves;
- an upsample with a stem skip connection, feeding a source->destination
  attention policy head over the 20x20 action grid and a mean-pool value head.

It uses LayerNorm only (no BatchNorm). Configure it with the `gf_*` fields in
`config.py`. It trains best with AdamW, warmup and clipping:

```sh
$ python3 train.py env_id=gess architecture=gessformer learning_rate=5e-4 \
    weight_decay=1e-4 warmup_steps=500 grad_clip_norm=1.0
```

### MLP (flat observations) and games with randomness

`architecture=mlp` is for games whose observation is a short vector of counters
rather than a board, such as `pig`. It is a residual MLP; `mlp_onehot_bins=N`
one-hot encodes each feature over `[0, N)` alongside the scaled raw values,
which matters when the value function is sharply non-linear in those counters
(in pig, whether banking now reaches 100).

Stochastic environments are supported: `train.py` gives every `env.step` a
random key, so each MCTS simulation samples its own chance outcome and the
search averages over them. `pgx.make_baseline_model("pig_v0")` is the classic
hold-at-20 policy, so `eval/vs_baseline/win_rate` is directly meaningful.

```sh
$ python3 train.py env_id=pig architecture=mlp mlp_onehot_bins=101 \
    selfplay_batch_size=1024 num_simulations=32 max_num_steps=256 \
    training_batch_size=4096 replay_buffer_iters=4
```

### Training data options

All default to the original behaviour; see `config.py` for details.

- `continue_games=true`: self-play games carry over between iterations instead
  of restarting. Steps from unfinished games are held back until the game ends,
  so they get real value targets.
- `replay_buffer_iters=N`, `num_updates_per_iter=U`: train on minibatches
  sampled from the last N iterations' worth of samples, U updates per iteration.
- `lr_schedule=cosine`, `lr_final_ratio=0.1`: cosine learning-rate decay over
  `max_num_iters`.
- `symmetry_augmentation=true` (Gess only): train on each sample under a random
  one of the 8 board rotations/reflections.
- `train_micro_batches=K`: accumulate gradients over K microbatches, for large
  `training_batch_size` on limited GPU memory.
- `save_data_state=true`: also save the replay buffer, held-back steps and
  in-progress games (`data_state.pkl` in the checkpoint directory, overwritten
  at each checkpoint; can be several GB). `resume_from=` the latest checkpoint
  then continues exactly as if uninterrupted, including after Ctrl+C.
- `selfplay_bf16=true`: run the self-play search network in bfloat16 (about 2x
  faster inference for GessFormer); training and evaluation stay float32.
- `playout_cap_prob=P`, `fast_num_simulations=N`: playout cap randomization.
  Each self-play step uses the full `num_simulations` search with probability P,
  otherwise an N-simulation search whose policy targets are not trained on.
- `mcts_eval_opponent=<checkpoint>`: during training, play an MCTS match
  against a fixed checkpoint every `mcts_eval_interval_hours` of wall-clock time
  (default 1.0; also at the start and end), logged as `eval/mcts/*`. Uses the
  same openings each time, so scores are comparable over the run. Tune with
  `mcts_eval_games`, `mcts_eval_batch_size` and `mcts_eval_simulations`.

### Comparing checkpoints

`model_tournament.py` plays two checkpoints against each other with MCTS.
`elo_ladder.py` plays every pair from a list and fits Elo ratings, caching pair
results in a JSON file so that adding a checkpoint only plays its new pairs:

```sh
$ python3 elo_ladder.py env_id=gess games_per_pair=256 num_simulations=32 \
    models=a=checkpoints/run_a/000060.ckpt,b=checkpoints/run_b/000060.ckpt,c=checkpoints/run_c/000040.ckpt
```

### Playing against an alpha-beta engine

`negamax.py` is a conventional alpha-beta search with a hand-written evaluation
function — no network, no rollouts. `interactive_tournament.py` takes it as the
player type `negamax`, alongside `random`, `me` and `model`:

```sh
$ python3 interactive_tournament.py env_id=epaminondas players=negamax,model \
    models=checkpoints/epaminondas_20260918013629/000160.ckpt \
    games=20 negamax_time_s=2 num_simulations=704 verbose=false
```

Only Epaminondas has an evaluation function so far; `negamax.make_evaluator` is
where another game's would be registered. The Epaminondas one implements the six
heuristics of the LEONIDAS agent in King and Peterson, *Epaminondas: Exploring
Combat Tactics*, ICGA Journal 37(3): mobility, material dominance, crossings,
center of mass, home row defense and territory. **The weights combining them are
not from the paper** — it defines the six terms but never publishes the
coefficients, and calls its own function unrefined — so they are set in
`EpaminondasWeights` and are the obvious thing to tune.

Two things worth knowing before reading results from it:

- **It is shallow.** The budget is per pgx action, matching how the MCTS agent
  is called, and one Epaminondas move is three actions. Measured on this
  machine: ~800 nodes/second on CPU, which reaches depth 1 (in full moves) from
  the opening in a 2 s budget and needs ~8 s for depth 2. The cost is
  `env.step` — about 0.23 ms per child, mostly recomputing legal moves — not
  the evaluation. A JAX env is a poor fit for a sequential tree search; the
  search is correct, but it is not a strong opponent, which is roughly what the
  paper reports for its own novice Alpha-Beta agent.
- **Run it on CPU.** The batches here are a few dozen states, so the search is
  dispatch-bound rather than compute-bound and the GPU is about *four times
  slower*: the same 2 s budget bought ~250-450 nodes on the GPU against ~1600 on
  the CPU. For a negamax-only tournament prefix the command with
  `JAX_PLATFORMS=cpu`. (A `negamax,model` tournament still wants the GPU for the
  model, so it pays the slower engine — worth knowing when reading think times.)
- **Construction compiles for several seconds.** `vmap(step)` is recompiled for
  every distinct batch shape, so the engine buckets batches to powers of two and
  compiles all of them up front (~14 s). Without that, compilation lands inside
  the per-move budget and the search gets ~5 nodes/second.

### Running a trained model elsewhere (ONNX)

`export_onnx.py` converts a checkpoint's policy/value network to ONNX, for
`onnxruntime-web` (WebGPU or WebAssembly) or any other ONNX runtime. It exports
the network only — the game rules and MCTS still have to exist on the target
platform. See the header of the script for the three commands (`dump`,
`convert`, `verify`); conversion needs `jax2onnx` in a throwaway virtualenv.

Measured on a GessFormer checkpoint (7.2M parameters): 29 MB at opset 20, 2073
nodes of standard operators, outputs within 0.012 of JAX on logits of scale 35
(value within 1e-4) and the same best move on every test position, at ~9 ms per
position on one CPU core.

## Reference

- [[Silver+18](https://www.science.org/doi/10.1126/science.aar6404)] "A general reinforcement learning algorithm that masters
chess, shogi, and go through self-play"
- [[Danihelka+22](https://openreview.net/forum?id=bERaNdoegnO)] "Policy improvement by planning with Gumbel"


## Change history

- **[#1107](https://github.com/sotetsuk/pgx/pull/1107)** Extract `compute_loss_input` ([wandb report](https://api.wandb.ai/links/sotetsuk/979hmps8)).
- **[#1106](https://github.com/sotetsuk/pgx/pull/1106)** Use `optax.softmax_cross_entropy` ([wandb report](https://api.wandb.ai/links/sotetsuk/8w0or84k)).
- **[#1088](https://github.com/sotetsuk/pgx/pull/1088)** Adjust to API v2 ([wandb report](https://api.wandb.ai/links/sotetsuk/0g44pjsg)).
- **[#1055](https://github.com/sotetsuk/pgx/pull/1055)** Use default Gumbel AlphaZero hyperparameters ([wandb report](https://api.wandb.ai/links/sotetsuk/o8752t54)).
- **[#1026](https://github.com/sotetsuk/pgx/pull/1026)** Initial version. Supposed to reproduce the [Pgx paper](https://openreview.net/forum?id=UvX8QfhfUx) results ([wandb report](https://api.wandb.ai/links/sotetsuk/5q30e5n9)).
