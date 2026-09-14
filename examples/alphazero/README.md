# AlphaZero example

A simple (Gumbel) AlphaZero [[Silver+18](https://www.science.org/doi/10.1126/science.aar6404), [Danihelka+22](https://openreview.net/forum?id=bERaNdoegnO)] example using [Mctx](https://github.com/deepmind/mctx) library. See [Pgx paper](https://openreview.net/forum?id=UvX8QfhfUx) for more details.

![](assets/pgx-az-training.png)

> [!NOTE]
> This implementation of AlphaZero demonstrates sufficient learning performance in environments including 9x9 Go, but it has some slight differences in learning details compared to the original AlphaZero and Gumbel AlphaZero. An implementation that addresses these differences and focuses on enhanced efficiency is currently under development and is expected to be released shortly.

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
