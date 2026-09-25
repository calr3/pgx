# g_hex experiments

g_hex: 21 triangles, each player places ten tiles valued 1-10, one per turn,
until one triangle is left empty (20 plies, always). The empty triangle's
neighbours are summed with sign (black positive, white negative) and the sign
decides the game; zero is a draw. Every game is exactly 20 plies and the last
mover chooses which of the final two cells stays empty.

Everything here is under **v1** rules (b838c2e: triangle 10 counts the tile on
4). All earlier `checkpoints/g_hex_*` were trained on v0 with the pre-2026-09
recipe (no replay buffer, constant LR); the strongest,
`g_hex_20260504192940/001740.ckpt` (158 h at 100 simulations), is used below as
a fixed reference, `old158h`. It plays with a slightly wrong idea of the rules.

Strength is measured by MCTS round robins on CPU (`JAX_PLATFORMS=cpu`, ~5
games/s at 32 simulations, fast enough to run beside a training run on 12
cores), cached in `elo_g_hex_E1.json`:

```sh
JAX_PLATFORMS=cpu taskset -c 12-23 python -u examples/alphazero/elo_ladder.py \
  env_id=g_hex models=A050=<ckpt>,... games_per_pair=256 num_simulations=32 \
  results_file=elo_g_hex_E1.json
```

## E1: transferred recipe, 32 vs 96 simulations at equal wall clock

Common settings (RECIPES.md "Defaults that transfer"; ResNet, since the 4x7
observation is not an even board and g_hex has no symmetry augmentation):

```sh
python -u examples/alphazero/train.py env_id=g_hex architecture=resnet \
  num_channels=128 num_layers=6 selfplay_bf16=true \
  playout_cap_prob=0.25 fast_num_simulations=8 continue_games=true \
  selfplay_batch_size=1024 max_num_steps=256 training_batch_size=4096 \
  num_updates_per_iter=256 replay_buffer_iters=4 \
  learning_rate=1e-3 weight_decay=1e-4 warmup_steps=100 grad_clip_norm=1.0 \
  lr_schedule=cosine save_data_state=false eval_interval=10 \
  mcts_eval_opponent=checkpoints/g_hex_20260504192940/001740.ckpt
```

| arm | sims | s/iter | iterations | wall clock | checkpoints |
|---|---|---|---|---|---|
| A | 32 | 24.4 | 610 | 4.14 h | `g_hex_20260925064902` (wandb `ahpomzkk`) |
| B | 96 | 90 | 200 | 5.00 h | `g_hex_20260925105841` (wandb `r4c3kow0`) |

B ran 20-25% slower than its 70 s/iteration probe (partly a CPU ladder
competing for the host), so it had 5.0 h to A's 4.14 h. B170 (4.1 h) is the
equal-wall-clock comparison point.

`save_data_state` was off: it writes 3.7 GB every `eval_interval`, which A would
have paid three times as often as B.

### Arm A: strongest after 20 minutes, then worse for four hours

In-training MCTS match against `old158h` (128 games): 43.4% at 1 h, 41.8% at
2 h, 40.2% at 3 h, 34.4% at 4 h, 34.0% at the end.

Round robin over A's trajectory, 256 games per pair, 32 simulations:

| player | Elo |
|---|---|
| **A050** (iteration 50, ~20 min) | **0** |
| old158h | -3 ± 10 |
| A150 | -43 ± 10 |
| A100 | -54 ± 10 |
| A450 | -63 ± 10 |
| A250 | -67 ± 10 |
| A350 | -68 ± 10 |
| A610 (final) | -92 ± 10 |
| A530 | -97 ± 10 |

Directly, A150 beat A610 **0.572 ± 0.025** over 256 games.

The self-play curves all looked like progress over the same stretch: policy
loss 1.31 -> 0.72, self-play draws 42% -> 15%, and value loss down to ~0.01.

**Suspected cause: self-play collapses onto a few lines.** Every game starts
from the empty board, and self-play's only exploration is Gumbel noise at scale
1.0 on the root logits (`train.py`, `gumbel_scale=1.0`), which stops changing
the chosen move once the policy's top logits are several units apart. The near-
zero value loss says the outcome of self-play positions is almost perfectly
predictable, i.e. the games have become repetitive, and the network is
increasingly specialised to them. Tournament games start with two random plies
and so leave that narrow set. Untested; the direct test is an arm that samples
the first few self-play plies (AlphaZero's temperature opening) or plays random
opening plies, measured head to head against A050.

`eval/vs_baseline/avg_R` (raw policy sampling against the 300-iteration v0
net) sat at about -0.45 for the whole run, far below what the MCTS results
imply. Consistent with a sharp but narrow policy; not used for ranking.

### Arm B, and the comparison: 96 simulations do not help

In-training match against `old158h`: 49.6% at 1 h, 48.8% at 2 h, 45.7% at 3 h,
40.6% at 4 h, 37.1% at the end. A few points above A at each hour, but that is
128 games against one opponent; the ladder does not bear it out.

Round robin, both arms, 256 games per pair (GPU, ~10 s a pair):

| player | hours | Elo |
|---|---|---|
| **A050** | 0.33 | **0** |
| old158h | (158) | -17 ± 10 |
| B020 | 0.5 | -41 ± 9 |
| A150 | 1.0 | -58 ± 9 |
| B040 | 0.9 | -62 ± 9 |
| B080 | 1.9 | -71 ± 9 |
| B120 | 2.9 | -90 ± 9 |
| A610 | 4.1 | -110 ± 9 |
| B170 | 4.1 | -111 ± 9 |
| B200 | 5.0 | -125 ± 10 |
| B010 | 0.25 | -166 ± 10 |

Direct head-to-heads:

| pair | score of first |
|---|---|
| A610 vs B170 (equal wall clock) | 0.518 |
| A610 vs B200 | 0.553 |
| A050 vs A610 | 0.637 |
| B020 vs B200 | 0.613 |
| A050 vs B020 (the two peaks) | 0.605 |

**Conclusions.**

- **Simulations: a tie.** At equal wall clock A610 and B170 are level (0.518),
  and at matched hours the two trajectories sit on top of each other (A150 v
  B040, A610 v B170). Tripling the search neither helps nor prevents the
  decline. Keep 32, which is 3x cheaper per iteration.
- **Both arms peak within the first half hour and then lose ~100 Elo** over
  the rest of the run, steadily if not monotonically, while every self-play
  curve improves.
  This is the finding that matters. It is not the schedule: nothing here
  suggests that a longer or differently-shaped cosine run would recover it.
- The first 20 minutes of the new recipe already match a 158 h run of the old
  one, and **A050 is the best g_hex v1 model** (`g_hex_20260925064902/000050.ckpt`).
- The in-training `eval/mcts/score` against one fixed opponent put B ahead of A
  by 6-8 points at every hour; the ladder says they are level. 128 games
  against a single opponent is not enough to rank two arms.

**Next: self-play opening diversity.** The collapse hypothesis above predicts
that the decline goes away if self-play keeps visiting varied positions. Test
by sampling the first few plies of each self-play game from the search policy
(AlphaZero's temperature opening) or playing a few random plies, needs a
`train.py` option; one ~2 h arm, ladder it against A050.

## E2: sampled self-play openings do not stop the decline

Arm A's command plus `selfplay_sample_plies=6` (the first 6 of 20 plies sampled
from the search's improved policy), `max_num_iters=290` (a 2 h schedule), and
`mcts_eval_opponent=` A050 every half hour. Checkpoints
`g_hex_20260925163040`, wandb `ysvec8fb`, 1.97 h.

In-training match against A050: 47.7% at 0.5 h, 44.9% at 1 h, 41.0% at 1.5 h,
33.2% at the end. Ladder (256 games per pair, cached in `elo_g_hex_E1.json`):

| player | Elo |
|---|---|
| **A050** | **0** |
| E2_050 | -6 ± 10 |
| E2_020 | -16 ± 10 |
| E2_100 | -34 ± 10 |
| A150 | -52 ± 10 |
| E2_150 | -61 ± 10 |
| E2_220 | -69 ± 10 |
| E2_290 | -101 ± 10 |
| A610 | -104 ± 10 |

At matched iterations E2 and A are level head to head: A050 v E2_050 0.508,
A150 v E2_150 0.508, and at the ends of their schedules A610 v E2_290 0.514.
E2_050 beat E2_290 0.619. Value loss again fell to ~0.01.

**The collapse hypothesis, in the form this option tests, is wrong**: varying
the first six plies neither delays nor softens the decline. The option stays
(off by default); it did no harm.

**What the decline tracks.** E2 reached A's final level in half the time: -101
at the end of a 2 h schedule, against A's -67 at 2 h and -104 at the end of its
4 h one. Measured by fraction of the cosine schedule completed, the two runs
coincide; measured by iterations, E2's end (-101) is well below A near 290
(-67). So the decline may be driven by the learning rate decaying, not by the
amount of training. Untested; the direct test is a constant-LR arm.
