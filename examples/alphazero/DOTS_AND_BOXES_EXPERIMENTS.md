# Dots and Boxes experiments

`dots_and_boxes` is 6x6 boxes (7x7 dots, 84 lines). The observation is the
13x13 lattice of dots, lines and boxes; the baseline (`dots_and_boxes_v0`) is
the classic greedy player - take any box, else give none away, else anything -
which an untrained network beats 0% of the time.

## D1: DotFormer against a ResNet, equal wall clock (~1.25 h each, side by side)

Both arms: the Epaminondas recipe (bf16 self-play, 32 simulations, playout cap
0.25 with 8 fast simulations, `continue_games`, batch 256, replay buffer 4,
cosine LR 5e-4) with 8-way symmetry augmentation, and a policy head that reads
each line's logit off its own lattice cell (`network.action_cells`).

- **A, DotFormer** (`architecture=boardformer bf_embed_dim=128 bf_num_layers=6
  bf_num_heads=4`): the 13x13 lattice padded to 14x14 with a board-mask plane,
  so each 2x2 patch-merged token is one dot with the lines to its right and
  below and the box between them - 49 tokens. 3.1M parameters, 1.6M of them the
  geometric attention bias. 105 iterations.
- **B, ResNet** (`architecture=resnet num_channels=128 num_layers=8
  num_heads=4 num_attention_layers=2 cell_policy_head=true`): full-resolution
  convolutions, the last two blocks attention over all 169 cells. 1.9M
  parameters; needs `train_micro_batches=2` on the 16 GB GPU. 88 iterations.

Iteration counts were sized from timing both arms running together, so each
cosine schedule finished in the same time.

| | vs greedy baseline (training eval, 128 games) |
|---|---|
| A, iterations 20 / 40 / 60 / 80 / 100 | 0.64 / 0.87 / 0.89 / 0.86 / 0.89 |
| B, iterations 20 / 40 / 60 / 80 | 0.84 / 0.88 / 0.90 / 0.92 |

**Head to head, final checkpoints** (`model_tournament.py`, 256 games in
seat-swapped pairs from 4-ply random openings, 32 simulations each): **A wins
154-98 with 4 draws, 0.609 +- 0.030, +77 Elo**, and about equally from either
seat (0.60 first, 0.62 second).

The baseline column would have picked B: it leads at every checkpoint, and it
cannot separate them at the end. Only the head-to-head does.

## D2: DotFormer for 8.7 hours (1,800 iterations)

D1's arm A unchanged but for the schedule: `max_num_iters=1800`, sized from its
iteration time alone (15.8 s) to finish the cosine schedule in about eight
hours, with `mcts_eval_opponent` set to D1's final DotFormer
(`dots_and_boxes_20260926050722/000105`). Run: `dots_and_boxes_20260926090930`.

Hourly score against D1's DotFormer (128 games): 0.72 at 1 h, 0.81, 0.93,
0.95, 0.98, 0.996 at 6 h, then 0.99 to the end - saturated, so it says nothing
about the last third. Head to head (`model_tournament.py`, 256 seat-swapped
games from 4-ply random openings, 32 simulations each), the final checkpoint
against:

| opponent | score | Elo |
|---|---|---|
| D1's DotFormer | 0.988 +- 0.007 | +770 |
| iteration 600 | 0.895 +- 0.018 | +371 |
| iteration 1,200 | 0.709 +- 0.027 | +155 |
| iteration 1,500 | 0.574 +- 0.028 | +52 |

- **Still improving when the schedule ended**: the last 300 iterations are worth
  +52 (about two standard errors), so a longer run should gain more.
- **Draws rise with strength**: 0 of 256 against D1's model, 8 against
  iteration 600, 25 against 1,200, 46 (18%) against 1,500. Close play between
  strong models ends 18-18 far more often.
- The fixed-opponent eval saturated by hour 6. For a longer run, move it to a
  late checkpoint of this run (1,500 is a fair match for 1,800).
