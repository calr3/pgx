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
