# Epaminondas experiments

Epaminondas is the second board game trained here, after Gess. It reuses Gess's
architecture (`boardformer`, GessFormer generalised to any even-sized board with
one action per cell) on a 14x12 board with 168 actions.

Before this file there was only a single ~15 minute pilot, recorded in
`RECIPES.md` as starting settings rather than a validated recipe.

## What makes Epaminondas different from Gess

**Three stages per move, not two.** A move picks the lead piece, then the rear of
the phalanx, then the destination. `current_player` flips only on the third.
Everything measured in *steps* therefore has to be divided by three to be read as
moves.

**Games are long.** `MAX_MOVES = 300` (`_src/games/epaminondas.py:59`), so a game
runs up to 900 steps, and under random play essentially every game hits the cap
(mean 858 steps over 512 games). The pilot's `max_num_steps=192` covered 64 moves
of a possible 300, which is why its value targets were starved.

**Most search nodes have very few legal actions.** Measured over 180 steps of
random play at batch 256:

| stage | what it picks | mean legal | median | fraction <= 4 |
|---|---|---|---|---|
| 0 | lead piece | 24.9 | 26 | 0.00 |
| 1 | rear of the phalanx | 3.4 | 3 | 0.77 |
| 2 | destination | 2.4 | 2 | 0.90 |

Two of every three nodes the search expands offer a median of two or three
moves. This is the regime where pig found mctx's default q-transform destructive:
`qtransform_completed_by_mix_value` rescales Q by the range across a node's
actions, and with two or three actions that range *is* the quantity being
measured, so only the sign of an advantage survives into the policy target. On
pig, switching to `completed_unscaled` tripled the score against optimal play at
equal wall clock (`PIG_EXPERIMENTS.md`, E3).

Gess never showed this because its policy head chooses among hundreds of cells.
Epaminondas is the first board game here with pig's shape.

## E1 vs E2: the q-transform, at equal wall clock

The one change under test is `qtransform`, with the step budget raised from the
pilot's 192 to 384 (128 moves) in both arms. Both runs use the RECIPES settings
otherwise, and `max_num_iters` is sized from a measured steady-state rate so the
cosine schedule completes inside the budget.

- **E1** `qtransform=completed_by_mix_value` (the default) - the corrected
  baseline, and the first Epaminondas run whose schedule finishes.
- **E2** `qtransform=completed_unscaled` - the pig fix.

Scored head to head with `model_tournament.py`, which is the honest comparison;
`eval/vs_baseline/*` is meaningless here because `epaminondas_v0` is an untrained
placeholder network, not an opponent.

Status: running. Results to follow.
