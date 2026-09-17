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

### Result: E2 collapsed into drawing every game

| iteration | 5 | 10 | 20 | 40 | 80 | 120 | 160 |
|---|---|---|---|---|---|---|---|
| E1 draw rate | 0.93 | 0.44 | 0.71 | 0.83 | 0.09 | 0.02 | **0.000** |
| E1 value loss | 0.05 | 0.18 | 0.14 | 0.10 | 0.36 | 0.44 | 0.37 |
| E2 draw rate | 0.91 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** |
| E2 value loss | 0.05 | 0.01 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 |

E1 is the healthy pattern: draws vanish, and `value_loss` *rises* as a result -
the same signature Gess showed, where a rising value loss tracked draws
disappearing rather than anything going wrong. E2 went the other way and stayed
there: every game reached the 300-move cap for 151 consecutive iterations, so
every value target was 0 and the value head trivially learned to predict 0.

So the pig result did **not** transfer, and "few legal actions" is not by itself
enough to justify `completed_unscaled`.

Two caveats on how much this comparison is worth:

- `train.py` only saved checkpoints on `eval_interval` multiples, so with
  `eval_interval=100` and 160 iterations both arms' newest checkpoint is
  iteration 100 - the low-LR tail of the cosine schedule was thrown away. Fixed
  (`train.py:661` now always saves the final iteration), but it means E1 and E2
  can only be compared at iteration 100, where E1 had 0.83 h of compute against
  E2's 0.71 h. That is equal-iteration, not equal wall clock, and it favours E1.
- The comparison was run under v0 rules, whose cap is a plain draw. See below.

## The move cap was a trap, not a tiebreak (v1 rules)

`rewards` returned `[0, 0]` for any game reaching `MAX_MOVES`, so **stalling was
an equilibrium**: a player who never commits scores 0 rather than -1, and
nothing pushes the search out of it. E1 escaped only around iteration 80; E2
never did.

Epaminondas is now **v1**: reaching the cap is decided on piece count, and only
an exact tie is a draw - exactly what Gess does with its captureless stalemate
(Gess made the same v0 -> v1 change). This is a deviation from the published
rules, which have no move limit at all, and is documented in the rules module.

A player who is behind can no longer shuffle to safety, which should remove the
attractor entirely rather than merely making it less attractive.

## E3 vs E4: the same q-transform question under v1 rules

Identical settings to E1/E2 but for the rules and `eval_interval=40` (so
intermediate checkpoints exist). This answers two things at once: whether the
material rule stops the stalling, and whether the q-transform makes any
difference once it does - E2's failure mode is gone, so the comparison is fair
for the first time.

Status: running. Results to follow.

## Queued: v2, a capture clock instead of an absolute cap

v1 fixed what the cap *pays* but not when it *fires*. `moves` counts from the
start of the game and never resets, where Gess resets `no_capture_turns` on any
capture, so Gess's cap only fires once a position has stopped progressing.

Two consequences, neither of which binds today (E3's games average ~27 moves,
nowhere near 300) but both of which matter in long games between strong models:

- a player ahead on material now has a reason to run the clock out, since
  reaching the cap pays them - an incentive v0 did not have, because it paid
  nothing;
- a game still being fought at move 300 is truncated and scored on material.

v2 will end the game after a fixed number of moves **without a capture**, with
material still deciding, and keep an absolute cap only as a loose backstop.
Captures are a sound progress measure here because pieces are only ever removed,
so at most 56 can occur and the capture clock alone bounds game length.

To be tested against v1 as its own equal-time comparison rather than assumed
better - v1 is already a large improvement over v0.
