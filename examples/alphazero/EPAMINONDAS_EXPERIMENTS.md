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

### Result: the rule change worked, and the q-transform question is settled

The material rule transformed the baseline arm. E3 against E1, same settings but
for the rules:

| iteration | 5 | 10 | 20 | 40 | 160 |
|---|---|---|---|---|---|
| E1 draw rate (v0) | 0.93 | 0.44 | 0.71 | 0.83 | 0.000 |
| **E3 draw rate (v1)** | 0.155 | 0.161 | 0.029 | **0.000** | 0.018 |
| E3 games finished | 219 | 168 | 138 | **1235** | 164 |

Draws are gone by iteration 20 where E1 was still drawing 83% of games at 40 and
did not escape until ~80. The knock-on effect is larger than the draw rate
itself: games stopped running to the cap, so the same 384-step budget completed
**~1,400 games an iteration instead of ~120-300** - an order of magnitude more
real value targets per unit of compute. The v0 cap was not only permitting a
degenerate policy, it was starving the data pipeline.

E4 repeated E2's failure, more slowly:

| iteration | 5 | 20 | 40 | 60 | 80 | 100 | 160 |
|---|---|---|---|---|---|---|---|
| E4 draw rate | 0.199 | 0.632 | 0.617 | 0.350 | 0.743 | 0.852 | **0.960** |
| E4 value loss | 0.420 | 0.191 | 0.180 | 0.162 | 0.094 | 0.052 | **0.029** |

Head to head at iteration 160, 128 games, 32 simulations, `max_num_steps=900`,
`random_opening_plies=3` (a multiple of the three plies per move, so openings
end on a move boundary):

```
A = E3 (default), B = E4 (completed_unscaled)
A wins 119 (93.0%) | B wins 0 (0.0%) | draws 9 (truncated: 0)
A score 0.965 ± 0.011   Elo diff A-B: +575
A as P0 0.953 | A as P1 0.977 | avg game length 324 plies
```

**`completed_unscaled` is actively harmful here**, shown under two rule sets and
confirmed by a 119-0 head-to-head. Caveats worth keeping: E4 ran 160 iterations
in 1.12 h against E3's 1.30 h (drawn games complete fewer games per iteration,
so there is less data to train on), making this equal-iteration rather than
equal-time - but the shortfall runs against the arm that already lost, so it
cannot explain the result.

### Why pig does not transfer

Pig's two actions are *hold* and *roll*, and the unscaled Q gap between them is
the real difference in win probability - exactly the quantity the policy target
should carry. Epaminondas's two or three actions at stages 1 and 2 are
structural continuations of one move (which rear, which destination), where the
raw Q spread is dominated by how the move happens to resolve rather than by how
much better one continuation is. Rank survives that; magnitude does not.

So the rule in `RECIPES.md` needed narrowing: a small *decision* space where the
Q difference means something, not merely a small action space.

## v2: a capture clock instead of an absolute cap

v1 fixed what the cap *pays* but not when it *fires*. `moves` counts from the
start of the game and never resets, where Gess resets `no_capture_turns` on any
capture, so Gess's cap only fires once a position has stopped progressing.

Two consequences, neither of which binds today (E3's games average ~27 moves,
nowhere near 300) but both of which matter in long games between strong models:

- a player ahead on material now has a reason to run the clock out, since
  reaching the cap pays them - an incentive v0 did not have, because it paid
  nothing;
- a game still being fought at move 300 is truncated and scored on material.

v2 ends the game after **60 moves without a capture**, material still deciding,
and keeps `MAX_MOVES = 600` only as a loose backstop. Captures are a sound
progress measure here because pieces are only ever removed, so at most 56 can
occur and the capture clock alone bounds game length; the backstop exists so
worst-case length stays predictable for sizing `max_num_steps`.

The window is deliberately generous - Gess uses 20 turns, but Epaminondas has a
14x12 board and a slow buildup, and cutting a live game short on material is the
failure this is meant to avoid. E3's games averaged 324 plies (108 moves) in the
tournament, so a 60-move quiet window should rarely bind.

### Result: v2 is worse than v1, and the window is why

E5 is E3 with only the rules changed, at 1.27 h against 1.30 h - a genuine
equal-time comparison.

| | E3 (v1) | E5 (v2) |
|---|---|---|
| final draw rate | 0.018 | 0.000 |
| final policy loss | 1.218 | 0.863 |
| final value loss | 0.261 | 0.399 |
| final steps/game | 599 | 202 |

```
A = E5 (v2), B = E3 (v1), 128 games, 32 sims, played under v2 rules
A wins 45 (35.2%) | B wins 79 (61.7%) | draws 4 (truncated: 0)
A score 0.367 ± 0.042   Elo diff A-B: -95
```

**v2 lost by ~95 Elo**, despite the games being played under v2's own rules,
and despite its training curves looking *better* on every metric that usually
signals progress - shorter decisive games, lower policy loss, no draws.

The cause shows up early. E5 spent its first ~40 iterations degenerate:

| iteration | 20 | 40 | 60 |
|---|---|---|---|
| E5 draw rate | **0.979** | 0.029 | 0.002 |
| E5 value loss | 0.016 | 0.316 | 0.363 |
| E3 draw rate | 0.029 | 0.000 | 0.000 |

A draw needs *exactly* equal material, and untrained play is quiet: early on
neither side captures, so the 60-move capture clock fires while both players
still hold all 28 pieces, and the game is scored a tie. v1's 300-move absolute
cap ran long enough that captures happened first. So the capture clock does not
only end stalled games - it ends *unskilled* ones, and roughly a quarter of the
160-iteration budget went to recovering from that.

The lesson is about how the window was chosen, not about capture clocks: 60 was
picked as "deliberately generous" against E3's 108-move average game length,
which is a **trained** statistic and the wrong reference for iteration 20. A cap
that depends on skill has to be sized for the *worst* play in the run, not the
best.

The incentive problem that motivated v2 is still real - under v1 a player ahead
on material can run the clock out - but it does not bite at this strength, and
the measured cost of fixing it does.

### Decision: v2 is kept, with the cost on the record

**v2 stays**, deliberately and against the head-to-head. The reasoning is that
v1's flaw is structural and gets worse exactly where it matters - a model strong
enough to hold a material lead is paid to stop playing - while v2's cost is a
one-off early-training tax that the run recovers from by iteration 40. Trading
95 Elo at pilot scale for a rule that does not reward stalling at full scale is
a judgement about where this is going, not a claim that v2 is stronger today.

What that means when reading results here:

- **The -95 Elo is a known, accepted cost, not a mystery.** Do not go looking
  for a regression to explain it, and do not compare v1-trained and v2-trained
  checkpoints as though the rules were held constant.
- The obvious lever if it becomes a problem is the window, not the mechanism:
  raise `MAX_MOVES_SINCE_CAPTURE` from 60 to ~150 so it cannot fire during early
  unskilled play, and re-run this comparison. Untested.
- E3 (v1) remains the strongest Epaminondas checkpoint measured so far, at
  `checkpoints/epaminondas_20260917191121/000160.ckpt`. E5 is the strongest
  under v2 rules.

## v3: the cap is decided on advancement, not material

Material was the wrong tiebreak, for the same reason a draw was: it does not
measure the thing the game is about. Epaminondas is won by getting up the board,
so the cap now compares piece counts **rank by rank, deepest first** - white's
count on row `HEIGHT-1-i` against black's on row `i` - and the first rank that
differs takes it (`_advancement_winner`).

The deepest pair is always level in any position the cap can be reached from: a
difference there *is* the win condition, so the game would already have ended.
The comparison therefore starts in practice at the second-to-home rank and walks
back down the board, and a draw needs the two sides to be exact mirrors rank for
rank.

Measured over 256 random games: **0 draws** (117 white wins, 139 black). Under
the material tiebreak, early play drew ~98% of games.

### Why this should also undo v2's cost

E5's -95 Elo came from ties, not from the capture clock as such: untrained play
is quiet, so the clock fired while both sides still held all 28 pieces, and
*equal material is a draw*. Advancement is almost never equal, so those same
games are now decided rather than drawn, and there is no reason for the
degenerate first 40 iterations to reappear.

Stated as a prediction before measuring, so it can be wrong: **v3 should not
show E5's early collapse, and should beat both E5 (v2) and E3 (v1)**. If the
early draw rate at iteration 20 is still near 1.0, the diagnosis above is wrong
and the capture clock itself is the problem.

### v4: no draws at all

The one position advancement cannot separate is an exact rank-for-rank mirror.
v4 resolves it by giving the game to **whoever captured most recently**, and if
neither player ever has, to **black** - compensation for moving second, in the
spirit of komi. `rewards` no longer has a zero branch: every terminal position
has a winner, so a draw is not representable.

That matters beyond tidiness. A draw is the one outcome a stalling player can
aim at without having to be better, and every rule change here has been chasing
that: v0 paid 0 for it, v1 and v2 paid it only on exact material ties, v3 made
it rare, v4 removes it. `train/value_loss` can no longer be driven down by
manufacturing zero targets.

Measured over 256 random games: 0 draws, 111 white wins, 145 black.

### E6, under v3, confirmed the diagnosis

E6 is E3 and E5's settings with only the rules changed. Draw rate by iteration:

| iteration | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| E6 (v3) | **0.000** | **0.000** | 0.012 | 0.007 |
| E5 (v2) | 0.996 | 0.972 | 0.979 | 0.029 |
| E3 (v1) | 0.155 | 0.161 | 0.029 | 0.000 |

v3 has **no degenerate phase at all**, and is cleaner from iteration 5 than v1
ever was. Its value loss stays at 0.44-0.50 rather than collapsing to 0.016, so
the value head has real targets from the start.

That confirms the prediction made above: E5's -95 Elo came from **ties**, not
from the capture clock. Equal material is common in quiet early play; equal
advancement is not. v3 keeps v2's anti-stalling property and drops its cost.

Note E6 trained under v3 while v4 is now current. The two differ only on exact
mirrors, which random play produced zero times in 256 games, so the difference
is unlikely to matter - but it is a rules difference, and head-to-heads
involving E6 are played under v4.

Status: E6 training; head-to-heads against E3 and E5 to follow.
