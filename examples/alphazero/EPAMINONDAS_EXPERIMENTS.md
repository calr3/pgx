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

### Result: the prediction was wrong. The capture clock is the problem

All 128-game matches, 32 simulations, played under v4 rules:

| match | A's score | Elo |
|---|---|---|
| E6 (v3) vs **E3 (v1)** | 0.344 ± 0.042 | **-112** |
| E6 (v3) vs E5 (v2) | 0.445 ± 0.044 | -38 |
| E5 (v2) vs **E3 (v1)** | 0.367 ± 0.011 | **-95** |

E6 lost to v1 by *more* than E5 did, with no degenerate phase at all. The
E6-E5 difference is ~1.25 standard errors and not significant; E6-E3 is 3.7
and is.

The prediction above - that removing the ties would remove the cost - is
**falsified**. Removing the ties removed every symptom (draws 0.000 from
iteration 5, value loss steady at 0.44) and made the model slightly worse. So
the cost was never the ties: it is the **60-move capture clock**, which both
E5 and E6 share and E3 does not.

Note what the models were measured under: all three played the tournament under
v4 rules, i.e. under the capture clock. E3 was trained without it and still wins
by 112 Elo **on the other rules' own terms**, which rules out "E3 is merely
better at the game it was trained on".

The likeliest explanation is the training horizon. Epaminondas has long
manoeuvring phases with no captures - building and repositioning phalanxes - and
a 60-move quiet limit cuts them off, so the model never sees a full strategic
arc. Game lengths line up: E3 599 steps per game, E6 449, E5 202, ranked exactly
as the models are.

### Where this leaves the rules

The evidence supports **v1's trigger** (an absolute cap, long horizon) and says
nothing against **v4's tiebreak cascade** (advancement, then last capture, then
black), which was never tested apart from the capture clock. They are
independent choices and the experiments confounded them.

The obvious next configuration, untested: **the absolute 300-move cap with the
v4 tiebreak** - the long horizon that wins, with draws still impossible. Failing
that, the capture clock with a much longer window (~150) would test the horizon
explanation directly.

What is now established:

- v1's trigger beats the 60-move capture clock by ~100 Elo, twice over.
- The tiebreak below it (material vs advancement vs no-draws) has never been
  measured on its own.

## E7 (v5): the two variables, finally separated

v5 is v1's trigger - an absolute 300-move cap - carrying v4's tiebreak
(advancement rank by rank, then the last capture, then black; no draws). E7 is
E3 and E6's settings with only the rules changed, 160 iterations in 1.30 h.

| comparison | what it isolates | A's score | Elo |
|---|---|---|---|
| E7 (v5) vs E3 (v1) | **the tiebreak**, horizon held fixed | 0.523 ± 0.044 | **+16** |
| E7 (v5) vs E6 (v3) | **the horizon**, tiebreak held fixed | 0.633 ± 0.043 | **+95** |

- **The tiebreak is free.** +16 Elo is half a standard error from parity, so
  advancement scoring and the removal of draws cost nothing in strength. They
  were adopted because they are the better rules, and the measurement says
  nothing stands in the way of that.
- **The capture clock was the whole cost.** +95 Elo at 3.1 standard errors,
  matching the -95 and -112 measured against it earlier. The three figures agree
  on a single quantity: ending games 60 quiet moves after the last capture costs
  about 100 Elo, because Epaminondas's manoeuvring phases are longer than that.

E7's self-play was clean throughout: draws 0.000 from iteration 1 (E3 drew 15%
early), value loss steady at 0.38-0.50, and early game lengths tracking E3's long
ones (945 steps at iteration 10) rather than E6's truncated ones (245).

**E7 is the strongest Epaminondas model measured**, at
`checkpoints/epaminondas_20260918013629/000160.ckpt`, and it plays the rules we
want to keep.

### What the sequence cost, and why

Four training runs (E4-E7, ~5.5 h) went into a question that two would have
answered, because v2 changed *when the cap fires* and v3/v4 changed *how it is
scored* without either being measured alone. The confound then produced a wrong
diagnosis - E5's loss was blamed on ties, E6 removed the ties and lost by more -
which cost a fifth run to correct.

Change one variable per comparison, even when the second change seems obviously
right.

## Sanity check: E7 vs random at 2 s/ply

`interactive_tournament.py` with `num_simulations=704`, which measured 2.18 s/ply
on this GPU (512 -> 1.3 s, 640 -> 1.8 s, 1024 -> 3.2 s; the first ply of a run is
~16 s of compilation and is excluded). Seats alternate every game.

**7-0 to E7**, stopped early once the point was made. This confirms the search
path end to end; it is not a strength measurement, because the same checkpoint
already scores 0.906 against random on raw policy argmax with *no* search, so a
random opponent has almost no resolution left at this level. Rank models against
each other, not against random.

Three bugs in the tournament script had to be fixed first, all of which would
have corrupted the result:

- **Scoring read `state._x.winner`.** Epaminondas at the 300-move cap leaves
  `winner == -1` and resolves the game in `rewards()` by advancement, so a capped
  game credited seat `(-1 + rotation) % 2` - a win awarded by parity rather than
  by who won. Now scored from `rewards`, which is authoritative for every
  termination path. Any game with a decided-but-unset winner hits this, so it is
  worth checking before trusting this script on a new env.
- **The single-legal-move short-circuit returned a 0-d array**, crashing on the
  first forced move. Epaminondas hits this constantly: a one-piece phalanx has
  exactly one legal rear.
- **No Epaminondas CLI existed**, so `get_cli` raised immediately.

`num_simulations` was also hardcoded at 6144 with no way to set it, and
`get_action` ran an extra *unjitted* forward pass every ply purely to print the
raw prior - which dominates the cost of a cheap search, so per-ply timings taken
with `verbose=true` measure the debug view rather than the search. Timing now
blocks on the result, since JAX dispatch is async.

## An alpha-beta opponent

`negamax.py` adds a conventional engine - hand-written evaluation, full-width
alpha-beta, no network - so a model can be measured against something whose
behaviour is understood. Its evaluation is the six LEONIDAS heuristics from King
and Peterson, *Epaminondas: Exploring Combat Tactics*, ICGA Journal 37(3); the
weights are ours, because the paper defines the terms but never publishes the
coefficients and calls its own function unrefined.

**6-0 against random** at a 0.5 s per-action budget, three games from each seat.
Over those 442 searches: depth 1 reached on 371, depth 2 on 29, 33 were forced
single-legal-move positions, and 9 ran out of clock before finishing depth 1
(those fall back to the best child by static evaluation, not to an arbitrary
move). Throughput was 632 nodes/second.

Beating random is not evidence of much - E7 does it on raw policy argmax with no
search - it only shows the engine plays legally and coherently. The point of it
is as a *fixed* reference: unlike a checkpoint, it does not move when the model
moves, so "model beats negamax at equal time" means the same thing next month as
it does today.

### What it costs to do a tree search over a JAX env

Worth knowing before anyone tries to make this deeper:

- **~800 nodes/second on CPU**, which buys depth 1 (in full moves) from the
  opening in 2 s and needs ~8 s for depth 2. Epaminondas averages a branching
  factor of 283, so depth 2 is ~80k moves; alpha-beta with good ordering is what
  makes even that borderline reachable.
- **The GPU is ~4x slower** - ~250-450 nodes per 2 s against ~1600 on CPU. The
  batches are a few dozen states, so the search is dispatch-bound, and none of
  the GPU's throughput is reachable. Run negamax-only tournaments under
  `JAX_PLATFORMS=cpu`.
- **The cost is `env.step`**, ~0.23 ms per child, mostly recomputing legal
  moves - not the evaluation, which is ~0.17 ms per board and was already halved
  by sharing run tables between the two sides.
- **jit recompiles per input shape.** The legal-move count differs at nearly
  every node, so exactly-sized `vmap(step)` batches cost one full XLA
  compilation *per node*: ~5 nodes/second, essentially all compilation. Batches
  are bucketed to powers of two and all buckets compiled up front.

The honest summary is that a JAX env is a poor fit for sequential tree search.
The search is correct and the engine is a useful fixed reference, but it is a
weak one, which is roughly what the paper reports for its own novice agent.

## E8: 64 simulations, at equal wall clock

**Question.** `num_simulations=32` was inherited from Gess and never tested here.
An Epaminondas move costs *three* network evaluations - lead, rear, destination -
so 32 simulations buy roughly a third of the per-move lookahead they buy in a
one-action game. Is 64 better at equal time?

**Setup.** E7's settings with `num_simulations=64` and nothing else changed.
A 3-iteration probe measured the new rate before launching, so that
`max_num_iters` could be sized for the cosine schedule to *complete* inside the
budget - an unfinished schedule would have confounded the comparison with a
truncated LR tail. 72 iterations, 1.034 h. Checkpoint
`checkpoints/epaminondas_20260918062619`.

The probe under-estimated: iteration 3 took 62 s, but steady state was 54 s, so
the run landed at 1.034 h against E7's 1.301 h. Rather than restart, the result
was **bracketed** against the two E7 checkpoints either side of it.

| comparison | E8's time vs opponent | E8 score | Elo |
|---|---|---|---|
| E8 vs E7 it120 (0.980 h) | **+5%** | 0.430 ± 0.044 | **-49** |
| E8 vs E7 it160 (1.301 h) | -21% | 0.258 ± 0.039 | **-184** |

128 games each, 32 simulations for both sides, `max_num_steps=900`, no draws and
no truncations.

**Conclusion. 64 simulations is worse at equal time; keep 32.** E8 loses even to
the checkpoint it out-spent by 5%, so the handicap does not explain it. Doubling
the search does not pay for halving the games: E8 saw 7.1M positions to E7's
15.7M.

This is the same shape as the Gess result that halving to 16 simulations cost
103 Elo. Both games are worse off either side of 32, so 32 looks like a genuine
optimum for this recipe rather than a Gess-specific accident - which is worth
knowing before scaling, since simulation count is exactly the sort of parameter
one is tempted to raise when given more hardware.

The bracket is the method to reuse: when a run misses its time budget, compare
it against checkpoints on **both** sides rather than restarting. Losing to the
favourable end settles it at no extra cost.

### A seat asymmetry worth following up

Both matches show the same thing, and so did the earlier baseline-vs-random
check (0.812 as player 0, 1.000 as player 1):

| | as player 0 | as player 1 |
|---|---|---|
| E8 vs E7 it120 | 0.391 | 0.469 |
| E8 vs E7 it160 | 0.203 | 0.312 |

Because the matches are seat-balanced, this says both sides do better as player
1: if E8 scores 0.391 as player 0, E7 scored 0.609 as player 1 in those games.
So **player 1 (black) appears to hold an advantage of roughly +0.08** in score.

It is probably *not* the v5 tiebreak, which hands black the last word only when
the 300-move cap is reached: these games averaged 153-190 plies, i.e. 51-63
moves, so the cap almost never fired. That points at move order itself - and
Epaminondas does give the second player a move to answer a crossing, so an
advantage there is not implausible.

Untested, and it matters for reading every result here: seat-balanced scores
cancel it, but any unbalanced measurement inherits it.

## v6 rules: a 100-move capture clock (adopted, not yet measured)

The move limit now counts moves **since the last capture**, resetting to zero on
every capture, at 100 rather than v2's 60. The tiebreak is unchanged from v4/v5:
advancement rank by rank from the opponent's home rank, then the last capturer,
then black. No draws.

**This is a design decision, not a measured improvement, and the prior evidence
is against it.** A 60-move capture clock lost twice, by 95 and 112 Elo (E5, E6),
and removing it was worth +95 (E7). The case for trying again at 100 is that an
absolute cap pays a player who is ahead to run the clock out, and that 60 may
have been too short a horizon rather than the idea being wrong - Epaminondas's
manoeuvring phases are long. Whether 100 is long enough is untested.

### What it does to game length

64 random-play games under v6, versus v5's hard bound of 900 actions
(300 moves x 3):

| | v5 | v6 |
|---|---|---|
| draws | 0 | 0 |
| moves | <= 300 | mean 218, median 100, max 691 |
| actions | <= 900 | mean 653, **max 2073** |
| games over 900 actions | impossible | **18 of 64 (28%)** |

**There is no longer an absolute bound.** Each capture resets the clock, so the
only limit is that captures are finite: 28 pieces a side, at least one removed
per capture, so at most ~55 captures and a worst case near 5600 moves (16,800
actions).

**Consequence for measurement.** Anything that caps steps per game must be
raised or it will truncate - and `model_tournament.py` scores a truncated game
as a **draw**, which silently reintroduces the one outcome these rules exist to
remove. The E1-E8 head-to-heads used `max_num_steps=900`, which was exactly v5's
bound and is now too small; 28% of random-play games exceed it. Use ~3000 for
v6, and check the reported "of which truncated" count is zero.

Trained models play far shorter games than random ones (E8's averaged 153-190
actions), so the practical impact is smaller than the random-play figures
suggest - but the tail is what truncation bites.

## E9: the first full-size run (in progress)

Every run from E1 to E8 was a ~1.3 h pilot at `selfplay_batch_size=256`. That
left three numbers unknown that the TPU rental has to be sized against: the
steady-state iteration cost at full batch, peak host RAM, and the size of
`data_state.pkl`. E9 is the run that measures them. It is also the first run
under **v6** rules, which are adopted but unmeasured.

```sh
python3 -u examples/alphazero/train.py env_id=epaminondas architecture=boardformer \
  selfplay_bf16=true num_simulations=32 playout_cap_prob=0.25 \
  fast_num_simulations=8 continue_games=true \
  selfplay_batch_size=1024 max_num_steps=384 training_batch_size=4096 \
  num_updates_per_iter=64 replay_buffer_iters=4 \
  learning_rate=5e-4 weight_decay=1e-4 warmup_steps=100 grad_clip_norm=1.0 \
  lr_schedule=cosine save_data_state=true eval_interval=20 max_num_iters=120
```

`checkpoints/epaminondas_20260918163258/`, budget ~4 h.

### Sizing: why `num_updates_per_iter` is 64 and not 384

A 5-iteration probe at full batch was run first. Two things it settled:

- **The replay buffer is 7.88 GiB** for `replay_buffer_iters=4` at this size,
  against 45 GiB free. Not a constraint.
- **Gess's 4x sample reuse does not transfer.** RECIPES justifies it as free
  "at full size, where training is ~4% of an iteration". Fitting iteration time
  against `num_updates` gives ~90-109 s of fixed self-play and a per-update cost
  that puts 384 updates at **19-25% of the iteration**, not 4%, because
  Epaminondas self-play is far cheaper per step than Gess's. Paying 25% of wall
  clock for an untested change was not worth it, so E9 holds the **0.67x reuse
  ratio that E1-E8 actually validated** and changes only the batch size.

The fit itself is unstable - 0.097 s/update on two points, 0.25 s on three -
because `trajectories.process` computes discounted returns in a Python loop over
`384 + tail_length` timesteps, so its cost tracks the held-back tail and the
regression misattributes that to `num_updates`. The two fits agree closely at 64
updates (106 s vs 116 s) and diverge at 256 (134 s vs 154 s), which is a second
reason to stay low. **Don't quote the per-update figure**; re-measure it with
`num_updates` varied at a fixed tail size if it ever matters.

### `max_pending_steps` under v6, and why it was left at 1024

With `continue_games=true`, steps wait in a tail until their game ends. The tail
is stored dense as `(tail_length, num_slots, ...)`, **not per-slot**, so
`max_pending_steps` costs `steps x 1024 slots x ~5.4 KB`: 5.7 GB at the default
1024, ~17 GB at 3072. Raising it is not free.

v6 removed the absolute game-length bound, so the obvious worry was that long
games would overflow the cap and enter the buffer without value targets. The
probe showed the cap holding: `value_target_fraction` ran 1.000, 1.000, 0.965,
0.973 over iterations 1-4. The memory is bounded by construction at ~5.7 GB, so
what a loose clock actually costs is target coverage, not RAM.

### The live signal to watch: `selfplay/steps_per_game`

The probe's near-random iterations got **longer** each time - 622, 716, 824 -
with `terminate_rate` falling 0.617 -> 0.521 -> 0.454, before turning at
iteration 4 (640 steps, 0.568). That is v6's unbounded length showing up exactly
where predicted, and then resolving as the net learned something.

If it is back near E8's ~160 actions by iteration 20, the 100-quiet-move clock
is behaving. If it is still above ~600, the clock is too loose and self-play is
spending its budget on manoeuvring that never resolves - which would be the
first real evidence against v6.

### Result: the scale-up lost by 118 Elo

E9 finished 120 iterations in **3.29 h**. Head to head against E7, 256 games, 32
simulations, `random_opening_plies=3`, `max_num_steps=3000`:

```
A = E9 (v6, batch 1024, 120 iters, 3.29 h)
B = E7 (v5, batch 256,  160 iters, 1.30 h)

A wins 86 (33.6%) | B wins 170 (66.4%) | draws 0 (of which truncated: 0)
A score 0.336 +/- 0.030   Elo diff A-B: -118
A as P0 0.281 | A as P1 0.391 | avg game length 334 plies
```

E9 had 4x the batch, 2.5x the wall clock, 1.5x the gradient steps (7,680 vs
5,120) and 4x the data, and is **118 Elo weaker** than a 1.3 h pilot. The
scale-up did not merely fail to help; this configuration cost more than all of
that compute was worth.

**The prime suspect is v6.** Capture clocks have now lost four times: -95 (v2 vs
v1), -112 (v2 vs v3), +95 when an absolute cap replaced one (E7), and -118 here.
The mechanism was visible live: finished games per iteration fell from 1,616 at
iteration 20 to ~500 by iteration 100 as the net strengthened, and
`value_target_fraction` slipped 1.000 -> 0.97. Same self-play compute, a third
of the terminal outcomes.

**It is confounded, and the confound is real.** E9 changed rules *and* scale
together. The competing explanation is that batch 1024 at `learning_rate=5e-4`
is mis-tuned - a 4x batch usually wants a larger LR, and E7's was kept unchanged
- which would also yield a weaker model with healthy curves. Nothing here
separates the two.

Two experiments do, and the first is cheap:

- **Rules, isolated (~1.3 h):** E7's exact pilot settings under v6, compared to
  E7 head to head. Any gap is purely the rules.
- **Scale, isolated (~3.3 h):** E9's settings under v5.

Until one of them is run, **do not attribute E9's loss to either cause**, and do
not treat full-size Epaminondas as a solved recipe.

### What E9 did establish

The sizing figures the TPU rental needed, all previously unknown:

| quantity | value |
|---|---|
| steady-state iteration | **~95-104 s** at batch 1024 (self-play ~90 s of it) |
| replay buffer | **7.88 GiB** at `replay_buffer_iters=4` |
| peak host RAM | **~30 GB** of 47, peaking during the data-state write, not training |
| `data_state.pkl` | **14.1 GB** (7.6 GB at iteration 20; it grew with the tail) |
| GPU memory | ~14.5 GiB of 16 GiB, no `train_micro_batches` needed |
| 120 iterations | 3.29 h |

Two of these are v6-inflated: `data_state.pkl` nearly doubled over the run
because longer games inflate the held-back tail, and the host-RAM peak follows
it. Under v5 both would be smaller.

### `eval/vs_baseline` was misleading in both directions

| iteration | 0 | 20 | 40 | 60 | 80 | 100 | 120 |
|---|---|---|---|---|---|---|---|
| win rate vs E7 | 0.004 | 0.006 | 0.190 | 0.192 | 0.237 | **0.131** | 0.290 |

The iteration-100 dip is ~4 standard errors on 256 games and looked, at the
time, like confirmation that the game-length blow-up was costing strength. It
recovered fully by iteration 120. **It was noise**, and a causal story was built
on it prematurely. The metric is raw policy sampling; CLAUDE.md already says it
is too noisy to rank runs, and a single point of it is worth less than that.

Note also that the final 0.290 - the run's best - accompanied a model that then
lost its head-to-head 0.336. The two are not inconsistent (different search
budgets), but it is a further reason to decide only on `model_tournament.py`.

### Trained v6 games are short; the random-play figures overstated the risk

`max_num_steps=3000` produced **0 truncated games** at an average length of 334
plies. The v6 section above warned that 900 was too small based on random play
(28% over 900 actions); for *trained* play 900 would have been ample. Keep 3000
as the safe default, but the truncation risk in practice is small.

### The seat asymmetry: not a bug, and probably not established either

E9 vs E7 gave A 0.281 as P0 and 0.391 as P1, which looked like a fourth sighting
of the player-1 advantage seen in the E8 matches and the baseline-vs-random
check. It was checked directly.

**There is no harness bug.** E7 against its own checkpoint, 256 games, same
settings, scored **0.500 +/- 0.031, Elo +0**. A seat or rotation mismatch in
`model_tournament.py` would have shown up here and did not, so no existing Elo
number in this log is contaminated.

**The rules are not structurally biased either.** 256 random-vs-random games on
CPU split 0.488 / 0.512 by seat. Of those, **69% ended on the quiet clock**
rather than by a real win, and even that tiebreak-decided subset split
0.481 / 0.525 - so "black wins ties" is not quietly handing player 1 games.

Quantified across every measurement rather than counted as sightings:

| test | games | P1 score | deviation | significance |
|---|---|---|---|---|
| random vs random | 256 | 0.512 | +0.012 | 0.4 sigma |
| E7 vs E7 (mirror) | 256 | 0.531 | +0.031 | 1.0 sigma |
| E9 vs E7 | 256 | 0.555 | +0.055 | 1.8 sigma |

Pooled over 768 games: **+0.033, ~1.8 sigma** (p ~ 0.07). Suggestive, not
established.

**The "four independent sightings" framing was wrong.** Each measurement is
individually consistent with zero, and pooling them still does not clear
significance; repetition of an underpowered comparison is not accumulating
evidence. There may be a real ~+0.03 (~20 Elo) edge to black, which would be
unsurprising given black moves second and wins the tiebreak, but demonstrating
it needs ~2,000 games, not 256.

Not worth chasing: a 20 Elo seat effect changes no decision here, and
`model_tournament.py` plays every pairing seat-swapped, so it cancels out of
every comparison.

## E10 (v7): the clock and verdict planes, and a design mistake in one of them

**Question.** The observation is fully canonical - own/opponent planes, rows
flipped for black - so the network could see neither the quiet-move clock nor
who wins if it fires. About 69% of random-play games are decided by that
machinery. v7 added two constant planes: 7 is `quiet_moves / MAX_QUIET_MOVES`,
8 is +1 if the mover wins should the game end now, else -1.

**Setup.** E7's exact pilot recipe (`selfplay_batch_size=256`,
`training_batch_size=2048`, `num_updates_per_iter=32`) run to 700 iterations
instead of 160 - 5.94 h against E7's 1.30 h. `checkpoints/epaminondas_20260919001750/`.
E9's batch-1024 config was deliberately not reused, since it had just lost by
118 Elo for reasons still unidentified.

### Result: -92 Elo, and a seat asymmetry three times E7's

```
A = E10 (v7 planes, 700 iters, 5.94 h)   B = E7 (160 iters, 1.30 h)

A wins 95 (37.1%) | B wins 161 (62.9%) | draws 0 (of which truncated: 0)
A score 0.371 +/- 0.030   Elo diff A-B: -92
A as P0 0.188 | A as P1 0.555   <- a 0.367 gap, ~12 sigma
avg game length 91.9 plies
```

Mirror matches, same settings, 256 games each:

| model vs itself | as P0 | as P1 | black's edge |
|---|---|---|---|
| E7 | 0.469 | 0.531 | +0.031 (1.0 sigma) |
| E10 | 0.398 | 0.602 | **+0.102 (3.3 sigma)** |

E10 is a materially different player by colour: respectable as black, weak as
white. E7's seat effect was indistinguishable from noise; E10's is not.

### Why: plane 8 degenerates into a colour identifier

Verified directly on the opening position:

```
white (player 0) verdict plane = -1.0
black (player 1) verdict plane = +1.0
last_capturer = -1
```

The start is an exact rank-for-rank mirror with nobody having captured, so the
tiebreak chain falls straight through to "black wins ties". **Every game
therefore begins with white told it is losing and black told it is winning**,
before a piece has moved, and the same holds in any symmetric position - which
is most of the opening phase.

This was a design error, not a bug: the plane computes exactly what it was
specified to compute. The specification was wrong. The stated virtue - "this is
how the black-wins-ties rule reaches a network that is never told which colour
it is" - is precisely the defect, because it breaks the canonical representation
wherever the position is symmetric, and it is symmetric exactly when the plane
carries no strategic content.

**Proposed fix:** scale the verdict by the clock,
`verdict * (quiet_moves / MAX_QUIET_MOVES)`. At the start the clock is 0, so the
plane is 0 for both colours and canonicalisation is preserved; it grows to +/-1
as a quiet stretch runs on, which is when the tiebreak actually decides
anything. The information then arrives in proportion to its relevance.
Untested.

### The larger problem: two scale-ups have now lost to a 1.3 h pilot

| run | what changed | wall clock | result vs E7 |
|---|---|---|---|
| E9 | 4x batch, v6 | 3.29 h | **-118 Elo** |
| E10 | 4.4x iterations, v7 planes | 5.94 h | **-92 Elo** |

E9 changed batch size, E10 changed neither batch size nor rules. The common
thread is that **both spent several times E7's compute and came out weaker**,
and no single explanation covers both. The seat defect accounts for part of
E10's loss but not all of it: E10 scored only 0.555 even as black, against a
model it out-trained 4.4x.

This is the finding that matters for the TPU rental, because a rental buys
exactly the thing that has now twice failed to convert into strength. Do not
rent until a run beats E7.

**Next experiment:** E7's exact recipe at E7's exact length (160 iterations,
~1.3 h) under v7. If that matches E7, the planes are neutral and the problem is
in longer schedules. If it also loses, the problem is neither scale nor
schedule, and the next suspects are the v7 planes themselves and whether E7 is
an unusually strong checkpoint rather than a typical one.

## E11 (v8): the cleanest experiment yet, and the plane loses 351 Elo

**Setup.** E7's recipe, matched to the iteration and near-matched on wall clock.
The *only* difference is the observation.

| | E7 | E11 |
|---|---|---|
| iterations | 160 | 160 |
| wall clock | 1.301 h | 1.273 h |
| frames | 15,728,640 | 15,728,640 |
| batch / train batch / updates | 256 / 2048 / 32 | 256 / 2048 / 32 |
| learning rate | 5e-4 cosine | 5e-4 cosine |
| observation planes | 7 | **8** |

`checkpoints/epaminondas_20260919064540/`. This is the first Epaminondas
comparison that moves one variable: E9 changed batch *and* rules, E10 changed
schedule *and* planes.

### Result

```
A = E11 (v8)   B = E7

A wins 30 (11.7%) | B wins 226 (88.3%) | draws 0 (of which truncated: 0)
A score 0.117 +/- 0.020   Elo diff A-B: -351
A as P0 0.125 | A as P1 0.109
avg game length 135.9 plies
```

### The mirror is the instrument, not the head-to-head

The head-to-head above shows almost no seat gap (0.125 vs 0.109), and reading
that as "the asymmetry is fixed" is a mistake: E11 is weak enough to lose from
both seats, which masks it. Against itself:

| model vs itself | as P0 | as P1 | black's score |
|---|---|---|---|
| E7 | 0.469 | 0.531 | 0.531 |
| E10 (v7, two planes) | 0.398 | 0.602 | 0.602 |
| **E11 (v8, signed plane)** | **0.227** | **0.773** | **0.773** |

**The v8 fix made the asymmetry worse than the defect it was meant to repair.**
The series tracks how much colour the observation leaks.

### Why scaling by the clock did not fix the leak

`quiet_moves` is zero only at the very start of a game and immediately after a
capture. By the second or third move, with no captures yet and the position
still near-symmetric, the tiebreak still resolves to black, so white sees
-0.02, -0.03, -0.04 ... and black the mirror image. v8 removed the leak from
exactly one position and let it return on the next move as a *ramp* - and a
smoothly ramping signal appears to be more learnable than a constant one.

The likely dynamic: the plane says black is favoured in every near-symmetric
position; self-play makes that self-fulfilling; the feedback loop runs away.
E11's self-play collapsed to 27-move games with `terminate_rate` 1.000 - black
rushes, white's policy rots.

### The motivating statistic was measured in the wrong regime

v7 was justified with "about 69% of games are decided by the clock machinery".
That figure came from **random** play, where games run 622+ steps. The clock
needs 100 quiet moves, and `quiet_moves` increments once per move:

| | steps | moves | can reach the clock? |
|---|---|---|---|
| E11 self-play (final) | 80 | ~27 | no |
| E11 vs E7 | 136 | ~45 | no |
| E7 self-play (E8) | 153-190 | 51-63 | no |

**In trained play the tiebreak essentially never fires.** The blind spot the
planes were added to fix is, in the regime that matters, largely not there - so
the planes contribute little information and all of the leak. The 69% figure was
quoted in the v7 code comments, RECIPES and the commit message before anyone
checked whether it applied to trained self-play. It does not.

### Training curves were the best of any run, and meant nothing

E11 finished at `policy_loss` 0.410 and `value_loss` 0.144, both far below E7's,
with `terminate_rate` 1.000 and 80-step games - and it is the weakest model
measured here by a wide margin. The same signature as the `completed_unscaled`
arm, which also had lower policy loss and shorter decisive games and lost
119-0. CLAUDE.md's rule held again: only a head-to-head decides.

### Caveat: one sample per arm

Nobody has measured this recipe's run-to-run variance, and E11's collapse has
the character of a degenerate self-play attractor. The mirror series makes the
plane the strong favourite, but "the plane caused it" currently rests on a
single run. **E12 repeats E11 with `seed=1`** to settle it: if E12 also collapses
the plane is guilty; if E12 lands near E7 then this recipe is unstable, several
of this log's conclusions have been noise read as signal, and that instability
matters far more for a rental than any plane does.

## E12: the seed repeat convicts the plane

**Question.** E11's collapse could have been a degenerate self-play attractor
rather than the observation plane: nobody had ever measured this recipe's
run-to-run variance. E12 is E11 with `seed=1` and nothing else changed.
`checkpoints/epaminondas_20260919145555/`, 160 iterations in 1.282 h against
E11's 1.273 h and E7's 1.301 h.

### Result: worse than E11

```
A = E12 (v8, seed 1)   B = E7

A wins 22 (8.6%) | B wins 234 (91.4%) | draws 0 (of which truncated: 0)
A score 0.086 +/- 0.018   Elo diff A-B: -411
A as P0 0.008 | A as P1 0.164
avg game length 135.1 plies
```

**As white, E12 scored 0.008 - one point in 128 games.**

### Both seeds, and the leak scales with how much colour the plane exposes

| run | observation | vs E7 | black's score in its own mirror |
|---|---|---|---|
| E7 | 7 planes | - | 0.531 (1.0 sigma) |
| E10 | v7, clock + raw verdict | -92 | 0.602 (3.3 sigma) |
| E11 | v8, signed clock, seed 0 | **-351** | **0.773 (8.8 sigma)** |
| E12 | v8, signed clock, seed 1 | **-411** | **0.672 (5.5 sigma)** |

Two independent seeds of a single-variable change, both catastrophic, both with
a large white handicap. **This is not seed luck.** The variance between seeds is
real (-351 vs -411, mirrors 0.773 vs 0.672) but small next to the effect.

### What this settles, and what it does not

Settled: **the observation plane is the cause of E10, E11 and E12's weakness.**
Any plane derived from the full tiebreak chain carries the black-wins-a-mirror
default, which is a colour signal in every near-symmetric position, and
self-play amplifies it into a policy that cannot play white.

Not settled: **E9's -118 remains unexplained.** E9 was 7-plane v6 with no leak,
so its loss has a different cause - batch size, learning rate, or something
else. That is still the open question for scaling, and the one that matters for
a rental.

### The fix, if the information is still wanted

The leak is entirely in the fallback, not the comparison. `_advancement_winner`
returns -1 for an exact mirror, and it is the `select(advantage >= 0, advantage,
tiebreak)` step that replaces that with "black". A plane carrying **advancement
only** - +1, -1, or **0 when tied** - is symmetric by construction: a mirrored
position reads 0 for both colours, and no amount of self-play can turn it into a
colour identifier. The black-default and `last_capturer` stay hidden, which
costs almost nothing, since trained games run 27-63 moves and never reach the
100-quiet-move clock anyway.

Untested. Given that two attempts at this plane have now cost 92, 351 and 411
Elo, the bar for a third should be a single-variable run against E7 at matched
wall clock, with the **mirror** checked before anything else.

## E13: the control that reframes E9 through E12

**Question.** Every comparison in this log is against E7, and nobody had checked
whether E7 is a typical run or a lucky one. E13 is E7's recipe with **seed=1**
and no observation plane (v9), 160 iterations in 1.269 h against E7's 1.301 h.
`checkpoints/epaminondas_20260919170010/`.

### Result: -140 Elo. E7 is not reproducible - because E7 is v5

```
A = E13 (7 planes, seed 1, v6 rules)   B = E7 (v5 rules)

A wins 79 (30.9%) | B wins 177 (69.1%) | draws 0 (of which truncated: 0)
A score 0.309 +/- 0.029   Elo diff A-B: -140
A as P0 0.234 | A as P1 0.383   avg game length 367 plies
```

Collecting every head-to-head against E7:

| run | rules | planes | iterations | vs E7 |
|---|---|---|---|---|
| **E13** | **v6** | **7** | **160** | **-140** |
| E9 | v6 | 7 | 120 @ batch 1024 | -118 |
| E10 | v7 | 9 | 700 | -92 |
| E11 | v8 | 8 | 160 | -351 |
| E12 | v8 | 8 | 160 | -411 |

**E7 is the only model trained under v5.** Every v6 run sits near -120 to -140,
and this log already records capture clocks costing 95 and 112 Elo (v2 vs v1, v2
vs v3). E13's -140 is that same penalty measured a third time. E7 is not a lucky
draw; it is trained under rules worth ~130 Elo more, and **the correct baseline
for a v6 run was never 0.**

### Correction: scaling is not broken

E9 - the batch-1024 run whose -118 prompted "do not rent until a run beats E7" -
scored **better than E13's -140**, at the same rules and plane count. The two are
about one standard error apart, i.e. indistinguishable. **E9's apparent failure
was the rules, not the batch size**, and the earlier conclusion that this recipe
does not reward more compute was drawn against a v5 yardstick that no v6 run
could match. Scaling looks neutral-to-fine.

### Correction: the mirror asymmetry is not a clean plane diagnostic

| run | planes | black in its own mirror | mirror game length |
|---|---|---|---|
| E7 (v5) | 7 | 0.531 | 342 plies |
| **E13 (v6)** | **7** | **0.625** | **498 plies** |
| E10 (v7) | 9 | 0.602 | - |
| E12 (v8) | 8 | 0.672 | - |
| E11 (v8) | 8 | 0.773 | 76 plies |

**E13 has no plane and still shows 0.625.** The claim in the E12 write-up that
the series "tracks how much colour the observation leaks" is wrong. Two
mechanisms produce similar numbers:

- **Long games reach the tiebreak**, which resolves to black. E13's mirror games
  run 498 plies (~166 moves), well past the 100-quiet-move clock. This is a
  property of v6, not of any observation.
- **The plane leaks colour directly**, which is the only thing that can explain
  E11: its mirror games averaged 76 plies, far too short to reach the clock, yet
  black scored 0.773.

Do not read a mirror asymmetry as evidence about the observation without also
checking the game length.

### What still stands: the plane is harmful

E11 and E12 differ from E13 only in the plane - same rules, same 160 iterations,
same batch, matched wall clock. -351 and -411 against E13's -140, so **the v8
plane costs roughly 210-270 Elo**. That rests on head-to-heads at a properly
matched baseline and is unaffected by the mirror confusion above. Unwiring it in
v9 was right; the magnitude claimed in the E11/E12 write-ups (350-410) was
inflated by measuring against a v5 model.

### The open question is now the rules, not scaling

v6's capture clock appears to cost ~130 Elo against v5, which is the third
consistent measurement of a capture clock being expensive. v6 is a deliberate
design choice taken on the grounds that an absolute cap pays whoever is ahead to
run the clock out, and it is kept for that reason - but its cost should be
recorded honestly rather than attributed to batch sizes and observation planes.

### E9 vs E13 head to head: scaling buys nothing

E9 and E13 are both v6 rules at 7 planes, so they play each other directly - no
need to chain through E7, and no v5 confound. 256 games, 32 simulations,
`max_num_steps=3000`:

```
A = E9 (batch 1024, 120 iters, 3.29 h)   B = E13 (batch 256, 160 iters, 1.27 h)

A wins 116 (45.3%) | B wins 140 (54.7%) | draws 0 (of which truncated: 0)
A score 0.453 +/- 0.031   Elo diff A-B: -33
A as P0 0.445 | A as P1 0.461   avg game length 475.8 plies
```

E9 spent **2.6x the wall clock, 4x the batch and 50% more gradient updates**
(7,680 vs 5,120) and came out slightly worse. At ~1.5 sigma, parity is not
excluded; an improvement is.

**This supersedes the "scaling looks fine" note in the E13 section above**, which
was inferred indirectly from -118 vs -140 against E7. Comparing two runs through
a third model of different rules is noisy; the direct head-to-head is the
instrument. Both statements can be true and only the second matters: E9 is not
118 Elo worse than a pilot (that was a v5-baseline artefact), *and* scaling from
batch 256 to 1024 currently gains nothing.

**Prime suspect: the learning rate.** E9 quadrupled the batch and kept
`learning_rate=5e-4`. More data per step at the same step size, with no gain, is
what an under-stepping optimizer looks like. The test is E9's config at
`learning_rate=1e-3` played against E9 itself - one variable, and the opponent
already exists. ~3.3 h. Nothing about a rental should be decided before it runs.

## E14: the learning rate was the problem, and scaling now works

**Question.** E9 spent 2.6x the wall clock at 4x the batch and gained nothing
(0.453 against E13). The untested suspect was the step size: E9 quadrupled the
batch and kept `learning_rate=5e-4`.

**Setup.** E9's config with `learning_rate=1e-3` and **the same seed**, so the
step size is the only difference in the run. 120 iterations in 3.133 h against
E9's 3.29 h. `checkpoints/epaminondas_20260919191528/`.

### Result: +63 Elo over E9, +38 over the pilot

```
A = E14 (lr 1e-3)   B = E9 (lr 5e-4)      A score 0.590 +/- 0.031   Elo +63
A = E14             B = E13 (pilot)       A score 0.555 +/- 0.031   Elo +38
```

Every comparison here is a direct head-to-head between v6, 7-plane models - no
chaining through E7:

| comparison | Elo |
|---|---|
| E9 (batch 1024, lr 5e-4) vs E13 (pilot) | **-33** |
| E14 (batch 1024, lr 1e-3) vs E13 (pilot) | **+38** |
| E14 vs E9 (learning rate isolated, same seed) | **+63** |

**Fixing the learning rate turned a scaling loss into a scaling win**, worth
about 70 Elo, consistently across both comparisons. E9's failure to convert more
compute was an under-stepping optimizer, not a ceiling in the recipe.

**Temper the magnitude.** E14 spent **2.5x E13's wall clock for +38 Elo**, at
~1.8 sigma. Scaling works; it is not lavishly efficient. Anyone sizing a rental
should plan on real but moderate returns.

### The mid-run eval would have given the wrong answer, again

| iteration | 40 | 60 | 80 | 100 | 120 |
|---|---|---|---|---|---|
| E9 (5e-4) win vs E7 | 0.190 | 0.192 | 0.237 | 0.131 | 0.290 |
| E14 (1e-3) win vs E7 | 0.052 | 0.128 | 0.259 | 0.457 | **0.473** |

At iteration 40 E14 was scoring a third of E9 and looked like a failure. It
finished 63% ahead. A higher learning rate decays from a higher peak, so the run
is still taking large steps in mid-schedule and converges late. **Third time in
this log that reading `eval/vs_baseline` early would have produced the wrong
conclusion.**

E14's 0.473 against E7 is the best any v6 model has reached - close to parity
with the v5 reference while carrying v6's ~130 Elo rules penalty.

### Next

The learning rate is now known to matter and is not known to be optimal. 1e-3
is 2x E9's; the batch is 4x the pilot's, and a linear rule would suggest 2e-3.
The obvious follow-up is E14's config at 2e-3 against E14, one variable, ~3.1 h.

### The seat asymmetry is a v6 rules property, not an observation one

E14's mirror, and the series it completes:

| model | rules | planes | black in its own mirror | mirror game length |
|---|---|---|---|---|
| E7 | **v5** | 7 | **0.531** | 342 plies |
| E13 | v6 | 7 | 0.625 | 498 plies |
| **E14** | v6 | 7 | **0.727** | 366 plies |
| E10 | v7 | 9 | 0.602 | - |
| E12 | v8 | 8 | 0.672 | - |
| E11 | v8 | 8 | 0.773 | 76 plies |

**Two explanations have now been tried and both are wrong.** The E12 write-up
said the series tracks how much colour the observation leaks - refuted by E13
and E14, which have no plane and still show 0.625 and 0.727. The E13 write-up
then said long games reach the black-favouring tiebreak - refuted by E14, whose
games are *shorter* than E13's (366 vs 498 plies) yet whose asymmetry is
*larger*.

What survives: **every v6 model shows a large black advantage and the single v5
model does not**, and the effect grows with strength - E14 is the strongest v6
model measured and has the biggest gap. The natural reading is that v6 contains
a real imbalance favouring black, most plausibly that the tiebreak's
black-wins-a-mirror default is something a strong player can steer towards, and
that stronger play exploits it harder.

**This is a rules design issue, not a bug**, and it is independent of the
observation: E14 has no extra plane. A strong v6 model playing black scores
about 0.73 against itself. Worth deciding whether that is acceptable, since it
means seat allocation matters a great deal in v6 - and note that
`model_tournament.py` plays every pairing seat-swapped, so it does not
contaminate any Elo in this log.

Not diagnosed further: nobody has measured what fraction of strong-play v6 games
actually end on the clock versus by a real win, which would separate "black
steers to the tiebreak" from "black simply has the better side under v6".

## Gauntlet vs a classical alpha-beta engine: the models are ~250-310 Elo behind

Every number above is relative - one of our checkpoints against another. This is
the first measurement against something outside the family, and it is
unflattering.

Run with `tdgauntlet` (`~/calr3gh/tdgauntlet`), a tournament server whose Rust
Epaminondas is an independent implementation of pgx's v6 rules, held to pgx by a
conformance test. Config `examples/az_v_alphabeta_gauntlet.toml`, a full round
robin, one minimatch per randomised **first move** (`plies = 1`, which is three
pgx actions; 114 distinct first moves exist). 120 games in 1 h 46 m.

```bash
# model clients, in the pgx environment, sharing the GPU
cd ~/calr3gh/tdgauntlet/clients/jax
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ~/.venv/bin/python3 -m tdg_jax \
    --port 9330 --sims 256 --allow-rules-mismatch \
    --checkpoint ~/calr3gh/pgx/checkpoints/epaminondas_20260918013629/000160.ckpt   # E7
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ~/.venv/bin/python3 -m tdg_jax \
    --port 9331 --sims 256 --allow-rules-mismatch \
    --checkpoint ~/calr3gh/pgx/checkpoints/epaminondas_20260919191528/000120.ckpt   # E14

cd ~/calr3gh/tdgauntlet && export PATH="$HOME/.cargo/bin:$PATH"
./target/release/tdgauntlet run examples/az_v_alphabeta_gauntlet.toml
```

### Result

```
player      score    Elo   counted     excluded      W-D-L   think ms
alphabeta  100.0%  >+999        37       3 (8%)     77-0-3        861
az-e14      18.5%   -257        27     13 (32%)    23-0-57       4149
az-e7       14.3%   -311        28     12 (30%)    20-0-60       5283

  az-e7  v az-e14     44.4%  over  9 counted, 11 excluded
  az-e7  v alphabeta   0.0%  over 19 counted,  1 excluded
  az-e14 v alphabeta   0.0%  over 18 counted,  2 excluded
```

**The engine won 77 of 80 games and took every counted minimatch from both
models.** 0.0% is not rounding: neither model ever won both games of an opening
against it. Our three wins were `az-e7` once (29 moves) and `az-e14` twice (83
and 270 moves).

**And the models had 5-6x the thinking time.** `budget_sims = 256` was chosen to
be roughly comparable to `budget_ms = 1000`; in practice the models averaged
4,149 ms and 5,283 ms per move against the engine's 861 ms. The gap is therefore
*understated* here, not overstated. A fair equal-time comparison would need the
models at far fewer simulations, or the engine given seconds.

A minimatch is scored only when it is not split 1-1, on the grounds that a split
means the opening decided it. Exclusion ran 8% for the engine and 30-32% for the
models.

### E7 v E14 came out backwards, and should not be believed

tdgauntlet has `az-e7` at 44.4% against `az-e14`, where pgx's own 256-game
tournament had E7 **+128 Elo** ahead. The gauntlet figure rests on **9 counted
minimatches** after an 11-of-20 exclusion rate. Prefer the pgx number; this is a
sample-size artefact, not a contradiction worth explaining.

### It does *not* corroborate the v6 black advantage - and that is a puzzle

White won **60 of 120** games overall, and **19 of 40** (0.475) in the
strength-balanced `az-e7` v `az-e14` pairing. No black advantage is visible.

That sits badly against the pgx mirror measurements taken the same day, where
black scored 0.625 (E13) and 0.727 (E14) against identical opponents. Three
differences could account for it, none tested:

- **Search size.** The pgx mirrors ran 32 simulations; this ran 256. A black edge
  that a shallow search cannot avoid may simply be avoidable with more search.
- **Identical versus different opponents.** A self-mirror pits a policy against
  its own blind spots; two different models do not.
- **The harness.** pgx and the Rust implementation are conformance-tested on
  rules, not on tournament bookkeeping.

An earlier note in this log offered the high exclusion rate as corroboration of
the black advantage. **That was wrong**: exclusion means colour-and-opening
decided a given minimatch, not that one colour wins systematically, and the
white-win rate here is 0.475. The v6 seat asymmetry is real in pgx at 32
simulations and absent here at 256; which regime is representative is open.

### What it means

Every relative gain in this log - the data pipeline, the architecture, the
learning-rate fix - is movement within a family that sits **250-310 Elo below a
competent classical engine given less thinking time**. That is worth knowing
before renting hardware to make the family marginally better: the gap is not of
a size that a 2x-scale run closes.

The engine is not a strawman - `crates/games` plus an alpha-beta with a
transposition table, 8 threads, and an evaluation tuned against pgx - but it is
also not a research artefact. A strong classical engine being far ahead of a
small AlphaZero run is the expected result at this scale; the value here is the
number, and a fixed external opponent to measure future runs against.

## E15 (v10): nine evaluation-feature planes - faster per iteration, level at equal time

**Question.** Give the network the nine terms of tdgauntlet's alpha-beta
evaluator as observation planes (the six LEONIDAS heuristics plus that engine's
three advancement terms), and does it learn faster? See `pgx/_src/games/
epaminondas.py`; verified against a port of `clients/alphabeta/src/eval.rs` and
against `negamax.py` over 336 positions, and antisymmetric under a colour flip,
so it cannot repeat the v7/v8 colour leak.

**Setup.** E14's config and seed, 70 iterations in **4.19 h**.
`checkpoints/epaminondas_20260921161241/`. The observation is the intended
difference; three others crept in and are listed under caveats.

### It learns much faster per iteration

| iteration | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 120 |
|---|---|---|---|---|---|---|---|---|
| E15 win vs E7 | 0.036 | 0.199 | 0.548 | **0.801** | 0.562 | 0.521 | 0.610 | - |
| E14 win vs E7 | - | 0.003 | - | 0.052 | - | 0.128 | - | 0.473 |

E15 reached by iteration 30 what took E14 about 120. **But the planes cost 2.18x
per iteration** (205 s against 94 s), almost all of it `mobility`, which runs
`_line_info` for both colours at every MCTS node.

### At equal wall clock it is a tie

```
E15 vs E14   A score 0.477 +/- 0.031   Elo -16    (E15 had 4.19 h to E14's 3.13 h)
E15 vs E7    A score 0.516 +/- 0.031   Elo +11
```

E15 is the first v6-rules model to draw level with E7. But against the matched
control it is **level, with 34% more wall clock**: the per-iteration gain is
exactly consumed by the per-iteration cost. **The planes do not help at equal
time.**

### The results are badly intransitive

| pairing | Elo |
|---|---|
| E15 vs E14 | -16 |
| E15 vs E7 | +11 |
| E14 vs E7 | -128 |

E15 ties E7, E7 beats E14 by 128, so E15 should beat E14 by ~139. It ties.
**A ~155 Elo violation**, far outside the +/- 0.031 standard errors. Style
matchups here are large enough that a single opponent does not order these
models, which is worth remembering before quoting any one number - including
these.

### Caveats: three deviations from E14

WSL's OOM killer took the run twice (iterations 24 and 43), so it was resumed
from checkpoints three times, and to fit memory it ran with
`replay_buffer_iters=2` (E14 used 4) and `max_pending_steps` capped, which left
`value_target_fraction` at **0.814** where E14 held 1.000. Each is small, but
E15 vs E14 is no longer strictly single-variable. Given the result is a tie,
they are unlikely to have decided it.

### `eval/vs_baseline` was wrong for the fourth time today

It had E15 at 0.61 against E7 where E14 managed 0.473, which reads as a clear
lead; the head-to-head says level. The iteration-40 spike of 0.801 was noise and
was over-read at the time. **This metric should not be used to rank runs at
all**, which is what CLAUDE.md already says.

## Why a classical alpha-beta crushes a model given the same heuristics

`tdgauntlet` gauntlet, same openings as the earlier one (`seed = 1909`), E15
under 256-simulation MCTS and as a raw policy, against the Rust alpha-beta:

```
player       score    Elo   counted   excluded     W-D-L   think ms
alphabeta   100.0%  >+999        40    0 (0%)    80-0-0        873
e15-mcts     50.0%     -0        40    0 (0%)   40-0-40       5154
e15-policy    0.0%  <-999        40    0 (0%)    0-0-80          43
```

**Zero exclusions across 60 minimatches** - a perfect total order, where the
earlier gauntlet excluded 8-32%. The gaps are large enough that no opening ever
flipped a result.

The natural objection is that the model now has the engine's own heuristics, so
why is it still crushed? The search numbers answer it:

| | nodes/move | depth | ms/move | nodes/sec |
|---|---|---|---|---|
| alphabeta | **398,336** | 4 | 886 | 449,589 |
| e15-mcts | **768** (256 sims x 3 stages) | - | 5,384 | ~143 |

**519x fewer positions per move, in six times the wall clock.** Every MCTS node
is a forward pass of a 4.5M-parameter network; an alpha-beta node is a few
hundred nanoseconds of board arithmetic and a nine-term dot product - about
3,150x the throughput.

"The same heuristics" is the misleading part. The network gets the nine *terms*;
it does not get the hand-tuned weights that say what each is worth (material 30,
crossing 250, ...), and it does not get the search, which is where nearly all of
alpha-beta's strength lives. Alpha-beta's evaluation is *cruder* than the
network's in principle - nine linear terms against 4.5M parameters - and it wins
anyway by resolving tactics exactly, four full moves deep, with a transposition
table. With 256 simulations against a branching factor of 114 and
`max_num_considered_actions=16`, the model gets ~16 visits per root move and
effectively sees one move ahead.

**The policy-only run is the direct evidence.** `e15-policy` lost **0-80** to
`e15-mcts` on identical weights, differing only in whether search runs. If the
heuristic features had been distilled into positional judgement the raw policy
would be respectable; it is helpless. The features reach the input but have not
become an evaluation the network can apply without search.

Epaminondas compounds this - a capture removes an entire enemy line and reaching
the far rank wins outright, so one unseen tactic is usually terminal - but the
search gap is the story, not the game. A competent alpha-beta beating a 4 h
AlphaZero run is the expected result at this scale, and closing it needs orders
of magnitude more training, not better input features.

## E16 (v11): per-square features win by 95 Elo at equal wall clock

**Question.** E15 showed the nine heuristics help per iteration and cost 2.18x
per iteration, netting a tie. Does encoding them **per square** rather than as
constants broadcast over all 168 cells change that?

**v11 does three things** (`pgx/_src/games/epaminondas.py`):

- drops `material`, `crossing`, `advancement` and `tiebreak` - each an exact
  linear functional of the two piece planes, so one pooling layer computes any
  of them and handing them over bought only bandwidth;
- replaces the scalar `mobility` with **`_travel`: eight per-square planes**,
  one per direction, giving at every piece the phalanx it heads and the room in
  front of it along each rank, file and diagonal;
- carries the clock raw instead of multiplied by the tiebreak, so it is
  linearly accessible and stays colour-blind.

**Setup.** E14's config and seed, 84 iterations in **3.01 h** against E14's
3.13 h - equal wall clock, which E15 never managed.

### Result

```
E16 vs E14   A score 0.633 +/- 0.030   Elo  +95   (3.01 h vs 3.13 h)
E16 vs E7    A score 0.785 +/- 0.026   Elo +225
```

| observation | vs E14 | wall clock |
|---|---|---|
| v10 - nine heuristics as broadcast scalars (E15) | -16, a tie | 34% *more* than E14 |
| **v11 - eight per-square travel planes + four scalars (E16)** | **+95** | **equal** |

**The same heuristics, encoded per square instead of as constants, are worth
about 111 Elo** - and cost less: 1.43x per iteration against v10's 2.18x,
because `_travel` reuses the run tables `legal_action_mask` already builds in
the same step, so XLA shares the scans and only the reductions are saved
(1.31 ms against the `mobility` scalar's 2.51 ms at batch 1024).

**E16 is by a distance the strongest model here.** E7 had beaten everything all
session - E9 -118, E13 -140, E14 -128, E15 +11 - and E16 takes it by 225.

### These results are transitive, unlike E15's

E16 vs E14 is +95 and E14 vs E7 is -128, predicting **+223** for E16 vs E7
against a measured **+225**. Two Elo of agreement, where E15's numbers violated
transitivity by ~155. The ordering here is real rather than a style matchup.

### Caveats

Two deviations from E14 remain, both forced by memory and both plausibly
*handicaps*, so +95 is if anything conservative: `replay_buffer_iters=2` where
E14 used 4, and `max_pending_steps=512`, which left `value_target_fraction` at
0.830.

`max_pending_steps` was first set to E14's 1024 to protect that fraction. With
v11's 19 planes the held-back tail is 1024 rows x 1024 slots x 13,446 B =
13.2 GiB, which put the run into swap and slowed it from 134 to 204 s per
iteration - the sizing was done with v10's 11,430-byte samples and not redone.
It was reverted to 512 and the run restarted. **Check the tail arithmetic
against the current sample size whenever the plane count changes.**

### The one number that did not mislead

`eval/vs_baseline` ran 0.017, 0.103, 0.200, 0.232, 0.542, 0.635, **0.706** - the
highest of any run, and this time the head-to-head agreed. That does not rehabilitate
it: it was wrong four times out of four earlier today, and one agreement is not
a record. Keep deciding on head-to-heads.

## E17 - finish E16's schedule: +386 Elo from letting the cosine land

E16 was a 264-iteration schedule stopped at **iteration 84**, because the
pilot's budget was 3 hours, not because it had converged. E17 resumes that
checkpoint and runs the schedule out.

```
resume_from=checkpoints/epaminondas_20260922033505/000084.ckpt
seed=0 architecture=boardformer selfplay_bf16=true continue_games=true
num_simulations=32 playout_cap_prob=0.25 fast_num_simulations=8
selfplay_batch_size=1024 max_num_steps=384 training_batch_size=4096
num_updates_per_iter=64 replay_buffer_iters=2 max_pending_steps=512
learning_rate=1e-3 weight_decay=1e-4 warmup_steps=100 grad_clip_norm=1.0
lr_schedule=cosine eval_interval=12 max_num_iters=264
```

This is a **deliberate warm restart** of the learning rate: iteration 84 of 264
puts the cosine back at ~7.8e-4, decaying to 1e-4 by the end. Policy loss duly
jumped to 1.33 and then fell to 0.72 over the run.

### Result

```
E17 vs E16   A score 0.906   Elo +386   (9.06 h total vs 3.01 h)
E17 vs E7    A score 0.941   Elo +484
E16 vs E7    A score 0.773   Elo +216   (the +225 on record, replayed)
```

Ladder fit, E7 anchored: **E17 +548 +/- 33**, E16 +194 +/- 23. Fit and
head-to-heads agree to ~30 Elo.

**Six more hours of the same recipe bought 386 Elo** - four times what the v11
observation itself was worth (+95). E16 was not a converged model being
improved on; it was a model stopped a third of the way through its schedule.
This is the standing "most gains arrive late, as the cosine decays" note
showing up at its full size, and it is worth remembering before reading any
future pilot as a converged result.

### First win over tdgauntlet's alpha-beta

50 games, 25 randomised first moves played twice with colours swapped, seed
1909 - the same openings E7 and E14 played. E17 at 256 sims, alphabeta at 1 s.

| player | score | Elo | counted | excluded | W-D-L | think ms |
|---|---|---|---|---|---|---|
| **e17** | **84.6%** | **+296** | 13 | 12 (48%) | 34-0-16 | 5013 |
| alphabeta | 15.4% | -296 | 13 | 12 (48%) | 16-0-34 | 813 |

On the same seed E7 scored **-311** and E14 **-257**, and E15's policy-only lost
**80-0**. E17 is the first model here to beat the alpha-beta at all.

Two caveats. Only **13 minimatches counted** - 48% split 1-1 and were excluded
as decided by the opening - and tdgauntlet's own rule of thumb puts the
standard error at 7-10 points for 20-50 counted minimatches, so 13 is worse
than that: the sign is solid, the margin is not. And E17 spent **5.0 s a move
against the alpha-beta's 0.8 s**, so this is not a like-for-like budget.

### The run died two iterations short

WSL was torn down at ~06:36, after iteration 262 of 264, with no traceback and
no `Killed` in the log - the same silent signature as the three earlier deaths,
and the second since `.wslconfig` was given 50 GB of memory and a 32 GB swap on
`E:`. Memory sat flat at 22 GiB of 49 for the whole run with swap untouched at
4/32, so in-VM pressure does **not** explain this one. The final checkpoint is
`000252.ckpt`, 95% of the schedule with the cosine already near its floor, and
it is what every number above was measured with.

`eval_interval=12` is what kept the cost to 12 iterations rather than the whole
run. Keep it there.
