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

### Result

Pending.
