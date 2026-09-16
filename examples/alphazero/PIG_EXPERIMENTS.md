# Pig experiments

Pig is the first stochastic game trained here, and the only one with a known
exact solution, which makes it a useful test of the training pipeline itself:
any gap to optimal play is a property of the method, not of an unknown opponent.

Measure everything with `pig_optimal.py`, which value-iterates the exact win
probabilities and reports, per checkpoint:

- `agree` - decision agreement with optimal play over every reachable state;
- `played` - agreement restricted to states the model actually reaches in play,
  where it has a real choice (after a 1, holding is forced). The honest number;
- `v_rmse` - value head against the true win probabilities, on [-1, 1];
- `vs optimal` - seat-balanced win rate against the optimal player.

Reference points on that last column: **optimal 0.500** (it concedes the first
move in half the games, and the first player wins 53.06%), **hold-at-20 0.461**,
**always-roll 0.000**.

## Setup

`architecture=mlp` with `mlp_onehot_bins=101`: pig's observation is six counters,
not a board, so there is no spatial structure to exploit, but the value function
turns sharply on thresholds ("does banking now reach 100?"). Each feature is
one-hot encoded over 0..100 next to its scaled raw value, which gives the trunk
a near-tabular input while keeping something smooth to generalise along.

## E1: baseline recipe, 60 iterations (~25 min)

`selfplay_batch_size=1024 num_simulations=32 training_batch_size=4096
replay_buffer_iters=4 lr_schedule=cosine learning_rate=1e-3`

| checkpoint | agree | played | v_rmse | vs optimal | hold@(0,0) |
|---|---|---|---|---|---|
| untrained | 0.479 | 0.502 | 0.679 | 0.096 | 2 |
| iteration 40 | 0.564 | 0.720 | 0.337 | 0.153 | 59 |

The **value head learns well** (RMSE 0.68 -> 0.34, tracking the true win
probability within ~0.05 at 0-0) while the **policy head stays nearly flat**:
p(roll) sits at 0.6-0.8 at every turn total and crosses 0.5 only near 60, so the
model rolls to ~59 where optimal play holds at 21.

### Why: mctx samples each chance outcome once per edge

mctx stores one sampled successor per tree edge. In a stochastic game, Q for
that edge therefore stays conditioned on a single die roll no matter how many
simulations run - extra simulations deepen the tree but never resample. Pig's
hold/roll value gap is only a few percent, so the policy-improvement signal is
buried in that variance and the target barely moves off the prior.

This is search variance, not a broken tree: replacing the network with an oracle
(exact optimal values and policy) made the same search hold correctly at turn
totals of 25, 40 and 45, with only a mild pull towards rolling at 30-35.

`config.chance_samples=N` averages an edge's value over N draws (default 1
leaves existing behaviour, and deterministic games, untouched). E2 tests it.

## E2: chance_samples=6, 25 iterations (~11 min)

Same recipe otherwise. Averaging the edge value over six draws costs only ~1.7x
per iteration (the extra evaluations batch into the same kernels), so this run
had less than half the baseline's wall clock.

It was **worse, not better**: `policy_loss` sat at 0.555 for all 25 iterations
(the baseline moved 0.57 -> 0.51), and the policy collapsed to holding at a turn
total of 77. Self-play itself stayed healthy (~72 steps per game, 87% real value
targets), so this is a target-quality problem, not a data problem.

The reason shows up in the oracle search: with only two actions,
`qtransform_completed_by_mix_value` normalises Q by the range across the tree -
which, with two actions, *is* the gap between them. Only the **sign** of that
tiny difference survives into the target, and the improved policy is quantised
to about {0, 0.198, 0.802, 1} whatever the true margin. Removing the chance
noise therefore did not sharpen the target; it just let a small constant bias in
the value head decide every sign, and the policy locked in and never escaped.
E1 drifted the same way in the other direction (hold threshold 59 at iteration
40, no threshold at all by 60).

## The knowledge is in the value head

Playing the *same networks* by one-ply expectimax over the six die faces - using
the value head only, and never consulting the policy head:

| model | as its policy head | as value head + expectimax |
|---|---|---|
| E1, iteration 60 | 0.100 (no threshold) | **0.428** (threshold 18) |
| E2, iteration 25 | 0.017 (threshold 77) | **0.382** (threshold 13) |

against hold-at-20's 0.461 and optimal's 0.500. The value head has learned the
game (its threshold of 18 is close to the true 21); the policy head, trained on
those quantised sign-only targets, throws that away.

## E3: qtransform=completed_unscaled, 60 iterations (~21 min)

E1's recipe with one change: `qtransform_completed_by_mix_value` with
`rescale_values=False`, which keeps Gumbel's completion of unvisited actions but
drops the per-node rescaling, so the size of an advantage - not just its sign -
reaches the policy target.

The oracle search predicted this: with rescaling it gets the probe wrong at turn
totals 30, 35 and 45 and returns quantised weights; without it, every probe is
right (rolls below 21, holds above). The rescaling was amplifying a
noise-sized Q difference into a large logit shift that overrode a correct prior.

| | agree | played | v_rmse | vs optimal |
|---|---|---|---|---|
| E1 iteration 60 | 0.560 | 0.715 | 0.298 | 0.100 |
| **E3 iteration 60** | **0.725** | **0.754** | **0.242** | **0.298** |
| E3, value head + expectimax | | 0.885 | | 0.448 |

Three times the score against optimal play at slightly less wall clock, and the
policy loss finally moves (0.595 -> 0.285, against E1's 0.57 -> 0.51 and E2's
flat 0.555). The value head improved too (0.428 -> 0.448 played by expectimax),
since better play makes better targets.

The policy head still trails its own value head (0.298 vs 0.448), so there is
more to get here - more iterations, and `chance_samples` is worth retesting now
that its failure mode (a sign-only target) is gone.

**For a stochastic game, search should expand
chance nodes rather than sample one successor per edge**, and a two-action game
needs a q-transform that preserves the magnitude of the advantage (e.g.
`qtransform_by_min_max` over a fixed [-1, 1]) rather than one that rescales it
to the range of the two actions being compared.
