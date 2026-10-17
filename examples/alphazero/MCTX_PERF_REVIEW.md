# mctx performance review (GPU/TPU)

Review of `google-deepmind/mctx` at `428bfb7` (2026-10-05), the version `pip
install mctx` tracks, for throughput on modern accelerators, and what each
finding means for this repo's self-play (Gumbel MuZero, 32 sims, batch 1024,
400 actions for Gess).

**Read this first:** the existing profile (`GESS_TPU_PLAN.md`, step 1) found
network inference is essentially all of a self-play move; tree operations were
negligible at 32 simulations. Nothing below changes that for the current Gess
recipe. The mctx-side costs grow with simulation count, with cheaper networks
(Pig, small MLPs), and with memory pressure, which is where these fixes pay.

All timings are **CPU only** (4 cores, JAX 0.11.2, a trivial stand-in network;
no GPU in the review container). They measure tree overhead, not end-to-end
self-play. Script: `bench_mctx_tree.py`.

## 1. `backward` rewrites whole `[B, N, A]` arrays on every step (measured, exact fix)

`search.backward` is `jax.vmap` over a `lax.while_loop`. When the loop
predicate is batched, JAX turns every loop iteration into
`select(pred, new_carry, old_carry)` over the whole carry. The carry is the
tree, so `children_values` and `children_visits` (each `[B, N, A]`) are fully
read and rewritten on every step up the tree. That makes each simulation cost
O(depth x B x N x A) instead of O(depth x B), and the search quadratic in
`num_simulations`. Confirmed in the compiled HLO: two full `[B,N,A]` selects
per iteration (`hlo.py` in the review scratchpad; the check is three lines).

Fix: batch the loop by hand. Gather and scatter B elements per step, give
finished rows an out-of-bounds index with `mode="drop"`, and loop while any row
is still active (`backward_batched` in the script, ~30 lines). Output is
**bit-identical** to stock mctx on every configuration tested.

| policy | B | A | sims | stock | batched backward | speed-up |
|---|---|---|---|---|---|---|
| gumbel | 128 | 400 | 32 | 97.6 ms | 64.8 ms | 1.51x |
| gumbel | 128 | 400 | 128 | 1173 ms | 296 ms | 3.96x |
| muzero | 128 | 400 | 32 | 1038 ms | 761 ms | 1.36x |
| muzero | 128 | 400 | 128 | 29.7 s | 12.5 s | 2.37x |

These times cover tree work only. Here that is a few percent of a move, so
expect a small end-to-end gain at 32 sims. The gain matters for deeper searches
(sim schedules like `64@250`, deep-search target generation in the plan), on TPU
where HBM bandwidth is the limit, and for cheap networks. This fix should go
upstream: it is self-contained and exact.

## 2. MuZero tie-break noise dominates `simulate` (measured; not used here)

`muzero_action_selection` draws `num_actions` Threefry uniforms for every node
visited by every simulation, scaled by 1e-7, only to break ties. Under `vmap`
the root/interior `lax.cond` becomes a select, so both branches run and the
noise is drawn twice. With the backward fix applied, 32-sim MuZero at B=128:

| variant | time |
|---|---|
| Threefry (default) | 750 ms |
| `jax_default_prng_impl=unsafe_rbg` | 553 ms |
| no tie-break noise | 263 ms |

Fixes: draw one `[B, A]` noise tensor per search (store it in `extra_data`, as
Gumbel does with `root_gumbel`), or derive it from the node index with a cheap
hash. On TPU, `rbg`/`unsafe_rbg` is the usual setting anyway. **This repo uses
Gumbel**, whose interior selection ignores its key, so this matters only for
`muzero_policy` / `stochastic_muzero_policy` users.

## 3. Gumbel's Sequential Halving is wide, but mctx runs it one action at a time (analysis)

`search` runs exactly one `recurrent_fn` call (batch B) per simulation, so a
move is `num_simulations` sequential network calls. With Gumbel at the root,
each Sequential Halving round visits every considered action once. Those visits
go into disjoint subtrees, so a round could be one call of batch B x width:

| m, sims | sequential calls | round widths |
|---|---|---|
| 16, 32 | 4 instead of 32 | 16, 8, 4, 4 |
| 16, 8 | 1 instead of 8 | 8 |
| 16, 64 | 15 instead of 64 | 16, 8, 8, 4x4, 2x8 |
| 16, 128 | 30 instead of 128 | ... |

The FLOPs are the same, so this helps only when one call at batch B leaves the
accelerator underused. That is the case for small per-device batches on TPU
(MXUs want large matmuls), small networks, and multi-host runs where per-call
latency adds up. It would not help a GPU already saturated at B=1024.

It is **not bit-identical**. Inside a round mctx re-scores after every visit:
the completed-Q rescaling and the `50 + max_visits` scale change globally, so a
later pick can depend on an earlier result. Picking each round's survivors at
the round boundary is textbook Sequential Halving and probably at least as good,
but per CLAUDE.md it needs a head-to-head before adoption. It is also the most
invasive change here: `search` and the Gumbel root selection would need a "wave"
mode.

**Cheaper first step for the TPU run:** raise the per-device
`selfplay_batch_size` until forward throughput stops improving. That reaches the
same utilisation with no mctx change, at the cost of memory, which is item 4.

## 4. Tree memory: the stored embedding is mostly observation (measured; repo-side)

mctx stores `[B, N, ...]` embeddings, and here the embedding is the whole
`pgx.State`, observation included. `recurrent_fn` only passes the stored state
to `env.step`, which recomputes the observation, so the stored copy is never
read. Measured per state, at B=1024 and N=33:

| env | state | observation | tree embeddings | of which observation | `[B,N,A]` arrays |
|---|---|---|---|---|---|
| gess | 8.6 KB | 7.8 KB (90%) | 278 MiB | 251 MiB | 6 x 52 MiB |
| epaminondas | 13.2 KB | 12.8 KB (97%) | 424 MiB | 411 MiB | 6 x 22 MiB |

The tree's largest component is the observations. Removing them (store the
state with a placeholder observation, and use the step's fresh observation only
for evaluation) would roughly halve tree memory for Gess and cut it by ~70% for
Epaminondas. It would also remove that scatter traffic from every simulation.
The freed memory can go to a larger per-device batch (item 3) or to fitting on
the 16 GB GPU. This is a change to `train.py`, not mctx. It needs care, since
the root and recurrent embeddings must share one pytree structure, and a
bit-identical check on CPU.

Smaller memory notes: `children_index`, `children_visits` and `parents` could be
`int16` when N < 32k. `children_discounts` is always ±1/0 in this repo but takes
4 bytes per edge. Neither is worth an mctx fork alone.

## 5. Smaller items (not measured; expected to be minor)

- `switching_action_selection_wrapper`'s `lax.cond` becomes a select under
  `vmap`. Every traversal step computes both root and interior selection: twice
  the softmax/qtransform work per step. Calling the root selection once per
  simulation, before the loop, avoids it.
- `simulate` splits an RNG key every depth step, even for Gumbel, which ignores
  it. That is cheap, but it is loop-carried state XLA cannot remove.
- On GPU, every iteration of a data-dependent `while_loop` copies its predicate
  back to the host to decide whether to continue. With 32 sims x (traversal +
  backward) that is ~10^3 syncs per move at ~10-20 µs each, i.e. ~10-20 ms
  against ~1 s of inference. Negligible now, but it grows with depth.
  `max_depth` bounds it.
- `train.py` uses `jax.pmap`. Everything in the search is batch-parallel, so
  `jit` with a batch `NamedSharding` (or `shard_map`) runs the same program and
  is the supported path for multi-host TPU slices. This affects portability
  more than speed.
- `recurrent_fn` evaluates the network on terminal children and then discards
  the value. Masking them out does not save FLOPs at a fixed batch shape, so
  leave it.

## Recommended order

1. Batched `backward` (exact, small, upstreamable). **Done:** `mctx_patch.py`
   replaces `mctx._src.search.backward` on import; `train.py` and
   `model_tournament.py` import it. `tests/test_mctx_patch.py` checks the tree
   against stock mctx (Gumbel and MuZero, with and without `max_depth`). Small
   CPU training runs, 3 iterations each, were bit-identical with and without the
   patch in params, optimizer state, RNG key and every logged metric: Pig with
   `chance_samples=2`, and Dots and Boxes with `continue_games` and playout cap
   randomization. No `game_version` bump is needed.
2. Strip observations from stored embeddings (memory, enables bigger batches).
3. On the TPU, tune per-device self-play batch size before anything else.
4. Only if 3 runs out of memory or the chips stay underused: wave-parallel
   Sequential Halving, adopted only after an equal-time head-to-head.
