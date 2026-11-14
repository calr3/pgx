# The mctx fork, and what to test on the GPU

Search runs on [calr3/mctx](https://github.com/calr3/mctx) (branch
`ccr-7d0dcbe7-3rlfby`, installed by `requirements.txt`), a fork of upstream
mctx 0.0.71. The fork's README lists its changes; in short:

| Change | Results | Expected effect |
|---|---|---|
| Root step of each simulation outside the traversal loop | bit-identical | less selection work per depth |
| Backward pass reads in its loop, writes once at the end | bit-identical | no whole-tree copy per level |
| New nodes written with slice updates, not scatters | bit-identical | cheaper writes (most on TPU) |
| Edge statistics stored once, on the child (`[B, N]`, not `[B, N, A]`) | bit-identical | 4 of 6 large arrays gone |
| `children_index` stored as int16 | bit-identical | halves one of the 2 remaining |
| `search_tree_bf16`: prior logits stored in bfloat16 (opt-in) | **differs slightly** | halves the other |

Everything above the last row is checked on CPU against the original upstream
code, which the fork carries as `mctx._src.tests.reference`:

    JAX_PLATFORMS=cpu python -m pytest -p no:xdist -q tests/test_mctx_fork.py

(The fork's own suite, `mctx/_src/tests/search_equivalence_test.py`, does the
same on a synthetic game, for plain, Gumbel and stochastic MuZero.)

## Measured so far (CPU only)

`bench_search.py`, Gess, 256 positions, 32 simulations, small ResNet
(32 channels, 2 blocks), all fork changes:

| Implementation | Median | Positions/s | XLA temp memory | Same move as reference |
|---|---|---|---|---|
| reference (upstream) | 1838 ms | 139 | 116.6 MiB | - |
| fork | 1435 ms (1.28x) | 178 | 58.8 MiB | 100% |
| fork_bf16 | 1530 ms (1.20x) | 167 | 57.3 MiB | 100% |

Earlier, with a random linear "network" and only the loop changes: Gess 512
positions 774 -> 540 ms; Epaminondas, where `env.step` dominates, 5.9 -> 5.2 s.

On CPU bfloat16 is emulated, so `fork_bf16` being slower and barely smaller
here says nothing about the GPU. A real network's inference is a far larger
share of a search than this small ResNet's, so expect a smaller relative
speedup in training; the memory saving carries over. Nothing has been measured
on a GPU or TPU yet.

## GPU test plan

Run from the repo root on the GPU machine.

**0. Setup.** `pip install -r examples/alphazero/requirements.txt`, then check
the GPU survived (CLAUDE.md: JAX and its CUDA plugin must stay in lockstep) and
that the fork is the installed mctx:

    python -c "import jax, mctx; print(jax.devices(), mctx.Tree.child_visits)"

**1. Correctness, on CPU.** The equivalence tests above. GPU runs of identical
code differ slightly, so exactness can only be shown on CPU.

**2. Search micro-benchmark.** `bench_search.py` times one self-play search per
implementation with the real network and reports compile time, run time, XLA
temp memory, and how often the implementations pick the same move:

    for env in "env_id=gess architecture=gessformer" "env_id=epaminondas architecture=boardformer"; do
      python -u examples/alphazero/bench_search.py $env selfplay_batch_size=1024 \
          num_simulations=32 selfplay_bf16=true out_json=bench_search.jsonl
    done

Prefer `checkpoint=<a trained checkpoint>` over random weights: a trained
network's search trees are shaped differently (deeper, narrower). Also try the
largest `selfplay_batch_size` that fits: the memory saved is only worth
something if it buys a bigger batch. `profile_dir=traces` writes a
`jax.profiler` trace per implementation (open in TensorBoard or Perfetto) to
see what share of a search is network versus tree bookkeeping.

**3. End-to-end training speed.** Short runs of the same recipe, varying only
the implementation, compared on `perf/selfplay_seconds` and
`perf/iteration_seconds` (logged every iteration):

    python -u examples/alphazero/train.py <recipe> mctx_impl=reference max_num_iters=20
    python -u examples/alphazero/train.py <recipe> mctx_impl=fork max_num_iters=20
    python -u examples/alphazero/train.py <recipe> mctx_impl=fork search_tree_bf16=true max_num_iters=20

Skip the first iteration or two (compilation). `mctx_impl` needs no reinstall:
`reference` is upstream code shipped inside the fork.

**4. Strength, only for `search_tree_bf16`.** It changes the search, so follow
CLAUDE.md: never adopt it on curves alone. Train two arms at equal wall-clock
time (identical except `search_tree_bf16`) and compare with
`model_tournament.py` / `elo_ladder.py`. If `bench_search.py` shows the bf16
search picks the fork's move almost always (as `selfplay_bf16` did, 96-97%)
and buys a meaningfully larger batch or speed, it is worth the tournament;
otherwise drop it.

Record the outcome in RECIPES.md if it changes a recommendation.

## Not done

- **Moving off `jax.pmap`** (to `jit` with sharded arrays). `pmap` still works
  in JAX 0.8 and is fine on one host, including a single-host TPU (v4-8,
  v5e-8). It only blocks multi-host TPU slices. The conversion touches every
  pmapped function in train.py and model_tournament.py (self-play, init,
  training step with its `pmean`, evaluation, checkpoint replication) and
  cannot be verified bit-for-bit, since a sharded mean reduces in a different
  order than `pmean`. Do it once the TPU type is known and needs it, checking
  on CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=4` that a
  short run matches the pmap version to float tolerance.
- **Caching softmax(prior logits)** per node: every visit recomputes it over
  all actions. Caching needs another `[B, N, A]` array and would not be
  bit-identical, so only worth it if a GPU profile shows selection math
  matters.
- **Subtree reuse** between Gess's two stages: changes the algorithm, so it
  needs a strength test, not just a speed test.
