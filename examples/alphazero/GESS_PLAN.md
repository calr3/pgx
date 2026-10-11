# Gess: forward plan

What to try next on the Gess line, and why. Results go in `GESS_EXPERIMENTS.md`, settled
recommendations in `RECIPES.md`; this file is for ideas not yet run. Judge every new
checkpoint against the web app's negamax (`tdgauntlet/clients/gess_negamax`) first and the
long-capture benchmark (`gess_slide_diag.py`) second: since E22, head-to-head gains within
the line have not reliably shown up against negamax.

## Running now

- **E25** (whole-move env `gess_joint`): E24 continued as a new run for ~34 h (iterations
  28 -> 183), LR re-warmed to 2e-4 and decayed to 1e-5. Afterwards: negamax matches on a
  series of its checkpoints, for a strength curve.

## Yardstick: negamax v0 (the app's engine), not v1

tdgauntlet's `clients/gess_negamax_rs` ports the app's negamax to native Rust (identical to the
app with its own weights, "v0"; 1.7x faster, but at 2 s both reach depth ~5) and refitted its
weights ("v1", `weights/v1.json`). v1 beats v0 head to head (0.82 per game at depth 4, 0.67 at
2 s), **but plays the networks worse**: in tdgauntlet `examples/gess_e25_negamax.toml` (2 s a
move, 50 openings x both colours, 128 sims) the models scored 90-94% against v1 and 67-86%
against v0. So v0 remains the yardstick for checkpoints. Results at 2 s a move:

| model | vs v0 (app .wasm) | vs v1 (native) |
|---|---|---|
| E25 it 183 | 79.8% | 93.8% |
| E25 it 120 | 85.7% | 93.9% |
| E25 it 60 | 78.6% | 91.2% |
| E24 it 28 | 66.7% | 90.0% |

(~26-34 counted minimatches each, so +/- 8-9 points.) E25 beats E24 against v0 but is flat from
iteration 60 to 183. Head to head (tdgauntlet `examples/gess_neural_round_robin.toml`, all at 128
sims, 50 openings x both colours), **E25 it 183 is the strongest network so far**: it beats E25
it 120 64.8% (27 counted minimatches), E24 96.0%, E23 100% and E22 98.9%, and each run beats
its predecessor decisively (E23 v E22 89%, E24 v E23 92%). A second round of evaluation ideas (new terms, 288 piece patterns,
quiescence) also failed to beat v1; see `TUNING.md` there.

## Next, after E25

- **Whole-move search and simulations: measured.** E25 it 183 against negamax v0 won 70% /
  75% / 79% of games at 128 / 512 / 2048 sims (the two-step E22: flat or worse), at 1.5 / 10 /
  201 s a move. A little gain, at a steep price; more self-play simulations are a weak lever.
- **No 2x2 patching: E29/E30, done.** One token per cell (the cheapest variant that fit, 6
  layers x 128, cuDNN attention) lost to the patched network at equal time 0.254 (minimatches
  0.8%). Patches stay. E30 (24 h from scratch) scored 0.683 against negamax, as E26/E28, yet
  E28 beat it 102-18: **negamax v0 no longer measures strength** at this level (E25-E30 all
  0.65-0.72 against it). Judge networks head to head.
- **Openings.** E28's policy puts 94% on l6-l7/l8/l9, which score worst in play-outs (0.39-0.46
  for black) while j6-g9 scores 0.655 and e3-e5 is the most robust. Self-play does not explore
  away from it. Candidates: more root noise or a sampled opening phase in self-play
  (`selfplay_sample_plies`, Dirichlet-style noise), or starting games from a spread of openings.
- **Capacity: E26 and E27, done.** Matched branches from E25 it 183 with two (E26) and four
  (E27) more transformer layers, starting as the identity. E26 beat E25 0.625 head to head;
  E27 did not beat E26 (0.458) and both scored 0.683 against negamax. Depth stops paying at 8
  layers here; width is untried. E26 is the best network.
- **Web app support for gess_joint networks.** The app's search (`gess-mcts.ts`) and encoder
  are two-step; it needs a legal-move list per node and the whole-move policy head's output
  (export: ONNX takes the move list as a second input). Only once a gess_joint model is the
  one to deploy.

## Tidy-ups (no retrain needed)

- **Drop the two always-zero observation planes** (2: source footprint, 3: stage) from
  `gess_joint`'s observation. They contribute nothing (zero inputs), cost almost nothing, and
  take ~a third of the replay buffer's observation storage. Since they are always zero, their
  stem-conv weight slices can be deleted from a checkpoint with the network's outputs exactly
  unchanged (the reverse of the clock planes' zero-widening): add a 4-plane observation as a
  new `gess_joint` version and strip E25's weights; no retraining.

## For a from-scratch run (only if one is ever justified)

Starting over is the one chance to change what a warm start locks in. Each is a separate
question; don't bundle them into one run.

- **Canonical orientation:** flip the board so the player to move is always at the bottom.
  Today black's home is always at the bottom and the network learns patterns from both sides
  (symmetry augmentation covers part of it); a canonical view is the AlphaZero norm.
- **Capacity:** a wider or deeper GessFormer, or dropping the 2x2 patch merge (long-range
  information currently reaches the network only at 2x2 granularity). The lever deferred
  since E17.
- **Rules-derived input planes** - e.g. which cells are legal piece centres for each side, how
  far each piece can slide, which moves win at once (`gess.winning_actions`). Information the
  network now has to infer from the stones. Whether this is still "pure" AlphaZero is the
  user's call (they ruled out tactical guards in play; input features are a different
  question, not yet decided).

## pgx/mctx code review (9 Oct): optimizations to test on the GPU

Where a self-play simulation's ~62 ms went (E26's network, batch 1024): network 36.5 ms, env
11.8 ms, mctx's own work ~14 ms. Found by reading the code; tested for exactness on CPU.

1. **mctx's backward pass copies the whole tree on every step up a path.** `search.backward`
   walks leaf to root in a `while_loop` vmapped over the batch, carrying the tree; with a
   batched predicate JAX runs the loop for the longest path and selects between old and new
   carry each pass, so every pass copies the [B, N, A] arrays (children_values and
   children_visits, ~200 MB each at 1024 x 65 x 768). `mctx_fast.py` walks the paths first
   with a small carry, then scatters every update once. Bit-identical: search trees and actions
   (`check_mctx_backward.py`), and 3 miniature training iterations (weights, optimizer state,
   RNG). Search bookkeeping on CPU: 2.4x faster at batch 64, 8.4x at 256. Opt-in:
   `mctx_fast_backward=true`. Likely most of mctx's ~14 ms a simulation; test with
   `bench_gess_selfplay.py` on and off. Probably also why the tdgauntlet JAX client was so slow
   at 2048 simulations (N = 2049 makes the copied arrays 30x larger).
2. **gess_joint applied each move twice** (done; exact). Its step went through Game.step twice,
   whose lax.cond on the stage evaluates both branches under vmap, so the stage-1 work (apply
   the move, check both rings) ran twice. Now calls the two stage functions directly: identical
   outputs and tests, ~3% faster env step on CPU (move generation and the observation dominate).
3. **Self-play batch size.** Time a self-play step at 2048 against 1024 (and max_num_steps
   halved, so samples per iteration stay equal): if the network or mctx underuse the GPU at
   1024, positions per second rise. Memory: the tree doubles too.
4. **Kernel launches and loop syncs.** Each simulation runs mctx's two vmapped while_loops (a
   few to ~10 passes each), every pass a set of small kernels and a device-to-host check of the
   loop condition. Try XLA flags for CUDA graphs / command buffers
   (`--xla_gpu_enable_command_buffer=...`, `--xla_gpu_graph_min_graph_size`) and see whether
   the gaps between kernels shrink in a profiler trace (`jax.profiler.trace` of one step).
5. **Smaller tree nodes.** Each node stores the full pgx State as its embedding (~12 KB: the
   float32 observation 7.8 KB, the move list as int32, the legal mask). Gathering and scattering
   it per simulation is cheap (~25 MB at batch 1024), so low priority; storing board + move list
   (int16) and computing the observation only for the network would cut tree memory ~3x,
   which matters only for big searches (opening analysis, high-simulation matches).
6. **Training-loop host work.** Each update gathers a 4096-sample minibatch on the host
   (~57 MB: float32 observations, policy targets, move lists) and copies it to the GPU, then
   syncs on `.item()`. Prefetch the next minibatch while the GPU trains, and store observations
   as float16 (half the buffer and the copy). First log per-phase times (self-play, host
   processing, training, eval) in train.py to see what this is worth: E28's 8.0 min an iteration
   leaves ~0.5 min unexplained by self-play and update times.
7. Already listed: the shorter move list (768 -> 576), and the network itself (GAB is ~25% of
   the patched network's FLOPs: share it across layers or lower its rank - a model change, so
   it needs a run).

**GPU session (11 Oct; E28's network, batch 1024, a full self-play step):**

| change | 64-sim step | 8-sim step |
|---|---|---|
| stock | 4178 ms | 475 ms |
| mctx_fast_backward | **3356 ms (-20%)** | 476 ms |
| batch 2048 (per 1024 positions) | 4164 / 3383 ms (stock / fast) | 488 / 487 ms |
| XLA command buffers off | 3322 ms | 480 ms |
| XLA while-loop double buffering | 3300 ms | 474 ms |

Adopted: `mctx_fast_backward` (now the default): self-play ~6.0 -> ~5.1 min an iteration, an
iteration ~8.0 -> ~7.1 min. Not worth it: a bigger batch (the GPU is saturated) and the XLA
flags (~1-2%, noise). Trace (`trace_selfplay.py`): the GPU is busy 94-95% of the time, so launch
gaps are not the problem; the stock run's `loop_select_fusion` (12% of GPU time) and most of its
memcpys - the backward pass's whole-tree copies - are gone with the fix.

**The scatter kernel (17% of GPU time) was the move generator's compaction**, found from the
dumped HLO (`--xla_dump_to`): `jnp.nonzero(size=768)` over 400 x 152 candidates compiles to a
running count and a scatter-add of 62M updates (1024 x 60,800) into 768 slots a position, all
atomics. Replaced by slot-wise search (each of the 768 slots finds its source, then its
candidate, by binary search over running counts); a first version that gathered each slot's
152-candidate row was as slow, as XLA fused it with building the table (768 x 152 scattered
reads a position). Exact (the gess_joint tests, an overflow test against jnp.nonzero, 16,834
real positions). A self-play step, E28's network, batch 1024:

| | 64 sims | 8 sims |
|---|---|---|
| stock | 4178 ms | 475 ms |
| + mctx_fast_backward | 3356 ms | 476 ms |
| + slot-wise compaction | **2698 ms** | **364 ms** |

Self-play ~35% faster at 64 sims, ~23% at 8; an iteration ~8.0 -> ~6.2 min expected. The top
kernels are now the network's (`trace_selfplay.py`: GPU busy 10.5 s for 4 steps, from 15.5).

Suggested GPU session (~30 min, before the next run; done 11 Oct, above): (a) `bench_gess_selfplay.py` with
mctx_fast_backward off/on at batch 1024 and 2048; (b) one profiler trace of a self-play step
with the winners; (c) the XLA flag variants. Adopt whatever is exact and faster.

## Next experiment: no 2x2 patching (one token per cell)

Asked for after E28 (user, 7 Oct). The 2x2 patch merge squeezes 4 cells into each of 100
tokens; a direct representation is one token per cell, 400 tokens. Tried before as E4's
RayFormer (line attention, one token per cell): +185 Elo over GessFormer at equal games but
-189 at equal time, its iterations ~50% slower; parked "until its self-play speed gap is
closed". So efficiency first.

Rough cost per position, 8 layers (multiply-adds): today ~0.7 G, of which the per-layer GAB
projection (128 -> 100^2 per head) is ~30%. At 400 tokens with GAB unchanged ~5.5 G (~8x; GAB
alone 160,000 outputs a head a layer, and a 2.6 GB bias tensor a layer at the self-play
batch); with a relative-position bias instead of GAB ~2.9 G (~4x); with line attention ~2.1 G
(~3x); line attention at width 128 ~1 G (~1.4x).

A. Efficiency (CPU now, GPU benchmark after E28):
   1. Relative-position attention bias (a learned (dr, dc) table per head) in place of GAB:
      cheap at any token count and geometric. Prerequisite: GAB is infeasible at 400 tokens.
   2. Fused attention (`jax.nn.dot_product_attention`, cuDNN, bfloat16): no 400x400 score
      matrix in memory.
   3. Size to an equal-cost budget: full attention with the offset bias, and line attention
      (RayFormer-style), at widths 192 and 128; `profile_gess_parts.py` /
      `bench_gess_selfplay.py` pick.
   4. Training memory: remat where needed, with train_bf16.
B. The 400-token network for gess_joint: no patch merge, no upsampling; the whole-move head
   reads the per-cell tokens directly. CPU tests of shapes and move/cell alignment.
C. The test: either a partial warm start from E26 (stem, layer weights at width 192, heads;
   new positional parameters) - approximate - or, cleaner, equal-time runs from scratch of
   the 400-token network and a 100-token GessFormer on today's pipeline (E2-E7's method).
   Go ahead only if the benchmark puts a 400-token network within ~1.5x today's iteration.

**Benchmark (8 Oct, GPU, `run_cells_bench.sh`; iteration = 256 self-play steps at 64 sims on a
quarter, 8 otherwise, plus 256 updates of 4096, all bfloat16):**

| network | forward (1024) | iteration | vs patched |
|---|---|---|---|
| patched (E26/E28), 8 layers, width 192 | 35.6 ms | ~8 min (measured 8.0) | 1x |
| cells, full attention, 8 heads, width 192 (JAX's own attention choice) | 234 ms | ~43 min | 5.4x |
| cells, line mask, 8 heads, width 192 | 232 ms | ~43 min | 5.4x |
| cells, ray attention (RayFormer's per-line blocks), 8 heads | 216 ms | ~41 min | 5.1x |
| cells, full attention, 8 heads, width 192, cuDNN | 120 ms | ~24 min | 3.0x |
| cells, full attention, 4 heads, width 192, cuDNN | 89 ms | ~19 min | 2.35x |
| cells, full attention, 4 heads, width 128, cuDNN | 67 ms | ~14.7 min | 1.8x |
| **cells, full attention, 4 heads, width 128, 6 layers, cuDNN** | **50 ms** | **~11.5 min** | **~1.4x** |

JAX's automatic attention never chose cuDNN's flash kernel; asked for explicitly it accepts the
offset bias and the line mask, with matching gradients, and is ~3x faster (the network now
asks for it in 16-bit floats on the GPU). A mask saves nothing; ray attention's per-line
reshuffles cost more than the attention they save. Past cuDNN, the per-token work at 4x the
tokens dominates, so width and depth are what bring the cost down. Chosen: the last row. The
test: E29 (that network) and E30 (the patched network), each from scratch for ~24 h.

## pgx performance (self-play throughput)

Self-play was ~95% of an iteration when last measured (E5). Most of it is believed to be the
environment: `gess_joint`'s step rebuilds the legal-move list (`legal_moves`) at every MCTS
expansion, checking all 400 cells as piece centres x 8 directions x 19 steps x 9 footprint
cells by gather (~550k indexed reads a position), for 1024 games x 64 simulations. Measured on
~29k positions from tdgauntlet games: legal piece centres median 139, max 164 (any block free of
opponent stones with an own stone round its centre is a piece); legal moves median 452, p99
511, max 541.

1. **Profile: done** (`profile_gess_parts.py`, E26's 8-layer network, GPU). Per self-play
   simulation at batch 1024 (62 ms, from `bench_gess_selfplay.py`): network forward 36.5 ms
   (bf16), env step 11.8 ms (new move generator), mctx's own work the remaining ~14 ms. A
   training update (batch 4096, float32) takes 1.10 s as configured (2 micro-batches, remat):
   4.7 min an iteration. So an iteration is now ~5.7 min self-play (network ~60%, env ~20%,
   mctx ~20%) + ~4.7 min training + eval. Remat off fits with 8 micro-batches and is faster:
   0.89 s an update (-19%, ~0.9 min an iteration; the same maths, so use it:
   `gf_remat=False train_micro_batches=8`); remat off with 1 or 2 micro-batches runs out of
   memory. Next levers: the network in both phases (bfloat16 training; the per-layer GAB bias
   and the move head's 768-way gather), then the shorter move list (item 3).
   **bfloat16 training: done** (`train_bf16=true`, `check_train_bf16.py`). An update 1.10 s
   -> 0.52 s (remat off, 4 micro-batches; remat off now fits even at 2). On E26's replay
   buffer: gradients' cosine with float32's 0.986-0.992 (two float32 batches: -0.08 to
   0.12), policy loss +0.15%, value loss equal; 240 updates from E26 at LR 2e-4 track
   float32's held-out loss within 0.003. First run with it: E28.
2. **Move generation by whole-board shifts** (in progress): per direction and step, blocked
   and reachable centres for all 400 at once as shifted 20x20 planes (beyond 2 steps a
   destination's block never overlaps the source's, so it is a shifted 3x3 dilation of
   occupancy), and the candidates of a centre taken in a fixed offset order so the list comes
   out sorted without the 160,000-cell table. Exactly the same move lists (CPU equivalence
   test on real and random positions); speed measured on GPU in the gap after E27.
   (Restricting the gather version to legal centres would save at most ~2.4x, as 139 of 400
   centres are legal: not worth it alone.)
   **Done** (`legal_moves`; the gather version kept as `legal_moves_reference`). GPU, batch
   1024 (`bench_gess_selfplay.py`, E26's network): the generator alone 52 -> 33 ms; a
   self-play step 6.31 -> 3.97 s at 64 sims (1.59x) and 0.83 -> 0.46 s at 8 sims (1.80x). At
   E26's playout cap (64 sims on a quarter of steps, 8 otherwise, 256 steps) self-play is
   ~9.4 -> ~5.7 min an iteration, so an iteration ~14.5 -> ~11 min; training (256 updates at
   batch 4096) is then ~5 of those 11 minutes - no longer the ~5% it was in E5.
3. **Shorter move list** (768 -> 576): policy head, tree arrays and buffer all scale with it;
   max seen 541. Needs a new env version and an overflow check.
4. **tdgauntlet JAX client at high simulation counts** is bound by one CPU core (2048 sims:
   98% CPU, 201 s a move): the search loop is driven from Python. Matches only, not training.
5. Smaller: one fused step instead of gess's two stage steps per move; check the GAB
   projection and attention in the profile (unlikely to dominate at 100 tokens).

## Ideas tried and parked

- Joint piece->destination policy head on the two-step env (`gf_joint_head`, E21): not needed;
  the network learns long captures once the targets allow (E22).
- Auxiliary immediate-win target (`aux_win_weight`, E23): cut self-play's missed wins 6.8% ->
  2.1%, no gain against negamax.
- Clock planes (E14, E19): level both times; kept (they cost <1%).
