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
- **Capacity: E26 and E27.** E25 made two (E26) and four (E27, running) transformer layers
  deeper, the new layers starting as the identity, as matched branches from E25 it 183. E26
  beat E25 0.625 head to head; see GESS_EXPERIMENTS.md.
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

## pgx performance (self-play throughput)

Self-play was ~95% of an iteration when last measured (E5). Most of it is believed to be the
environment: `gess_joint`'s step rebuilds the legal-move list (`legal_moves`) at every MCTS
expansion, checking all 400 cells as piece centres x 8 directions x 19 steps x 9 footprint
cells by gather (~550k indexed reads a position), for 1024 games x 64 simulations. Measured on
~29k positions from tdgauntlet games: legal piece centres median 139, max 164 (any block free of
opponent stones with an own stone round its centre is a piece); legal moves median 452, p99
511, max 541.

1. **Profile first** (~10 min of GPU between runs): one self-play iteration under the JAX
   profiler, splitting env step / network / mctx bookkeeping. Not measurable on CPU.
2. **Move generation by whole-board shifts** (in progress): per direction and step, blocked
   and reachable centres for all 400 at once as shifted 20x20 planes (beyond 2 steps a
   destination's block never overlaps the source's, so it is a shifted 3x3 dilation of
   occupancy), and the candidates of a centre taken in a fixed offset order so the list comes
   out sorted without the 160,000-cell table. Exactly the same move lists (CPU equivalence
   test on real and random positions); speed measured on GPU in the gap after E27.
   (Restricting the gather version to legal centres would save at most ~2.4x, as 139 of 400
   centres are legal: not worth it alone.)
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
