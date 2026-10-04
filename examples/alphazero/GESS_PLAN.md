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
iteration 60 to 183. A second round of evaluation ideas (new terms, 288 piece patterns,
quiescence) also failed to beat v1; see `TUNING.md` there.

## Next, after E25

- **Does whole-move search turn simulations into strength?** E25 at 128 / 512 / 2048 sims
  against negamax. The two-step tree did not improve with more search (E22: 0.50, 0.50, 0.38);
  a whole-move tree spends each simulation on a whole move. Also compare at equal think time:
  E24 took 0.78 s a move at 128 sims, E23 2.6 s.
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

## Ideas tried and parked

- Joint piece->destination policy head on the two-step env (`gf_joint_head`, E21): not needed;
  the network learns long captures once the targets allow (E22).
- Auxiliary immediate-win target (`aux_win_weight`, E23): cut self-play's missed wins 6.8% ->
  2.1%, no gain against negamax.
- Clock planes (E14, E19): level both times; kept (they cost <1%).
