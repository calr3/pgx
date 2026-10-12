# A.I. Wars

[A.I. Wars (The Insect Mind)](https://www.tacticalneuronics.com/ai-wars/) by John A. Reder
(Tactical Neuronics, 1996-2026) is a programming game: players write the AI of a robot
insect, a *Cybug*, in the scripting language CAICL, and Cybugs fight on a 45 x 32 grid
with guns, missiles, grenades, mines, shields and cloaks. Nobody steers during a battle,
so a battle is a pure function of the two scripts, the map and the dice.

This port reproduces the web edition's engine (`engine_core.js` + `engine_caicl.js`, a
port of the VB6 original) tick for tick. `tools/ai_wars/conformance.py` checks that
against the reference engine itself. Random battles between the 164 bundled Cybugs on
the 34 bundled maps match on every bug's state and the whole map after every tick: 200
of 200 in the final check (and all of several hundred more during development).

## Usage

```py
import jax, pgx
from pgx.ai_wars import AIWars, run_battles

env = AIWars(sources=[src_a, src_b, src_c], names=["a", "b", "c"], maps=[map_text])
print(env.warnings)                    # compile-time approximations, per program

# many battles at once (the fast path): program indices into the library
res = run_battles(env, [(0, 1), (1, 0), (0, 2), (2, 0)] * 64, seed=0, lanes=256)
res["winner"], res["reward"], res["score"], res["ticks"]

# or as a pgx env: init draws a matchup, each step plays one tick
state = env.init(jax.random.PRNGKey(0))
while not state.terminated:
    state = env.step(state, 0)
```

`pgx.make("ai_wars")` gives an env with two small demonstration scripts and a generated
arena. The game's own maps and Cybug library are not in the repo; `tools/ai_wars/fetch.sh`
downloads and decrypts them.

| | |
|---|---|
| Players | 2 (the engine supports up to 10 bugs; `Engine(dims, n_bugs)`) |
| Actions | 1 (a battle has no decisions; `step` plays a tick) |
| Observation | `[32, 45, 9]`: walls, flags, mines, own bug, enemy bug, own missile/grenade, enemy missile/grenade, strategy node, own heading |
| Rewards | +1 / -1 for the engine's winner and loser; 0 for an exact tie |
| Termination | one bug (or IFF team) left, 2000 ticks, or 101 ticks without a move |

The engine's winner is the living bug with the highest score, ties going to the lower
slot. The env (and `run_battles`' `reward`) calls an exact tie - equal scores, both alive
or both dead - a draw, since otherwise slot 1 would win every battle in which nothing
happens. `res["winner"]` is the engine's verdict.

## Layout

- `pgx/_src/games/ai_wars_caicl.py` compiles CAICL source to tables (`Library`, `lower`).
- `pgx/_src/games/ai_wars.py` is the engine (`Engine`): the battle state, the tick, the
  interpreter over the compiled tables, and `run_queue`, which plays a list of battles on
  a fixed number of lanes.
- `pgx/ai_wars.py` is the env, plus `run_battles` and `make_jobs`.
- `tools/ai_wars/` holds the reference harness (`run_ref.js`), `fetch.sh` and
  `conformance.py`.

## How CAICL is compiled

The reference interpreter works on text. On every execution of a line it prefixes the
open block headers' conditions, substitutes `~vN`/`~hN` variables and `#tokens` word by
word, and matches the result against command prefixes, often at fixed character columns.
A variable can hold any text: a whole command (`assign vm turn left` ... `~vm`), part of a
label (`goto l #scan`), or text with a number in it.

The compiler turns each line into *variants*. Numbers stay symbolic and are read at run
time. A variable that may hold a string (found by a fixpoint over every `assign` in the
library) is expanded into one variant per string it can hold. Where a fixed-column parse
depends on how many characters a number prints as, the number case is split by width. At
run time the interpreter picks the variant from the current values. Programs that may
meet are compiled together (`Library`), since they share the string table and the hive
(team) variables.

## Writing Cybugs: CAICL in brief

One command per line; `;` starts a comment line; `label:` marks a jump target. Case does
not matter. Lines run top to bottom and wrap around at the end.

**Each turn** a bug gets 10 *sub-clicks*. A line costs one, a scan three. A *mega*
command ends the turn at once: turning, `fire weapon`, launches, shield and cloak
changes, `attempt repairs`, `discharge energy`, `self destruct` and data links. Moving
does not end the turn: a bug can move up to 10 cells a tick.

| Command | Effect |
|---|---|
| `move forward`, `move backward` | costs `burn` fuel; a flag repairs (or overloads an undamaged bug), a mine does 5 damage |
| `turn left`, `turn right` | costs `burn` fuel; mega |
| `fire weapon` | gun, shield down, 1 ammo: 5..1 damage over 5 cells (2..0 if the target is shielded) |
| `launch missile` | 10 ammo, 300 fuel: flies 4 cells a tick, 3x3 blast (9 direct / 7 splash) |
| `launch grenade` | 5 ammo, 200 fuel: flies 3 cells, 3x3 blast (7 / 5) |
| `set grenade fuse N` | changes the range of a grenade in flight; a launch resets it to 3 |
| `lay mines on/off` | drop a mine (2 ammo) on every cell left |
| `raise shield`, `lower shield` | shield heats 1/tick (2 cloaked); at 30 it drops and the bug freezes 10 ticks; launching with it up mostly misfires on yourself |
| `cloak on/off` | costs 10 fuel and 2 ammo a tick |
| `attempt repairs` | shield down: 10% chance of -1 damage; failure raises `burn` |
| `discharge energy` | 2 damage to all 8 neighbours, 1 to self; clears adjacent flags and mines |
| `self destruct` | dies; neighbours take its remaining health as damage |
| `scan forward/left/right`, `long range scan`, `scan perimeter`, `cross scan`, `corner scan`, `scan position N`, `scan relative N`, `gps scan x X y Y` | set the scan result: nothing, barrier, enemy, mine, friend, flag (self for long range) |
| `generate random` | `#random` = 1..4 |
| `assign vN text`, `math vN = a + b * c ...` (left to right), `cmath vN = expr` (BASIC-style expression, functions `abs int fix sgn sqr rnd sin cos tan atn exp log`) | variables `v1`-`v9`, `v0`, `va`-`vz`; hive (team) variables `h...` |
| `goto label`, `gosub label`, `return` | gosub nests 5 deep |
| `iff code X` | join team X; the battle ends when only one team is alive |
| `data link command C`, `data link resource fuel/ammo N` | with a bug back to back: overwrite its current line with C, or (shield up) give it fuel/ammo |
| `queen` | losing this bug costs 1000 points |

**Conditions**, all of the form `if ... then <command>`, chainable (`if a then if b then
...`): `if value A op B` (`> < = <> >= <=`), `if scan found X`, `if bump barrier`, `if not
bump barrier`, `if shield is up/down`, `if facing D`, `if not facing D`, `if missile
ready`, `if missile not ready`, `if grenade ready`, `if grenade not ready`, `if damage is
op P` (percent), `if fuel is op P` (percent), `if ammo is op N`, `if random is N`,
`if x coordinate is op N`, `if y coordinate is op N`, `if no ammo`. A line ending in
`then` opens a block closed by `end if`.

**Tokens**: `#x_pos #y_pos #cur_head #cur_fuel #cur_ammo #cur_life #cur_score #burn
#shield #heatlevel #cloakstatus #scan #random #alive #inbound #strat_x #strat_y
#enemy_x/y/h/d #friend_x/y/h/d`, and constants `#max_fuel #max_life #set_ammo #damsetN`.

Traps worth knowing, all faithful to the original:

- Block bodies are not skipped. Each body line runs with the header's conditions in
  front of it and costs its sub-click even when they fail. A `goto` out of an open block
  leaves its condition applied to the lines that follow, until an `end if` closes it.
- The fixed-column conditions (`if x/y coordinate is`, `if damage/fuel/ammo is`, `if
  random is`) read their number from two fixed columns and the action from a fixed
  column after it. A one-digit number must be written `05`. A variable or token that
  prints as one character (`#enemy_x` = 7) shifts the action text and silently turns it
  into a no-op. `if damage is > 0 then` as a block header garbles every body line.
- Variables hold text. `assign v1 05` keeps `05`, which matters for labels:
  `goto l ~v1` looks for `l 05:`.

## Fidelity

`tools/ai_wars/conformance.py` runs battles in both engines, giving the reference
pgx's random draws, and compares them after every tick. Known differences, all rare:

- Randomness: the k-th draw of bug b's turn in tick t is
  `bits(fold_in(fold_in(key, t), b))[k] >> 8` over 2^24, where the reference calls
  `Math.random()`.
- Arithmetic is float32, where the reference uses doubles; only `cmath` can tell.
- Text holding several numbers (`assign v0 math ~v0 + ~v1`) becomes an unknown word.
  Labels computed from more than 3 numbers are unsupported. So are `iff code` and
  `data link command` texts built from variables.
- A malformed block header disables its body lines; the reference garbles them, which
  almost always amounts to the same.
- `data link command` overwrites the target's next executable line, where the reference
  overwrites the line its pointer is on, which can be blank. A bug remembers its last 8
  overwritten lines.
- Nesting deeper than 50 blocks disables the excess (the reference degrades past 50 in
  its own way); a turn is cut off after 600 interpreter iterations (the reference would
  hang on a script that only opens and closes blocks).

## Performance

Under `vmap` every battle computes every command, so CPU throughput is modest. Measured
on a 24-core desktop CPU (one process, which uses about 2 cores), random pairings of the
bundled Cybugs:

| | ticks/s | battles/s (~375 ticks each) |
|---|---|---|
| `run_battles`, 256 lanes, steady state | ~2,900 | ~7.5 |
| reference JS engine, one core, no tracing | ~9,700 | ~22 |

On top of the steady rate, a run has a tail: once the queue is empty, the slowest battle
still has to finish, and its length is sequential (about 5 minutes on that CPU, for a
2000-tick battle between scripts with long `cmath` lines). So give `run_battles`
thousands of battles at a time.

The design targets accelerators, where one step covers tens of thousands of battles for
about the cost of one; it has not been measured on a GPU yet. Use `run_battles` /
`Engine.run_queue` rather than stepping the env. The queue keeps every lane busy, where
lanes of a plain `vmap(run)` wait for the longest turn and the longest battle.
