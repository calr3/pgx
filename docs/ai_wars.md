# A.I. Wars

[A.I. Wars (The Insect Mind)](https://www.tacticalneuronics.com/ai-wars/) by John A. Reder
(Tactical Neuronics) is a programming game: players write the AI of a robot insect, a
*Cybug*, in the scripting language CAICL, and Cybugs fight on a 45 x 32 grid. Nobody
steers during a battle, so a battle is a pure function of the two scripts, the map and
the dice.

The game lives in the [aiwars_jax](https://github.com/calr3/aiwars_jax) package: the
CAICL compiler, a battle engine that reproduces the web edition tick for tick (checked
against the reference engine), maps, `run_battles`, and a CAICL reference for writing
Cybugs. This env wraps it in the pgx API. Install it next to pgx without letting pip
touch JAX:

```sh
pip install --no-deps -e path/to/aiwars_jax
```

## Usage

```py
import jax, pgx
from pgx.ai_wars import AIWars
from aiwars_jax import run_battles

env = AIWars(sources=[src_a, src_b, src_c], names=["a", "b", "c"], maps=[map_text])
state = env.init(jax.random.PRNGKey(0))   # a random matchup, map and placement
while not state.terminated:
    state = env.step(state, 0)              # one tick

# evaluating scripts: many battles at once, much faster than stepping the env
res = run_battles(env.arena, [(0, 1), (1, 0)] * 128, lanes=256)
```

`pgx.make("ai_wars")` gives an env with aiwars_jax's two demonstration scripts and a
generated arena.

| | |
|---|---|
| Players | 2 |
| Actions | 1 (a battle has no decisions; `step` plays a tick) |
| Observation | `[32, 45, 9]`: walls, flags, mines, own bug, enemy bug, own missile/grenade, enemy missile/grenade, strategy node, own heading |
| Rewards | +1 / -1 for the winner and loser; 0 for an exact tie (equal scores, both alive or both dead) |
| Termination | one bug (or IFF team) left, 2000 ticks, or 101 ticks without a move |
| Version | v0 |
