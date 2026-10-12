# Copyright 2026 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A.I. Wars: Cybug battles between CAICL scripts, as a pgx env.

The game itself - the CAICL compiler, the battle engine, maps and `run_battles` - is the
`aiwars_jax` package (github.com/calr3/aiwars_jax); this module only wraps it in the pgx
API. The players of A.I. Wars write programs, not moves: a battle is decided entirely by
the two scripts (and the map and dice). So this env has no decisions to make - the
action space has a single action, and each `step` plays one tick of the battle.
`init(key)` draws a matchup from the env's library: two different programs (or one of
`pairs`), a map, start cells, headings and the strategy node.

To evaluate scripts, use `aiwars_jax.run_battles(env.arena, matchups)`, which plays many
battles at once and is much faster than stepping the env.

Rewards are `aiwars_jax.rewards`: +1 / -1 for the winner and loser, 0 for an exact tie.
"""

from typing import Optional, Sequence

import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

try:
    import aiwars_jax
    from aiwars_jax import engine as W
except ImportError as e:  # pragma: no cover
    raise ImportError("the ai_wars env needs the aiwars_jax package: "
                      "pip install --no-deps -e <path to github.com/calr3/aiwars_jax>") from e


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((W.ROWS, W.COLS, aiwars_jax.arena.N_PLANES), dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(1, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: W.GameState = None  # type: ignore

    @property
    def env_id(self) -> core.EnvId:
        return "ai_wars"


class AIWars(core.Env):
    """A.I. Wars battles between two CAICL programs.

    Args:
        sources: CAICL program texts (plain, as the editor shows them). Default:
            aiwars_jax's two demonstration scripts.
        names: their names.
        maps: .map file texts. Default: a generated open arena.
        pairs: optional [P, 2] program indices; `init` draws one of these matchups
            instead of any two different programs.
        rules: battle settings (`aiwars_jax.Rules`).
    """

    def __init__(self, sources: Optional[Sequence[str]] = None, names: Optional[Sequence[str]] = None,
                 maps: Optional[Sequence[str]] = None, pairs=None, rules: W.Rules = W.Rules()):
        super().__init__()
        self.arena = aiwars_jax.Arena(sources, names, maps, rules)
        if len(self.arena.library.programs) < 2 and pairs is None:
            raise ValueError("need at least two programs")
        self.pairs = None if pairs is None else jnp.asarray(pairs, jnp.int32)

    @property
    def warnings(self):
        """Compile-time approximations, per program."""
        return self.arena.warnings

    def _init(self, key: PRNGKey) -> State:
        return State(_x=self.arena.new_battle(key, self.pairs))  # type: ignore

    def _step(self, state: core.State, action: Array, key) -> State:
        del action, key
        assert isinstance(state, State)
        x = self.arena.engine.tick(self.arena.T, state._x)
        terminated = ~x.running
        rewards = jnp.where(terminated, aiwars_jax.rewards(x), 0.0)
        return state.replace(_x=x, terminated=terminated, rewards=rewards)  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return aiwars_jax.observe(state._x, player_id)

    @property
    def id(self) -> core.EnvId:
        return "ai_wars"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
