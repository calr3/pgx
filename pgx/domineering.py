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

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.games.domineering import Game, GameState
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey


@dataclass
class State(core.State):
    """State for the game Domineering."""

    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    observation: Array = jnp.dstack([
        jnp.ones((8, 8), jnp.bool),
        jnp.ones((8, 8), jnp.bool),
    ])
    legal_action_mask: Array = jnp.ones(8 * 7, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "domineering"


class Domineering(core.Env):
    """Environment for the game Domineering."""

    def __init__(self):
        super().__init__()
        self._game = Game()

    def _init(self, key: PRNGKey) -> State:
        current_player = jnp.int32(jax.random.bernoulli(key))
        return State(current_player=current_player, _x=self._game.init())  # type:ignore

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        x = self._game.step(state._x, action)
        state = state.replace(  # type: ignore
            current_player=1 - state.current_player,
            _x=x,
        )
        assert isinstance(state, State)
        legal_action_mask = self._game.legal_action_mask(state._x)
        terminated = self._game.is_terminal(state._x)
        rewards = self._game.rewards(state._x)
        should_flip = state.current_player != state._x.color
        rewards = jax.lax.select(should_flip, jnp.flip(rewards), rewards)
        rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))
        return state.replace(  # type: ignore
            legal_action_mask=legal_action_mask,
            rewards=rewards,
            terminated=terminated,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return self._game.observe(state._x)

    @property
    def id(self) -> core.EnvId:
        return "domineering"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
