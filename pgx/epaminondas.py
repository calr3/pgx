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
from pgx._src.games.epaminondas import Game, GameState, HEIGHT, N, WIDTH
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey


@dataclass
class State(core.State):
    """Public pgx state for Epaminondas.

    The action space has 168 elements (one per square of the 14x12 board).
    Three consecutive actions form a full move:
      Step 1 - the lead piece of the moving group (stage 0).
      Step 2 - its rear piece; the lead square again means a single piece (stage 1).
      Step 3 - the destination of the lead piece (stage 2).
    `current_player` is unchanged across the three steps of one move.
    """

    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((HEIGHT, WIDTH, 7), dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(N, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "epaminondas"


class Epaminondas(core.Env):
    """Pgx environment for Epaminondas."""

    def __init__(self):
        super().__init__()
        self._game = Game()

    def _init(self, key: PRNGKey) -> State:
        del key  # fixed starting position
        x = self._game.init()
        return State(  # type: ignore
            current_player=jnp.int32(0),
            _x=x,
            legal_action_mask=self._game.legal_action_mask(x),
        )

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        was_last_stage = state._x.stage == 2
        x = self._game.step(state._x, action)

        # The turn passes only after the third action of a move.
        current_player = jax.lax.select(
            was_last_stage, jnp.int32(1 - state.current_player), state.current_player
        )
        terminated = self._game.is_terminal(x)
        rewards = self._game.rewards(x)
        rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))

        return state.replace(  # type: ignore
            current_player=current_player,
            _x=x,
            terminated=terminated,
            rewards=rewards,
            legal_action_mask=self._game.legal_action_mask(x),
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return self._game.observe(state._x, jnp.int32(player_id))

    @property
    def id(self) -> core.EnvId:
        return "epaminondas"

    @property
    def version(self) -> str:
        # v1: reaching the move cap is decided on piece count (v0: a draw).
        # v2: the cap counts moves since the last capture, not since the start.
        # v3: the cap is decided rank by rank on advancement, not on piece count.
        return "v3"

    @property
    def num_players(self) -> int:
        return 2
