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
from pgx._src.games.gess import Game, GameState, N, BOARD_SIZE, MIN_IDX, MAX_IDX, EMPTY, BLACK, WHITE
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey


@dataclass
class State(core.State):
    """Public pgx state for Gess.

    The action space has 400 elements (one per cell of the 20×20 grid).
    Two consecutive actions form a full move:
      Step 1 – choose the source centre (stage 0).
      Step 2 – choose the destination centre (stage 1).
    `current_player` is unchanged between the two steps of the same move.
    """
    current_player:    Array = jnp.int32(0)
    observation:       Array = jnp.zeros((MAX_IDX - MIN_IDX + 1, MAX_IDX - MIN_IDX + 1, 4), dtype=jnp.float32)
    rewards:           Array = jnp.float32([0.0, 0.0])
    terminated:        Array = jnp.bool_(False)
    truncated:         Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(N, dtype=jnp.bool_)
    _step_count:       Array = jnp.int32(0)
    _x:                GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "gess"


class Gess(core.Env):
    """Pgx environment for Gess."""

    def __init__(self):
        super().__init__()
        self._game = Game()

    def _init(self, key: PRNGKey) -> State:
        del key  # Gess has a fixed starting position; key unused.
        x = self._game.init()
        return State(  # type: ignore
            current_player=jnp.int32(0),
            _x=x,
            legal_action_mask=self._game.legal_action_mask(x),
        )

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        was_stage0 = state._x.stage == 0
        x = self._game.step(state._x, action)

        # current_player switches only after the full two-part move (stage 1).
        new_current_player = jax.lax.select(
            was_stage0,
            state.current_player,
            jnp.int32(1 - state.current_player),
        )

        terminated = self._game.is_terminal(x)
        rewards    = self._game.rewards(x)
        rewards    = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))

        return state.replace(  # type: ignore
            current_player=new_current_player,
            _x=x,
            terminated=terminated,
            rewards=rewards,
            legal_action_mask=self._game.legal_action_mask(x),
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return self._game.observe(state._x, jnp.int8(player_id))

    @property
    def id(self) -> core.EnvId:
        return "gess"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
