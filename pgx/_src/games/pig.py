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

from typing import NamedTuple, Optional
import functools
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey


PLAYER_COUNT = 2
TARGET = 100  # The score that a player's total must reach or exceed to win.

class GameState(NamedTuple):
    """Internal state for the game Pig."""
    color: Array = jnp.int32(0)
    totals: Array = jnp.zeros(PLAYER_COUNT, jnp.int32)  # The total banked score across the game.
    turn_total: Array = jnp.int32(0)  # The score accumulated across this turn, *including* the value just rolled.
    last_roll: Array = jnp.int32(0)   # The value of the dice just rolled.
    winner: Array = jnp.int32(-1)     # The winning player, or -1 if no winner yet.


def _roll(key: PRNGKey) -> Array:
    return jax.random.randint(key, shape=(), minval=1, maxval=7, dtype=jnp.int32)


class Game:
    """The game representation of Pig."""

    def init(self, key: PRNGKey) -> GameState:
        state = GameState()
        first_roll = _roll(key)
        return state._replace(
            turn_total = first_roll,
            last_roll = first_roll,
        )


    def step(self, state: GameState, action: Array, key: PRNGKey) -> GameState:
        next_roll = _roll(key)

        return state._replace(
            color = jax.lax.select(action == 0, (state.color + 1) % PLAYER_COUNT, state.color),
            totals = jax.lax.select((action == 0) & (state.last_roll > 1),
                                     state.totals.at[state.color].set(state.totals[state.color] + state.turn_total),
                                     state.totals),
            turn_total = jax.lax.select(action == 0, next_roll, state.turn_total + next_roll),
            last_roll = next_roll,
            winner = jax.lax.select((action == 0) & (state.totals[state.color] + state.turn_total >= TARGET), state.color, state.winner),
        )


    def observe(self, state: GameState) -> Array:
        return jnp.hstack([
            jnp.roll(state.totals, -state.color),
            jnp.tile(state.turn_total, PLAYER_COUNT),
            jnp.tile(state.last_roll, PLAYER_COUNT),
        ])


    def legal_action_mask(self, state: GameState) -> Array:
        return jnp.bool([True, state.last_roll > 1])


    def is_terminal(self, state: GameState) -> Array:
        return state.winner >= 0


    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            (jnp.ones(PLAYER_COUNT, jnp.float32) * -1).at[state.winner].set(1.0),
            jnp.zeros(PLAYER_COUNT, jnp.float32),
        )


