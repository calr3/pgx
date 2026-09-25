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
from pgx._src.games.pig import Game, GameState, PLAYER_COUNT, TARGET, _roll
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

@dataclass
class State(core.State):
    """State for the game Pig."""

    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.zeros(PLAYER_COUNT, jnp.float32)
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    observation: Array = jnp.zeros(3 * PLAYER_COUNT + 1, dtype=jnp.int32)
    legal_action_mask: Array = jnp.ones(2, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "pig"


class Pig(core.Env):
    """Environment for the game Pig."""

    def __init__(self):
        super().__init__()
        self._game = Game()
        # The die is rolled straight from the key `step` is given, so a key that
        # rolls a face forces it. One per face, found by trying keys in turn.
        faces = {}
        i = 0
        while len(faces) < 6:
            key = jax.random.PRNGKey(i)
            faces.setdefault(int(_roll(key)), key)
            i += 1
        self._chance_keys = jnp.stack([faces[r] for r in range(1, 7)])

    def _init(self, key: PRNGKey) -> State:
        x = self._game.init(key)
        current_player = x.color
        return State(current_player=current_player,
                     _x=x,
                     observation=self._game.observe(x),
                     legal_action_mask=self._game.legal_action_mask(x))  # type:ignore

    def _step(self, state: core.State, action: Array, key: PRNGKey) -> State:
        assert isinstance(state, State)
        x = self._game.step(state._x, action, key)
        state = state.replace(  # type: ignore
            current_player=x.color,
            _x=x,
        )
        assert isinstance(state, State)
        legal_action_mask = self._game.legal_action_mask(state._x)
        terminated = self._game.is_terminal(state._x)
        # `rewards` is indexed by player, and this env keeps player ids equal to
        # the game's colors, so it needs no rotation.
        rewards = self._game.rewards(state._x)
        rewards = jax.lax.select(terminated, rewards, jnp.zeros(PLAYER_COUNT, jnp.float32))
        return state.replace(  # type: ignore
            legal_action_mask=legal_action_mask,
            rewards=rewards,
            terminated=terminated,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return self._game.observe(state._x, player_id)

    @property
    def id(self) -> core.EnvId:
        return "pig"

    @property
    def version(self) -> str:
        # v1: the observation gains a seventh feature, the total holding now
        # would bank. The rules are v0's.
        return "v1"

    @property
    def chance_keys(self) -> Array:
        """Step keys for every chance outcome of a step, equally likely:
        `chance_keys[r - 1]` rolls an `r`. Lets a search average over the die
        exactly rather than sample it."""
        return self._chance_keys

    @property
    def num_players(self) -> int:
        return PLAYER_COUNT
