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
    observation: Array = jnp.zeros((HEIGHT, WIDTH, 16), dtype=jnp.float32)
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
        # v4: an exact mirror falls back to the last capture, then to black. No draws.
        # v5: back to an absolute 300-move cap; the capture clock cost ~100 Elo.
        # v6: a capture clock again, but 100 quiet moves rather than 60, keeping
        #     v4's rank-by-rank tiebreak. There is no absolute bound in v6.
        # v7: the observation gains two planes - the quiet-move clock and the
        #     verdict if the game ended now. Both were invisible before, and
        #     they decide most games. 7 channels -> 9, so v6 and earlier
        #     checkpoints cannot be loaded against this env.
        # v8: those two become one signed clock plane. v7's raw verdict read -1
        #     for white and +1 for black in any symmetric position, including
        #     the opening, so it leaked colour and cost a 3.3 sigma seat
        #     asymmetry; scaling it by the clock makes it 0 there. 9 -> 8
        #     channels, so v7 checkpoints are retired too - plane 7 changed
        #     meaning, so narrowing cannot rescue them.
        # v9: the clock/verdict plane is unwired, back to 7 planes. Both forms
        #     of it leaked the tiebreak's black-wins-a-mirror default as a
        #     colour signal and cost 92, 351 and 411 Elo (E10, E11, E12).
        #     v6-era checkpoints load again; v7/v8 ones are retired.
        # v10: the observation gains nine evaluation-feature planes, the terms
        #      of tdgauntlet's alpha-beta engine (the six LEONIDAS heuristics
        #      plus its three advancement terms), each scaled to about [-1, 1].
        #      Every term is "ours minus theirs", so the vector is antisymmetric
        #      under a colour flip and leaks no colour, unlike v7/v8. 7 planes
        #      -> 16; planes 0-6 are unchanged, so older checkpoints still play
        #      once narrowed to the leading 7.
        return "v10"

    @property
    def num_players(self) -> int:
        return 2
