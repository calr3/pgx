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
from pgx._src.games.dots_and_boxes import GRID, LINES, SIZE, Game, GameState, h_line, v_line
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

__all__ = ["DotsAndBoxes", "State", "SIZE", "h_line", "v_line", "line_name", "parse_line"]

_FILES = "abcdefghijklmnopqrstuvwxyz"[: SIZE + 1]


def _dot(r: int, c: int) -> str:
    return f"{_FILES[c]}{r + 1}"


def line_name(action: int) -> str:
    """A line as the two dots it joins, columns a.. from the left and rows 1..
    from the top: `a1-b1` (horizontal), `a1-a2` (vertical)."""
    action = int(action)
    if action < (SIZE + 1) * SIZE:
        r, c = divmod(action, SIZE)
        return f"{_dot(r, c)}-{_dot(r, c + 1)}"
    r, c = divmod(action - (SIZE + 1) * SIZE, SIZE + 1)
    return f"{_dot(r, c)}-{_dot(r + 1, c)}"


def parse_line(text: str):
    """The action for two adjacent dots, in either order and separated by `-`
    or a space (`b3-c3`, `c3 b3`); None for anything else."""
    parts = text.replace("-", " ").split()
    if len(parts) != 2:
        return None
    dots = []
    for p in parts:
        if len(p) < 2 or p[0] not in _FILES or not p[1:].isdigit():
            return None
        r, c = int(p[1:]) - 1, _FILES.index(p[0])
        if not 0 <= r <= SIZE:
            return None
        dots.append((r, c))
    (r1, c1), (r2, c2) = sorted(dots)
    if r1 == r2 and c2 == c1 + 1:
        return h_line(r1, c1)
    if c1 == c2 and r2 == r1 + 1:
        return v_line(r1, c1)
    return None


@dataclass
class State(core.State):
    """State for Dots and Boxes."""

    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((GRID, GRID, 8), dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(LINES, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        return "dots_and_boxes"


class DotsAndBoxes(core.Env):
    """Dots and Boxes on 6 x 6 boxes (7 x 7 dots): 84 lines, one action each.

    Completing a box earns another move, so the player to move does not simply
    alternate. Which seat plays colour 0 (who draws first) is drawn at random,
    as in pgx's other two-player games; `current_player` follows the colour.
    """

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
        # The seat changes exactly when the colour does: not after a box.
        current_player = jax.lax.select(
            x.color == state._x.color, state.current_player, 1 - state.current_player
        )
        state = state.replace(current_player=current_player, _x=x)  # type: ignore
        assert isinstance(state, State)
        terminated = self._game.is_terminal(x)
        # Rewards are by colour; colour 0 is the seat that started, which is
        # the current player exactly when the current colour is 0.
        seat_of_color0 = jax.lax.select(x.color == 0, current_player, 1 - current_player)
        rewards = self._game.rewards(x)
        rewards = jax.lax.select(seat_of_color0 == 0, rewards, jnp.flip(rewards))
        rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))
        return state.replace(  # type: ignore
            legal_action_mask=self._game.legal_action_mask(x),
            rewards=rewards,
            terminated=terminated,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        color = jax.lax.select(
            player_id == state.current_player, state._x.color, 1 - state._x.color
        )
        return self._game.observe(state._x, color)

    def pretty(self, state: State) -> str:
        return self._game.pretty(state._x)

    @property
    def id(self) -> core.EnvId:
        return "dots_and_boxes"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2
