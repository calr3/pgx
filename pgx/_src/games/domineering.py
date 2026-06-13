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

import jax
import jax.numpy as jnp
from jax import Array


class GameState(NamedTuple):
    """Internal state for the game Domineering on an 8x8 board."""

    color: Array = jnp.int32(0)
    # 8x8 board
    # [[ 0,  1,  2,  3,  4,  5,  6,  7],
    #  [ 8,  9, 10, 11, 12, 13, 14, 15],
    #  [16, 17, 18, 19, 20, 21, 22, 23],
    #  [24, 25, 26, 27, 28, 29, 30, 31],
    #  [32, 33, 34, 35, 36, 37, 38, 39],
    #  [40, 41, 42, 43, 44, 45, 46, 47],
    #  [48, 49, 50, 51, 52, 53, 54, 55],
    #  [56, 57, 58, 59, 60, 61, 62, 63]]
    board: Array = jnp.ones((8, 8), jnp.bool_)  # True (available), False (occupied)
    winner: Array = jnp.int32(-1)


class Game:
    """The game representation of Domineering on an 8x8 board."""

    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        """Performs a step in the Domineering game.

        Args:
          state: The current game state.
          action: The chosen action, representing the index of the top/left
            square of the domino to be placed.

        Returns:
          The new game state after the action has been applied.
        """

        # Transpose the move index so that the same action is mapped to the same square.
        x = action % 7
        y = action // 7

        new_board = state.board.at[y, x].set(False).at[y, x + 1].set(False).transpose()

        def can_play(move_mask):
            return new_board.flatten()[move_mask].all()

        # Game is over if the player next to play has no legal moves.
        has_next_move = jax.vmap(can_play)(_MASK_CACHE).any()

        return state._replace(  # type: ignore
            color=1 - state.color,
            board=new_board,
            winner=jax.lax.select(has_next_move, -1, state.color),
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        return jnp.dstack([
            # Feature 0: is square empty. Tranposed so current player is always horizontal.
            state.board,
            # Feature 1: whose turn (1-hot).
            (color == 0) * jnp.ones_like(state.board, dtype=jnp.bool_),
        ])

    def legal_action_mask(self, state: GameState) -> Array:
        # To be legal, a move must have its own square and a neighbour free, and not be
        # on the edge of the board. The relevant definition of neighbour and edge
        # depends on the player's direction.
        playable = state.board & jnp.roll(state.board, shift=-1, axis=1)
        # Strip off the last column (which is never playable, and is also set incorrectly).
        return playable[:, :-1].flatten()

    def is_terminal(self, state: GameState) -> Array:
        return state.winner >= 0  # Game always ends with a winner.

    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )


def _make_mask_cache():
    move_masks = []
    for x in range(7):
        for y in range(8):
            move_masks.append(jnp.array([y * 8 + x, y * 8 + x + 1]))
    return jnp.array(move_masks)


# Precomputed masks for required empty squares for each possible move.
_MASK_CACHE = _make_mask_cache()
