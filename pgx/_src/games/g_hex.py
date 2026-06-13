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
    """Internal state for the game GHex."""

    #       ____________
    #      /\    /\    /\                    0
    #     / 0\ 1/ 2\ 3/ 4\                   1
    #    /____\/____\/____\                  2
    #   /\    /\    /\    /\                 3
    #  / 5\ 6/ 7\ 8/ 9\10/11\                4
    # /____\/____\/____\/____\               5
    # \    /\    /\                          6
    #  \12/13\14/15\                         7
    #   \/____\/____\______                  8
    #    \    /\    /\    /                  9
    #     \16/17\18/19\20/                  10
    #      \/____\/____\/                   11
    # Always-empty triangle: 21
    board: Array = jnp.zeros(22, jnp.int32)
    tiles: Array = jnp.ones((2, 10), jnp.bool_)
    color: Array = jnp.int32(0)
    winner: Array = jnp.int32(-1)

    def h(self, i):
      if self.board[i] == 10:
         return " T"
      if self.board[i] == -10:
          return "-T"
      return f"{(self.board[i]):2}"

    def pretty(self):
      return f"""
           ____________
          /\\    /\\    /\\                    0
         /{self.h(0)}\\{self.h(1)}/{self.h(2)}\\{self.h(3)}/{self.h(4)}\\                   1
        /____\\/____\\/____\\                  2
       /\\    /\\    /\\    /\\                 3
      /{self.h(5)}\\{self.h(6)}/{self.h(7)}\\{self.h(8)}/{self.h(9)}\\{self.h(10)}/{self.h(11)}\\                4
     /____\\/____\\/____\\/____\\               5
     \\    /\\    /\\                          6
      \\{self.h(12)}/{self.h(13)}\\{self.h(14)}/{self.h(15)}\\                         7
       \\/____\\/____\\______                  8
        \\    /\\    /\\    /                  9
         \\{self.h(16)}/{self.h(17)}\\{self.h(18)}/{self.h(19)}\\{self.h(20)}/                  10
          \\/____\\/____\\/                   11
           """


    def debug_me(self):
        new_board = self.board
        golden_sum = (new_board[_ADJ].sum(axis=1) * (new_board[:21] == 0)).sum()
        return {
           "sum": golden_sum,
           "sign":  jnp.sign(golden_sum),
           "winner": _WINNER_BY_SIGN[1 + jnp.sign(golden_sum)],
           "board": self.pretty(),
        }


_ADJ = jnp.array([
             [1, 6, 21],   #  0
             [0, 2, 21],   #  1
             [1, 3, 8],    #  2
             [2, 4, 21],   #  3
             [3, 10, 21],  #  4
             [6, 12, 21],  #  5
             [0, 5, 7],    #  6
             [6, 8, 14],   #  7
             [2, 7, 9],    #  8
             [8, 10, 21],  #  9
             [9, 11, 21],  # 10
             [10, 21, 21], # 11
             [5, 13, 21],  # 12
             [12, 14, 16], # 13
             [7, 13, 15],  # 14
             [14, 18, 21], # 15
             [13, 17, 21], # 16
             [16, 18, 21], # 17
             [15, 17, 19], # 18
             [18, 20, 21], # 19
             [19, 21, 21], # 20
         ], dtype=jnp.int32)

_WINNER_BY_SIGN = jnp.array([1, -1, 0], dtype=jnp.int32)
_TILE_VAL = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.int32)

def _golden_sum(new_board):
    return (new_board[_ADJ].sum(axis=1) * (new_board[:21] == 0)).sum()


class Game:
    """The game representation of GHex on an 8x8 board."""

    def init(self) -> GameState:
        return GameState()


    def h(self, state:GameState, i):
      if state.board[0][i] == 10:
          return "BT"
      if state.board[0][i] == -10:
          return "WT"
      if state.board[0][i] == 0:
          return "  "
      if state.board[0][i] > 0:
          return f"B{state.board[0][i]}"
      else:
          return f"W{abs(state.board[0][i])}"


    def pretty_game(self, state: GameState):
      return f"""
           ____________
          /\\    /\\    /\\
         /{self.h(state,0)}\\{self.h(state,1)}/{self.h(state,2)}\\{self.h(state,3)}/{self.h(state,4)}\\
        /____\\/____\\/____\\
       /\\    /\\    /\\    /\\
      /{self.h(state,5)}\\{self.h(state,6)}/{self.h(state,7)}\\{self.h(state,8)}/{self.h(state,9)}\\{self.h(state,10)}/{self.h(state,11)}\\
     /____\\/____\\/____\\/____\\
     \\    /\\    /\\
      \\{self.h(state,12)}/{self.h(state,13)}\\{self.h(state,14)}/{self.h(state,15)}\\
       \\/____\\/____\\______
        \\    /\\    /\\    /
         \\{self.h(state,16)}/{self.h(state,17)}\\{self.h(state,18)}/{self.h(state,19)}\\{self.h(state,20)}/
          \\/____\\/____\\/
           """


    def step(self, state: GameState, action: Array) -> GameState:
        """Performs a step in the GHex game.

        Args:
          state: The current game state.
          action: The chosen action.

        Returns:
          The new game state after the action has been applied.
        """
        triangle = action // 10  # was 20
        tile_i = action % 10

        new_board = state.board.at[triangle].set((1 + tile_i) * (1 - 2 * state.color))
        golden_sum = _golden_sum(new_board)

        return state._replace(  # type: ignore
            board=new_board,
            tiles=state.tiles.at[state.color, tile_i].set(False),
            color=1 - state.color,
            winner=jax.lax.select(state.tiles.sum() > 1, -1, _WINNER_BY_SIGN[1 + jnp.sign(golden_sum)])
        )


    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color

        virtual_up =   jnp.zeros(24, dtype=jnp.int32).at[1].set(1) .at[3].set(color)
        virtual_down = jnp.zeros(24, dtype=jnp.int32).at[1].set(-1).at[3].set(color)

        tile_features = jnp.concatenate([state.tiles[color] * _TILE_VAL, state.tiles[1 - color] * -_TILE_VAL])
        fstack = jnp.vstack([state.board[:21] * (1 - 2 * color),
                             jnp.array([
                                1, -1,  1, -1,  1,
                                1, -1,  1, -1,  1, -1,  1,
                               -1,  1, -1,  1,
                               -1,  1, -1,  1, -1,
                             ], dtype=jnp.int32),
                             jnp.ones(21, dtype=jnp.int32),
                             jnp.ones(21, dtype=jnp.int32) * color,
                             (state.board[:21] == 0) * jnp.transpose(jnp.broadcast_to(tile_features, (21, 20))),
                            ]).transpose()

        return jnp.array([
          [virtual_down, fstack[0],  fstack[1],  fstack[2],  fstack[3],    fstack[4],   virtual_down],
          [fstack[5],    fstack[6],  fstack[7],  fstack[8],  fstack[9],    fstack[10],  fstack[11]],
          [fstack[12],   fstack[13], fstack[14], fstack[15], virtual_down, virtual_up,  virtual_down],
          [virtual_up,   fstack[16], fstack[17], fstack[18], fstack[19],   fstack[20],  virtual_up],
        ])


    def legal_action_mask(self, state: GameState) -> Array:
        tile_available = state.tiles[state.color]
        def can_play(bl):
            return tile_available & bl
        return jax.vmap(can_play)(state.board[:21] == 0).flatten()


    def is_terminal(self, state: GameState) -> Array:
        return state.tiles.sum() == 0


    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )
