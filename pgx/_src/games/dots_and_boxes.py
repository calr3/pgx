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

"""Dots and Boxes on a board of SIZE x SIZE boxes ((SIZE + 1) x (SIZE + 1) dots).

Players take turns drawing a line between two adjacent dots. A player who
completes the fourth side of a box (or two boxes at once) owns it and must move
again. When every line is drawn, whoever owns more boxes wins; equal counts
draw.

Lines are the actions. Horizontal lines come first, row by row from the top:
line `r * SIZE + c` joins dots (r, c) and (r, c + 1), for r in 0..SIZE and c in
0..SIZE-1. Then vertical lines: line `H + r * (SIZE + 1) + c` joins dots (r, c)
and (r + 1, c), for r in 0..SIZE-1 and c in 0..SIZE. Box (r, c) is bounded by
horizontal lines (r, c) and (r + 1, c) and vertical lines (r, c) and (r, c + 1).
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

SIZE = 6  # boxes per side
BOXES = SIZE * SIZE
H = (SIZE + 1) * SIZE  # horizontal lines
V = SIZE * (SIZE + 1)  # vertical lines
LINES = H + V
GRID = 2 * SIZE + 1  # the lattice of dots, lines and boxes the observation is laid on


def h_line(r: int, c: int) -> int:
    """The horizontal line from dot (r, c) to dot (r, c + 1)."""
    return r * SIZE + c


def v_line(r: int, c: int) -> int:
    """The vertical line from dot (r, c) to dot (r + 1, c)."""
    return H + r * (SIZE + 1) + c


def _box_lines() -> np.ndarray:
    out = np.zeros((BOXES, 4), dtype=np.int32)
    for r in range(SIZE):
        for c in range(SIZE):
            out[r * SIZE + c] = [h_line(r, c), h_line(r + 1, c), v_line(r, c), v_line(r, c + 1)]
    return out


def _line_cells() -> np.ndarray:
    """Each line's (row, col) on the lattice: dots at even/even, boxes at odd/odd."""
    out = np.zeros((LINES, 2), dtype=np.int32)
    for r in range(SIZE + 1):
        for c in range(SIZE):
            out[h_line(r, c)] = [2 * r, 2 * c + 1]
    for r in range(SIZE):
        for c in range(SIZE + 1):
            out[v_line(r, c)] = [2 * r + 1, 2 * c]
    return out


_BOX_LINES = jnp.asarray(_box_lines())  # (BOXES, 4): the four sides of each box
_LINE_CELLS = jnp.asarray(_line_cells())  # (LINES, 2)
_LINE_SLOTS = jnp.zeros((GRID, GRID), jnp.float32).at[_LINE_CELLS[:, 0], _LINE_CELLS[:, 1]].set(1.0)


class GameState(NamedTuple):
    """Internal state for Dots and Boxes."""

    color: Array = jnp.int32(0)
    lines: Array = jnp.zeros(LINES, jnp.bool_)  # drawn or not, in action order
    owner: Array = jnp.full(BOXES, -1, jnp.int32)  # the colour owning each box, -1 if open
    winner: Array = jnp.int32(-1)  # -1 while going or drawn; see `rewards`


def _on_boxes(values: Array) -> Array:
    """Place one value per box on the lattice (odd, odd cells); zero elsewhere."""
    grid = jnp.zeros((GRID, GRID), values.dtype)
    return grid.at[1::2, 1::2].set(values.reshape(SIZE, SIZE))


class Game:
    """The game representation of Dots and Boxes."""

    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        lines = state.lines.at[action].set(True)
        complete = lines[_BOX_LINES].all(axis=1)
        taken = complete & (state.owner < 0)
        owner = jnp.where(taken, state.color, state.owner)
        # Completing a box earns another move; otherwise the turn passes.
        color = jax.lax.select(taken.any(), state.color, 1 - state.color)
        counts = jnp.stack([(owner == 0).sum(), (owner == 1).sum()])
        over = lines.all()
        winner = jax.lax.select(
            over & (counts[0] != counts[1]), jnp.argmax(counts).astype(jnp.int32), jnp.int32(-1)
        )
        return state._replace(color=color, lines=lines, owner=owner, winner=winner)  # type: ignore

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        """(GRID, GRID, 8) from `color`'s side (the mover's by default), on the
        lattice of dots (even, even), lines (one even, one odd) and boxes
        (odd, odd):

        0. lines drawn
        1. boxes `color` owns
        2. boxes the opponent owns
        3. where lines can be (constant)
        4. open boxes with three sides drawn: the mover can take them now
        5. open boxes with two sides drawn: what chains are made of
        6. `color`'s box count / BOXES, everywhere
        7. the opponent's box count / BOXES, everywhere

        Planes 4-7 are derivable from the others, but a conv net would have to
        learn to count sides and to add up the whole board to see them.
        """
        if color is None:
            color = state.color
        drawn = jnp.zeros((GRID, GRID), jnp.float32).at[_LINE_CELLS[:, 0], _LINE_CELLS[:, 1]].set(
            state.lines.astype(jnp.float32)
        )
        sides = state.lines[_BOX_LINES].sum(axis=1)
        open_ = state.owner < 0
        mine = (state.owner == color).astype(jnp.float32)
        theirs = (state.owner == 1 - color).astype(jnp.float32)
        ones = jnp.ones((GRID, GRID), jnp.float32)
        return jnp.stack(
            [
                drawn,
                _on_boxes(mine),
                _on_boxes(theirs),
                _LINE_SLOTS,
                _on_boxes((open_ & (sides == 3)).astype(jnp.float32)),
                _on_boxes((open_ & (sides == 2)).astype(jnp.float32)),
                ones * mine.sum() / BOXES,
                ones * theirs.sum() / BOXES,
            ],
            axis=-1,
        )

    def legal_action_mask(self, state: GameState) -> Array:
        return ~state.lines

    def is_terminal(self, state: GameState) -> Array:
        return state.lines.all()

    def rewards(self, state: GameState) -> Array:
        """Indexed by colour: +1 / -1 once decided, zeros while going or drawn."""
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )

    def box_counts(self, state: GameState) -> Array:
        return jnp.stack([(state.owner == 0).sum(), (state.owner == 1).sum()])

    def pretty(self, state: GameState) -> str:
        """The board in text: `+` dots, `---` and `|` lines, and each box's
        owner (0 or 1) in its middle."""
        lines = np.asarray(state.lines)
        owner = np.asarray(state.owner)
        rows = []
        for r in range(SIZE + 1):
            rows.append(
                "+" + "+".join("---" if lines[h_line(r, c)] else "   " for c in range(SIZE)) + "+"
            )
            if r == SIZE:
                break
            cells = []
            for c in range(SIZE + 1):
                cells.append("|" if lines[v_line(r, c)] else " ")
                if c < SIZE:
                    o = owner[r * SIZE + c]
                    cells.append(f" {o} " if o >= 0 else "   ")
            rows.append("".join(cells))
        return "\n".join(rows)
