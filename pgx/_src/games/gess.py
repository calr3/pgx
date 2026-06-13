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

# Gess is played on a 20×20 grid; the playing area is the inner 18×18 cells
# (indices 1–18 in each axis).  The outermost ring (index 0 and 19) acts as
# an invisible border: any stone that lands there is removed.
#
# A move is represented as two consecutive actions:
#   Stage 0 – choose the source centre (0..399, row-major on the 20×20 grid).
#   Stage 1 – choose the destination centre (same space).
#
# Board values: 0 = empty, 1 = black (player 0, moves first), 2 = white (player 1).
# Both stages share the same 400-wide legal_action_mask.

from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

BOARD_SIZE = 20            # total grid side (including border ring)
N          = BOARD_SIZE ** 2  # 400 flat cells
MIN_IDX    = 1             # first playing-area row/col index
MAX_IDX    = 18            # last  playing-area row/col index

# Board cell values (see header note above).
EMPTY = 0                  # empty cell
BLACK = 1                  # black stone (player 0, moves first)
WHITE = 2                  # white stone (player 1)

# The game is drawn once this many consecutive turns (full moves) pass without
# a capture of any kind (opponent captures and self-captures both count).
DRAW_NO_CAPTURE_TURNS = 20

# Eight compass directions as a JAX array (8, 2) of (dr, dc) pairs.
_DIRS_ARR: Array = jnp.int32([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])

# Row/column offsets for all 9 cells in a 3×3 footprint (row-major order).
_SLOT_DR: Array = jnp.int32([-1, -1, -1,  0,  0,  0,  1,  1,  1])
_SLOT_DC: Array = jnp.int32([-1,  0,  1, -1,  0,  1, -1,  0,  1])

# Steps 1..19 — the most a piece centre can ever travel within the 20×20 grid.
_STEPS: Array = jnp.arange(1, BOARD_SIZE, dtype=jnp.int32)


# ─── GameState ──────────────────────────────────────────────────────────────

class GameState(NamedTuple):
    color:  Array = jnp.int32(0)            # 0 = black, 1 = white
    board:  Array = jnp.zeros(N, jnp.int8)  # 0=empty  1=black  2=white
    stage:  Array = jnp.int32(0)            # 0=pick source  1=pick dest
    source: Array = jnp.int32(0)            # source centre chosen in stage 0
    winner: Array = jnp.int32(-1)           # -1=ongoing  0=black wins  1=white wins
    # Consecutive completed turns with no capture; the game is drawn once it
    # reaches DRAW_NO_CAPTURE_TURNS (see is_terminal).
    no_capture_turns: Array = jnp.int32(0)


# ─── Game class ──────────────────────────────────────────────────────────────

class Game:
    def init(self) -> GameState:
        return GameState(board=_make_init_board())

    def step(self, state: GameState, action: Array) -> GameState:
        return jax.lax.cond(
            state.stage == 0,
            lambda: _step_stage0(state, action),
            lambda: _step_stage1(state, action),
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        return _observe(state, color)

    def legal_action_mask(self, state: GameState) -> Array:
        src_mask = _legal_source_mask(state)
        dst_mask = _legal_dest_mask(state, state.source)
        return jnp.where(state.stage == 0, src_mask, dst_mask)

    def is_terminal(self, state: GameState) -> Array:
        # A win, or a draw from too many consecutive captureless turns. A win
        # always destroys a ring (hence captures), so it resets no_capture_turns;
        # the two conditions therefore never collide.
        return (state.winner >= 0) | (state.no_capture_turns >= DRAW_NO_CAPTURE_TURNS)

    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1.0, -1.0]).at[state.winner].set(1.0),
            jnp.zeros(2, jnp.float32),
        )


# ─── Step helpers ────────────────────────────────────────────────────────────

def _step_stage0(state: GameState, action: Array) -> GameState:
    return state._replace(stage=jnp.int32(1), source=action)


def _step_stage1(state: GameState, action: Array) -> GameState:
    mover_stone = jnp.int8(state.color + 1)   # 1 for black, 2 for white
    opp_stone   = jnp.int8(2 - state.color)

    new_board = _apply_move(state.board, state.source, action, mover_stone)

    # A capture (opponent or self) always removes at least one stone, and a move
    # never adds stones, so the total count strictly decreases iff something was
    # captured. No capture leaves every stone merely relocated → count unchanged.
    captured = jnp.sum(new_board != 0) < jnp.sum(state.board != 0)
    no_capture_turns = jnp.where(
        captured, jnp.int32(0), state.no_capture_turns + 1
    )

    mover_has_ring = _has_ring(new_board, mover_stone)
    opp_has_ring   = _has_ring(new_board, opp_stone)

    # Rules: if neither has a ring the player who just moved loses;
    # otherwise a player who has no ring loses.
    # ⟹ check mover first: if mover has no ring → opp wins.
    terminal = ~mover_has_ring | ~opp_has_ring
    winner = jax.lax.select(
        terminal,
        jax.lax.select(~mover_has_ring, jnp.int32(1 - state.color), state.color),
        jnp.int32(-1),
    )

    return state._replace(
        color=jnp.int32(1 - state.color),
        board=new_board,
        stage=jnp.int32(0),
        winner=winner,
        no_capture_turns=no_capture_turns,
    )


# ─── Move application ────────────────────────────────────────────────────────

def _apply_move(board: Array, source: Array, dest: Array, mover_stone: Array) -> Array:
    """Return the board after moving the piece centred at `source` to `dest`."""
    cell_idx = jnp.arange(N, dtype=jnp.int32)
    cell_r   = cell_idx // BOARD_SIZE
    cell_c   = cell_idx  % BOARD_SIZE

    src_r = source // BOARD_SIZE;  src_c = source % BOARD_SIZE
    dst_r = dest   // BOARD_SIZE;  dst_c = dest   % BOARD_SIZE

    in_src = (jnp.abs(cell_r - src_r) <= 1) & (jnp.abs(cell_c - src_c) <= 1)
    in_dst = (jnp.abs(cell_r - dst_r) <= 1) & (jnp.abs(cell_c - dst_c) <= 1)

    # 1. Lift the mover's own stones from the source footprint.
    cleared = jnp.where(in_src & (board == mover_stone), jnp.int8(0), board)
    # 2. Displace every stone at the destination footprint (captures).
    cleared = jnp.where(in_dst, jnp.int8(0), cleared)

    # 3. Place the lifted stones at the destination.
    #    Stone at source (src_r+dr, src_c+dc) lands at (dst_r+dr, dst_c+dc).
    #    For each destination cell (cell_r, cell_c), its donor source cell is:
    #      (cell_r - dst_r + src_r, cell_c - dst_c + src_c).
    src_r_for = cell_r - dst_r + src_r
    src_c_for = cell_c - dst_c + src_c
    src_inbnd  = ((src_r_for >= 0) & (src_r_for < BOARD_SIZE) &
                  (src_c_for >= 0) & (src_c_for < BOARD_SIZE))
    safe_src   = jnp.clip(src_r_for * BOARD_SIZE + src_c_for, 0, N - 1)
    src_had_own = src_inbnd & (board[safe_src] == mover_stone)

    # Stones landing on the border ring (row/col 0 or 19) are removed.
    on_border  = ((cell_r == 0) | (cell_r == BOARD_SIZE - 1) |
                  (cell_c == 0) | (cell_c == BOARD_SIZE - 1))
    place_here = in_dst & src_had_own & ~on_border

    return jnp.where(place_here, mover_stone, cleared)


# ─── Legal-move masks ────────────────────────────────────────────────────────

def _legal_source_mask(state: GameState) -> Array:
    """Boolean mask (400,): True for each cell that is a valid piece centre.

    A centre is valid when the 3×3 footprint contains no opponent stone AND
    at least one own stone among the eight surrounding cells (not the centre).
    """
    b         = state.board.reshape(BOARD_SIZE, BOARD_SIZE)
    own_stone = jnp.int8(state.color + 1)
    opp_stone = jnp.int8(2 - state.color)

    own_2d = (b == own_stone)
    opp_2d = (b == opp_stone)

    # Pad by 1 so that slice [1+dr : BOARD_SIZE+1+dr] always covers the full grid.
    own_p = jnp.pad(own_2d, 1, constant_values=False)
    opp_p = jnp.pad(opp_2d, 1, constant_values=False)

    opp_in_fp       = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
    own_in_surround = jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            sl = opp_p[1 + dr: BOARD_SIZE + 1 + dr, 1 + dc: BOARD_SIZE + 1 + dc]
            opp_in_fp = opp_in_fp | sl
            if not (dr == 0 and dc == 0):
                sl2 = own_p[1 + dr: BOARD_SIZE + 1 + dr, 1 + dc: BOARD_SIZE + 1 + dc]
                own_in_surround = own_in_surround | sl2

    return (~opp_in_fp & own_in_surround).flatten()


def _legal_dest_mask(state: GameState, source: Array) -> Array:
    """Boolean mask (400,): True for reachable destination centres from `source`.

    All 19 potential steps per direction are evaluated in parallel:
      1. Compute (19, 9) footprint-cell coordinates for every step at once.
      2. Check clearance against the board with a (19, 9) gather — no 400-wide
         table lookup needed; footprint membership is tested arithmetically.
      3. Replace the sequential `lax.scan` carry with a prefix cumsum to find
         the "first blocker" boundary, which XLA can parallelise.
    """
    board     = state.board
    own_stone = jnp.int8(state.color + 1)

    src_r      = source // BOARD_SIZE
    src_c      = source  % BOARD_SIZE
    has_center = (board[source] == own_stone)
    max_steps  = jax.lax.select(has_center, jnp.int32(BOARD_SIZE), jnp.int32(3))
    occupied   = (board != 0)          # (N,) bool – any stone on the board

    def _one_dir(dr_dc):
        """Valid destinations and their flat indices for a single direction."""
        dr, dc = dr_dc[0], dr_dc[1]

        # Direction enabled: own stone at the footprint cell one step that way.
        dir_r       = src_r + dr
        dir_c       = src_c + dc
        dir_inbnd   = ((dir_r >= 0) & (dir_r < BOARD_SIZE) &
                       (dir_c >= 0) & (dir_c < BOARD_SIZE))
        safe_dir    = jnp.clip(dir_r * BOARD_SIZE + dir_c, 0, N - 1)
        dir_enabled = dir_inbnd & (board[safe_dir] == own_stone)

        # Destination centres for steps 1..19  →  (19,)
        nr        = src_r + dr * _STEPS
        nc        = src_c + dc * _STEPS
        in_bounds = ((nr >= 0) & (nr < BOARD_SIZE) &
                     (nc >= 0) & (nc < BOARD_SIZE))
        in_range  = _STEPS <= max_steps
        safe_dest = jnp.clip(nr * BOARD_SIZE + nc, 0, N - 1)

        # 3×3 footprint cells for every destination  →  (19, 9)
        fp_r     = nr[:, None] + _SLOT_DR[None, :]
        fp_c     = nc[:, None] + _SLOT_DC[None, :]
        fp_valid = ((fp_r >= 0) & (fp_r < BOARD_SIZE) &
                    (fp_c >= 0) & (fp_c < BOARD_SIZE))
        safe_fp  = jnp.clip(fp_r * BOARD_SIZE + fp_c, 0, N - 1)

        # Is this footprint cell transparent (part of the source's own footprint)?
        in_src = (jnp.abs(fp_r - src_r) <= 1) & (jnp.abs(fp_c - src_c) <= 1)

        # Clearance hit: in dest footprint, not transparent, occupied  →  (19,)
        has_blocker = (
            jnp.any(fp_valid & ~in_src & occupied[safe_fp], axis=1) & in_bounds
        )

        # blocked_before[k] = any blocker at steps 0..k-1 (prefix cumsum).
        blocked_before = jnp.concatenate([
            jnp.zeros(1, jnp.int32),
            jnp.cumsum(has_blocker.astype(jnp.int32))[:-1],
        ]) > 0  # (19,)

        is_valid = dir_enabled & ~blocked_before & in_bounds & in_range
        return is_valid.astype(jnp.int32), safe_dest   # (19,), (19,)

    # Evaluate all 8 directions in one vectorised pass  →  (8, 19) each.
    # vmap makes the direction-parallelism explicit to XLA: a single fused
    # (8, 19, 9) board-gather and one (8×19) scatter rather than 8 separate ops.
    is_valid_8, safe_dest_8 = jax.vmap(_one_dir)(_DIRS_ARR)

    return (jnp.zeros(N, dtype=jnp.int32)
            .at[safe_dest_8.flatten()]
            .add(is_valid_8.flatten())) > 0


# ─── Ring detection ──────────────────────────────────────────────────────────

def _has_ring(board: Array, stone_val: Array) -> Array:
    """True iff `stone_val` has at least one ring anywhere on the playing area.

    A ring: centre cell empty, all eight surrounding cells have `stone_val`.
    Centre must be inside [MIN_IDX..MAX_IDX] so all neighbours are in-bounds.
    """
    b    = board.reshape(BOARD_SIZE, BOARD_SIZE)
    own  = (b == stone_val)
    empty = (b == 0)

    centre_empty = empty[MIN_IDX: MAX_IDX + 1, MIN_IDX: MAX_IDX + 1]  # (18, 18)
    surround_all = jnp.ones((MAX_IDX - MIN_IDX + 1,) * 2, dtype=jnp.bool_)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            surround_all = surround_all & own[MIN_IDX + dr: MAX_IDX + 1 + dr,
                                               MIN_IDX + dc: MAX_IDX + 1 + dc]

    return jnp.any(centre_empty & surround_all)


# ─── Observation ─────────────────────────────────────────────────────────────

def _observe(state: GameState, color: Array) -> Array:
    """Return an (18, 18, 4) float32 observation from `color`'s perspective.

    The border ring (row/col 0 and 19) is always empty by the game rules, so
    the observation covers only the 18×18 playing area (rows/cols 1–18).

    Channels:
      0 – own stones
      1 – opponent stones
      2 – source footprint indicator (filled when stage == 1)
      3 – stage indicator (0.0 or 1.0)
    """
    own_stone = jnp.int8(color + 1)
    opp_stone = jnp.int8(2 - color)
    b = state.board.reshape(BOARD_SIZE, BOARD_SIZE)
    p = slice(MIN_IDX, MAX_IDX + 1)   # slice for the playing area

    own = (b[p, p] == own_stone).astype(jnp.float32)
    opp = (b[p, p] == opp_stone).astype(jnp.float32)

    src_r = state.source // BOARD_SIZE
    src_c = state.source  % BOARD_SIZE
    rr = jnp.arange(MIN_IDX, MAX_IDX + 1, dtype=jnp.int32)[:, None]
    cc = jnp.arange(MIN_IDX, MAX_IDX + 1, dtype=jnp.int32)[None, :]
    in_src_fp = ((jnp.abs(rr - src_r) <= 1) &
                 (jnp.abs(cc - src_c) <= 1) &
                 (state.stage == 1)).astype(jnp.float32)

    play_size  = MAX_IDX - MIN_IDX + 1   # 18
    stage_plane = jnp.full((play_size, play_size), state.stage.astype(jnp.float32))

    return jnp.stack([own, opp, in_src_fp, stage_plane], axis=-1)  # (18, 18, 4)


# ─── Initial position ────────────────────────────────────────────────────────

def _make_init_board() -> Array:
    """Standard Gess opening: 43 stones per player, one ring each."""
    board = np.zeros(N, dtype=np.int8)
    # White (player 1, stone=2) occupies rows 1,2,3,6; black mirrors at 19-r.
    white_rows: list = [
        (1, [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]),
        (2, [1, 2, 3, 5, 7, 8, 9, 10, 12, 14, 16, 17, 18]),
        (3, [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]),
        (6, [2, 5, 8, 11, 14, 17]),
    ]
    for r, cols in white_rows:
        for c in cols:
            board[r * BOARD_SIZE + c]                      = 2  # white
            board[(BOARD_SIZE - 1 - r) * BOARD_SIZE + c]  = 1  # black (mirror)
    return jnp.array(board, dtype=jnp.int8)
