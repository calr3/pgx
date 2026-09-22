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

# Epaminondas (Robert Abbott, 1975), played on a 14 x 12 board.
#
# Each player starts with 28 pieces filling the two rows nearest them. A
# *phalanx* is two or more friendly pieces adjacent in a straight line
# (orthogonal or diagonal). A turn moves either a single piece one square in any
# direction to an empty square, or a phalanx along its own line: every piece
# moves the same number of squares, at most the number of pieces moving, and a
# phalanx may be split (any contiguous sub-run that includes the leading piece).
# Nothing may move onto or over a friendly piece or over an enemy piece.
#
# Capture: the lead piece may land on an enemy piece if the enemy line starting
# there and continuing *away* in the direction of movement is strictly shorter
# than the moving phalanx; that whole enemy line is removed and the movement
# stops there.
#
# Objective: at the start of your turn, if you have more pieces on your
# opponent's back rank than they have on yours, you win — so an incursion gives
# the opponent one turn to capture it or match it. A player may not move a piece
# onto the opponent's back rank if that would make the whole position
# left-to-right symmetric (this stops a mirroring draw).
#
# A move is played as three actions, each naming a square (0..167):
#   Stage 0 - the lead piece (the one that moves furthest forward).
#   Stage 1 - the rear piece of the moving group; the lead square itself means
#             a single piece. This fixes the direction and the group size.
#   Stage 2 - the destination of the lead piece.

from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

WIDTH = 14
HEIGHT = 12
N = WIDTH * HEIGHT  # 168 squares, indexed row-major: idx = row * WIDTH + col

# Board values.
EMPTY = 0
WHITE = 1  # player 0, moves first, back rank is row 0
BLACK = 2  # player 1, back rank is row HEIGHT - 1

# The game ends once this many moves have passed *without a capture*. The
# counter resets to zero on every capture, so a game that keeps trading stays
# alive; only a quiet stretch ends it. It is then decided on advancement: piece
# counts are compared rank by rank starting from the opponent's home rank, and
# the first rank that differs wins it (see _advancement_winner). An exact
# rank-for-rank mirror falls back to whoever captured last, and then to black,
# who moves second - so no game is ever drawn.
#
# This deviates from the published rules, which have no move limit at all; the
# limit exists only to bound self-play.
#
# History (EPAMINONDAS_EXPERIMENTS.md). A 60-move capture clock was tried twice
# and cost ~100 Elo both times (E5, E6): Epaminondas has long manoeuvring phases
# with no captures, and cutting them short stopped the model seeing a full
# strategic arc. An absolute 300-move cap replaced it (E7, +95 Elo). This is a
# capture clock again but at 100 moves, on the reasoning that an absolute cap
# pays a player who is ahead to run the clock out, and that 60 was simply too
# short a horizon rather than the idea being wrong.
#
# Note there is no longer an absolute bound. Each capture removes at least one
# piece and each side starts with 28, so at most 55 captures can occur and the
# worst case is ~56 * 100 = 5600 moves (16,800 actions). That is far beyond any
# game seen in practice - E8's averaged ~60 moves - but anything that caps steps
# per game (tournaments, ladders) has to allow for it or it will truncate.
MAX_QUIET_MOVES = 100

# The eight directions, and the index of each one's opposite.
_DIRS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int32
)
_OPP_DIR = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
NUM_DIRS = 8
# The longest line on the board is 14 squares, so at most 13 steps of movement,
# and a phalanx of at most 14 pieces.
MAX_STEP = max(WIDTH, HEIGHT) - 1
MAX_GROUP = max(WIDTH, HEIGHT)


def _build_rays():
    """Ray tables: for each direction, square and step count k (0..MAX_STEP),
    the index of the square k steps away, and whether it is on the board."""
    idx = np.zeros((NUM_DIRS, N, MAX_STEP + 1), dtype=np.int32)
    ok = np.zeros((NUM_DIRS, N, MAX_STEP + 1), dtype=bool)
    for d, (dr, dc) in enumerate(_DIRS):
        for s in range(N):
            r0, c0 = divmod(s, WIDTH)
            for k in range(MAX_STEP + 1):
                r, c = r0 + dr * k, c0 + dc * k
                if 0 <= r < HEIGHT and 0 <= c < WIDTH:
                    idx[d, s, k] = r * WIDTH + c
                    ok[d, s, k] = True
    return jnp.asarray(idx), jnp.asarray(ok)


RAY_IDX, RAY_OK = _build_rays()
# JAX copies, for indexing with traced values.
_DIRS_J = jnp.asarray(_DIRS)
_OPP_DIR_J = jnp.asarray(_OPP_DIR)
_STEPS = jnp.arange(MAX_STEP + 1, dtype=jnp.int32)  # 0..MAX_STEP


class GameState(NamedTuple):
    color: Array = jnp.int32(0)  # 0 = white to move, 1 = black
    board: Array = jnp.zeros(N, jnp.int8)
    stage: Array = jnp.int32(0)  # 0 = pick lead, 1 = pick rear, 2 = pick destination
    lead: Array = jnp.int32(0)  # chosen in stage 0
    rear: Array = jnp.int32(0)  # chosen in stage 1 (== lead for a single piece)
    winner: Array = jnp.int32(-1)  # -1 = ongoing, else the winning player
    moves: Array = jnp.int32(0)  # completed full moves, for reporting
    quiet_moves: Array = jnp.int32(0)  # moves since the last capture; ends the game
    last_capturer: Array = jnp.int32(-1)  # who captured most recently; -1 = nobody has


class Game:
    def init(self) -> GameState:
        return GameState(board=_init_board())

    def step(self, state: GameState, action: Array) -> GameState:
        return jax.lax.switch(
            state.stage,
            [
                lambda: state._replace(stage=jnp.int32(1), lead=action, rear=action),
                lambda: state._replace(stage=jnp.int32(2), rear=action),
                lambda: _apply_move(state, action),
            ],
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        return _observe(state, color)

    def legal_action_mask(self, state: GameState) -> Array:
        return jax.lax.switch(
            state.stage,
            [
                lambda: _lead_mask(state),
                lambda: _rear_mask(state),
                lambda: _dest_mask(state),
            ],
        )

    def is_terminal(self, state: GameState) -> Array:
        return (state.winner >= 0) | (state.quiet_moves >= MAX_QUIET_MOVES)

    def rewards(self, state: GameState) -> Array:
        return jnp.float32([-1.0, -1.0]).at[_clock_winner(state)].set(1.0)


# ─── Board helpers ───────────────────────────────────────────────────────────


def _clock_winner(state: GameState) -> Array:
    """Who wins if the game ends in this position: 0 white, 1 black.

    Reaching the cap goes to whoever has come further up the board, compared
    rank by rank. If the two are exact mirrors it goes to whoever captured most
    recently, and failing that to black, who compensates for moving second.
    Every game therefore has a winner: there are no draws.

    Shared by `rewards` and by the observation's verdict plane, so the value the
    network is shown is by construction the outcome it will be scored against.
    """
    tiebreak = jax.lax.select(state.last_capturer >= 0, state.last_capturer, jnp.int32(1))
    advantage = _advancement_winner(state.board)
    winner = jax.lax.select(advantage >= 0, advantage, tiebreak)
    return jax.lax.select(state.winner >= 0, state.winner, winner)


def _advancement_winner(board: Array) -> Array:
    """Whoever has come further up the board: 0 white, 1 black, -1 tied.

    Compares piece counts row by row, deepest first: white's count on row
    HEIGHT-1-i against black's count on row i, for i = 0, 1, 2, ... The first
    row that differs decides.

    White advances towards row HEIGHT-1 and black towards row 0, so row pair i
    is "i rows from the opponent's home rank" for both players. The i = 0 pair -
    pieces actually on the opponent's home rank - is level in any position the
    cap can be reached from, because a difference there is the win condition and
    would have ended the game already; so in practice this starts at the
    second-to-home rank and walks back down the board.

    This scores the objective rather than material: the game is won by getting
    up the board, so a player who is further up is the one making progress.
    Ties need the two sides to be mirror images rank for rank, which makes draws
    almost impossible.
    """
    b = board.reshape(HEIGHT, WIDTH)
    # diff[i] = white on row HEIGHT-1-i, minus black on row i.
    diff = (b == WHITE).sum(axis=1)[::-1] - (b == BLACK).sum(axis=1)
    decided = diff != 0
    first = jnp.argmax(decided)  # the deepest rank where the two differ
    return jnp.where(
        ~decided.any(),
        jnp.int32(-1),
        jnp.where(diff[first] > 0, jnp.int32(0), jnp.int32(1)),
    )


def _init_board() -> Array:
    """Each player fills the two rows nearest them: 28 pieces each."""
    board = np.zeros(N, dtype=np.int8)
    board[: 2 * WIDTH] = WHITE
    board[N - 2 * WIDTH :] = BLACK
    return jnp.asarray(board)


def _stone(color: Array) -> Array:
    """Board value of `color`'s pieces (0 -> WHITE, 1 -> BLACK)."""
    return jnp.int8(color + 1)


def _runs(mask: Array, d: int) -> Array:
    """Per square, the number of consecutive True cells along direction `d`,
    counting the square itself (0 if it is False)."""
    vals = mask[RAY_IDX[d]] & RAY_OK[d]  # (N, MAX_STEP + 1)
    return jnp.cumprod(vals, axis=1).sum(axis=1).astype(jnp.int32)


def _all_runs(mask: Array) -> Array:
    """`_runs` for every direction, stacked: (NUM_DIRS, N)."""
    return jnp.stack([_runs(mask, d) for d in range(NUM_DIRS)])


def _line_info(board: Array, color: Array):
    """Per direction and square: how far own/enemy/empty lines run.

    Returns (own_back, enemy_runs, empty_runs), each (NUM_DIRS, N):
      own_back[d, s]   own pieces from s extending back against d (the largest
                       group s can lead in direction d, so >= 1 iff s is ours)
      enemy_runs[d, s] enemy pieces from s continuing along d
      empty_runs[d, s] empty squares from s continuing along d
    """
    own = board == _stone(color)
    enemy = (board != EMPTY) & ~own
    empty = board == EMPTY
    runs = _all_runs(own)
    own_back = jnp.stack([runs[_OPP_DIR[d]] for d in range(NUM_DIRS)])
    return own_back, _all_runs(enemy), _all_runs(empty)


def _move_legality_for_lead(board: Array, color: Array, lead: Array):
    """Legality of every (direction, group size, distance) for one lead square.

    Shape (NUM_DIRS, MAX_GROUP + 1, MAX_STEP + 1), indexed [d, m, k]: moving the
    m pieces ending at `lead` (extending back against d) k squares along d.
    Index 0 of the m/k axes is unused.
    """
    own_back, enemy_runs, empty_runs = _line_info(board, color)
    enemy = (board != EMPTY) & (board != _stone(color))
    empty = board == EMPTY
    dirs = jnp.arange(NUM_DIRS)

    rays = RAY_IDX[:, lead]  # (NUM_DIRS, MAX_STEP + 1) squares along each direction
    on_board = RAY_OK[:, lead]
    max_group = own_back[dirs, lead][:, None, None]

    # Squares strictly between lead and target must be empty: k - 1 of them,
    # starting one step along d (vacuously true for k == 1).
    empties_ahead = jnp.where(on_board[:, 1], empty_runs[dirs, rays[:, 1]], 0)

    m = jnp.arange(MAX_GROUP + 1, dtype=jnp.int32)[None, :, None]
    k = _STEPS[None, None, :]
    path_clear = (k <= 1) | (empties_ahead[:, None, None] >= k - 1)
    can_land = empty[rays][:, None, :] | (
        enemy[rays][:, None, :] & (enemy_runs[dirs[:, None], rays][:, None, :] < m)
    )
    return (
        (m >= 1) & (k >= 1) & (k <= m) & (m <= max_group)
        & on_board[:, None, :] & path_clear & can_land
    )


def _group_of(lead: Array, rear: Array):
    """Direction index and size of the group from `rear` to `lead`.

    For rear == lead (a single piece) the direction is undefined and returned as
    0 with size 1; callers handle that case separately.
    """
    lr, lc = lead // WIDTH, lead % WIDTH
    rr, rc = rear // WIDTH, rear % WIDTH
    dr, dc = jnp.sign(lr - rr), jnp.sign(lc - rc)
    size = jnp.maximum(jnp.abs(lr - rr), jnp.abs(lc - rc)) + 1
    d = jnp.argmax((_DIRS_J[:, 0] == dr) & (_DIRS_J[:, 1] == dc)).astype(jnp.int32)
    return d, size.astype(jnp.int32)


def _is_mirror_symmetric(board: Array) -> Array:
    b = board.reshape(HEIGHT, WIDTH)
    return jnp.all(b == b[:, ::-1])


def _back_rank_rows():
    return jnp.int32([0, HEIGHT - 1])


def _apply_move(state: GameState, dest: Array) -> GameState:
    """Play the move (lead, rear, dest) and hand the turn to the opponent."""
    board = _do_move(state.board, state.color, state.lead, state.rear, dest)
    color = 1 - state.color
    moves = state.moves + 1
    # A move never adds a piece, and a capture always removes at least one, so
    # the piece count alone says whether this move captured.
    captured = (board != EMPTY).sum() < (state.board != EMPTY).sum()
    last_capturer = jnp.where(captured, state.color, state.last_capturer)
    # The clock runs on quiet moves only, and a capture puts it back to zero.
    quiet_moves = jnp.where(captured, jnp.int32(0), state.quiet_moves + 1)
    # The player about to move wins if they have more pieces on the opponent's
    # back rank than the opponent has on theirs.
    winner = jax.lax.select(_wins_at_turn_start(board, color), color, jnp.int32(-1))
    # A player with no legal move loses (the rules forbid passing; this cannot
    # normally arise).
    stuck = ~_lead_mask(GameState(color=color, board=board, stage=jnp.int32(0))).any()
    winner = jax.lax.select((winner < 0) & stuck, 1 - color, winner)
    return GameState(
        color=color, board=board, stage=jnp.int32(0), lead=jnp.int32(0), rear=jnp.int32(0),
        winner=winner, moves=moves, quiet_moves=quiet_moves, last_capturer=last_capturer,
    )


def _do_move(board: Array, color: Array, lead: Array, rear: Array, dest: Array) -> Array:
    """Board after moving the group `rear`..`lead` so that `lead` lands on `dest`."""
    single = lead == rear
    d_group, size = _group_of(lead, rear)
    d_single, _ = _group_of(dest, lead)  # direction of a single-piece step
    d = jax.lax.select(single, d_single, d_group)
    m = jax.lax.select(single, jnp.int32(1), size)

    lr, lc = lead // WIDTH, lead % WIDTH
    tr, tc = dest // WIDTH, dest % WIDTH
    k = jnp.maximum(jnp.abs(tr - lr), jnp.abs(tc - lc)).astype(jnp.int32)

    back = _OPP_DIR_J[d]
    offsets = _STEPS  # 0..MAX_STEP
    in_group = offsets < m
    # Squares vacated: lead, lead - d, ... (m of them, against d).
    src_cells = RAY_IDX[back, lead]
    # Squares occupied after the move: the same group shifted k steps along d.
    dst_cells = RAY_IDX[back, dest]

    src_mask = jnp.zeros(N, bool).at[src_cells].max(in_group)
    dst_mask = jnp.zeros(N, bool).at[dst_cells].max(in_group)

    # Captured: the enemy line from `dest` continuing along d.
    enemy = (board != EMPTY) & (board != _stone(color))
    cap_len = jax.lax.select(enemy[dest], _all_runs(enemy)[d, dest], jnp.int32(0))
    cap_cells = RAY_IDX[d, dest]
    cap_mask = jnp.zeros(N, bool).at[cap_cells].max(offsets < cap_len)

    board = jnp.where(src_mask | cap_mask, jnp.int8(EMPTY), board)
    return jnp.where(dst_mask, _stone(color), board)


def _wins_at_turn_start(board: Array, color: Array) -> Array:
    """True if `color`, about to move, has more pieces on the opponent's back
    rank than the opponent has on `color`'s back rank."""
    b = board.reshape(HEIGHT, WIDTH)
    own_home, opp_home = jax.lax.select(color == 0, _back_rank_rows(), jnp.flip(_back_rank_rows()))
    mine_there = (b[opp_home] == _stone(color)).sum()
    theirs_here = (b[own_home] == _stone(1 - color)).sum()
    return mine_there > theirs_here


# ─── Legal action masks ──────────────────────────────────────────────────────


def _lead_mask(state: GameState) -> Array:
    """Squares that can lead a move: an own piece with at least one legal move.

    Computed without enumerating group sizes and distances: for each direction a
    lead can move quietly if there is any empty square ahead, and can capture the
    first piece ahead if that piece is an enemy whose line is shorter than the
    group and lies within reach.
    """
    own_back, enemy_runs, empty_runs = _line_info(state.board, state.color)
    enemy = (state.board != EMPTY) & (state.board != _stone(state.color))
    dirs = jnp.arange(NUM_DIRS)[:, None]

    group = own_back  # (NUM_DIRS, N): pieces this square can lead in direction d
    first, first_ok = RAY_IDX[:, :, 1], RAY_OK[:, :, 1]
    empties_ahead = jnp.where(first_ok, empty_runs[dirs, first], 0)

    quiet = (group >= 1) & (empties_ahead >= 1)

    # The first piece along d sits just past the empty squares.
    steps = jnp.clip(empties_ahead + 1, 0, MAX_STEP)[:, :, None]
    blocker = jnp.take_along_axis(RAY_IDX, steps, axis=2)[:, :, 0]
    blocker_ok = jnp.take_along_axis(RAY_OK, steps, axis=2)[:, :, 0]
    capture = (
        blocker_ok & enemy[blocker]
        & (empties_ahead + 1 <= group)
        & (enemy_runs[dirs, blocker] < group)
    )
    return (quiet | capture).any(axis=0)


def _rear_mask(state: GameState) -> Array:
    """Rear squares for the chosen lead: the lead itself (single piece) or the
    rear of a contiguous own group that has a legal move."""
    lead = state.lead
    legal = _move_legality_for_lead(state.board, state.color, lead)  # (d, m, k)
    per_dir = legal.any(axis=-1)  # (d, m): some distance is legal

    # Rear square for direction d and group size m is lead - (m - 1) * d.
    back = _OPP_DIR
    rear_idx = jnp.stack([RAY_IDX[back[d], lead] for d in range(NUM_DIRS)])  # (d, MAX_STEP + 1)
    # A group of size m has its rear at offset m - 1 from the lead.
    sizes = _STEPS + 1  # group size for each offset
    ok = jnp.stack([per_dir[d, sizes] & (sizes >= 2) for d in range(NUM_DIRS)])
    # Group of size m occupies offsets 0..m-1, so its rear is at offset m - 1.
    mask = jnp.zeros(N, bool)
    for d in range(NUM_DIRS):
        mask = mask.at[rear_idx[d]].max(ok[d] & RAY_OK[back[d], lead])
    # A single piece (rear == lead) is legal when the piece can move at all.
    single_ok = legal[:, 1, 1].any()
    return mask.at[lead].max(single_ok)


def _dest_mask(state: GameState) -> Array:
    """Destinations for the chosen (lead, rear) group."""
    lead, rear = state.lead, state.rear
    legal = _move_legality_for_lead(state.board, state.color, lead)  # (d, m, k)
    single = lead == rear
    d_group, size = _group_of(lead, rear)
    m = jax.lax.select(single, jnp.int32(1), size)

    def for_dir(d):
        ok = legal[d, m]  # (k,)
        return jnp.zeros(N, bool).at[RAY_IDX[d, lead]].max(ok & RAY_OK[d, lead])

    all_dirs = jnp.stack([for_dir(d) for d in range(NUM_DIRS)])
    mask = jax.lax.select(single, all_dirs.any(axis=0), all_dirs[d_group])
    return mask & _symmetry_allowed(state, mask)


def _symmetry_allowed(state: GameState, candidates: Array) -> Array:
    """Forbid destinations that put a piece on the opponent's back rank and make
    the whole position left-to-right symmetric.

    Only the lead piece can reach further forward than the rest of its group, so
    a move puts a piece on that rank exactly when the lead lands there. That
    leaves WIDTH candidate destinations to simulate instead of all N.
    """
    opp_back_row = jax.lax.select(state.color == 0, jnp.int32(HEIGHT - 1), jnp.int32(0))
    row_cells = opp_back_row * WIDTH + jnp.arange(WIDTH, dtype=jnp.int32)

    def symmetric_after(dest):
        board = _do_move(state.board, state.color, state.lead, state.rear, dest)
        return _is_mirror_symmetric(board)

    # Simulate only the destinations that could trigger the rule, and only those
    # that are actually on offer.
    forbidden = jax.vmap(symmetric_after)(row_cells) & candidates[row_cells]
    return jnp.ones(N, bool).at[row_cells].set(~forbidden)


# ─── Observation ─────────────────────────────────────────────────────────────

_DIST_J = jnp.sqrt(
    ((jnp.arange(N) // WIDTH) - (HEIGHT - 1) / 2.0) ** 2
    + ((jnp.arange(N) % WIDTH) - (WIDTH - 1) / 2.0) ** 2
).astype(jnp.float32)

# Roughly the largest magnitude each scalar feature can reach, used to divide
# them into about [-1, 1] before they become observation planes.
#
# The alpha-beta engine these terms come from multiplies them by weights of 0.5
# to 250 and sums the result, so their raw scales never matter there. Here they
# are network *inputs*, and `BoardFormer` applies its stem convolution to the
# observation with no normalisation in front of it. Unscaled, these arrive far
# larger than the 0/1 piece planes and dominate the stem at initialisation.
#
# Structural bounds, not measured ones: 28 pieces a side, 14 columns, 8
# neighbours per square. They only have to be the right order of magnitude.
_FEATURE_SCALE_J = jnp.float32([
    14.0,        # home_row_defense: a full home rank
    28.0 * 8.0,  # territory: 8 neighbours per piece
    8.0,         # centrality: half the board diagonal
    1.0,         # clock: already a fraction in [0, 1]
])

# Travel distances are at most the length of the longest line.
_TRAVEL_SCALE = float(MAX_STEP)

# Flipping the board vertically for black maps direction (dr, dc) to (-dr, dc).
# Direction-indexed planes must therefore have their channels permuted as well
# as their rows flipped, or black reads "towards row 0" data out of the channel
# white uses for "towards row 11" - silently, with nothing to crash on. The
# permutation is its own inverse.
_FLIP_DIR_PERM = jnp.asarray(
    [int(np.flatnonzero((_DIRS[:, 0] == -dr) & (_DIRS[:, 1] == dc))[0]) for dr, dc in _DIRS]
)

# The same thing for a left-right mirror, which maps (dr, dc) to (dr, -dc). The
# game is exactly invariant under it - the board, the starting position and the
# eight move directions are all symmetric about the vertical axis - so it is a
# valid training-time augmentation, and `examples/alphazero/symmetry.py` uses
# this to permute the travel channels when it mirrors a sample. Also its own
# inverse.
MIRROR_DIR_PERM = jnp.asarray(
    [int(np.flatnonzero((_DIRS[:, 0] == dr) & (_DIRS[:, 1] == -dc))[0]) for dr, dc in _DIRS]
)


def _longest_run_jax(row_bool: Array) -> Array:
    """Longest contiguous run of True values in a 1D boolean array."""
    def step(acc, val):
        new_acc = jnp.where(val, acc + jnp.int32(1), jnp.int32(0))
        return new_acc, new_acc
    _, runs = jax.lax.scan(step, jnp.int32(0), row_bool)
    return runs.max().astype(jnp.float32)


def _territory_jax(board: Array, mine: Array, theirs: Array) -> Array:
    """Per piece, adjacent squares on the board not occupied by enemy."""
    nb = RAY_IDX[:, :, 1]  # (NUM_DIRS, N)
    on_board = RAY_OK[:, :, 1]  # (NUM_DIRS, N)
    not_enemy = (board[nb] != theirs) & on_board  # (NUM_DIRS, N)
    is_mine = board == mine  # (N,)
    return (not_enemy & is_mine[None, :]).sum().astype(jnp.float32)


def _centrality_side(board: Array, stone: Array) -> Array:
    """Mean Euclidean distance of `stone` pieces from the center of the board."""
    mask = board == stone
    n = mask.sum()
    total = (mask * _DIST_J).sum()
    return jnp.where(n > 0, total / n.astype(jnp.float32), jnp.float32(0.0))


def _travel(board: Array, color: Array) -> Array:
    """(NUM_DIRS, N): how far the group led by each square can move each way.

    `own_back[d, s]` is the largest group `s` can lead along `d`, and a group
    may move at most its own length, so the distance is that capped by the empty
    squares ahead. Zero where `s` is not `color`'s piece.

    This is the per-square quantity the old scalar `mobility` reduced to a mean
    and a max before broadcasting the result over all 168 cells. Keeping it
    per-square is what lets the network see phalanx length and room to move
    along each vertical and diagonal, which is where this game's structure is.
    The run tables it needs are already built for `legal_action_mask` in the
    same step, so the scans are shared.
    """
    own_back, _, empty_runs = _line_info(board, color)  # (NUM_DIRS, N)
    dirs = jnp.arange(NUM_DIRS)[:, None]
    ahead = jnp.where(RAY_OK[:, :, 1], empty_runs[dirs, RAY_IDX[:, :, 1]], 0)
    is_own = board == _stone(color)
    return jnp.where(is_own[None, :], jnp.minimum(own_back, ahead), 0)


def _eval_features(state: GameState, color: Array) -> Array:
    """Four scalar features from `color`'s perspective:
    0 home_row_defense, 1 territory, 2 centrality, 3 clock.

    These are what remains of the nine terms of tdgauntlet's alpha-beta
    evaluator (`clients/alphabeta/src/eval.rs`) once the ones a network gets for
    free are dropped. `material`, `crossing`, `advancement` and `tiebreak` are
    all exact *linear functionals* of the two piece planes - sums of them, or
    row-weighted sums - so a single pooling layer computes any of them and
    handing them over buys nothing but bandwidth. `mobility` is now the
    per-square `_travel` planes instead of a mean and a max.

    What is left is not linear in the planes: `home_row_defense` is a longest
    contiguous run, `territory` a product of two masks, `centrality` a ratio,
    and `clock` is not a function of the board at all - `quiet_moves` appears
    nowhere else in the observation, which makes it the one feature here the
    network could not otherwise obtain.

    The first three are "ours minus theirs" and so antisymmetric under a colour
    flip; `clock` is colour-blind. Either way a mirrored position reads
    identically to both sides, which is what keeps this from repeating the
    v7/v8 colour leak - that plane leaked because the tiebreak's black-wins-a-
    mirror default is *not* antisymmetric.

    Divided by `_FEATURE_SCALE_J` on the way out; see the note there.
    """
    b = state.board
    b_2d = b.reshape(HEIGHT, WIDTH)
    own_st = _stone(color)
    opp_st = _stone(1 - color)
    own_home = jax.lax.select(color == 0, 0, HEIGHT - 1)
    opp_home = jax.lax.select(color == 0, HEIGHT - 1, 0)

    home_row_defense = _longest_run_jax(b_2d[own_home] == own_st) - _longest_run_jax(
        b_2d[opp_home] == opp_st
    )
    territory = _territory_jax(b, own_st, opp_st) - _territory_jax(b, opp_st, own_st)
    centrality = _centrality_side(b, opp_st) - _centrality_side(b, own_st)
    clock = jnp.minimum(state.quiet_moves, MAX_QUIET_MOVES).astype(jnp.float32) / float(
        MAX_QUIET_MOVES
    )

    return jnp.stack([home_row_defense, territory, centrality, clock]) / _FEATURE_SCALE_J


def _observe(state: GameState, color: Array) -> Array:
    """(HEIGHT, WIDTH, 19) float32 from `color`'s perspective.

    Rows are flipped for black so that the player to move always looks "up" the
    board: own back rank is row 0.

    Channels:
      0 own pieces, 1 opponent pieces
      2 lead marker, 3 rear marker
      4-6 constant planes one-hot encoding the stage
      7-10 scalar features, broadcast: home_row_defense, territory, centrality,
        clock (see `_eval_features`)
      11-18 `_travel`, one plane per direction: how far the group led by this
        square could move that way

    The travel planes are the point of this observation. A scalar broadcast over
    168 cells spends a whole plane carrying one number; travel varies per square
    and tells the network, at every piece, the phalanx it heads and the room in
    front of it along each vertical, rank and diagonal - which is where this
    game's tactics live.
    """
    b = state.board.reshape(HEIGHT, WIDTH)
    own = (b == _stone(color)).astype(jnp.float32)
    opp = (b == _stone(1 - color)).astype(jnp.float32)

    def marker(idx, active):
        return (jnp.zeros(N, jnp.float32).at[idx].set(active.astype(jnp.float32))).reshape(HEIGHT, WIDTH)

    lead = marker(state.lead, state.stage >= 1)
    rear = marker(state.rear, state.stage >= 2)
    stage_planes = [jnp.full((HEIGHT, WIDTH), (state.stage == i).astype(jnp.float32)) for i in range(3)]

    features = _eval_features(state, color)
    feature_planes = [jnp.full((HEIGHT, WIDTH), features[i]) for i in range(features.shape[0])]

    # (NUM_DIRS, N) -> (HEIGHT, WIDTH, NUM_DIRS)
    travel = _travel(state.board, color).astype(jnp.float32) / _TRAVEL_SCALE
    travel = travel.reshape(NUM_DIRS, HEIGHT, WIDTH).transpose(1, 2, 0)

    planes = jnp.concatenate(
        [jnp.stack([own, opp, lead, rear, *stage_planes, *feature_planes], axis=-1), travel],
        axis=-1,
    )
    # For black the rows are mirrored, which also mirrors what each direction
    # means, so the travel channels are permuted to match. Without that, black
    # reads "towards row 0" out of the channel white uses for "towards row 11".
    flipped = jnp.flip(planes, axis=0)
    flipped = jnp.concatenate(
        [flipped[..., :-NUM_DIRS], flipped[..., -NUM_DIRS:][..., _FLIP_DIR_PERM]], axis=-1
    )
    return jax.lax.select(color == 0, planes, flipped)

