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

"""Gess with one action per whole move: env id ``gess_joint``.

Same rules and game state as ``gess`` (pgx/_src/games/gess.py), but a move is a
single action instead of two (piece, then destination). Every position carries
a list of its legal moves, ``_moves``: entry i is ``source * 400 + destination``
(cells of the 20x20 grid), sorted ascending, padded with -1; action i plays
entry i. Strong-play positions have a median of ~350 legal moves and at most
528 in 9,433 sampled, so the list holds MAX_MOVES = 768; a position with more
would drop the moves past the cap (``_overflow`` records that it happened).

The observation is ``gess``'s v2 observation at stage 0, so its source-footprint
and stage planes are always zero and a ``gess`` network reads it unchanged.
"""

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.games.gess import Game, GameState, N, _legal_dest_mask, _legal_source_mask
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

MAX_MOVES = 768
_CELLS = jnp.arange(N, dtype=jnp.int32)
_SIDE = 20  # the 20x20 grid of cells (pgx/_src/games/gess.py: BOARD_SIZE)
_DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_STEPS = range(1, _SIDE)

# A centre's candidate moves, one per (direction, step), in ascending order of
# the destination's flat offset from the source. The order does not depend on
# the source, so a source-major (N, len(_CANDIDATES)) table of these comes out
# sorted by source * N + destination. Two candidates may share an offset (e.g.
# 19 steps east and one step south-west), but never both lie on the board for
# the same source, so at most one of them is ever a legal move.
_CANDIDATES = sorted(((dr * _SIDE + dc) * k, d, k) for d, (dr, dc) in enumerate(_DIRS) for k in _STEPS)
_CANDIDATE_OFFSETS = jnp.int32([o for o, _, _ in _CANDIDATES])


def _padded(a: Array) -> Array:
    """`a` (20x20) padded with False by _SIDE on every side, for _shift."""
    return jnp.pad(a, _SIDE, constant_values=False)


def _shift(pad: Array, dr: int, dc: int) -> Array:
    """b[r, c] = a[r + dr, c + dc], False off the grid, from pad = _padded(a)."""
    return pad[_SIDE + dr : 2 * _SIDE + dr, _SIDE + dc : 2 * _SIDE + dc]


def legal_moves(x: GameState) -> tuple[Array, Array]:
    """(moves (MAX_MOVES,) int32, overflow bool) for a stage-0 game state: the
    legal moves as source * N + destination, ascending, padded with -1.

    The same rules as gess's _legal_dest_mask, for all 400 centres at once as
    shifted 20x20 planes rather than by per-centre gathers: a direction is open
    if an own stone sits there in the piece; step k is reachable if no earlier
    step's destination block holds a stone outside the source's own block, and
    k <= 3 unless an own stone is at the centre. Beyond two steps the two blocks
    cannot overlap, so "a stone in the destination block" is a shifted 3x3
    dilation of the occupancy. legal_moves_reference is the per-centre version;
    tests check that the two agree.
    """
    board = x.board.reshape(_SIDE, _SIDE)
    own = board == (x.color + 1).astype(board.dtype)
    occ = board != 0
    src = _legal_source_mask(x).reshape(_SIDE, _SIDE)
    far = src & own  # an own stone at the centre: any distance

    own_p, occ_p = _padded(own), _padded(occ)
    on_grid_p = _padded(jnp.ones((_SIDE, _SIDE), jnp.bool_))
    occ3 = jnp.zeros((_SIDE, _SIDE), jnp.bool_)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            occ3 = occ3 | _shift(occ_p, a, b)
    occ3_p = _padded(occ3)

    valid = {}
    for d, (dr, dc) in enumerate(_DIRS):
        open_ = src & _shift(own_p, dr, dc)
        blocked_before = jnp.zeros((_SIDE, _SIDE), jnp.bool_)
        for k in _STEPS:
            on_grid = _shift(on_grid_p, k * dr, k * dc)
            ok = open_ & ~blocked_before & on_grid
            valid[(d, k)] = ok if k <= 3 else ok & far
            if k > 2:
                hit = _shift(occ3_p, k * dr, k * dc)
            else:
                hit = jnp.zeros((_SIDE, _SIDE), jnp.bool_)
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        r, c = k * dr + a, k * dc + b
                        if abs(r) > 1 or abs(c) > 1:  # outside the source's block
                            hit = hit | _shift(occ_p, r, c)
            blocked_before = blocked_before | hit

    table = jnp.stack([valid[(d, k)].reshape(-1) for _, d, k in _CANDIDATES], axis=1)  # (N, C)
    flat = table.reshape(-1)
    (idx,) = jnp.nonzero(flat, size=MAX_MOVES, fill_value=-1)
    c = len(_CANDIDATES)
    source = idx // c
    moves = source * N + source + _CANDIDATE_OFFSETS[idx % c]
    return jnp.where(idx >= 0, moves, -1).astype(jnp.int32), flat.sum() > MAX_MOVES


def legal_moves_reference(x: GameState) -> tuple[Array, Array]:
    """legal_moves by gess's per-centre destination masks (the original version)."""
    src = _legal_source_mask(x)
    table = jax.vmap(lambda s: _legal_dest_mask(x, s))(_CELLS) & src[:, None]  # (N, N)
    flat = table.reshape(-1)
    (idx,) = jnp.nonzero(flat, size=MAX_MOVES, fill_value=-1)
    return idx.astype(jnp.int32), flat.sum() > MAX_MOVES


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((18, 18, 6), dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.zeros(MAX_MOVES, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()
    # The legal moves (see module docstring) and whether any were cut off.
    _moves: Array = jnp.full(MAX_MOVES, -1, dtype=jnp.int32)
    _overflow: Array = jnp.bool_(False)

    @property
    def env_id(self) -> core.EnvId:
        return "gess_joint"


class GessJoint(core.Env):
    """Gess, one action per move (see the module docstring)."""

    def __init__(self):
        super().__init__()
        self._game = Game()

    def _init(self, key: PRNGKey) -> State:
        del key  # fixed starting position
        x = self._game.init()
        moves, overflow = legal_moves(x)
        return State(  # type: ignore
            current_player=jnp.int32(0),
            _x=x,
            _moves=moves,
            _overflow=overflow,
            legal_action_mask=moves >= 0,
        )

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        move = state._moves[action]
        x = self._game.step(state._x, move // N)  # choose the piece
        x = self._game.step(x, move % N)  # and its destination
        terminated = self._game.is_terminal(x)
        rewards = jax.lax.select(terminated, self._game.rewards(x), jnp.zeros(2, jnp.float32))
        moves, overflow = legal_moves(x)
        return state.replace(  # type: ignore
            current_player=jnp.int32(1 - state.current_player),
            _x=x,
            terminated=terminated,
            rewards=rewards,
            _moves=moves,
            _overflow=overflow,
            legal_action_mask=moves >= 0,
        )

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return self._game.observe(state._x, jnp.int8(player_id))

    @property
    def id(self) -> core.EnvId:
        return "gess_joint"

    @property
    def version(self) -> str:
        # v1: gess v1's rules with gess v2's observation, one action per move.
        return "v1"

    @property
    def num_players(self) -> int:
        return 2
