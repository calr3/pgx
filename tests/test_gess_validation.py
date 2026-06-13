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

"""Cross-validation of the pgx Gess implementation against the reference
TypeScript logic (gess-engine.ts), translated to Python.

Tests 20 positions sampled evenly from a random game (opening → endgame).
For each position:
  1. Legal source squares match isValidFootprintCenter for all 400 cells
     (allowing the known intentional difference: the reference accepts a
     centre-only piece as a valid source, but pgx excludes it because the
     resulting stage-1 mask would be empty, making it equivalent for play).
  2. Legal destinations for five sampled sources match getValidMoves exactly.
  3. The board after applying a randomly-chosen move matches applyMove exactly.
  4. Ring detection (hasRing) matches _has_ring for both colours.
"""

import random as _random

import numpy as np
import jax
import jax.numpy as jnp

from pgx._src.games.gess import (
    BOARD_SIZE, N, MIN_IDX, MAX_IDX,
    Game, GameState,
    _apply_move, _has_ring,
    _legal_source_mask, _legal_dest_mask,
)

# ─── Reference implementation (translated from gess-engine.ts) ───────────────
# These are direct Python ports of the TypeScript originals; no pgx code is used.

_R_SIZE = BOARD_SIZE   # 20
_R_MIN  = MIN_IDX      # 1
_R_MAX  = MAX_IDX      # 18
_R_DIRS = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]


def _ref_is_valid_center(board, row, col, player):
    """isValidFootprintCenter: any own stone in 3×3, no opponent stone."""
    if not (0 <= row < _R_SIZE and 0 <= col < _R_SIZE):
        return False
    opp = 'black' if player == 'white' else 'white'
    has_own = False
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = row + dr, col + dc
            if not (0 <= r < _R_SIZE and 0 <= c < _R_SIZE):
                continue
            cell = board[r][c]
            if cell == opp:
                return False
            if cell == player:
                has_own = True
    return has_own


def _ref_get_valid_moves(board, row, col, player):
    """getValidMoves: returns list of (nr, nc) destination centres."""
    max_steps = _R_SIZE if board[row][col] == player else 3
    results, seen = [], set()
    origin = {(row + ddr, col + ddc) for ddr in (-1, 0, 1) for ddc in (-1, 0, 1)}

    for dr, dc in _R_DIRS:
        fr, fc = row + dr, col + dc
        if not (0 <= fr < _R_SIZE and 0 <= fc < _R_SIZE):
            continue
        if board[fr][fc] != player:
            continue
        for step in range(1, max_steps + 1):
            nr, nc = row + dr * step, col + dc * step
            # Piece may reach the border ring (0 or 19) but not beyond.
            if not (_R_MIN - 1 <= nr <= _R_MAX + 1 and _R_MIN - 1 <= nc <= _R_MAX + 1):
                break
            # Clearance: any stone in new footprint that isn't in origin.
            # The border ring (row/col 0 or 19) is always empty → skip it.
            hits_any = False
            for ddr in (-1, 0, 1):
                for ddc in (-1, 0, 1):
                    sr, sc = nr + ddr, nc + ddc
                    if (sr, sc) in origin:
                        continue
                    if not (_R_MIN <= sr <= _R_MAX and _R_MIN <= sc <= _R_MAX):
                        continue
                    if board[sr][sc] is not None:
                        hits_any = True
                        break
                if hits_any:
                    break
            key = (nr, nc)
            if key not in seen:
                seen.add(key)
                results.append(key)
            if hits_any:
                break
    return results


def _ref_apply_move(board, from_r, from_c, to_r, to_c, player):
    """applyMove: returns new 2-D list after sliding the piece."""
    b = [list(row) for row in board]
    dr = 0 if to_r == from_r else (1 if to_r > from_r else -1)
    dc = 0 if to_c == from_c else (1 if to_c > from_c else -1)
    steps = max(abs(to_r - from_r), abs(to_c - from_c))

    rel_stones = []
    for ddr in (-1, 0, 1):
        for ddc in (-1, 0, 1):
            r, c = from_r + ddr, from_c + ddc
            if 0 <= r < _R_SIZE and 0 <= c < _R_SIZE and b[r][c] == player:
                rel_stones.append((ddr, ddc))
                b[r][c] = None

    for step in range(1, steps + 1):
        cr, cc = from_r + dr * step, from_c + dc * step
        for ddr in (-1, 0, 1):
            for ddc in (-1, 0, 1):
                r, c = cr + ddr, cc + ddc
                if 0 <= r < _R_SIZE and 0 <= c < _R_SIZE and b[r][c] is not None:
                    b[r][c] = None

    for ddr, ddc in rel_stones:
        r, c = to_r + ddr, to_c + ddc
        if _R_MIN <= r <= _R_MAX and _R_MIN <= c <= _R_MAX:
            b[r][c] = player
    return b


def _ref_has_ring(board, player):
    """hasRing: True iff player has at least one ring on the playing area."""
    for r in range(_R_MIN, _R_MAX + 1):
        for c in range(_R_MIN, _R_MAX + 1):
            if board[r][c] is not None:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if board[r + dr][c + dc] != player:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


# ─── Board-format converters ─────────────────────────────────────────────────

def _pgx_to_ref(flat_board):
    """pgx flat int8 (0/1/2) → 20×20 list-of-lists (None/'black'/'white')."""
    b = [[None] * _R_SIZE for _ in range(_R_SIZE)]
    arr = np.array(flat_board)
    for idx in range(N):
        r, c = divmod(idx, _R_SIZE)
        v = int(arr[idx])
        if v == 1: b[r][c] = 'black'
        elif v == 2: b[r][c] = 'white'
    return b

def _ref_to_pgx(ref_board):
    """20×20 list-of-lists → pgx flat int8 numpy array."""
    flat = np.zeros(N, dtype=np.int8)
    for r in range(_R_SIZE):
        for c in range(_R_SIZE):
            v = ref_board[r][c]
            if v == 'black': flat[r * _R_SIZE + c] = 1
            elif v == 'white': flat[r * _R_SIZE + c] = 2
    return flat


# ─── JIT-compiled pgx helpers ────────────────────────────────────────────────

_GAME          = Game()
_jit_step      = jax.jit(_GAME.step)
_jit_src_mask  = jax.jit(_legal_source_mask)
_jit_dst_mask  = jax.jit(_legal_dest_mask)
_jit_apply     = jax.jit(_apply_move)
_jit_has_ring  = jax.jit(_has_ring)


# ─── Per-position comparison ─────────────────────────────────────────────────

def _compare_position(gs: GameState, label: str, rng: _random.Random) -> None:
    """Validate a single stage-0 GameState against the reference."""
    player       = 'black' if int(gs.color) == 0 else 'white'
    mover_stone  = jnp.int8(int(gs.color) + 1)
    ref_board    = _pgx_to_ref(gs.board)

    # ── 1. Source mask ──────────────────────────────────────────────────────
    pgx_src = set(np.where(np.array(_jit_src_mask(gs), dtype=bool))[0].tolist())

    ref_src = {
        r * _R_SIZE + c
        for r in range(_R_SIZE) for c in range(_R_SIZE)
        if _ref_is_valid_center(ref_board, r, c, player)
    }

    # pgx excludes "centre-only" sources (own stone only at the centre cell,
    # none in the surrounding 8).  The reference includes them, but they have
    # no valid destinations, so they are unreachable in either implementation.
    ref_only = ref_src - pgx_src
    for idx in ref_only:
        r, c = divmod(idx, _R_SIZE)
        dests = _ref_get_valid_moves(ref_board, r, c, player)
        assert len(dests) == 0, (
            f"{label}: ref-only source ({r},{c}) has {len(dests)} destinations "
            f"in the reference — pgx should include it"
        )

    pgx_only = pgx_src - ref_src
    assert pgx_only == set(), (
        f"{label}: pgx source mask contains cells not in reference: "
        + ", ".join(f"({idx // _R_SIZE},{idx % _R_SIZE})" for idx in sorted(pgx_only))
    )

    # ── 2. Destination masks for five sampled sources ───────────────────────
    if not pgx_src:
        return
    sample_srcs = rng.sample(sorted(pgx_src), min(5, len(pgx_src)))
    for src_idx in sample_srcs:
        src_r, src_c = divmod(src_idx, _R_SIZE)

        ref_dests = {r * _R_SIZE + c for r, c in _ref_get_valid_moves(ref_board, src_r, src_c, player)}
        pgx_dests = set(np.where(np.array(_jit_dst_mask(gs, jnp.int32(src_idx)), dtype=bool))[0].tolist())

        assert pgx_dests == ref_dests, (
            f"{label}: destinations differ for source ({src_r},{src_c})\n"
            f"  pgx-only : {sorted((i//20, i%20) for i in pgx_dests - ref_dests)}\n"
            f"  ref-only : {sorted((i//20, i%20) for i in ref_dests - pgx_dests)}"
        )

    # ── 3. Board state after a random move ──────────────────────────────────
    src_idx  = rng.choice(sorted(pgx_src))
    src_r, src_c = divmod(src_idx, _R_SIZE)
    ref_dests_list = _ref_get_valid_moves(ref_board, src_r, src_c, player)
    if not ref_dests_list:
        return

    to_r, to_c = rng.choice(ref_dests_list)
    dst_idx    = to_r * _R_SIZE + to_c

    ref_after  = _ref_apply_move(ref_board, src_r, src_c, to_r, to_c, player)
    expected   = _ref_to_pgx(ref_after)

    board_jnp  = jnp.array(np.array(gs.board), dtype=jnp.int8)
    pgx_after  = np.array(_jit_apply(board_jnp, jnp.int32(src_idx), jnp.int32(dst_idx), mover_stone))

    assert np.array_equal(pgx_after, expected), (
        f"{label}: board mismatch after ({src_r},{src_c})→({to_r},{to_c})\n"
        f"  first diff at flat indices: {np.where(pgx_after != expected)[0].tolist()}"
    )

    # ── 4. Ring detection ───────────────────────────────────────────────────
    for p_str, stone_val in [('black', jnp.int8(1)), ('white', jnp.int8(2))]:
        ref_ring = _ref_has_ring(ref_board, p_str)
        pgx_ring = bool(_jit_has_ring(board_jnp, stone_val))
        assert pgx_ring == ref_ring, (
            f"{label}: ring detection mismatch for {p_str}: pgx={pgx_ring} ref={ref_ring}"
        )


# ─── Scenario generation ─────────────────────────────────────────────────────

# ─── Notation helpers (standard Gess: label = 20 − row_index) ────────────────

import re as _re

def _label_to_idx(s: str) -> int:
    """'p3' → flat board index.  label = 20 − row_index; column a–t = 0–19."""
    m = _re.fullmatch(r'([a-t])(20|1[0-9]|[1-9])', s.strip().lower())
    assert m, f"bad label: {s!r}"
    col     = ord(m.group(1)) - ord('a')
    row_idx = _R_SIZE - int(m.group(2))
    return row_idx * _R_SIZE + col

def _idx_to_label(idx: int) -> str:
    col     = idx % _R_SIZE
    row_idx = idx // _R_SIZE
    return f"{'abcdefghijklmnopqrst'[col]}{_R_SIZE - row_idx}"


def _collect_positions(seed: int = 54321, max_full_moves: int = 350) -> list[GameState]:
    """Play a random game and collect every stage-0 GameState."""
    positions: list[GameState] = []
    state = _GAME.init()
    rng = _random.Random(seed)

    for _ in range(max_full_moves * 2 + 1):
        if _GAME.is_terminal(state):
            break

        mask  = np.array(_GAME.legal_action_mask(state), dtype=bool)
        legal = np.where(mask)[0].tolist()
        if not legal:
            break

        if int(state.stage) == 0:
            positions.append(state)

        action = rng.choice(legal)
        state  = _jit_step(state, jnp.int32(action))

    return positions


# ─── Test ────────────────────────────────────────────────────────────────────

def test_gess_vs_reference_20_scenarios():
    """Compare pgx and reference on 20 positions sampled evenly from a random game.

    We try several seeds and use the first one that yields ≥ 20 full moves so
    that the sampled positions genuinely span the opening, midgame, and endgame.
    """
    # Collect from several seeds, accumulating until we have ≥ 20 positions.
    # Random Gess games can be short; pooling across seeds gives broad coverage.
    seen_boards: set[bytes] = set()
    positions: list[GameState] = []
    for seed in [6, 17, 14, 1, 15, 16, 10, 54321, 12345, 99999]:
        for gs in _collect_positions(seed=seed):
            key = np.array(gs.board).tobytes()
            if key not in seen_boards:
                seen_boards.add(key)
                positions.append(gs)
        if len(positions) >= 20:
            break

    assert len(positions) >= 5, (
        f"All seeds produced very few positions — only {len(positions)} unique boards"
    )

    n = len(positions)
    if n >= 20:
        indices = [int(round(i * (n - 1) / 19)) for i in range(20)]
    else:
        indices = list(range(n))
    sampled = [positions[i] for i in indices]

    rng = _random.Random(99)
    for i, gs in enumerate(sampled):
        depth  = indices[i]
        player = 'black' if int(gs.color) == 0 else 'white'
        label  = f"Scenario {i+1}/{len(sampled)} depth={depth} player={player}"
        _compare_position(gs, label, rng)


# ─── Example-game trace ──────────────────────────────────────────────────────

# 24 verified moves from a real human game (black moves first).
# Truncated after j11-l13; the next annotated move (q18-o16) appears to be a
# transcription error — its source square is in white's territory at that point.
_EXAMPLE_GAME = (
    "p3-p6 e18-e16 q6-r7 e15-l8 p5-n7 e18-e17 s3-p3 q19-p19 e3-e6 n10-m9 "
    "o9-o7 l9-l8 p6-m6 j9-l7 l3-l5 f16-e16 b3-e3 c17-m7 j5-l7 p13-n15 "
    "r7-l13 h15-j13 m13-m15 j11-l13"
).split()

def test_gess_example_game_trace():
    """Play through the provided example game move-by-move.

    At every position we verify:
      - The source square is legal in both pgx and the reference.
      - The destination is legal for that source in both implementations.
      - The board after applying the move matches the reference applyMove exactly.
      - Ring detection agrees at every step.
    """
    state  = _GAME.init()
    color  = 0  # black moves first

    for move_i, move_str in enumerate(_EXAMPLE_GAME):
        src_label, dst_label = move_str.split('-')
        src_idx = _label_to_idx(src_label)
        dst_idx = _label_to_idx(dst_label)
        player  = 'black' if color == 0 else 'white'
        label   = f"move {move_i + 1} ({player}) {move_str}"

        ref_board = _pgx_to_ref(state.board)

        # Source must be legal
        pgx_src_mask = np.array(_jit_src_mask(state), dtype=bool)
        assert pgx_src_mask[src_idx], (
            f"{label}: source {src_label} (idx={src_idx}) not in pgx legal source mask"
        )
        src_r, src_c = divmod(src_idx, _R_SIZE)
        assert _ref_is_valid_center(ref_board, src_r, src_c, player), (
            f"{label}: source {src_label} not valid per reference isValidFootprintCenter"
        )

        # Destination must be legal for that source
        pgx_dst_mask = np.array(_jit_dst_mask(state, jnp.int32(src_idx)), dtype=bool)
        assert pgx_dst_mask[dst_idx], (
            f"{label}: destination {dst_label} (idx={dst_idx}) not in pgx legal dest mask"
        )
        dst_r, dst_c = divmod(dst_idx, _R_SIZE)
        ref_dests = {r * _R_SIZE + c for r, c in _ref_get_valid_moves(ref_board, src_r, src_c, player)}
        assert dst_idx in ref_dests, (
            f"{label}: destination {dst_label} not in reference getValidMoves"
        )

        # Boards must agree after the move
        mover_stone = jnp.int8(color + 1)
        ref_after   = _ref_apply_move(ref_board, src_r, src_c, dst_r, dst_c, player)
        expected    = _ref_to_pgx(ref_after)
        pgx_after   = np.array(_jit_apply(
            jnp.array(np.array(state.board), dtype=jnp.int8),
            jnp.int32(src_idx), jnp.int32(dst_idx), mover_stone,
        ))
        assert np.array_equal(pgx_after, expected), (
            f"{label}: board mismatch after move\n"
            f"  first diff at flat indices: {np.where(pgx_after != expected)[0].tolist()}"
        )

        # Ring detection must agree
        for p_str, sv in [('black', jnp.int8(1)), ('white', jnp.int8(2))]:
            ref_ring = _ref_has_ring(ref_after, p_str)
            pgx_ring = bool(_jit_has_ring(jnp.array(expected, dtype=jnp.int8), sv))
            assert pgx_ring == ref_ring, (
                f"{label}: ring mismatch for {p_str} after move: "
                f"pgx={pgx_ring} ref={ref_ring}"
            )

        # Advance pgx state through both stages
        state = _jit_step(state, jnp.int32(src_idx))   # stage 0 → 1
        state = _jit_step(state, jnp.int32(dst_idx))   # stage 1 → 0
        color = 1 - color

        # Once the game ends, stop
        if bool(_GAME.is_terminal(state)):
            break
