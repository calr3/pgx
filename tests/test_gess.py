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
import numpy as np
import pytest

import pgx
from pgx.gess import Gess, State
from pgx._src.games.gess import (
    BOARD_SIZE, N, MIN_IDX, MAX_IDX, DRAW_NO_CAPTURE_TURNS,
    Game, GameState,
    _apply_move, _has_ring, _legal_source_mask, _legal_dest_mask,
    _make_init_board,
)

env   = Gess()
game  = Game()
_init = jax.jit(env.init)
_step = jax.jit(env.step)

BLACK = jnp.int8(1)
WHITE = jnp.int8(2)


# ─── helpers ─────────────────────────────────────────────────────────────────

def idx(r, c):
    return r * BOARD_SIZE + c

def make_board(*stones):
    """Build a board from (stone_val, r, c) triples."""
    b = np.zeros(N, dtype=np.int8)
    for val, r, c in stones:
        b[idx(r, c)] = val
    return jnp.array(b, dtype=jnp.int8)

def make_state(color=0, board=None, stage=0, source=0, winner=-1):
    if board is None:
        board = jnp.zeros(N, dtype=jnp.int8)
    return GameState(
        color=jnp.int32(color),
        board=board,
        stage=jnp.int32(stage),
        source=jnp.int32(source),
        winner=jnp.int32(winner),
    )


# ─── initial position ────────────────────────────────────────────────────────

def test_init_stone_counts():
    board = _make_init_board()
    assert int(jnp.sum(board == BLACK)) == 43, "black should have 43 stones"
    assert int(jnp.sum(board == WHITE)) == 43, "white should have 43 stones"

def test_init_rings():
    board = _make_init_board()
    assert bool(_has_ring(board, BLACK)), "black should start with a ring"
    assert bool(_has_ring(board, WHITE)), "white should start with a ring"

def test_init_border_empty():
    board = _make_init_board().reshape(BOARD_SIZE, BOARD_SIZE)
    assert int(jnp.sum(board[0,  :])) == 0, "top border must be empty"
    assert int(jnp.sum(board[19, :])) == 0, "bottom border must be empty"
    assert int(jnp.sum(board[:,  0])) == 0, "left border must be empty"
    assert int(jnp.sum(board[:, 19])) == 0, "right border must be empty"

def test_pgx_init():
    state = _init(jax.random.PRNGKey(0))
    assert state.current_player == 0
    assert not state.terminated
    assert not state.truncated
    assert state.rewards.tolist() == [0.0, 0.0]
    assert state._x.stage == 0
    assert state._x.winner == -1
    assert state.observation.shape == (MAX_IDX - MIN_IDX + 1, MAX_IDX - MIN_IDX + 1, 4)


# ─── ring detection ──────────────────────────────────────────────────────────

def test_has_ring_true():
    # 8 black stones surrounding (5,5)
    stones = [(BLACK, r, c) for r in range(4, 7) for c in range(4, 7) if not (r==5 and c==5)]
    board  = make_board(*stones)
    assert bool(_has_ring(board, BLACK))
    assert not bool(_has_ring(board, WHITE))

def test_has_ring_false_centre_occupied():
    # Centre also occupied → not a ring
    stones = [(BLACK, r, c) for r in range(4, 7) for c in range(4, 7)]
    board  = make_board(*stones)
    assert not bool(_has_ring(board, BLACK))

def test_has_ring_false_missing_neighbour():
    stones = [(BLACK, r, c) for r in range(4, 7) for c in range(4, 7) if not (r==5 and c==5)]
    board  = make_board(*stones)
    # Remove one corner stone
    board  = board.at[idx(4, 4)].set(0)
    assert not bool(_has_ring(board, BLACK))

def test_has_ring_near_playing_area_edge():
    # Ring centred at (2, 2): all 8 neighbours are within the playing area.
    stones = [(BLACK, r, c) for r in range(1, 4) for c in range(1, 4) if not (r==2 and c==2)]
    board  = make_board(*stones)
    assert bool(_has_ring(board, BLACK))

    # Ring centred at (18, 18): far corner of the playing area, also valid.
    stones2 = [(BLACK, r, c) for r in range(17, 20) for c in range(17, 20)
               if not (r==18 and c==18) and r < BOARD_SIZE and c < BOARD_SIZE]
    board2  = make_board(*stones2)
    # The neighbours at row/col 19 are border cells – always empty in real play –
    # so this ring is impossible without storing stones on the border.
    # In a synthetic board with those cells filled, _has_ring would report True;
    # in real play it never happens because _apply_move clears border cells.
    assert bool(_has_ring(board2, BLACK))


# ─── legal source mask ───────────────────────────────────────────────────────

def test_legal_source_no_own():
    state = make_state(color=0, board=jnp.zeros(N, jnp.int8))
    mask  = _legal_source_mask(state)
    assert not bool(jnp.any(mask)), "no own stones → no valid sources"

def test_legal_source_opponent_in_footprint():
    # Black stone at (5,5); white stone at (5,6) blocks the footprint
    board = make_board((BLACK, 5, 5), (WHITE, 5, 6))
    state = make_state(color=0, board=board)
    mask  = _legal_source_mask(state)
    assert not bool(mask[idx(5, 5)]), "opponent in footprint → invalid source"

def test_legal_source_only_centre():
    # Black stone only at centre (5,5), none in surrounding 8 → invalid
    board = make_board((BLACK, 5, 5))
    state = make_state(color=0, board=board)
    mask  = _legal_source_mask(state)
    assert not bool(mask[idx(5, 5)])

def test_legal_source_valid():
    # Black stone at (5,5) and (4,5) → centre (5,5) has own stone above it
    board = make_board((BLACK, 5, 5), (BLACK, 4, 5))
    state = make_state(color=0, board=board)
    mask  = _legal_source_mask(state)
    assert bool(mask[idx(5, 5)]), "surrounding stone present → valid source"

def test_legal_source_ring_valid():
    # A ring centred at (5,5): 8 surrounding cells occupied, centre empty
    stones = [(BLACK, r, c) for r in range(4, 7) for c in range(4, 7) if not (r==5 and c==5)]
    board  = make_board(*stones)
    state  = make_state(color=0, board=board)
    mask   = _legal_source_mask(state)
    assert bool(mask[idx(5, 5)]), "ring centre is a valid source"


# ─── legal destination mask ──────────────────────────────────────────────────

def test_legal_dest_no_direction():
    # Stone only at centre → no direction → no destinations
    board = make_board((BLACK, 5, 5))
    state = make_state(color=0, board=board)
    mask  = _legal_dest_mask(state, jnp.int32(idx(5, 5)))
    assert not bool(jnp.any(mask))

def test_legal_dest_one_direction():
    # Empty centre at (5,5), black stone only at (4,5): north direction enabled,
    # no centre stone → max 3 steps.
    board = make_board((BLACK, 4, 5))   # centre (5,5) stays empty
    state = make_state(color=0, board=board)
    mask  = _legal_dest_mask(state, jnp.int32(idx(5, 5)))
    # Steps 1,2,3 north from (5,5) = (4,5),(3,5),(2,5)
    assert bool(mask[idx(4, 5)]), "1 step north should be reachable"
    assert bool(mask[idx(3, 5)]), "2 steps north should be reachable"
    assert bool(mask[idx(2, 5)]), "3 steps north should be reachable"
    assert not bool(mask[idx(1, 5)]), "4 steps north: out of range (no centre stone)"

def test_legal_dest_centre_stone_unlimited():
    # Centre stone at (5,5) + north neighbour at (4,5) → unlimited north movement.
    board = make_board((BLACK, 5, 5), (BLACK, 4, 5))
    state = make_state(color=0, board=board)
    mask  = _legal_dest_mask(state, jnp.int32(idx(5, 5)))
    # board[5*20+5] == BLACK → has_center=True → max_steps=BOARD_SIZE
    assert bool(mask[idx(1, 5)]), "4 steps north reachable with centre stone"

def test_legal_dest_blocked_by_stone():
    # Blocker at (4,5): footprint of destination (3,5) overlaps it,
    # so the piece stops at (3,5) and cannot reach (2,5).
    board = make_board((BLACK, 5, 5), (BLACK, 4, 5), (WHITE, 4, 5))
    # Wait – (4,5) can't be both BLACK and WHITE; let's use a blocker further out.
    # Black at (5,5)[centre] + (4,5)[north]: footprint around (5,5).
    # Blocker at (2,6) (outside source footprint, but inside dest footprint for step 2).
    board = make_board((BLACK, 5, 5), (BLACK, 4, 5), (WHITE, 2, 6))
    state = make_state(color=0, board=board)
    mask  = _legal_dest_mask(state, jnp.int32(idx(5, 5)))
    # Step 1 north: centre=(4,5). clearance = fp(4,5) \ fp(5,5) = {(3,4),(3,5),(3,6),(4,4),(4,6)}.
    #   (2,6) is NOT in clearance, so no blocking at step 1 → (4,5) reachable.
    # Step 2 north: centre=(3,5). clearance includes (2,4),(2,5),(2,6).
    #   (2,6) has WHITE → blocking! Piece stops at (3,5).
    assert bool(mask[idx(4, 5)]), "step 1 should be reachable"
    assert bool(mask[idx(3, 5)]), "step 2 (stopping cell) should be reachable"
    assert not bool(mask[idx(2, 5)]), "step 3 blocked by stone at step 2"


# ─── apply_move ──────────────────────────────────────────────────────────────

def test_apply_move_simple():
    # Move a single stone from (5,5) north by 1 step: source=(5,5), dest=(4,5).
    # Stone is at the north cell of source footprint → north direction enabled.
    # Black has stones at (4,5) and (5,5) (both needed for source validity,
    # but here we just test _apply_move directly).
    board = make_board((BLACK, 4, 5), (BLACK, 5, 5))
    new_b = _apply_move(board, jnp.int32(idx(5, 5)), jnp.int32(idx(4, 5)), BLACK)
    # Source footprint of (5,5) covers rows 4-6, cols 4-6.
    # Dest footprint of (4,5) covers rows 3-5, cols 4-6.
    # Stone at (4,5): offset (-1,0) in source → lands at (3,5) in dest.
    # Stone at (5,5): offset (0,0) in source → lands at (4,5) in dest.
    assert int(new_b[idx(3, 5)]) == BLACK, "stone from (4,5) should move to (3,5)"
    assert int(new_b[idx(4, 5)]) == BLACK, "stone from (5,5) should move to (4,5)"
    assert int(new_b[idx(5, 5)]) == 0,     "source centre should be cleared"

def test_apply_move_capture():
    # Move black piece onto a white stone; white stone should be removed.
    board = make_board((BLACK, 5, 5), (BLACK, 4, 5), (WHITE, 3, 5))
    new_b = _apply_move(board, jnp.int32(idx(5, 5)), jnp.int32(idx(4, 5)), BLACK)
    # Dest footprint (4,5) covers (3..5, 4..6). White at (3,5) is in dest fp → cleared.
    # Black stone from (5,5) offset (0,0) → lands at (4,5).
    # Black stone from (4,5) offset (-1,0) → lands at (3,5).
    assert int(new_b[idx(3, 5)]) == BLACK, "capturing stone should occupy (3,5)"
    assert int(new_b[idx(4, 5)]) == BLACK

def test_apply_move_border_removal():
    # Stone near the top edge: if dest pushes it onto border row 0, it's removed.
    board = make_board((BLACK, 2, 5), (BLACK, 1, 5))
    # Move source=(2,5) to dest=(1,5): stone at (1,5) offset (-1,0) → (0,5) = border!
    new_b = _apply_move(board, jnp.int32(idx(2, 5)), jnp.int32(idx(1, 5)), BLACK)
    assert int(new_b[idx(0, 5)]) == 0,     "stone landing on border must be removed"
    assert int(new_b[idx(1, 5)]) == BLACK  # stone from (2,5) offset (0,0) → (1,5)

def test_apply_move_own_stone_capture():
    # Own stone in destination footprint but not part of moving piece gets displaced.
    board = make_board((BLACK, 5, 5), (BLACK, 4, 5), (BLACK, 3, 5))
    # Moving source=(5,5) to dest=(4,5): dest fp covers (3..5,4..6).
    # (3,5) is in dest fp but not in src fp (src is 4..6,4..6): it gets displaced.
    # Moving piece: (4,5) offset(-1,0)→(3,5); (5,5) offset(0,0)→(4,5).
    new_b = _apply_move(board, jnp.int32(idx(5, 5)), jnp.int32(idx(4, 5)), BLACK)
    # (3,5) cleared first, then placed again by moving stone from (4,5).
    assert int(new_b[idx(3, 5)]) == BLACK
    assert int(new_b[idx(4, 5)]) == BLACK
    assert int(new_b[idx(5, 5)]) == 0


# ─── win / terminal conditions ───────────────────────────────────────────────

def test_winner_mover_destroys_own_ring():
    # Build a minimal ring for black at (10,10), and nothing for white.
    # Then after white "plays", check that black wins (white has no ring).
    # We test _step_stage1 indirectly via game.step.
    ring_stones = [(BLACK, r, c)
                   for r in range(9, 12) for c in range(9, 12)
                   if not (r == 10 and c == 10)]
    board = make_board(*ring_stones)
    # White has one stone at (5,5) and (4,5) to form a valid source, but no ring.
    board = board.at[idx(5, 5)].set(WHITE)
    board = board.at[idx(4, 5)].set(WHITE)

    state = make_state(color=1, board=board, stage=1, source=idx(5, 5))
    # White moves to (3,5): vacuous destination; white still has no ring.
    x = game.step(state, jnp.int32(idx(3, 5)))
    assert int(x.winner) == 0, "black wins because white never had a ring"

def test_no_winner_mid_game():
    board = _make_init_board()
    state = make_state(color=0, board=board, stage=1, source=idx(6, 8))
    # Move black's ring-adjacent piece slightly; both players still have rings.
    x = game.step(state, jnp.int32(idx(6, 9)))
    assert int(x.winner) == -1, "game should continue mid-play"


# ─── two-stage pgx step ──────────────────────────────────────────────────────

def test_two_stage_current_player():
    key   = jax.random.PRNGKey(0)
    state = _init(key)
    assert int(state.current_player) == 0
    assert int(state._x.stage)       == 0

    # Stage 0: pick any valid source
    src_mask = state.legal_action_mask
    src_idx  = int(jnp.argmax(src_mask))
    state    = _step(state, jnp.int32(src_idx))

    assert int(state.current_player) == 0, "current_player must not change after stage 0"
    assert int(state._x.stage)       == 1

    # Stage 1: pick any valid destination
    dst_mask = state.legal_action_mask
    dst_idx  = int(jnp.argmax(dst_mask))
    state    = _step(state, jnp.int32(dst_idx))

    assert int(state.current_player) == 1, "current_player switches after stage 1"
    assert int(state._x.stage)       == 0

def test_two_stage_no_self_termination():
    # Play several full moves; game should not terminate spontaneously.
    key   = jax.random.PRNGKey(42)
    state = _init(key)
    for _ in range(6):   # 3 full moves per player
        for _stage in range(2):
            mask  = state.legal_action_mask
            act   = int(jnp.argmax(mask))
            state = _step(state, jnp.int32(act))
        assert not state.terminated, "game should not terminate after a few normal moves"

def test_make_gess():
    env2 = pgx.make("gess")
    assert isinstance(env2, Gess)
    state = env2.init(jax.random.PRNGKey(0))
    assert state.legal_action_mask.shape == (N,)
    assert state.observation.shape == (MAX_IDX - MIN_IDX + 1, MAX_IDX - MIN_IDX + 1, 4)

def test_mover_loses_when_both_final_rings_destroyed():
    # Black's only ring at (10,10), white's only ring at (10,14).
    # Black moves (10,6)→(10,12); destination footprint (rows 9-11, cols 11-13)
    # clips 3 stones from black's ring (cols 11) and 3 from white's ring (cols 13),
    # destroying both. Neither player has a ring → mover (black) loses.
    black_ring = [(BLACK, r, c) for r in range(9, 12) for c in range(9, 12)
                  if not (r == 10 and c == 10)]
    white_ring = [(WHITE, r, c) for r in range(9, 12) for c in range(13, 16)
                  if not (r == 10 and c == 14)]
    board = make_board(*black_ring, *white_ring, (BLACK, 10, 6), (BLACK, 10, 7))

    state = make_state(color=0, board=board, stage=1, source=idx(10, 6))
    x = game.step(state, jnp.int32(idx(10, 12)))

    assert int(x.winner) == 1, (
        "white should win: both final rings destroyed → mover (black) loses"
    )

# ─── draw: 20 consecutive turns without a capture ────────────────────────────

def _draw_test_board():
    """Board where black can shuffle a piece back and forth without capturing.

    Black ring at (5,5), white ring at (15,15) (so neither side is in danger),
    plus a lone black piece at (10,5) with a north neighbour at (9,5) that can
    slide into empty space, capturing nothing.
    """
    black_ring = [(BLACK, r, c) for r in range(4, 7) for c in range(4, 7)
                  if not (r == 5 and c == 5)]
    white_ring = [(WHITE, r, c) for r in range(14, 17) for c in range(14, 17)
                  if not (r == 15 and c == 15)]
    return make_board(*black_ring, *white_ring, (BLACK, 10, 5), (BLACK, 9, 5))


def test_no_capture_counter_increments():
    # A captureless move bumps the counter by one and does not draw early.
    state = make_state(color=0, board=_draw_test_board(), stage=1,
                       source=idx(10, 5))
    x = game.step(state, jnp.int32(idx(9, 5)))   # slide north, captures nothing
    assert int(x.no_capture_turns) == 1, "captureless turn should increment counter"
    assert not bool(game.is_terminal(x))


def test_draw_after_20_no_capture_turns():
    # Counter sitting one below the threshold; a final captureless turn draws.
    state = make_state(color=0, board=_draw_test_board(), stage=1,
                       source=idx(10, 5))
    state = state._replace(
        no_capture_turns=jnp.int32(DRAW_NO_CAPTURE_TURNS - 1)
    )
    x = game.step(state, jnp.int32(idx(9, 5)))   # 20th captureless turn

    assert int(x.no_capture_turns) == DRAW_NO_CAPTURE_TURNS
    assert bool(game.is_terminal(x)), "game should be drawn after 20 captureless turns"
    assert int(x.winner) == -1, "a draw has no winner"
    assert game.rewards(x).tolist() == [0.0, 0.0], "a draw rewards both players 0"


def test_capture_resets_no_capture_counter():
    # Same setup, but a white stone in the destination footprint gets captured,
    # which resets the counter and prevents the draw.
    board = _draw_test_board().at[idx(8, 5)].set(WHITE)  # lands in dest footprint
    state = make_state(color=0, board=board, stage=1, source=idx(10, 5))
    state = state._replace(
        no_capture_turns=jnp.int32(DRAW_NO_CAPTURE_TURNS - 1)
    )
    x = game.step(state, jnp.int32(idx(9, 5)))

    assert int(x.no_capture_turns) == 0, "a capture resets the counter"
    assert not bool(game.is_terminal(x)), "capturing avoids the draw even at the threshold"


def test_self_capture_resets_no_capture_counter():
    # A stone pushed onto the border ring is removed (a self-capture), which
    # counts as a capture and resets the counter.
    # Black piece at (1,5)+(2,5): moving source (2,5)→(1,5) shoves the (1,5)
    # stone onto border row 0, removing it.
    black_ring = [(BLACK, r, c) for r in range(4, 7) for c in range(4, 7)
                  if not (r == 5 and c == 5)]
    white_ring = [(WHITE, r, c) for r in range(14, 17) for c in range(14, 17)
                  if not (r == 15 and c == 15)]
    board = make_board(*black_ring, *white_ring, (BLACK, 2, 5), (BLACK, 1, 5))
    state = make_state(color=0, board=board, stage=1, source=idx(2, 5))
    state = state._replace(
        no_capture_turns=jnp.int32(DRAW_NO_CAPTURE_TURNS - 1)
    )
    x = game.step(state, jnp.int32(idx(1, 5)))

    assert int(x.no_capture_turns) == 0, "a self-capture resets the counter"
    assert not bool(game.is_terminal(x))


def test_win_on_threshold_turn_is_not_a_draw():
    # Counter one below the draw threshold, but this move wins outright by
    # destroying the opponent's only ring. The win destroys a ring (a capture),
    # which resets the counter, so the result is a decisive win — never a draw.
    # (Reuses the win setup: black keeps ring B, white's only ring is destroyed.)
    black_ring_a = [(BLACK, r, c) for r in range(9, 12) for c in range(9, 12)
                    if not (r == 10 and c == 10)]
    black_ring_b = [(BLACK, r, c) for r in range(14, 17) for c in range(9, 12)
                    if not (r == 15 and c == 10)]
    white_ring   = [(WHITE, r, c) for r in range(9, 12) for c in range(13, 16)
                    if not (r == 10 and c == 14)]
    board = make_board(*black_ring_a, *black_ring_b, *white_ring,
                       (BLACK, 10, 6), (BLACK, 10, 7))
    state = make_state(color=0, board=board, stage=1, source=idx(10, 6))
    state = state._replace(no_capture_turns=jnp.int32(DRAW_NO_CAPTURE_TURNS - 1))

    x = game.step(state, jnp.int32(idx(10, 12)))

    assert int(x.winner) == 0, "black wins outright on the would-be draw turn"
    assert int(x.no_capture_turns) == 0, "the winning capture resets the counter"
    assert bool(game.is_terminal(x))
    assert game.rewards(x).tolist() == [1.0, -1.0], "decisive win, not a draw"


def test_self_destruction_loss_on_threshold_turn_is_not_a_draw():
    # Counter one below the threshold, but the mover destroys its OWN only ring
    # this move (a self-capture) and loses. The self-capture resets the counter,
    # so it is a decisive loss rather than a draw.
    black_ring = [(BLACK, r, c) for r in range(4, 7) for c in range(4, 7)
                  if not (r == 5 and c == 5)]
    white_ring = [(WHITE, r, c) for r in range(14, 17) for c in range(14, 17)
                  if not (r == 15 and c == 15)]
    board = make_board(*black_ring, *white_ring, (BLACK, 10, 5), (BLACK, 9, 5))
    state = make_state(color=0, board=board, stage=1, source=idx(10, 5))
    state = state._replace(no_capture_turns=jnp.int32(DRAW_NO_CAPTURE_TURNS - 1))

    # Black slams its moving piece onto its own ring at (5,5), wiping it out.
    # Black has no other ring → mover loses; white (ring at (15,15)) wins.
    x = game.step(state, jnp.int32(idx(5, 5)))

    assert int(x.winner) == 1, "black destroyed its own only ring → white wins"
    assert int(x.no_capture_turns) == 0, "the self-capture resets the counter"
    assert bool(game.is_terminal(x))
    assert game.rewards(x).tolist() == [-1.0, 1.0], "decisive loss, not a draw"


def test_pgx_draw_terminates_with_zero_rewards():
    # End-to-end through the pgx env: a draw sets terminated and zero rewards.
    public = State(  # type: ignore
        current_player=jnp.int32(0),
        _x=make_state(color=0, board=_draw_test_board(), stage=1,
                      source=idx(10, 5))._replace(
            no_capture_turns=jnp.int32(DRAW_NO_CAPTURE_TURNS - 1)
        ),
    )
    public = _step(public, jnp.int32(idx(9, 5)))
    assert bool(public.terminated), "draw should terminate the pgx episode"
    assert public.rewards.tolist() == [0.0, 0.0]
    assert int(public._x.no_capture_turns) == DRAW_NO_CAPTURE_TURNS


def test_mover_wins_when_own_nonfinal_ring_and_opp_final_ring_destroyed():
    # Black has TWO rings: ring A at (10,10) and ring B at (15,10).
    # White has ONE ring at (10,14).
    # Black moves (10,6)→(10,12); destination footprint destroys ring A and white's
    # ring, but ring B (rows 14-16) is untouched. Black still has a ring → black wins.
    black_ring_a = [(BLACK, r, c) for r in range(9, 12) for c in range(9, 12)
                    if not (r == 10 and c == 10)]
    black_ring_b = [(BLACK, r, c) for r in range(14, 17) for c in range(9, 12)
                    if not (r == 15 and c == 10)]
    white_ring   = [(WHITE, r, c) for r in range(9, 12) for c in range(13, 16)
                    if not (r == 10 and c == 14)]
    board = make_board(*black_ring_a, *black_ring_b, *white_ring,
                       (BLACK, 10, 6), (BLACK, 10, 7))

    state = make_state(color=0, board=board, stage=1, source=idx(10, 6))
    x = game.step(state, jnp.int32(idx(10, 12)))

    assert int(x.winner) == 0, (
        "black should win: ring B survives even though ring A and white's only ring "
        "were both destroyed"
    )
