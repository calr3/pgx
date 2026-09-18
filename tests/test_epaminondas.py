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
from pgx.epaminondas import Epaminondas, State
from pgx._src.games.epaminondas import (
    BLACK, EMPTY, HEIGHT, MAX_QUIET_MOVES, N, WHITE, WIDTH, Game, GameState,
)

env = Epaminondas()
game = Game()


# ─── helpers ─────────────────────────────────────────────────────────────────

def idx(r, c):
    return r * WIDTH + c


def rc(i):
    return (int(i) // WIDTH, int(i) % WIDTH)


def make_state(white=(), black=(), color=0, **kwargs) -> GameState:
    """GameState with only the listed (row, col) squares occupied."""
    board = np.zeros(N, dtype=np.int8)
    for r, c in white:
        board[idx(r, c)] = WHITE
    for r, c in black:
        board[idx(r, c)] = BLACK
    return GameState(color=jnp.int32(color), board=jnp.asarray(board), **kwargs)


def legal_squares(state: GameState):
    return {rc(i) for i in np.flatnonzero(np.asarray(game.legal_action_mask(state)))}


def play(state: GameState, lead, rear, dest) -> GameState:
    """Play one full move as three actions."""
    for square in (lead, rear, dest):
        state = game.step(state, jnp.int32(idx(*square)))
    return state


def occupied(state: GameState):
    board = np.asarray(state.board)
    return (
        {rc(i) for i in np.flatnonzero(board == WHITE)},
        {rc(i) for i in np.flatnonzero(board == BLACK)},
    )


# ─── setup and basic movement ────────────────────────────────────────────────

def test_initial_position():
    state = game.init()
    board = np.asarray(state.board).reshape(HEIGHT, WIDTH)
    assert (board == WHITE).sum() == 28 and (board == BLACK).sum() == 28
    assert set(np.where(board == WHITE)[0]) == {0, 1}
    assert set(np.where(board == BLACK)[0]) == {HEIGHT - 2, HEIGHT - 1}
    assert int(state.color) == 0 and int(state.stage) == 0
    # Only the second row can lead a move; the back row is completely blocked.
    assert legal_squares(state) == {(1, c) for c in range(WIDTH)}


def test_single_piece_moves_to_any_empty_neighbour():
    state = make_state(white=[(5, 5)], black=[(6, 6)])
    assert legal_squares(state) == {(5, 5)}  # the only piece that can lead
    state = game.step(state, jnp.int32(idx(5, 5)))
    assert (5, 5) in legal_squares(state)  # rear == lead: a single piece
    state = game.step(state, jnp.int32(idx(5, 5)))
    # All eight neighbours except the one occupied by the enemy piece: a lone
    # piece is a phalanx of one and can never capture.
    assert legal_squares(state) == {
        (4, 4), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5),
    }


def test_phalanx_moves_up_to_its_length():
    # Three white pieces in a row; lead (5, 5) moving right.
    state = make_state(white=[(5, 3), (5, 4), (5, 5)])
    state = game.step(state, jnp.int32(idx(5, 5)))
    assert legal_squares(state) == {(5, 5), (5, 4), (5, 3)}  # single, pair, triple
    triple = game.step(state, jnp.int32(idx(5, 3)))
    assert legal_squares(triple) == {(5, 6), (5, 7), (5, 8)}  # up to three squares
    pair = game.step(state, jnp.int32(idx(5, 4)))
    assert legal_squares(pair) == {(5, 6), (5, 7)}  # split: only two squares


def test_phalanx_cannot_move_onto_or_over_pieces():
    state = make_state(white=[(5, 3), (5, 4), (5, 5), (5, 8)], black=[(5, 7)])
    state = play(state, (5, 5), (5, 3), (5, 6))
    white, _ = occupied(state)
    assert white == {(5, 4), (5, 5), (5, 6), (5, 8)}  # moved one square right

    # The friendly piece at (5, 8) blocks; the enemy at (5, 7) may only be
    # captured, never jumped.
    state = make_state(white=[(5, 3), (5, 4), (5, 5), (5, 8)], black=[(5, 7)])
    state = game.step(state, jnp.int32(idx(5, 5)))
    state = game.step(state, jnp.int32(idx(5, 3)))
    assert legal_squares(state) == {(5, 6), (5, 7)}  # (5, 7) is a capture


def test_phalanx_moves_diagonally_and_backwards():
    state = make_state(white=[(3, 3), (4, 4), (5, 5)])
    # Any of the three can lead: the ends can lead the phalanx along its
    # diagonal, and every piece can also step somewhere as a single piece.
    assert legal_squares(state) == {(3, 3), (4, 4), (5, 5)}
    state = play(state, (5, 5), (3, 3), (8, 8))
    white, _ = occupied(state)
    assert white == {(6, 6), (7, 7), (8, 8)}


# ─── capture ─────────────────────────────────────────────────────────────────

def test_capture_removes_the_whole_enemy_line():
    # White phalanx of three captures an enemy line of two.
    state = make_state(white=[(5, 2), (5, 3), (5, 4)], black=[(5, 5), (5, 6), (7, 7)])
    state = play(state, (5, 4), (5, 2), (5, 5))
    white, black = occupied(state)
    assert white == {(5, 3), (5, 4), (5, 5)}  # lead stops on the first enemy square
    assert black == {(7, 7)}  # both pieces of the enemy line are removed


def test_capture_needs_a_strictly_shorter_enemy_line():
    # Three against three: the enemy line is not shorter, so no capture.
    state = make_state(white=[(5, 2), (5, 3), (5, 4)], black=[(5, 5), (5, 6), (5, 7)])
    state = game.step(state, jnp.int32(idx(5, 4)))
    state = game.step(state, jnp.int32(idx(5, 2)))
    assert legal_squares(state) == set()  # blocked entirely

    # Against two it is legal again.
    state = make_state(white=[(5, 2), (5, 3), (5, 4)], black=[(5, 5), (5, 6)])
    state = game.step(state, jnp.int32(idx(5, 4)))
    state = game.step(state, jnp.int32(idx(5, 2)))
    assert legal_squares(state) == {(5, 5)}


def test_enemy_line_counted_along_the_direction_of_movement():
    # The enemy pieces lie across the line of movement, so the line being
    # captured is length one even though the enemy has three pieces adjacent.
    state = make_state(
        white=[(5, 2), (5, 3)], black=[(5, 4), (4, 4), (6, 4)]
    )
    state = play(state, (5, 3), (5, 2), (5, 4))
    white, black = occupied(state)
    assert white == {(5, 3), (5, 4)}
    assert black == {(4, 4), (6, 4)}  # only the piece in the line is captured


def test_single_piece_cannot_capture():
    state = make_state(white=[(5, 4)], black=[(5, 5)])
    state = play(state, (5, 4), (5, 4), (4, 4))  # forced to move elsewhere
    white, black = occupied(state)
    assert white == {(4, 4)} and black == {(5, 5)}


# ─── objective ───────────────────────────────────────────────────────────────

def test_reaching_the_back_rank_gives_the_opponent_one_turn():
    state = make_state(white=[(10, 5)], black=[(6, 9), (6, 10)], color=0)
    state = play(state, (10, 5), (10, 5), (11, 5))  # white steps onto black's back rank
    assert int(state.winner) == -1 and not bool(game.is_terminal(state))  # black replies first
    assert int(state.color) == 1

    # Black ignores it: white wins at the start of its turn.
    ignored = play(state, (6, 10), (6, 9), (6, 11))
    assert int(ignored.winner) == 0 and bool(game.is_terminal(ignored))
    assert np.asarray(game.rewards(ignored)).tolist() == [1.0, -1.0]


def test_matching_the_incursion_avoids_defeat():
    state = make_state(white=[(10, 5)], black=[(1, 9), (1, 10)], color=0)
    state = play(state, (10, 5), (10, 5), (11, 5))
    # Black puts a piece on white's back rank too: equal counts, so nobody wins.
    state = play(state, (1, 10), (1, 9), (0, 11))
    assert int(state.winner) == -1 and not bool(game.is_terminal(state))


def test_capturing_the_intruder_avoids_defeat():
    state = make_state(white=[(10, 5)], black=[(11, 7), (11, 8)], color=0)
    state = play(state, (10, 5), (10, 5), (11, 5))
    white, _ = occupied(state)
    assert white == {(11, 5)}
    # Black's phalanx of two captures the lone white piece on its back rank.
    state = play(state, (11, 7), (11, 8), (11, 5))
    white, black = occupied(state)
    assert white == set() and black == {(11, 5), (11, 6)}
    assert int(state.winner) == 1  # white has no pieces left and cannot move


def test_symmetry_rule_forbids_a_mirrored_back_rank():
    # White (10, 3) stepping to (11, 3) would mirror white (11, 10) and leave
    # the whole board left-to-right symmetric, so it is forbidden.
    state = make_state(white=[(10, 3), (11, 10)], black=[(0, 3), (0, 10)], color=0)
    state = game.step(state, jnp.int32(idx(10, 3)))
    state = game.step(state, jnp.int32(idx(10, 3)))
    dests = legal_squares(state)
    assert (11, 3) not in dests
    assert (11, 4) in dests  # a non-symmetric square on the back rank is fine

    # The same move is allowed when the result is not symmetric.
    state = make_state(white=[(10, 3), (11, 9)], black=[(0, 3), (0, 10)], color=0)
    state = game.step(state, jnp.int32(idx(10, 3)))
    state = game.step(state, jnp.int32(idx(10, 3)))
    assert (11, 3) in legal_squares(state)


def test_the_quiet_move_cap_ends_the_game():
    state = make_state(white=[(5, 5), (9, 1)], black=[(7, 7)], color=0,
                       quiet_moves=jnp.int32(MAX_QUIET_MOVES - 1))
    assert not bool(game.is_terminal(state))
    state = play(state, (5, 5), (5, 5), (4, 5))
    assert int(state.quiet_moves) == MAX_QUIET_MOVES and bool(game.is_terminal(state))
    # White's piece on row 9 is two ranks from black's home; black's best is
    # seven from white's, so white is further up the board.
    assert np.asarray(game.rewards(state)).tolist() == [1.0, -1.0]


def test_quiet_move_cap_is_decided_on_advancement():
    # Reaching the cap is not a draw: it goes to whoever has come further up the
    # board, compared rank by rank from the opponent's home rank inwards. A
    # plain draw there made stalling a safe equilibrium - a player who never
    # commits scores 0 rather than -1 - and self-play converged on it.
    def capped(white, black):
        state = make_state(white=white, black=black, color=0,
                           quiet_moves=jnp.int32(MAX_QUIET_MOVES))
        assert bool(game.is_terminal(state))
        return np.asarray(game.rewards(state)).tolist()

    # White is one rank from black's home (row 10), black is four from white's.
    assert capped([(10, 3)], [(4, 7)]) == [1.0, -1.0]
    assert capped([(4, 7)], [(1, 3)]) == [-1.0, 1.0]

    # Advancement decides, not material: one piece further up beats five at home.
    assert capped([(10, 3)], [(11, 0), (11, 1), (11, 2), (11, 3), (11, 4)]) == [1.0, -1.0]

    # Level at the second-to-home rank, so the next rank back decides it.
    assert capped([(10, 3), (9, 2)], [(1, 5)]) == [1.0, -1.0]

    # A rank-for-rank mirror is the only position advancement cannot separate;
    # test_mirror_falls_back_to_the_last_capture covers what happens then.
    assert capped([(1, 3), (3, 5)], [(10, 3), (9, 5)]) == [1.0, -1.0]  # white a rank deeper


def test_mirror_falls_back_to_the_last_capture():
    # Advancement ties only on an exact mirror. It then goes to whoever captured
    # most recently, and if nobody ever has, to black - who moves second. No
    # game is ever drawn.
    mirror = dict(white=[(1, 3), (2, 5)], black=[(10, 3), (9, 5)])

    def capped(**kwargs):
        state = make_state(color=0, quiet_moves=jnp.int32(MAX_QUIET_MOVES),
                           **mirror, **kwargs)
        assert bool(game.is_terminal(state))
        return np.asarray(game.rewards(state)).tolist()

    assert capped(last_capturer=jnp.int32(0)) == [1.0, -1.0]
    assert capped(last_capturer=jnp.int32(1)) == [-1.0, 1.0]
    assert capped() == [-1.0, 1.0]  # no capture all game: black, for moving second


def test_a_capture_resets_the_quiet_clock():
    """The clock counts moves since the last capture, so a capture zeroes it.

    This is what separates v6 from v5's absolute cap: a game that keeps trading
    never reaches the limit, however long it runs.
    """
    # One move short of the cap, with a capture available. Black keeps a second
    # piece well away from the action: capturing its last one would end the game
    # by leaving it with no legal move, which is a win rather than a cap.
    state = make_state(white=[(5, 4), (5, 5)], black=[(5, 6), (9, 9)], color=0,
                       quiet_moves=jnp.int32(MAX_QUIET_MOVES - 1))
    assert not bool(game.is_terminal(state))
    state = play(state, (5, 5), (5, 4), (5, 6))  # captures
    assert int(state.quiet_moves) == 0
    assert not bool(game.is_terminal(state))

    # The same position without a capture available ends instead.
    quiet = make_state(white=[(5, 5)], black=[(9, 9)], color=0,
                       quiet_moves=jnp.int32(MAX_QUIET_MOVES - 1))
    quiet = play(quiet, (5, 5), (5, 5), (4, 5))
    assert int(quiet.quiet_moves) == MAX_QUIET_MOVES
    assert bool(game.is_terminal(quiet))


def test_total_moves_still_counts_up_across_captures():
    """`moves` is now only for reporting, and must not reset when the clock does."""
    state = make_state(white=[(5, 4), (5, 5)], black=[(5, 6)], color=0)
    state = play(state, (5, 5), (5, 4), (5, 6))  # captures
    assert int(state.moves) == 1 and int(state.quiet_moves) == 0


def test_capturing_records_the_capturer():
    state = make_state(white=[(5, 4), (5, 5)], black=[(5, 6)], color=0)
    assert int(state.last_capturer) == -1
    state = play(state, (5, 5), (5, 4), (5, 6))
    assert int(state.last_capturer) == 0  # white took the piece

    # A quiet move leaves it alone.
    quiet = make_state(white=[(5, 5)], black=[(7, 7)], color=0, last_capturer=jnp.int32(1))
    quiet = play(quiet, (5, 5), (5, 5), (4, 5))
    assert int(quiet.last_capturer) == 1


def test_a_win_beats_the_cap_tiebreak():
    # The tiebreak only decides when nobody has won. A player who wins on the
    # same move that reaches the cap still wins, even while further back.
    state = make_state(white=[(5, 5)], black=[(7, 7), (2, 9)], color=0)
    state = state._replace(quiet_moves=jnp.int32(MAX_QUIET_MOVES), winner=jnp.int32(0))
    assert bool(game.is_terminal(state))
    assert np.asarray(game.rewards(state)).tolist() == [1.0, -1.0]


# ─── env wiring ──────────────────────────────────────────────────────────────

def test_env_turn_order_and_observation():
    state = env.init(jax.random.PRNGKey(0))
    assert state.observation.shape == (HEIGHT, WIDTH, 8)
    assert env.num_actions == N and env.num_players == 2
    step = jax.jit(env.step)
    players = [int(state.current_player)]
    for action in (idx(1, 3), idx(1, 3), idx(2, 3)):
        state = step(state, jnp.int32(action))
        players.append(int(state.current_player))
    assert players == [0, 0, 0, 1]  # the turn passes only after the third action

    # Black sees the board flipped, so its own back rank is row 0 for both.
    white_view = env.observe(state, jnp.int32(0))
    black_view = env.observe(state, jnp.int32(1))
    assert np.asarray(white_view[..., 0]).sum() == 28  # own pieces
    assert np.allclose(np.asarray(white_view[..., 0]), np.asarray(black_view[..., 1])[::-1])


def test_the_signed_clock_plane_tracks_the_quiet_move_count():
    # Just after a capture the plane is zero whatever the position, so neither
    # side is told anything about the tiebreak before it can matter.
    state = make_state(white=[(10, 3)], black=[(4, 7)], color=0)
    for seat in (0, 1):
        assert np.asarray(game.observe(state, jnp.int32(seat))[..., 7]).max() == 0.0

    half = state._replace(quiet_moves=jnp.int32(MAX_QUIET_MOVES // 2))
    plane = np.asarray(game.observe(half, jnp.int32(0))[..., 7])
    assert plane.min() == plane.max() == pytest.approx(0.5)  # constant over the board


def test_the_signed_clock_plane_says_who_wins_if_the_game_ends_now():
    # White is further up the board, so white wins on advancement.
    state = make_state(white=[(10, 3)], black=[(4, 7)], color=0)
    state = state._replace(quiet_moves=jnp.int32(MAX_QUIET_MOVES))
    assert np.asarray(game.rewards(state)).tolist() == [1.0, -1.0]

    # Each side is told the same outcome from its own point of view, and the
    # magnitude is the clock either way.
    assert np.asarray(game.observe(state, jnp.int32(0))[..., 7]).max() == pytest.approx(1.0)
    assert np.asarray(game.observe(state, jnp.int32(1))[..., 7]).max() == pytest.approx(-1.0)


def test_the_opening_position_leaks_no_colour():
    # The regression that cost E10 a 3.3 sigma seat asymmetry: the opening is an
    # exact mirror with no captures, so a raw verdict resolved to black and told
    # white it was losing before a piece had moved. Scaled by the clock, which
    # is zero there, both sides see the same thing.
    opening = game.init()
    white = np.asarray(game.observe(opening, jnp.int32(0)))
    black = np.asarray(game.observe(opening, jnp.int32(1)))
    assert white[..., 7].max() == 0.0 and black[..., 7].max() == 0.0
    # And the whole observation is identical under the canonical view: both are
    # already stated from the mover's side, so they match without any flip.
    assert np.allclose(white, black)


def test_the_signed_clock_carries_the_black_wins_mirrors_rule():
    # An exact rank-for-rank mirror with nobody having captured: black wins it,
    # and that is recoverable from neither the stones nor the canonical view.
    state = make_state(white=[(5, 5)], black=[(6, 5)], color=0,
                       quiet_moves=jnp.int32(MAX_QUIET_MOVES))
    assert int(state.last_capturer) == -1
    assert np.asarray(game.rewards(state)).tolist() == [-1.0, 1.0]
    assert np.asarray(game.observe(state, jnp.int32(0))[..., 7]).max() == pytest.approx(-1.0)
    assert np.asarray(game.observe(state, jnp.int32(1))[..., 7]).max() == pytest.approx(1.0)

    # It follows the last capturer once there has been one.
    took = state._replace(last_capturer=jnp.int32(0))
    assert np.asarray(game.observe(took, jnp.int32(0))[..., 7]).max() == pytest.approx(1.0)


def test_random_playthrough_stays_legal():
    state = env.init(jax.random.PRNGKey(1))
    step = jax.jit(env.step)
    rng = np.random.default_rng(0)
    for _ in range(200):
        if bool(state.terminated):
            break
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        assert legal.size > 0
        state = step(state, jnp.int32(int(rng.choice(legal))))
        board = np.asarray(state._x.board)
        assert (board == WHITE).sum() <= 28 and (board == BLACK).sum() <= 28


def test_api():
    pgx.api_test(env, 3, use_key=False)
