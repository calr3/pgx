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

import pgx
from pgx.epaminondas import Epaminondas, State
from pgx._src.games.epaminondas import (
    BLACK, EMPTY, HEIGHT, MAX_MOVES, MAX_MOVES_SINCE_CAPTURE, N, WHITE, WIDTH, Game, GameState,
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


def test_capture_resets_the_clock():
    # The cap counts moves since the last capture, not moves since the start:
    # an absolute cap decided on material pays a player who is ahead to run the
    # clock out, and truncates games that are still being fought.
    state = make_state(white=[(5, 4), (5, 5)], black=[(5, 6)], color=0,
                       moves_since_capture=jnp.int32(30))
    state = play(state, (5, 5), (5, 4), (5, 6))  # phalanx of two takes a lone piece
    assert int((np.asarray(state.board) != EMPTY).sum()) == 2  # a piece came off
    assert int(state.moves_since_capture) == 0

    # A quiet move advances it instead.
    state = make_state(white=[(5, 5)], black=[(7, 7)], color=0, moves_since_capture=jnp.int32(10))
    state = play(state, (5, 5), (5, 5), (4, 5))
    assert int(state.moves_since_capture) == 11


def test_capture_clock_ends_the_game_on_material():
    state = make_state(white=[(5, 5), (9, 1)], black=[(7, 7)], color=0,
                       moves_since_capture=jnp.int32(MAX_MOVES_SINCE_CAPTURE - 1))
    state = play(state, (5, 5), (5, 5), (4, 5))
    assert int(state.moves_since_capture) == MAX_MOVES_SINCE_CAPTURE
    assert bool(game.is_terminal(state))
    assert np.asarray(game.rewards(state)).tolist() == [1.0, -1.0]  # white up 2-1


def test_move_cap_is_decided_on_material():
    # Reaching the cap is not a draw: it goes to whoever has more pieces left.
    # A plain draw there made stalling a safe equilibrium - a player who never
    # commits scores 0 rather than -1 - and self-play converged on it.
    def capped(white, black):
        state = make_state(white=white, black=black, color=0, moves=jnp.int32(MAX_MOVES - 1))
        state = play(state, white[0], white[0], (white[0][0] - 1, white[0][1]))
        assert int(state.moves) == MAX_MOVES and bool(game.is_terminal(state))
        return np.asarray(game.rewards(state)).tolist()

    assert capped([(5, 5), (9, 1), (9, 3)], [(7, 7)]) == [1.0, -1.0]  # white up 3-1
    assert capped([(5, 5)], [(7, 7), (2, 9), (2, 11)]) == [-1.0, 1.0]  # black up 3-1
    assert capped([(5, 5), (9, 1)], [(7, 7), (2, 9)]) == [0.0, 0.0]  # level: the only draw


def test_a_win_beats_material_at_the_cap():
    # Material only decides when nobody has won. A player who wins on the same
    # move that reaches the cap still wins, even a piece down.
    state = make_state(white=[(5, 5)], black=[(7, 7), (2, 9)], color=0)
    state = state._replace(moves=jnp.int32(MAX_MOVES), winner=jnp.int32(0))
    assert bool(game.is_terminal(state))
    assert np.asarray(game.rewards(state)).tolist() == [1.0, -1.0]


# ─── env wiring ──────────────────────────────────────────────────────────────

def test_env_turn_order_and_observation():
    state = env.init(jax.random.PRNGKey(0))
    assert state.observation.shape == (HEIGHT, WIDTH, 7)
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
