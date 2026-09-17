"""Tests for the alpha-beta engine in examples/alphazero/negamax.py.

The engine lives under examples/ rather than in the package, so the path is
prepended the same way pgx/_src/baseline.py does it for the network module.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pgx
from pgx._src.games.epaminondas import HEIGHT, N, WIDTH

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "alphazero")
    ),
)

from negamax import (  # noqa: E402
    EpaminondasEvaluator,
    EpaminondasWeights,
    NegamaxEngine,
    _bucket,
    make_evaluator,
)

WHITE, BLACK = 1, 2


def board_with(pieces) -> np.ndarray:
    """A (1, N) board holding just `pieces`, each an (row, col, value) triple."""
    b = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
    for r, c, v in pieces:
        b[r, c] = v
    return b.reshape(1, N)


@pytest.fixture(scope="module")
def ev():
    return EpaminondasEvaluator()


def masks(board):
    return board == WHITE, board == BLACK, board == 0


# -- the six heuristics, against hand-computed values --------------------


def test_material_is_the_piece_difference(ev):
    own, opp, _ = masks(board_with([(5, 5, WHITE), (5, 6, WHITE), (6, 6, BLACK)]))
    assert ev._material(own, opp) == pytest.approx([1.0])


def test_crossing_counts_pieces_on_the_opponents_home_rank(ev):
    # White's target is row HEIGHT-1 (black's home); black's target is row 0.
    board = board_with([(HEIGHT - 1, 3, WHITE), (0, 2, BLACK), (0, 3, BLACK)])
    own, opp, _ = masks(board)
    assert ev._crossing(own, opp, np.array([0])) == pytest.approx([1.0 - 2.0])
    # From black's seat the same position is worth the opposite.
    assert ev._crossing(opp, own, np.array([1])) == pytest.approx([2.0 - 1.0])


def test_home_row_defense_is_the_longest_contiguous_run(ev):
    board = board_with(
        [(0, 1, WHITE), (0, 2, WHITE), (0, 3, WHITE), (0, 4, WHITE), (0, 7, WHITE)]
        + [(HEIGHT - 1, 0, BLACK), (HEIGHT - 1, 1, BLACK)]
    )
    own, opp, _ = masks(board)
    # Four connected (the lone piece at column 7 does not extend the run) vs two.
    assert ev._home_row_defense(own, opp, np.array([0])) == pytest.approx([2.0])


def test_territory_counts_non_enemy_neighbours(ev):
    own, opp, empty = masks(board_with([(5, 5, WHITE)]))
    assert ev._territory(own, opp, empty) == pytest.approx([8.0])
    # A corner piece has only three neighbours on the board.
    own, opp, empty = masks(board_with([(0, 0, WHITE)]))
    assert ev._territory(own, opp, empty) == pytest.approx([3.0])
    # An adjacent enemy removes exactly one of the eight.
    own, opp, empty = masks(board_with([(5, 5, WHITE), (5, 6, BLACK)]))
    # White: 8 - 1 enemy neighbour = 7. Black sees the same, so they cancel.
    assert ev._territory(own, opp, empty) == pytest.approx([0.0])


def test_centrality_prefers_the_middle(ev):
    # White in the centre, black in a corner: white is the more central.
    own, opp, _ = masks(board_with([(5, 6, WHITE), (0, 0, BLACK)]))
    assert ev._centrality(own, opp)[0] > 0
    own, opp, _ = masks(board_with([(0, 0, WHITE), (5, 6, BLACK)]))
    assert ev._centrality(own, opp)[0] < 0


def test_mobility_of_a_phalanx_grows_with_its_length(ev):
    """A group of n can advance up to n squares, so longer phalanxes score more."""
    one = board_with([(5, 5, WHITE)])
    three = board_with([(5, 3, WHITE), (5, 4, WHITE), (5, 5, WHITE)])
    m1 = ev._mobility(*masks(one))[0]
    m3 = ev._mobility(*masks(three))[0]
    assert m3 > m1
    # The lone piece moves one square in each of eight directions: mean 1, max 1.
    # (The paper's worked example says 1.125 here; see _mobility's docstring.)
    assert m1 == pytest.approx(2.0)


def test_evaluation_is_antisymmetric(ev):
    """The same position scored from either seat must give opposite values."""
    rng = np.random.default_rng(0)
    boards = rng.integers(0, 3, size=(32, N)).astype(np.int8)
    white = ev.score(boards, np.zeros(32, dtype=np.int64))
    black = ev.score(boards, np.ones(32, dtype=np.int64))
    assert white == pytest.approx(-black)


def test_batched_scoring_matches_one_at_a_time(ev):
    rng = np.random.default_rng(1)
    boards = rng.integers(0, 3, size=(16, N)).astype(np.int8)
    colors = rng.integers(0, 2, size=16)
    batched = ev.score(boards, colors)
    single = np.array([ev.score(boards[i : i + 1], colors[i : i + 1])[0] for i in range(16)])
    assert batched == pytest.approx(single)


def test_weights_are_applied(ev):
    """Zeroing a term must change the score of a position that term rates."""
    # Black is deliberately off row 0: a black piece on white's home rank would
    # cancel white's crossing and make the term zero for both weightings.
    board = board_with([(HEIGHT - 1, 3, WHITE), (5, 0, BLACK)])
    colors = np.zeros(1, dtype=np.int64)
    with_crossing = EpaminondasEvaluator().score(board, colors)[0]
    without = EpaminondasEvaluator(EpaminondasWeights(crossing=0.0)).score(board, colors)[0]
    assert with_crossing != without


# -- search -------------------------------------------------------------


def test_bucket_rounds_up_to_a_power_of_two():
    assert [_bucket(n) for n in (1, 2, 3, 5, 8, 9, 168)] == [1, 2, 4, 8, 8, 16, 256]


@pytest.fixture(scope="module")
def env():
    return pgx.make("epaminondas")


def test_engine_returns_a_legal_action(env):
    engine = NegamaxEngine(env, make_evaluator("epaminondas"), time_limit_s=1.0)
    state = jax.jit(env.init)(jax.random.PRNGKey(0))
    action, stats = engine.select_action(state)
    assert bool(state.legal_action_mask[action])
    assert stats.depth_reached >= 1
    assert stats.nodes > 0


def test_engine_completes_a_full_move_before_evaluating(env):
    """Depth is counted in moves, so depth 1 must search past the two stages.

    Epaminondas needs three actions per move. If depth were counted in pgx
    actions, a depth-1 search would stop after choosing a lead and evaluate a
    board that had not changed at all, making every lead look identical.
    """
    engine = NegamaxEngine(env, make_evaluator("epaminondas"), time_limit_s=5.0)
    state = jax.jit(env.init)(jax.random.PRNGKey(0))
    _, stats = engine.select_action(state)
    # One full move from the opening is far more than the ~14 lead choices.
    assert stats.nodes > 100


def position(board: np.ndarray, color: int = 0):
    """A playable pgx state with `board` on it and `color` to move.

    Note there is no "win in one move" position to test against: a crossing only
    decides the game at the *start* of the crosser's next turn, because the rules
    give the opponent a move to answer it. So a tactical test has to be about
    material, not about mate.
    """
    from pgx._src.games.epaminondas import Game

    game = Game()
    x = game.init()
    # GameState is a NamedTuple (_replace); the pgx State is a dataclass (replace).
    x = x._replace(
        board=jnp.asarray(board.reshape(N), dtype=jnp.int8),
        color=jnp.int32(color),
        stage=jnp.int32(0),
    )
    state = jax.jit(pgx.make("epaminondas").init)(jax.random.PRNGKey(0))
    return state.replace(
        _x=x,
        current_player=jnp.int32(color),
        legal_action_mask=game.legal_action_mask(x),
        terminated=jnp.bool_(False),
    )


def test_engine_takes_a_free_capture(env):
    """A phalanx of two can take a lone piece three squares away; it should.

    White's group at (5,4)-(5,5) may advance up to two squares along the rank,
    so landing on the black piece at (5,6) captures it. Every other move leaves
    material unchanged, and material is the dominant term, so the engine has to
    pick the capture even at depth 1.
    """
    board = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
    board[5, 4] = board[5, 5] = WHITE
    board[5, 6] = BLACK
    board[8, 0] = board[8, 1] = BLACK  # so black still has material elsewhere
    state = position(board, color=0)

    engine = NegamaxEngine(env, make_evaluator("epaminondas"), time_limit_s=3.0)
    step = jax.jit(env.step)
    before = int((np.asarray(state._x.board) == BLACK).sum())

    # Play the three actions of one full move.
    for _ in range(3):
        action, _ = engine.select_action(state)
        assert bool(state.legal_action_mask[action])
        state = step(state, jnp.int32(action), jax.random.PRNGKey(0))
        if int(state.current_player) != 0:
            break

    after = int((np.asarray(state._x.board) == BLACK).sum())
    assert after < before, "engine declined a free capture"


def test_unknown_game_is_rejected():
    with pytest.raises(ValueError, match="No negamax evaluation function"):
        make_evaluator("go_9x9")
