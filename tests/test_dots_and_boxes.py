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
from pgx._src.games.dots_and_boxes import (
    _BOX_LINES,
    BOXES,
    GRID,
    LINES,
    SIZE,
    Game,
    GameState,
    h_line,
    v_line,
)
from pgx.dots_and_boxes import DotsAndBoxes, line_name, parse_line

env = DotsAndBoxes()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)
game = Game()


def box(r, c):
    return r * SIZE + c


def sides(r, c):
    """Box (r, c)'s lines: top, bottom, left, right."""
    return [h_line(r, c), h_line(r + 1, c), v_line(r, c), v_line(r, c + 1)]


def play(lines, key=0):
    state = init(jax.random.PRNGKey(key))
    for line in lines:
        assert bool(state.legal_action_mask[line]), f"line {line} is not legal"
        state = step(state, jnp.int32(line))
    return state


def test_geometry():
    assert (SIZE, BOXES, LINES, GRID) == (6, 36, 84, 13)
    box_lines = np.asarray(_BOX_LINES)
    assert all(len(set(b)) == 4 for b in box_lines)
    # Every line is a side of one box (on the edge) or two (inside).
    borders = np.bincount(box_lines.ravel(), minlength=LINES)
    assert set(borders) == {1, 2}
    assert (borders == 1).sum() == 4 * SIZE
    assert (borders == 2).sum() == LINES - 4 * SIZE
    # Box (r, c)'s right side is box (r, c + 1)'s left, its bottom (r + 1, c)'s top.
    assert box_lines[box(2, 3)][3] == box_lines[box(2, 4)][2]
    assert box_lines[box(2, 3)][1] == box_lines[box(3, 3)][0]


def test_init():
    state = init(jax.random.PRNGKey(0))
    assert state.legal_action_mask.shape == (LINES,)
    assert state.legal_action_mask.all()
    assert state.observation.shape == (GRID, GRID, 8)
    assert env.observation_shape == (GRID, GRID, 8)
    assert env.num_actions == LINES
    assert not state.terminated
    assert int(state._x.color) == 0
    seats = {int(init(jax.random.PRNGKey(k)).current_player) for k in range(20)}
    assert seats == {0, 1}, "the seat that draws first is random"


def test_a_line_that_completes_nothing_passes_the_turn():
    state = init(jax.random.PRNGKey(0))
    first = int(state.current_player)
    state = step(state, jnp.int32(h_line(0, 0)))
    assert not bool(state.legal_action_mask[h_line(0, 0)])
    assert int(state.current_player) == 1 - first
    assert int(state._x.color) == 1


def test_completing_a_box_takes_it_and_moves_again():
    # Three sides by alternating players, then the fourth by whoever is next.
    top, bottom, left, right = sides(0, 0)
    state = play([top, bottom, left])
    mover, color = int(state.current_player), int(state._x.color)
    state = step(state, jnp.int32(right))
    assert int(state._x.owner[box(0, 0)]) == color
    assert int(state.current_player) == mover, "a completed box earns another move"
    assert int(state._x.color) == color
    assert (np.asarray(state._x.owner) >= 0).sum() == 1


def test_one_line_can_complete_two_boxes():
    # Boxes (1, 1) and (1, 2) share a vertical line; draw everything else first.
    shared = v_line(1, 2)
    others = [line for line in sides(1, 1) + sides(1, 2) if line != shared]
    state = play(others)
    color = int(state._x.color)
    state = step(state, jnp.int32(shared))
    owner = np.asarray(state._x.owner)
    assert owner[box(1, 1)] == color and owner[box(1, 2)] == color
    assert (owner >= 0).sum() == 2


def test_the_game_ends_when_every_line_is_drawn():
    state = init(jax.random.PRNGKey(3))
    for i, line in enumerate(np.random.default_rng(3).permutation(LINES)):
        assert not bool(state.terminated), f"over after {i} lines"
        state = step(state, jnp.int32(line))
    assert bool(state.terminated)
    assert (np.asarray(state._x.owner) >= 0).all(), "every box is taken"


def counts_by_seat(state):
    """Boxes owned by each seat, from the colours and whose seat plays colour 0."""
    counts = np.asarray(game.box_counts(state._x))
    color0 = int(state.current_player) if int(state._x.color) == 0 else 1 - int(state.current_player)
    return counts if color0 == 0 else counts[::-1]


def test_rewards_go_to_the_seat_with_more_boxes():
    outcomes = set()
    for k in range(24):
        state = init(jax.random.PRNGKey(k))
        rng = np.random.default_rng(k)
        while not bool(state.terminated):
            legal = np.flatnonzero(np.asarray(state.legal_action_mask))
            state = step(state, jnp.int32(rng.choice(legal)))
        mine = counts_by_seat(state)
        rewards = np.asarray(state.rewards)
        assert mine.sum() == BOXES
        expected = np.sign(mine - mine[::-1]).astype(np.float32)
        np.testing.assert_array_equal(rewards, expected, err_msg=str(mine))
        outcomes.add(tuple(rewards))
    assert (1.0, -1.0) in outcomes and (-1.0, 1.0) in outcomes, "both seats should win some"


def test_a_tie_is_a_draw():
    # Hand-built: all but one line drawn, 18 boxes each, the last line completes nothing new.
    lines = jnp.ones(LINES, jnp.bool_)
    owner = jnp.asarray([i % 2 for i in range(BOXES)], jnp.int32)
    x = GameState(color=jnp.int32(0), lines=lines, owner=owner, winner=jnp.int32(-1))
    assert bool(game.is_terminal(x))
    np.testing.assert_array_equal(game.rewards(x), [0.0, 0.0])


def test_the_last_line_decides_the_winner():
    # Every line but box (5, 5)'s right side is drawn; colour 0 holds 17 and colour 1 holds 18.
    last = sides(5, 5)[3]
    lines = jnp.ones(LINES, jnp.bool_).at[last].set(False)
    owner = jnp.asarray([0] * 17 + [1] * 18 + [-1], jnp.int32)
    x = GameState(color=jnp.int32(0), lines=lines, owner=owner, winner=jnp.int32(-1))
    x = game.step(x, jnp.int32(last))
    assert int(x.owner[box(5, 5)]) == 0
    assert bool(game.is_terminal(x))
    np.testing.assert_array_equal(game.rewards(x), [0.0, 0.0])  # 18-18
    x = GameState(color=jnp.int32(1), lines=lines, owner=owner, winner=jnp.int32(-1))
    x = game.step(x, jnp.int32(last))
    np.testing.assert_array_equal(game.rewards(x), [-1.0, 1.0])  # 17-19


def test_observe():
    top, bottom, left, right = sides(0, 0)
    state = play([top, bottom, left])  # three sides of box (0, 0), and a box to be had
    obs = np.asarray(state.observation)
    assert obs[0, 1, 0] == 1 and obs[2, 1, 0] == 1 and obs[1, 0, 0] == 1  # the lines drawn
    assert obs[1, 2, 0] == 0  # the right side is not
    assert obs[:, :, 0].sum() == 3
    assert obs[:, :, 3].sum() == LINES, "one slot per line"
    assert obs[1, 1, 4] == 1 and obs[:, :, 4].sum() == 1, "box (0, 0) has three sides"
    # Boxes (0, 1) and (1, 0) have one side each (shared with (0, 0)); nothing has two.
    assert obs[:, :, 5].sum() == 0

    state = step(state, jnp.int32(right))
    mover = int(state.current_player)
    mine = np.asarray(observe(state, jnp.int32(mover)))
    theirs = np.asarray(observe(state, jnp.int32(1 - mover)))
    assert mine[1, 1, 1] == 1 and mine[1, 1, 2] == 0, "the mover took box (0, 0)"
    assert theirs[1, 1, 1] == 0 and theirs[1, 1, 2] == 1, "and the other side sees it as theirs"
    assert mine[0, 0, 6] == 1 / BOXES and mine[0, 0, 7] == 0
    assert theirs[5, 5, 6] == 0 and theirs[5, 5, 7] == 1 / BOXES
    assert mine[:, :, 4].sum() == 0, "no box has three sides left"


def test_pretty():
    top, bottom, left, right = sides(0, 0)
    text = env.pretty(play([top, bottom, left, right]))
    rows = text.splitlines()
    assert len(rows) == 2 * SIZE + 1
    assert rows[0].startswith("+---+   +")
    assert rows[1].startswith("| 0 |") or rows[1].startswith("| 1 |")


def test_api():
    pgx.api_test(pgx.make("dots_and_boxes"), 3, use_key=False)


def test_lines_are_named_by_their_dots():
    assert line_name(h_line(0, 0)) == "a1-b1"
    assert line_name(v_line(0, 0)) == "a1-a2"
    assert line_name(h_line(6, 5)) == "f7-g7"
    assert line_name(v_line(5, 6)) == "g6-g7"
    for action in range(LINES):
        assert parse_line(line_name(action)) == action
    assert parse_line("b1 a1") == h_line(0, 0), "either order"
    for bad in ("a1-c1", "a1-b2", "a1", "h1-h2", "a7-a8", "a1-a1", "xx"):
        assert parse_line(bad) is None, bad


def test_the_greedy_baseline_takes_boxes_and_gives_none_away():
    greedy = pgx.make_baseline_model("dots_and_boxes_v0")

    def choice(state):
        logits, _ = greedy(state.observation[None])
        logits = jnp.where(state.legal_action_mask, logits[0], -jnp.inf)
        return np.flatnonzero(np.asarray(logits == logits.max()))

    top, bottom, left, right = sides(0, 0)
    # Box (0, 0) has three sides: the only best move is its fourth.
    assert list(choice(play([top, bottom, left]))) == [right]
    # Box (2, 2) has two sides: its other two would each leave it with three.
    state = play([sides(2, 2)[0], sides(2, 2)[1]])
    best = set(choice(state))
    assert sides(2, 2)[2] not in best and sides(2, 2)[3] not in best
    # Its neighbours above and below have one side each, so their lines are safe.
    assert len(best) == LINES - 2 - 2, "every other line gives nothing away"
