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
from pgx.pig import Pig, PLAYER_COUNT, TARGET
from pgx._src.games.pig import Game, GameState

env = Pig()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)

ACTION_STOP = jnp.int32(0)
ACTION_CONTINUE = jnp.int32(1)

game = Game()
step_key = jax.random.PRNGKey(42)


def make_x(color=0, totals=None, turn_total=0, last_roll=3, winner=-1):
    """Create a controlled GameState for isolated unit testing."""
    if totals is None:
        totals = [0, 0]
    return GameState(
        color=jnp.int32(color),
        totals=jnp.int32(totals),
        turn_total=jnp.int32(turn_total),
        last_roll=jnp.int32(last_roll),
        winner=jnp.int32(winner),
    )


def test_init():
    key = jax.random.PRNGKey(1921882)
    state = init(key=key)
    assert state.current_player == 0
    assert (state.rewards == jnp.array([0.0, 0.0])).all()
    assert not state.terminated
    assert not state.truncated
    assert (state._x.totals == jnp.int32([0, 0])).all()
    assert state._x.turn_total == 3  # init sets turn_total = last_roll = first_roll
    assert state._x.last_roll == 3  # value produced by PRNGKey(1921882)
    assert (state.legal_action_mask == jnp.bool_([1, 1])).all()


def test_legal_action():
    # Stop is always legal; continue is legal only when last_roll > 1.
    for roll in range(1, 7):
        mask = game.legal_action_mask(make_x(last_roll=roll))
        assert bool(mask[0]), f"stop should always be legal (last_roll={roll})"
        assert bool(mask[1]) == (roll > 1), f"continue legal iff last_roll > 1, got roll={roll}"


def test_stop():
    # Stopping should bank turn_total into totals and switch the current player.
    x = make_x(color=0, totals=[50, 20], turn_total=10, last_roll=3)
    x2 = game.step(x, ACTION_STOP, step_key)
    assert x2.color == 1, "stopping should switch to the other player"
    assert x2.totals[0] == 60, "turn_total (10) should be banked into player 0's total (50)"
    assert x2.totals[1] == 20, "other player's total should be unchanged"
    assert x2.turn_total == x2.last_roll, "new player's turn_total starts as their first roll"
    assert 1 <= x2.last_roll <= 6, "a new die should be rolled for the next player"

    # Stopping from player 1's turn should switch back to player 0.
    x = make_x(color=1, totals=[30, 40], turn_total=7, last_roll=5)
    x2 = game.step(x, ACTION_STOP, step_key)
    assert x2.color == 0
    assert x2.totals[1] == 47


def test_continue():
    # Continuing should add last_roll to turn_total without switching player.
    x = make_x(color=0, totals=[20, 30], turn_total=5, last_roll=4)
    x2 = game.step(x, ACTION_CONTINUE, step_key)
    assert x2.color == 0, "continuing should not switch player"
    assert (x2.totals == jnp.int32([20, 30])).all(), "continuing should not change totals"
    assert x2.turn_total == 5 + x2.last_roll, "turn_total should be old_turn_total + new_roll"
    assert 1 <= x2.last_roll <= 6, "a new die should be rolled"


def test_pig_out():
    # Rolling a 1 forces stop: turn_total is lost and the player switches.
    x = make_x(color=0, totals=[30, 40], turn_total=15, last_roll=1)

    mask = game.legal_action_mask(x)
    assert bool(mask[0]), "stop must be legal on a pig out"
    assert not bool(mask[1]), "continue must be illegal when last_roll == 1"

    x2 = game.step(x, ACTION_STOP, step_key)
    assert x2.color == 1, "pig out should switch player"
    assert x2.totals[0] == 30, "turn_total should be lost, not banked"
    assert x2.totals[1] == 40, "other player's total should be unchanged"
    assert x2.turn_total == x2.last_roll, "new player's turn_total starts as their first roll"


def test_win():
    # Stopping when turn_total + total reaches TARGET should record a winner.
    x = make_x(color=0, totals=[95, 50], turn_total=5, last_roll=3)
    x2 = game.step(x, ACTION_STOP, step_key)
    assert x2.winner == 0, "player 0 should win when banking 5 brings total from 95 to 100"
    assert game.is_terminal(x2)

    # Stopping exactly at TARGET should also win (boundary check).
    x = make_x(color=0, totals=[99, 0], turn_total=1, last_roll=2)
    x2 = game.step(x, ACTION_STOP, step_key)
    assert x2.winner == 0, "stopping at exactly TARGET should win"

    # Stopping below TARGET should not win.
    x = make_x(color=0, totals=[80, 50], turn_total=5, last_roll=3)
    x2 = game.step(x, ACTION_STOP, step_key)
    assert x2.winner == -1, "no winner when banked score is below TARGET (80 + 5 = 85)"
    assert not game.is_terminal(x2)

    # Player 1 can also win.
    x = make_x(color=1, totals=[50, 97], turn_total=3, last_roll=4)
    x2 = game.step(x, ACTION_STOP, step_key)
    assert x2.winner == 1, "player 1 should win when banking reaches TARGET"

    # Stopping one short of TARGET should not win even if the incoming roll for
    # the next player would have pushed the total over (PRNGKey(2) rolls a 6).
    # A buggy winner check of `totals + turn_total + next_roll >= TARGET` would
    # incorrectly declare a winner here (93 + 6 + 6 = 105), but the correct
    # check `totals + turn_total >= TARGET` should not (93 + 6 = 99 < 100).
    x = make_x(color=0, totals=[93, 50], turn_total=6, last_roll=3)
    x2 = game.step(x, ACTION_STOP, jax.random.PRNGKey(2))  # next_roll == 6
    assert x2.winner == -1, "winner should not be set when banked total is below TARGET"


def test_rewards():
    assert (game.rewards(make_x(winner=-1)) == jnp.float32([0.0, 0.0])).all()
    r0 = game.rewards(make_x(winner=0))
    assert r0[0] == 1.0 and r0[1] == -1.0
    r1 = game.rewards(make_x(winner=1))
    assert r1[0] == -1.0 and r1[1] == 1.0


def test_observe():
    # Observation: [current_total, opponent_total, turn_total x2, last_roll x2].
    x = make_x(color=0, totals=[30, 50], turn_total=7, last_roll=4)
    obs = game.observe(x)
    assert obs[0] == 30, "first element should be current player's total"
    assert obs[1] == 50, "second element should be opponent's total"
    assert (obs[2:4] == 7).all(), "elements 2-3 should be turn_total tiled"
    assert (obs[4:6] == 4).all(), "elements 4-5 should be last_roll tiled"

    # From player 1's perspective the totals should be swapped.
    x = make_x(color=1, totals=[30, 50], turn_total=7, last_roll=4)
    obs = game.observe(x)
    assert obs[0] == 50, "current player (1) should see their own total first"
    assert obs[1] == 30


def test_api():
    import pgx
    environment = pgx.make("pig")
    # pgx.api_test(environment, 3, use_key=False)  # TODO
    # pgx.api_test(environment, 3, use_key=True)   # TODO
