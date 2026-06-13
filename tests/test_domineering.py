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
import pytest
from pgx.domineering import Domineering

env = Domineering()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def test_init():
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert state.current_player == 1
    assert (state.rewards == jnp.array([0.0, 0.0])).all()
    assert not state.terminated
    assert not state.truncated


def test_legal_action():
    key = jax.random.PRNGKey(0)
    _, sub_key = jax.random.split(key)

    state = init(sub_key)
    # fmt: off
    assert (state.legal_action_mask == jnp.array(
        [1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1])).all()
    # fmt:on
    assert (state.rewards == jnp.array([0.0, 0.0])).all()

    state = step(state, 1)
    # fmt: off
    assert (state.legal_action_mask == jnp.array(
        [1, 1, 1, 1, 1, 1, 1,
         0, 1, 1, 1, 1, 1, 1,
         0, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1])).all()
    # fmt:on
    assert (state.rewards == jnp.array([0.0, 0.0])).all()

    state = step(state, 4)
    # fmt: off
    assert (state.legal_action_mask == jnp.array(
        [0, 0, 0, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         0, 1, 1, 1, 1, 1, 1,
         0, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,])).all()
    # fmt:on
    assert (state.rewards == jnp.array([0.0, 0.0])).all()

    state = step(state, 51)
    # fmt: off
    assert (state.legal_action_mask == jnp.array(
        [1, 1, 1, 0, 0, 0, 1,
         0, 1, 1, 1, 1, 1, 1,
         0, 1, 1, 1, 1, 1, 0,
         1, 1, 1, 1, 1, 1, 0,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1,])).all()
    # fmt:on
    assert (state.rewards == jnp.array([0.0, 0.0])).all()


@pytest.mark.skip(reason="WIP: win-check move sequences are unfinished")
def test_win_check():
    key = jax.random.PRNGKey(6)

    # Arbitrary game in which V quickly loses.
    state = init(key)
    assert state.current_player == 0
    #moves = [8, 15, 24, 39, 40, 55, 56, 14, 10, 38, 26, 54, 42, 13, 58, 37, 52, 12, 35]
    #for move in [7, 14, 21, 35, :
    #    assert not state.terminated
    #    assert state.legal_action_mask[move]
    #    state = step(state, move)
    assert state.terminated
    assert state._x.winner == 0
    assert (state.rewards == jnp.array([1.0, -1.0])).all()

    return

    # Arbitrary game in which H quickly loses.
    state = init(key)
    assert state.current_player == 0
    moves = [0, 9, 24, 33, 2, 49, 26, 35, 4, 6, 28, 22, 36, 51, 44, 53, 46, 31, 10, 12, 54, 15, 18, 13, 62, 34]
    for move in moves:
        assert not state.terminated
        assert state.legal_action_mask[move]
        state = step(state, move)
    assert state.terminated
    assert state._x.winner == 1
    assert (state.rewards == jnp.array([-1.0, 1.0])).all()


def test_observe():
    key = jax.random.PRNGKey(0)
    state = init(key)

    assert (
        observe(state)
        == jnp.bool_(
            [
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
            ]
        )
    ).all()

    state = step(state, 1) # 1 = (0, 1)
    assert (
        observe(state)
        == jnp.bool_(
            [
                [[ True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False]],
                [[False, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False]],
                [[False, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False]],
                [[ True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False]],
                [[ True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False]],
                [[ True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False]],
                [[ True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False]],
                [[ True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False]],
            ]
        )
    ).all()

    state = step(state, 55) # 55 = (6, 7)
    obs = observe(state)
    assert (
        observe(state)
        == jnp.bool_(
            [
                [[ True,  True], [False, True], [False, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [False, True]],
                [[ True,  True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [ True, True], [False, True]],
            ]
        )
    ).all()


def test_api():
    import pgx

    environment = pgx.make("domineering")
    pgx.api_test(environment, 3, use_key=False)
    pgx.api_test(environment, 3, use_key=True)
