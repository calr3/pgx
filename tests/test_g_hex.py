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
from pgx.g_hex import GHex, black, white

jnp.set_printoptions(linewidth=7*20+15)

env = GHex()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)

def test_init():
    key = jax.random.PRNGKey(1)
    state = init(key=key)
    assert state.current_player == 0
    assert (state.rewards == jnp.array([0.0, 0.0])).all()
    assert not state.terminated
    assert not state.truncated
    assert (state._x.tiles[0] == jnp.ones(10)).all()
    assert (state._x.tiles[1] == jnp.ones(10)).all()


def test_legal_action():
    key = jax.random.PRNGKey(0)
    _, sub_key = jax.random.split(key)

    state = init(sub_key)
    # fmt: off
    assert (state.legal_action_mask == jnp.array([
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
       ], dtype=bool)).all()
    # fmt:on
    assert (state.rewards == jnp.array([0.0, 0.0])).all()

    state = step(state, black(4, 5))
    # fmt: off
    assert (state.legal_action_mask == jnp.array([
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        False,False, False, False, False, False, False, False, False, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
       ], dtype=bool)).all()
    # fmt:on
    assert (state.rewards == jnp.array([0.0, 0.0])).all()

    state = step(state, black(10, 20))
    # fmt: off
    assert (state.legal_action_mask == jnp.array([
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
       False, False, False, False, False, False, False, False, False, False,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        False, False, False, False, False, False, False, False, False, False,
       ], dtype=bool)).all()
    # fmt:on
    assert (state.rewards == jnp.array([0.0, 0.0])).all()

    state = step(state, black(1, 0))
    print (state.legal_action_mask)
    print ("end of mask\n\n")
    # fmt: off
    assert (state.legal_action_mask == jnp.array([
       False, False, False, False, False, False, False, False, False, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
       False, False, False, False, False, False, False, False, False, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
        True,  True,  True,  True,  True,  True,  True,  True,  True, False,
       False, False, False, False, False, False, False, False, False, False,
      ], dtype=bool)).all()
    # fmt:on
    assert (state.rewards == jnp.array([0.0, 0.0])).all()


def test_win_check():
    key = jax.random.PRNGKey(6)

    # Arbitrary game in which color==1 wins.
    state = init(key)
    assert state.current_player == 0
    moves = [
      black(10, 0),
      white(10, 1),
      black(9, 2),
      white(9, 3),
      black(8, 4),
      white(8, 5),
      black(7, 6),
      white(7, 7),
      black(6, 8),
      white(6, 9),
      black(5, 10),
      white(5, 11),
      black(4, 12),
      white(4, 13),
      black(3, 14),
      white(3, 15),
      black(2, 16),
      white(2, 17),
      black(1, 18),
      white(1, 19),
    ]
    for move in moves:
        assert not state.terminated
        assert state.legal_action_mask[move]
        state = step(state, move)
    assert state.terminated
    assert state._x.winner == 1
    assert (state.rewards == jnp.array([-1.0, 1.0])).all()

    # Arbitrary game which ends in a draw (both players have the 7 next to triangle 0).
    state = init(key)
    assert state.current_player == 0
    moves = [
      black(10, 6),
      white(10, 1),
      black(9, 2),
      white(9, 3),
      black(8, 4),
      white(8, 5),
      black(7, 7),
      white(7, 20),
      black(6, 8),
      white(6, 9),
      black(5, 10),
      white(5, 11),
      black(4, 12),
      white(4, 13),
      black(3, 14),
      white(3, 15),
      black(2, 16),
      white(2, 17),
      black(1, 18),
      white(1, 19),
    ]
    for move in moves:
        assert not state.terminated
        assert state.legal_action_mask[move]
        state = step(state, move)
    assert state.terminated
    print (state._x.debug_me())
    print (state._x.debug_me()["board"])
    assert state._x.winner == -1
    assert (state.rewards == jnp.array([0.0, 0.0])).all()

    # Arbitrary game in which color==0 wins.
    state = init(key)
    assert state.current_player == 0
    moves = [
      black(10, 6),
      white(10, 20),
      black(9, 2),
      white(9, 3),
      black(8, 4),
      white(8, 5),
      black(7, 7),
      white(7, 1),
      black(6, 8),
      white(6, 9),
      black(5, 10),
      white(5, 11),
      black(4, 12),
      white(4, 13),
      black(3, 14),
      white(3, 15),
      black(2, 16),
      white(2, 17),
      black(1, 18),
      white(1, 19),
    ]
    for move in moves:
        assert not state.terminated
        assert state.legal_action_mask[move]
        state = step(state, move)
    assert state.terminated
    print (state._x.debug_me())
    print (state._x.debug_me()["board"])
    assert state._x.winner == 0
    assert (state.rewards == jnp.array([1.0, -1.0])).all()

def test_observe():
    key = jax.random.PRNGKey(0)

    state = init(key)
    assert (observe(state) == jnp.int32([
      [
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual down-triangle
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual down-triangle
      ],
      [
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
      ],
      [
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual down-triangle
        [0,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual up-triangle
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual down-triangle
      ],
      [
        [0,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual up-triangle
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual up-triangle
      ],
    ])).all()

    state = step(state, black(1, 15))
    assert (observe(state) == jnp.int32([
      [
        [0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual down-triangle
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual down-triangle
      ],
      [
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
      ],
      [
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [-1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # row 15
        [0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual down-triangle
        [0,  1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual up-triangle
        [0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual down-triangle
      ],
      [
        [0,  1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual up-triangle
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0, -1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        [0,  1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0], # virtual up-triangle
      ],
    ])).all()

    state = step(state, white(10, 13))
    #assert (observe(state) == jnp.int32([
    #  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, -10, 0, 1, 0, 0, 0, 0, 0],
    #  [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #])).all()


def test_api():
    import pgx

    environment = pgx.make("g_hex")
    pgx.api_test(environment, 3, use_key=False)
    pgx.api_test(environment, 3, use_key=True)
