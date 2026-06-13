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
from pgx.heckmeck import Heckmeck

jnp.set_printoptions(linewidth=7*20+15)

env = Heckmeck()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)
key = jax.random.PRNGKey(69)

def test_init():
    state = init(key=key)
    assert state.current_player == 0
    assert (state.rewards == jnp.array([0.0, 0.0, 0.0])).all()
    assert not state.terminated
    assert not state.truncated
    assert (state._x.dice_rolled == jnp.int32([1, 1, 0, 0, 1, 1])).all()


def test_legal_action():
    key = jax.random.PRNGKey(0)
    _, sub_key = jax.random.split(key)

    state = init(sub_key)
    assert (state._x.dice_rolled == jnp.int32([0, 0, 0, 1, 1, 2])).all()
    # fmt: off
   # assert (state.observation[0][0] == jnp.zeros(29)).all()
    assert (state.legal_action_mask == jnp.array([
        False, False, False,  True,  True,  True,
        False, False, False, False, False, False,
        False,
       ], dtype=bool)).all()
    # fmt:on
    assert (state.rewards == jnp.float32([0, 0, 0])).all()

    return

    state = step(state, 0, key)  # Take the 1 worm and keep going.
    assert state._x.color == 0
    assert (state._x.grill == jnp.ones(17, dtype=jnp.bool_)).all()
    assert (state._x.stacks == jnp.zeros((3, 16))).all()
    assert (state._x.dice_rolled == jnp.int32([1, 2, 2, 1, 1, 0])).all()
    assert (state._x.dice_taken == jnp.int32([1, 0, 0, 0, 0, 0])).all()

    state = step(state, 2, key)  # Take the 2 twos and keep going.
    assert state._x.color == 0
    assert (state._x.grill == jnp.ones(17, dtype=jnp.bool_)).all()
    assert (state._x.stacks == jnp.zeros((3, 16))).all()
    assert (state._x.dice_rolled == jnp.int32([1, 1, 2, 1, 0, 0])).all()
    assert (state._x.dice_taken == jnp.int32([1, 0, 2, 0, 0, 0])).all()

    state = step(state, 1, jax.random.PRNGKey(12))  # Take the 1 and keep going (with a handy key that rolls a lot of 5s).
    assert state._x.color == 0
    assert (state._x.grill == jnp.ones(17, dtype=jnp.bool_)).all()
    assert (state._x.stacks == jnp.zeros((3, 16))).all()
    assert (state._x.dice_rolled == jnp.int32([0, 0, 0, 0, 0, 4])).all()
    assert (state._x.dice_taken == jnp.int32([1, 1, 2, 0, 0, 0])).all()

    state = step(state, 11, key)  # Take the fives and stop, making a total of 5+2*2+1+4*5 = 30 (tile #10).
    assert state._x.color == 1
    assert (state._x.grill == jnp.ones(17, dtype=jnp.bool_).at[10].set(False)).all()
    assert (state._x.stacks == jnp.zeros((3, 16)).at[0, 0].set(10)).all()
    assert (state._x.dice_rolled == jnp.int32([2, 2, 2, 1, 1, 0])).all()
    assert (state._x.dice_taken == jnp.zeros(6)).all()

    state = step(state, 12, key)  # Bust P1 immediately.
    assert (state._x.stacks == jnp.zeros((3, 16)).at[0, 0].set(10)).all()
    state = step(state, 12, key)  # Bust P2 immediately.
    assert (state._x.stacks == jnp.zeros((3, 16)).at[0, 0].set(10)).all()

    state = step(state, 12, key)  # Bust P0, who had a non-empty stack. The 30 goes back and now the 36 gets flipped.
    assert state._x.color == 1
    assert (state._x.grill == jnp.ones(17, dtype=jnp.bool_).at[16].set(False)).all()
    assert (state._x.stacks == jnp.zeros((3, 16))).all()
    #assert (state._x.dice_rolled == jnp.int32([2, 2, 2, 1, 1, 0])).all()
    assert (state._x.dice_taken == jnp.zeros(6)).all()


def test_bust():
    key = jax.random.PRNGKey(0)
    _, sub_key = jax.random.split(key)

    state = init(sub_key)
    assert (state._x.dice_rolled == jnp.int32([0, 0, 0, 1, 1, 2])).all()
    assert (state.rewards == jnp.float32([0.0, 0.0, 0.0])).all()

    state = step(state, 12, key)  # Bust myself while stack is empty (TODO: not legal)
    assert state._x.color == 1
    assert (state._x.grill == jnp.ones(9, dtype=jnp.bool_)).all()
    assert (state._x.stacks == jnp.zeros((3, 8))).all()
    assert (state._x.dice_rolled == jnp.int32([0, 1, 2, 1, 0, 0])).all()
    assert (state._x.dice_taken == jnp.int32([0, 0, 0, 0, 0, 0])).all()


    return  # TODO
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
    assert (state.rewards == jnp.array([0.0, 0.0, 0.0])).all()

    state = step(state, 1, key)
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
    assert (state.rewards == jnp.array([0.0, 0.0, 0.0])).all()

    state = step(state, black(1, 0), key)
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
    assert (state.rewards == jnp.array([0.0, 0.0, 0.0])).all()


def test_win_check():
    key = jax.random.PRNGKey(6)

    # Arbitrary game in which color==1 wins.
    state = init(key)
    assert state.current_player == 0
    return  # TODO
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
        state = step(state, move, key)
    assert state.terminated
    assert state._x.winner == 1
    assert (state.rewards == jnp.array([-1.0, 1.0, 0.0])).all()

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
        state = step(state, move, key)
    assert state.terminated
    print (state._x.debug_me())
    print (state._x.debug_me()["board"])
    assert state._x.winner == -1
    assert (state.rewards == jnp.array([0.0, 0.0, 0.0])).all()

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
        state = step(state, move, key)
    assert state.terminated
    print (state._x.debug_me())
    print (state._x.debug_me()["board"])
    assert state._x.winner == 0
    assert (state.rewards == jnp.array([1.0, -1.0, 0.0])).all()

def test_observe():
    key = jax.random.PRNGKey(0)

    state = init(key)
    observe(state)
    return  # TODO

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

    state = step(state, black(1, 15), key)
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

    state = step(state, white(10, 13), key)
    #assert (observe(state) == jnp.int32([
    #  [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, -10, 0, 1, 0, 0, 0, 0, 0],
    #  [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #])).all()


def test_api():
    import pgx

    environment = pgx.make("heckmeck")
    #pgx.api_test(environment, 3, use_key=False)  # TODO
    #pgx.api_test(environment, 3, use_key=True)   # TODO
