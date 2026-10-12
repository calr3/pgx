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

# The game's rules are tested in aiwars_jax (including against the reference engine);
# these tests cover the pgx wrapper.

import jax
import jax.numpy as jnp
import numpy as np

import pgx
from pgx.ai_wars import AIWars


def test_api():
    env = pgx.make("ai_wars")
    pgx.api_test(env, 1, use_key=False)


def test_steps_follow_the_engine():
    env = AIWars()
    state = jax.jit(env.init)(jax.random.PRNGKey(3))
    x = state._x
    step = jax.jit(env.step)
    tick = jax.jit(lambda s: env.arena.engine.tick(env.arena.T, s))
    for _ in range(40):
        state = step(state, jnp.int32(0))
        x = tick(x)
        if bool(state.terminated):
            break
    assert np.array_equal(np.asarray(state._x.grid), np.asarray(x.grid))
    assert np.array_equal(np.asarray(state._x.score), np.asarray(x.score))


def test_rewards_at_the_end():
    env = AIWars()
    state = jax.jit(env.init)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    while not bool(state.terminated):
        assert float(jnp.abs(state.rewards).sum()) == 0.0
        state = step(state, jnp.int32(0))
    w = int(state._x.winner)
    tie = int(state._x.score[0]) == int(state._x.score[1]) and int(state._x.dead[0]) == int(state._x.dead[1])
    expect = [0.0, 0.0] if tie or w < 0 else ([1.0, -1.0] if w == 0 else [-1.0, 1.0])
    assert state.rewards.tolist() == expect


def test_pairs():
    env = AIWars(pairs=[[1, 0]])
    state = jax.jit(env.init)(jax.random.PRNGKey(1))
    assert state._x.prog.tolist() == [1, 0]
