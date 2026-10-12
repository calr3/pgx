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

import os
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pgx
from pgx._src.games import ai_wars as W
from pgx._src.games import ai_wars_caicl as C
from pgx.ai_wars import DEFAULT_IFFS, AIWars, device_tables, run_battles

IDLE = "idle:\ngoto idle\n"


def open_map():
    return ["6" * W.COLS] + ["6" + "0" * (W.COLS - 2) + "6" for _ in range(W.ROWS - 2)] + ["6" * W.COLS]


def battle(script, other=IDLE, start=(10, 10, 2), other_start=(40, 28, 1), ticks=1, rows=None, node=(1, 1)):
    """Run `ticks` ticks of `script` (bug 0) against `other` on an open map."""
    lib = C.Library([script, other], ["a", "b"], default_iffs=DEFAULT_IFFS)
    T = C.lower(lib)
    eng = W.Engine(T.dims, 2)
    TT = device_tables(T)
    iffs = [lib.strings.ids[s] for s in DEFAULT_IFFS[:2]]
    st = eng.init(jax.random.PRNGKey(0), TT, [0, 1], W.map_grid(rows or open_map()),
                  np.array([start, other_start]), (node[0], node[1], True), iffs)
    tick = jax.jit(eng.tick)
    for _ in range(ticks):
        st = tick(TT, st)
    return st, lib


def test_moves_do_not_end_a_turn():
    # moving is not a "mega" command: ten moves fit in one turn of ten sub-clicks
    st, _ = battle("move forward\n" * 12)
    assert int(st.x[0]) == 20 and int(st.y[0]) == 10


def test_turning_ends_the_turn():
    st, _ = battle("turn right\nturn right\nturn right\n")
    assert int(st.h[0]) == 3  # one turn per tick


def test_block_bodies_cost_a_sub_click_each():
    # the header's condition is false, yet each body line still costs a sub-click
    body = "if random is 3 then\n" + "move forward\n" * 8 + "end if\nmove forward\nmove forward\nmove forward\n"
    st, _ = battle(body)
    assert int(st.x[0]) == 12  # 8 skipped lines + 2 real moves = 10 sub-clicks


def test_fixed_column_condition_needs_two_digits():
    # "if x coordinate is < N then ..." reads N from two fixed columns: a one-digit N
    # shifts the action text by one character and turns it into a no-op
    st, _ = battle("if x coordinate is < 50 then turn right\n")
    assert int(st.h[0]) == 3
    st, _ = battle("if x coordinate is < 5 then turn right\n", start=(1, 10, 2))
    assert int(st.h[0]) == 2
    st, _ = battle("if x coordinate is < 05 then turn right\n", start=(1, 10, 2))
    assert int(st.h[0]) == 3


def test_variable_width_decides_the_parse():
    # the same line works or not depending on how many digits #x_pos prints as
    script = "if y coordinate is = #x_pos then turn right\nidle:\ngoto idle\n"
    st, _ = battle(script, start=(12, 12, 2))
    assert int(st.h[0]) == 3
    st, _ = battle(script, start=(5, 5, 2))
    assert int(st.h[0]) == 2


def test_commands_held_in_variables():
    st, lib = battle("assign v1 turn right\n~v1\n")
    assert int(st.h[0]) == 3
    assert "turn right" in lib.strings.strings


def test_computed_goto():
    script = "goto h #cur_head\nh 1:\nturn left\nh 2:\nturn right\nturn right\n"
    st, _ = battle(script, start=(10, 10, 2))
    assert int(st.h[0]) == 3


def test_grenade_fuse_is_reset_at_launch():
    # "set grenade fuse" is overwritten by the launch: grenades always fly 3 cells (but
    # setting it while one is in flight does change that grenade's range)
    script = "set grenade fuse 9\nlaunch grenade\nidle:\ngoto idle\n"
    st, _ = battle(script, other_start=(15, 10, 1), ticks=3)
    assert int(st.damage[1]) == 0  # 5 cells away: out of reach
    st, _ = battle(script, other_start=(13, 10, 1), ticks=3)
    assert int(st.damage[1]) == 7  # direct hit, shield down


def test_template_strings():
    # demon.ai's "assign v3 = #cur_life": text plus a number, substituted later as words
    script = "assign v3 = #cur_life\nif value #cur_life > ~v3 then turn right\n"
    st, lib = battle(script)
    assert not any("unknown" in w for w in lib.programs[0].warnings)
    # "if value 0 > = 0 then turn right": the action becomes "then turn right", a no-op
    assert int(st.h[0]) == 2


def test_api():
    env = pgx.make("ai_wars")
    pgx.api_test(env, 1, use_key=False)


def test_queue_matches_ticks():
    # run_queue (lanes interleaving turns and battles) against plain tick-by-tick play
    env = AIWars()
    matchups = [(0, 1), (1, 0)] * 4
    res = run_battles(env, matchups, seed=11, lanes=3, rounds_iters=4)
    from pgx.ai_wars import make_jobs

    jobs = make_jobs(env.library, env.maps, matchups, 11)
    run = jax.jit(lambda k, p, g, s, nd, iff: env.engine.run(
        env.T, env.engine.init(k, env.T, p, g, s, (nd[0], nd[1], nd[2] > 0), iff)))
    for i in range(len(matchups)):
        j = {k: v[i] for k, v in jobs.items()}
        st = run(j["key"], j["prog"], jnp.asarray(env.maps.grids)[j["map"]], j["starts"], j["node"], j["iff"])
        assert int(st.winner) == res["winner"][i]
        assert int(st.turn) == res["ticks"][i]
        assert np.array_equal(np.asarray(st.score), res["score"][i])


SRC = os.environ.get("AIWARS_SRC", os.path.expanduser("~/calr3gh/aiwars_src"))


@pytest.mark.skipif(not (os.path.isdir(os.path.join(SRC, "plain")) and shutil.which(os.environ.get("NODE", "node"))),
                    reason="reference engine not fetched (tools/ai_wars/fetch.sh) or node missing")
def test_matches_reference_engine():
    tool = os.path.join(os.path.dirname(__file__), "..", "tools", "ai_wars", "conformance.py")
    r = subprocess.run([sys.executable, tool, "--n", "4", "--seed", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
