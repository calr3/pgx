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

"""A.I. Wars: Cybug battles between CAICL scripts.

The players of A.I. Wars write programs, not moves: a battle is decided entirely by the
two scripts (and the map and dice). So this env has no decisions to make - the action
space has a single action, and each `step` plays one tick of the battle. `init(key)`
draws a matchup from the env's library: two different programs, a map, start cells,
headings and the strategy node.

For evaluating scripts, use `run_battles` (or `Engine.run_queue`), which plays many
battles at once without the env wrapper and is much faster than stepping the env.

Rewards: +1 / -1 for the engine's winner (the living bug with the higher score) and
loser. The engine breaks a tie in score towards the lower slot; here an exact tie (both
alive or both dead, equal scores) is a draw instead, since otherwise slot 1 would win
every battle where nothing happens.
"""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import pgx.core as core
from pgx._src.games import ai_wars as W
from pgx._src.games import ai_wars_caicl as C
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

N_PLANES = 9
DEFAULT_IFFS = [str(40000 + s * 1001) for s in range(1, 11)]  # the Battle Setup's FFA codes

# Two small demonstration scripts, so that `pgx.make("ai_wars")` works out of the box.
DEMO_SCRIPTS = {
    "hunter": """name Hunter
author pgx
; turn towards the nearest enemy, shoot when it is in line, use flags to repair
main:
scan forward
if scan found enemy then fire weapon
if scan found enemy then goto main
if value #enemy_x > #x_pos then gosub east
if value #enemy_x < #x_pos then gosub west
if value #enemy_y > #y_pos then gosub south
if value #enemy_y < #y_pos then gosub north
move forward
if bump barrier then turn right
if damage is > 50 then attempt repairs
goto main
east:
if not facing east then turn right
return
west:
if not facing west then turn left
return
south:
if not facing south then turn right
return
north:
if not facing north then turn left
return
""",
    "wanderer": """name Wanderer
author pgx
; wander at random, lobbing grenades and missiles at anything ahead
loop:
generate random
if random is 1 then turn left
if random is 2 then turn right
long range scan
if scan found enemy then if missile ready then launch missile
scan forward
if scan found enemy then if grenade ready then launch grenade
move forward
move forward
if bump barrier then turn left
goto loop
""",
}


def demo_map() -> str:
    """An open arena with a few walls, flags and mines, and two 'Z' start cells."""
    rows = []
    for i in range(W.ROWS):
        r = []
        for j in range(W.COLS):
            if i in (0, W.ROWS - 1) or j in (0, W.COLS - 1):
                r.append("6")
            elif (i, j) in ((4, 4), (W.ROWS - 5, W.COLS - 5)):
                r.append("Z")
            elif j in (15, 29) and 8 <= i <= 23:
                r.append("2")
            elif (i, j) in ((15, 22), (16, 22), (5, 38), (26, 6)):
                r.append("F")
            elif (i, j) in ((10, 22), (21, 22)):
                r.append("9")
            else:
                r.append("0")
        rows.append("".join(r))
    return "\n".join(rows)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((W.ROWS, W.COLS, N_PLANES), dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(1, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: W.GameState = None  # type: ignore

    @property
    def env_id(self) -> core.EnvId:
        return "ai_wars"


class MapSet:
    """Maps as arrays for placing bugs inside jit: two bugs (slots 1 and 2)."""

    def __init__(self, texts: Sequence[str], names: Optional[Sequence[str]] = None):
        infos = [W.parse_map(t, n) for t, n in zip(texts, names or [""] * len(texts))]
        self.infos = infos
        zmax = max([len(m.z_cells) for m in infos] + [1])
        self.grids = np.stack([W.map_grid([r.replace("Z", "0") for r in m.rows]) for m in infos])
        self.fixed = np.full((len(infos), 2, 2), -1, np.int32)
        self.zcells = np.zeros((len(infos), zmax, 2), np.int32)
        self.nz = np.zeros(len(infos), np.int32)
        # node candidates: interior '0' cells, plus 'Z' cells that no bug takes
        self.node_ok = np.zeros((len(infos), W.ROWS, W.COLS), np.bool_)
        for k, m in enumerate(infos):
            for s in (1, 2):
                if s in m.fixed:
                    self.fixed[k, s - 1] = m.fixed[s]
            for i, z in enumerate(m.z_cells):
                self.zcells[k, i] = z
            self.nz[k] = len(m.z_cells)
            for i in range(1, W.ROWS - 1):
                for j in range(1, W.COLS - 1):
                    self.node_ok[k, i, j] = m.rows[i][j] in "0Z"

    def sample(self, key, m):
        """(starts [2, 3], node (x, y)) for map m, as the web Battle Setup places them."""
        kz, kh, kn = jax.random.split(key, 3)
        fixed = jnp.asarray(self.fixed)[m]
        nz = jnp.asarray(self.nz)[m]
        zc = jnp.asarray(self.zcells)[m]
        zmax = zc.shape[0]
        order = jnp.argsort(jnp.where(jnp.arange(zmax) < nz, jax.random.uniform(kz, (zmax,)), 2.0))
        has = fixed[:, 0] >= 0
        k = jnp.cumsum(~has) - (~has)  # Z cells taken by earlier slots
        z = zc[order[jnp.clip(k, 0, zmax - 1)]]
        dflt = jnp.array(W.DEFAULT_POS[:2], jnp.int32)
        pos = jnp.where(has[:, None], fixed, jnp.where((k < nz)[:, None], z, dflt))
        heads = jax.random.randint(kh, (2,), 1, 5)
        ok = jnp.asarray(self.node_ok)[m].at[pos[:, 1], pos[:, 0]].set(False)
        logits = jnp.where(ok.reshape(-1), 0.0, -jnp.inf)
        cell = jax.random.categorical(kn, logits)
        node = jnp.stack([cell % W.COLS, cell // W.COLS])
        return jnp.concatenate([pos, heads[:, None]], axis=1), node


class AIWars(core.Env):
    """A.I. Wars battles between two CAICL programs.

    Args:
        sources: CAICL program texts (plain, as the editor shows them). Default: two
            demonstration scripts.
        names: their names.
        maps: .map file texts. Default: a generated open arena.
        pairs: optional [P, 2] program indices; `init` draws one of these matchups
            (in either order as given) instead of any two different programs.
        rules: battle settings.
    """

    def __init__(self, sources: Optional[Sequence[str]] = None, names: Optional[Sequence[str]] = None,
                 maps: Optional[Sequence[str]] = None, pairs=None, rules: W.Rules = W.Rules()):
        super().__init__()
        if sources is None:
            names, sources = list(DEMO_SCRIPTS), list(DEMO_SCRIPTS.values())
        if len(sources) < 2 and pairs is None:
            raise ValueError("need at least two programs")
        self.library = C.Library(sources, names, settings=C.Settings(max_damage=rules.max_damage),
                                 default_iffs=DEFAULT_IFFS)
        self.tables = C.lower(self.library)
        self.T = device_tables(self.tables)
        self.engine = W.Engine(self.tables.dims, 2, rules)
        self.maps = MapSet(maps if maps is not None else [demo_map()])
        self.pairs = None if pairs is None else jnp.asarray(pairs, jnp.int32)
        self.iffs = jnp.array([self.library.strings.ids[s] for s in DEFAULT_IFFS[:2]], jnp.int32)

    @property
    def warnings(self):
        """Compile-time approximations, per program."""
        return dict(zip(self.tables.names, self.tables.warnings))

    def _init(self, key: PRNGKey) -> State:
        kp, km, ks, kb = jax.random.split(key, 4)
        P = len(self.library.programs)
        if self.pairs is None:
            a = jax.random.randint(kp, (), 0, P)
            b = (a + jax.random.randint(jax.random.fold_in(kp, 1), (), 1, P)) % P
            progs = jnp.stack([a, b])
        else:
            progs = self.pairs[jax.random.randint(kp, (), 0, self.pairs.shape[0])]
        m = jax.random.randint(km, (), 0, self.maps.grids.shape[0])
        starts, node = self.maps.sample(ks, m)
        x = self.engine.init(kb, self.T, progs, jnp.asarray(self.maps.grids)[m], starts,
                             (node[0], node[1], True), self.iffs)
        return State(_x=x)  # type: ignore

    def _step(self, state: core.State, action: Array, key) -> State:
        del action, key
        assert isinstance(state, State)
        x = self.engine.tick(self.T, state._x)
        terminated = ~x.running
        return state.replace(_x=x, terminated=terminated, rewards=jnp.where(terminated, rewards(x), 0.0))  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return observe(state._x, player_id)

    @property
    def id(self) -> core.EnvId:
        return "ai_wars"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def rewards(x: W.GameState) -> Array:
    w = x.winner
    tie = (x.score[0] == x.score[1]) & (x.dead[0] == x.dead[1])
    r = jnp.where(w == 0, jnp.float32([1.0, -1.0]), jnp.float32([-1.0, 1.0]))
    return jnp.where(tie | (w < 0), jnp.zeros(2, jnp.float32), r)


def observe(x: W.GameState, player_id) -> Array:
    """[ROWS, COLS, 9] planes from `player_id`'s side: walls, flags, mines, own bug,
    enemy bug, own missile/grenade, enemy missile/grenade, strategy node, then the own
    bug's heading (1..4)/4 on its cell."""
    g = W.Engine.board(x)
    me, other = player_id + 1, 2 - player_id
    yy, xx = jnp.mgrid[0:W.ROWS, 0:W.COLS]

    def at(px, py, on):
        return ((xx == px) & (yy == py) & on).astype(jnp.float32)

    i, o = player_id, 1 - player_id
    proj = lambda k: jnp.maximum(at(x.mis_x[k], x.mis_y[k], x.mis_on[k] == 1), at(x.gr_x[k], x.gr_y[k], x.gr_on[k] == 1))  # noqa: E731
    planes = [g == W.WALL, g == W.FLAG, g == W.MINE, g == me, g == other, proj(i), proj(o),
              at(x.node_x, x.node_y, x.node_on), (g == me) * x.h[i] / 4.0]
    return jnp.stack([p.astype(jnp.float32) for p in planes], axis=-1)


def device_tables(tables: C.Tables):
    return {k: jnp.asarray(v.astype(np.float32) if v.dtype == np.float64 else v)
            for k, v in tables.arrays.items()}


def make_jobs(library: C.Library, maps: MapSet, matchups, seed: int = 0):
    """Battle specs for `Engine.run_queue`: one battle per (prog_a, prog_b[, map]) in
    `matchups` (a random map when none is given), with start cells, headings and node
    drawn as the web Battle Setup does. Battle i gets key fold_in(PRNGKey(seed), i)."""
    matchups = [tuple(int(v) for v in mu) for mu in matchups]
    n = len(matchups)
    rng = np.random.default_rng(seed)
    iffs = [library.strings.ids[s] for s in DEFAULT_IFFS[:2]]
    base = jax.random.PRNGKey(seed)
    keys = np.asarray(jax.vmap(lambda i: jax.random.fold_in(base, i))(jnp.arange(n)))
    out = dict(key=keys.astype(np.uint32), prog=np.zeros((n, 2), np.int32), map=np.zeros(n, np.int32),
               starts=np.zeros((n, 2, 3), np.int32), node=np.zeros((n, 3), np.int32),
               iff=np.tile(np.array(iffs, np.int32), (n, 1)))
    for i, mu in enumerate(matchups):
        m = mu[2] if len(mu) > 2 else int(rng.integers(len(maps.infos)))
        rows, starts = W.place(maps.infos[m], 2, rng)
        cands = W.node_candidates(rows)
        node = cands[int(rng.integers(len(cands)))]
        out["prog"][i] = mu[:2]
        out["map"][i] = m
        out["starts"][i] = [[x, y, int(rng.integers(1, 5))] for x, y in starts]
        out["node"][i] = [node[0], node[1], 1]
    return {k: jnp.asarray(v) for k, v in out.items()}


def run_battles(env: AIWars, matchups, seed: int = 0, lanes: int = 256, rounds_iters: int = 16):
    """Play one battle per matchup (program indices into `env.library`, optionally with
    a map index), `lanes` at a time, and return numpy arrays: winner (0/1, -1 none),
    ticks, score [n, 2], dead [n, 2], damage [n, 2], reward [n, 2] (the env's rewards).

    Every distinct (n, lanes, rounds_iters) compiles once; pad `matchups` to a fixed
    size when calling this repeatedly."""
    jobs = make_jobs(env.library, env.maps, matchups, seed)
    lanes = min(lanes, len(matchups))
    res = {k: np.asarray(v) for k, v in jax.device_get(
        _runner(env, lanes, rounds_iters)(env.T, jobs, jnp.asarray(env.maps.grids))).items()}
    tie = (res["score"][:, 0] == res["score"][:, 1]) & (res["dead"][:, 0] == res["dead"][:, 1])
    rw = np.where(res["winner"][:, None] == np.arange(2), 1.0, -1.0)
    res["reward"] = np.where((tie | (res["winner"] < 0))[:, None], 0.0, rw).astype(np.float32)
    return res


_RUNNERS = {}


def _runner(env: AIWars, lanes: int, rounds_iters: int):
    key = (id(env), lanes, rounds_iters)
    if key not in _RUNNERS:
        _RUNNERS[key] = jax.jit(lambda T, jobs, maps: env.engine.run_queue(T, jobs, maps, lanes, rounds_iters))
    return _RUNNERS[key]
