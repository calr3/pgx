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

"""A.I. Wars (The Insect Mind, John A. Reder, Tactical Neuronics) battle engine.

Cybugs - programmed robot insects - fight on a 45 x 32 grid. Nobody steers them during
a battle: each runs a script in CAICL, the game's scripting language, and the battle is
a pure function of the scripts, the map and the random draws. This module is a port of
the web edition's engine (`engine_core.js` + `engine_caicl.js`, itself a port of the VB6
original), with CAICL compiled ahead of time by `ai_wars_caicl`.

Rules, in brief (every number is the engine default):

- A battle runs in ticks (2000 at most). In each tick every bug, in slot order, gets a
  turn: its shield heats (1 a tick, 2 when cloaked; overheating at 30 drops the shield
  and freezes the bug for 10 ticks), then its script runs until it has used 10
  sub-clicks. A line costs one sub-click and a scan three; a "mega" command (turning,
  firing, shield and cloak changes, repairs, launches, data link) ends the turn. Block
  `if` bodies are not skipped: each body line runs, prefixed with its header's
  conditions, and costs its sub-click whether or not they hold.
- After the script: standing on the strategy node scores 10 and gives 1 fuel; a cloak
  costs 10 fuel and 2 ammo; the bug's missile flies 4 cells and its grenade 2.
- Moving costs `burn` fuel (1 at the start; every hit raises it). Walking onto a flag
  restores a damaged bug (damage 0, +30 ammo, +350 fuel, +50 score) but overloads an
  undamaged one (damage 5). Mines do 5 damage. A bug dies at 10 damage, leaving a flag.
- Weapons: the gun (1 ammo, shield down, 5-cell line, 5..1 damage by range), missiles
  (10 ammo, 300 fuel, 3x3 blast), grenades (5 ammo, 200 fuel, flies 3 cells), energy
  discharge (hits the 8 neighbours, 1 damage to self), mines, self-destruct. A raised
  shield reduces damage taken; launching with it up usually misfires on yourself.
- The battle ends when at most one bug (or one IFF team) is left, at the tick limit,
  or after 101 ticks in which nobody moved. The winner is the living bug with the
  highest score, ties going to the lower slot.

Faithfulness: the port reproduces the reference engine tick for tick, including its
quirks (moves do not end a turn; `set grenade fuse` is overwritten to 3 at launch; dead
bugs still score on the strategy node). Known differences, all rare:

- Randomness: the reference uses `Math.random()`. Here the k-th draw of bug b's turn in
  tick t is `bits(fold_in(fold_in(key, t), b))[k] >> 8` over 2^24, so a battle is a
  pure function of its key. Feeding those draws to the reference (see the conformance
  harness) reproduces it exactly.
- Arithmetic is float32 where the reference uses doubles (only `cmath` can tell).
- `data link command` overwrites the target's next *executable* line, where the
  reference overwrites the line its pointer is on, which can be a blank line. A bug
  remembers the last `PATCHES` overwritten lines; older ones revert.
- Block nesting deeper than `NEST_DEPTH` disables the excess bodies (the reference
  degrades past 50 in its own way).
- See `ai_wars_caicl` for the compile-time approximations.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pgx._src.games import ai_wars_caicl as C

COLS, ROWS = 45, 32
OPEN, WALL, FLAG, MINE = 0, 20, 30, 40
MAX_SLOTS = 10
NEST_DEPTH = 50
GOSUB_DEPTH = 5
DRAWS_PER_TURN = 32
MAX_ITERS_PER_TURN = 600  # loop iterations per turn; only header/end-if loops get near it
CONDS_PER_ITER = 3  # conditions of one header or line checked per iteration
EXPR_PER_ITER = 3  # math terms or cmath RPN steps per iteration
EXPR_STACK = 16
HIVE_ROWS = 16  # distinct IFFs a battle can see
PATCHES = 8  # distinct lines of one bug a data link can overwrite before the oldest is lost

# Headings 1..4 = N, E, S, W; index 0 unused.
DX = np.array([0, 0, 1, 0, -1], np.int32)
DY = np.array([0, -1, 0, 1, 0], np.int32)
# 8 neighbours n, ne, e, se, s, sw, w, nw, then the centre (blast order)
NBX = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0], np.int32)
NBY = np.array([-1, -1, 0, 1, 1, 1, 0, -1, 0], np.int32)
# scan position 1..8 offsets: N NE E SE S SW W NW
PX = np.array([0, 0, 1, 1, 0, -1, -1, -1, 0], np.int32)
PY = np.array([0, -1, -1, 0, 1, 1, 0, -1, -1], np.int32)
# left / right of heading
LX = np.array([0, -1, 0, 1, 0], np.int32)
LY = np.array([0, 0, -1, 0, 1], np.int32)
RX = np.array([0, 1, 0, -1, 0], np.int32)
RY = np.array([0, 0, 1, 0, -1], np.int32)

DEFAULT_POS = [(4, 3), (40, 27), (4, 27), (40, 3), (22, 1), (22, 30), (1, 15), (43, 16),
               (18, 16), (26, 15)]


class Rules(NamedTuple):
    """Battle settings (the web edition's Battle Setup options)."""

    max_turns: int = 2000
    max_damage: int = 10
    start_fuel: int = 2500
    start_ammo: int = 30
    overheat_limit: int = 30
    shield_overheat_disabled: bool = False
    smart_clicks_disabled: bool = False
    iff_disabled: bool = False
    end_if_no_move: bool = True
    battle_ends_queen_death: bool = False


class GameState(NamedTuple):
    key: jax.Array
    turn: jax.Array  # ticks completed
    running: jax.Array
    movecount: jax.Array
    tcnt: jax.Array
    winner: jax.Array  # bug index, -1 while running or if nobody qualifies
    node_x: jax.Array
    node_y: jax.Array
    node_on: jax.Array
    grid: jax.Array  # [ROWS+2, COLS+2]: 0 open, 1..10 bug slot, 20 wall, 30 flag, 40 mine,
    #                  with a border of wall around the map (see `board`)
    prog: jax.Array  # [S] program index in the tables
    x: jax.Array
    y: jax.Array
    h: jax.Array
    damage: jax.Array
    fuel: jax.Array
    ammo: jax.Array
    score: jax.Array
    burn: jax.Array
    fuel_rate: jax.Array
    shield: jax.Array
    overheat: jax.Array
    penalty: jax.Array
    cloak: jax.Array
    mine: jax.Array
    dead: jax.Array
    fstat: jax.Array
    queen: jax.Array
    team: jax.Array
    iff: jax.Array  # string id
    scan: jax.Array  # index into ai_wars_caicl.SCAN_RESULTS
    random: jax.Array
    barrier: jax.Array
    mem: jax.Array  # compact line pointer
    nest_level: jax.Array
    nest: jax.Array  # [S, NEST_DEPTH] global rows of the open block headers
    curnest: jax.Array
    subnest: jax.Array  # [S, GOSUB_DEPTH]
    mis_on: jax.Array
    mis_x: jax.Array
    mis_y: jax.Array
    mis_h: jax.Array
    gr_on: jax.Array
    gr_x: jax.Array
    gr_y: jax.Array
    gr_h: jax.Array
    gr_d: jax.Array
    gr_t: jax.Array
    vnum: jax.Array  # [S, V] variable values
    vsid: jax.Array  # [S, V] string id, or -1 for a number
    hkey: jax.Array  # [HIVE_ROWS] IFF string id owning each hive row, -1 free
    hnum: jax.Array  # [HIVE_ROWS, H]
    hsid: jax.Array
    patch_line: jax.Array  # [S, PATCHES] compact lines overwritten by a data link, or -1
    patch_row: jax.Array  # [S, PATCHES] the injected line now standing there
    patch_n: jax.Array  # [S] patches made so far (slots are reused round-robin)


i32 = jnp.int32
f32 = jnp.float32


def _pow10():
    return jnp.array([10.0 ** k for k in range(12)], f32)


def num_width(v):
    """Characters in String(v) for an integer-valued v."""
    a = jnp.abs(jnp.trunc(v))
    digits = 1 + jnp.sum(a[..., None] >= _pow10()[1:10], axis=-1)
    return digits + (v < 0)


def lead_digits(v, m):
    """parseInt of the first m characters of String(v) (v an integer)."""
    w = num_width(v)
    p = _pow10()[jnp.clip(w - m, 0, 11)]
    a = jnp.floor(jnp.abs(v) / p)
    return jnp.where(v < 0, jnp.where(m <= 1, 0.0, -a), a)


class Engine:
    """A battle engine for `n_bugs` Cybugs whose programs come from `tables`.

    Every method is pure and jittable; the tables are passed in (as a dict of
    arrays) so that one compiled engine serves any set of programs of the same shape.
    """

    def __init__(self, dims: dict, n_bugs: int, rules: Rules = Rules()):
        self.S = n_bugs
        self.rules = rules
        self.dims = dict(dims)
        self.R = int(dims["rows"])
        self.K = int(dims["keys"])
        self.CH = int(dims["choices"])
        self.MC = int(dims["max_conds"])
        self.MT = int(dims["max_terms"])
        self.MR = int(dims["max_rpn"])
        self.ML = int(dims["max_labels"])
        self.NV = int(dims["vars"])
        self.NH = int(dims["hives"])

    # ------------------------------------------------------------------ init
    def init(self, key, T, progs, grid, starts, node, iffs, teams=None):
        """Start a battle.

        progs: [S] program indices into T. grid: [ROWS, COLS] cell values (bugs not
        yet placed). starts: [S, 3] (x, y, heading). node: (x, y, enabled).
        iffs: [S] string ids of the starting IFF codes.
        """
        S = self.S
        r = self.rules
        progs = jnp.asarray(progs, i32)
        starts = jnp.asarray(starts, i32)
        x, y, h = starts[:, 0], starts[:, 1], starts[:, 2]
        grid = jnp.pad(jnp.asarray(grid, i32), 1, constant_values=WALL)
        grid = grid.at[y + 1, x + 1].set(jnp.arange(1, S + 1, dtype=i32))
        iffs = jnp.asarray(iffs, i32)
        teams = jnp.zeros(S, i32) if teams is None else jnp.asarray(teams, i32)
        # one hive row per distinct starting IFF
        hkey = jnp.full(HIVE_ROWS, -1, i32)

        def add(hk, iff):
            present = jnp.any(hk == iff)
            free = jnp.argmax(hk < 0)
            return jnp.where(present, hk, hk.at[free].set(iff)), None

        hkey, _ = jax.lax.scan(add, hkey, iffs)
        z = jnp.zeros(S, i32)
        return GameState(
            key=key, turn=i32(0), running=jnp.bool_(True), movecount=i32(0), tcnt=i32(0),
            winner=i32(-1), node_x=i32(node[0]), node_y=i32(node[1]),
            node_on=jnp.bool_(node[2]), grid=grid, prog=progs, x=x, y=y, h=h,
            damage=z, fuel=z + r.start_fuel, ammo=z + r.start_ammo, score=z, burn=z + 1,
            fuel_rate=z + 1, shield=z, overheat=z, penalty=z, cloak=z, mine=z, dead=z,
            fstat=z, queen=z, team=teams, iff=iffs, scan=z, random=z, barrier=z, mem=z,
            nest_level=z, nest=jnp.zeros((S, NEST_DEPTH), i32), curnest=z,
            subnest=jnp.zeros((S, GOSUB_DEPTH), i32), mis_on=z, mis_x=z, mis_y=z, mis_h=z,
            gr_on=z, gr_x=z, gr_y=z, gr_h=z, gr_d=z, gr_t=z,
            vnum=jnp.zeros((S, self.NV), f32), vsid=jnp.full((S, self.NV), -1, i32),
            hkey=hkey, hnum=jnp.zeros((HIVE_ROWS, self.NH), f32),
            hsid=jnp.full((HIVE_ROWS, self.NH), -1, i32),
            patch_line=jnp.full((S, PATCHES), -1, i32), patch_row=jnp.zeros((S, PATCHES), i32),
            patch_n=z,
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def mg(st, x, y):
        # anything off the map lands on the wall border
        return st.grid[jnp.clip(y, -1, ROWS) + 1, jnp.clip(x, -1, COLS) + 1]

    @staticmethod
    def mgset(grid, x, y, v, cond=True):
        inb = (x >= 0) & (x < COLS) & (y >= 0) & (y < ROWS) & cond
        yc, xc = jnp.clip(y, 0, ROWS - 1) + 1, jnp.clip(x, 0, COLS - 1) + 1
        return grid.at[yc, xc].set(jnp.where(inb, v, grid[yc, xc]))

    @staticmethod
    def board(st):
        """The map without its wall border: [ROWS, COLS]."""
        return st.grid[..., 1:-1, 1:-1]

    @staticmethod
    def set1(arr, b, v, cond=True):
        return arr.at[b].set(jnp.where(cond, v, arr[b]))

    def hive_row(self, st, b):
        return jnp.argmax(st.hkey == st.iff[b])

    def tokens(self, st, b):
        """Values of the run-time #tokens for bug b (ai_wars_caicl.TOKENS, then INTERNAL)."""
        S = self.S
        alive = st.dead == 0
        mine_iff = st.iff[b]
        d = jnp.abs(st.x - st.x[b]) + jnp.abs(st.y - st.y[b])
        other = (jnp.arange(S) != b) & alive

        def nearest(friend):
            cand = other & ((st.iff == mine_iff) == friend)
            dd = jnp.where(cand, d, 99999)
            j = jnp.argmin(dd)  # first minimum = lowest slot
            ok = jnp.any(cand)
            return [jnp.where(ok, v[j], 0) for v in (st.damage, st.h, st.x, st.y)]

        e = nearest(False)
        fr = nearest(True)
        scan_tok = jnp.where(st.scan[b] >= 6, 0, st.scan[b])
        ammo, fuel = st.ammo[b], st.fuel[b]
        vals = [fuel, ammo, st.damage[b], st.h[b], st.score[b],
                jnp.where(st.node_on, st.node_x, 0), jnp.where(st.node_on, st.node_y, 0),
                st.random[b], st.burn[b], jnp.sum(alive), scan_tok, st.shield[b],
                st.overheat[b], st.cloak[b], st.x[b], st.y[b]] + e + fr + [self.inbound(st, b)]
        vals += [st.barrier[b], (st.mis_on[b] == 0) & (ammo >= 10) & (fuel >= 300),
                 (st.gr_on[b] == 0) & (ammo >= 5) & (fuel >= 200), st.scan[b]]
        return jnp.stack([jnp.asarray(v, f32) for v in vals])

    def inbound(self, st, b):
        x, y = st.x[b], st.y[b]

        def dirs(on, px, py, ph):
            return jnp.where(on & (px == x) & (py > y) & (ph == 1), 3,
                   jnp.where(on & (px == x) & (py < y) & (ph == 3), 1,
                   jnp.where(on & (py == y) & (px > x) & (ph == 4), 2,
                   jnp.where(on & (py == y) & (px < x) & (ph == 2), 4, 0))))

        m = dirs(st.mis_on == 1, st.mis_x, st.mis_y, st.mis_h)
        g = dirs(st.gr_on == 1, st.gr_x, st.gr_y, st.gr_h)
        per = jnp.where(m > 0, m, g)  # per slot: missile checked before grenade
        first = jnp.argmax(per > 0)
        return jnp.where(jnp.any(per > 0), per[first], 0)

    # ---------------------------------------------------------- operands
    def values(self, st, b):
        """Everything an operand can read, as one vector: tokens, variables, hive."""
        hr = self.hive_row(st, b)
        tok = self.tokens(st, b)
        vals = jnp.concatenate([tok, st.vnum[b], st.hnum[hr]])
        sids = jnp.concatenate([jnp.full(C.N_TOKENS, -1, i32), st.vsid[b], st.hsid[hr]])
        return vals, sids

    def _slot(self, src, idx):
        off = jnp.where(src == C.SRC_VAR, C.N_TOKENS, jnp.where(src == C.SRC_HIVE, C.N_TOKENS + self.NV, 0))
        hi = jnp.where(src == C.SRC_VAR, self.NV, jnp.where(src == C.SRC_HIVE, self.NH, C.N_TOKENS)) - 1
        return off + jnp.clip(idx, 0, hi)

    def operand(self, vals, row, base):
        """Read an operand (src, idx, mode, m, lit) stored at row[base:base+5]."""
        src = row[base].astype(i32)
        mode = row[base + 2].astype(i32)
        v = jnp.where((src == C.SRC_LIT) | (src == C.SRC_RESULT), row[base + 4],
                      vals[self._slot(src, row[base + 1].astype(i32))])
        v = jnp.where(mode == C.MODE_INT, jnp.trunc(v), v)
        return jnp.where(mode == C.MODE_LEAD, lead_digits(v, row[base + 3].astype(i32)), v)

    def source_value(self, vals, sids, src, idx):
        """(num, sid) of a key hole or copied value."""
        k = self._slot(src, idx)
        return vals[k], sids[k]

    # ---------------------------------------------------------- variants
    def variant(self, T, vals, sids, gl):
        """The compiled variant of global line `gl` for the current values."""
        src = T["line_keysrc"][gl]  # [K, 2]
        width = T["line_keywidth"][gl]
        nch = T["line_nchoice"][gl]
        choices = T["line_choices"][gl]  # [K, CH]
        num, sid = self.source_value(vals, sids, src[:, 0], src[:, 1])
        wcls = -1 - jnp.minimum(num_width(num), C.MAX_WIDTH).astype(i32)
        cls = jnp.where(sid >= 0, sid, jnp.where(width, wcls, C.CLASS_NUM))
        used = choices[:, 0] != -99
        hit = choices == cls[:, None]
        found = jnp.any(hit, axis=1) | ~used
        idx = jnp.where(used, jnp.argmax(hit, axis=1), 0)
        stride = jnp.cumprod(jnp.concatenate([nch[1:], jnp.ones(1, i32)])[::-1])[::-1]
        return T["line_var0"][gl] + jnp.sum(idx * stride), jnp.all(found)

    def conds_ok(self, T, vals, v, ci):
        """Conditions ci .. ci+CONDS_PER_ITER-1 of variant v all hold (those that exist);
        also returns how many conditions v has."""
        act = T["acts"][v]
        c0 = act[C.ACT["c0"]].astype(i32)
        cn = act[C.ACT["cn"]].astype(i32)
        k = ci + jnp.arange(CONDS_PER_ITER)
        rows = T["conds"][jnp.clip(c0 + k, 0, T["conds"].shape[0] - 1)]
        a = jax.vmap(lambda r: self.operand(vals, r, C.COND["a0_src"]))(rows)
        bb = jax.vmap(lambda r: self.operand(vals, r, C.COND["b0_src"]))(rows) * rows[:, C.COND["scale"]]
        op = rows[:, C.COND["op"]].astype(i32)
        ok = jnp.select([op == C.OP_GT, op == C.OP_LT, op == C.OP_EQ, op == C.OP_NE,
                         op == C.OP_GE, op == C.OP_LE],
                        [a > bb, a < bb, a == bb, a != bb, a >= bb, a <= bb], False)
        return jnp.all(ok | (k >= cn)), cn

    # ------------------------------------------------------------ the turn
    def run_turn(self, T, st, b, draws):
        """read_code for bug b: execute lines until its sub-clicks are spent.

        Each loop iteration does a bounded amount of work, so that battles batched under
        vmap stay cheap: it fetches a line if none is pending, then checks up to
        CONDS_PER_ITER conditions (of one open block header, or of the line itself), and
        evaluates up to EXPR_PER_ITER steps of a math/cmath expression. A line needing
        more than that takes several iterations. Conditions have no side effects and the
        bug's state does not change while its line is pending, so splitting the work
        does not change the result."""
        carry = dict(st=st, **self.turn_carry())

        def body(c):
            c = self.turn_step(T, b, draws, c)
            c["iters"] = c["iters"] + 1
            c["done"] = c["done"] | (c["iters"] >= MAX_ITERS_PER_TURN)
            return c

        return jax.lax.while_loop(lambda c: ~c["done"], body, carry)["st"]

    @staticmethod
    def turn_carry():
        """The interpreter's per-turn registers, reset at the start of a bug's turn."""
        z = i32(0)
        return dict(sub=z, nd=z, done=jnp.bool_(False), iters=z, pend=jnp.bool_(False), gl=z,
                    lvl=z, ci=z, ok=jnp.bool_(True), cnt=z, epc=z,
                    stk=jnp.zeros(EXPR_STACK, f32), sp=z, err=jnp.bool_(False))

    def turn_step(self, T, b, draws, c):
        st = c["st"]
        prog = st.prog[b]
        n = T["n_lines"][prog]
        fetch = ~c["pend"]

        # --- fetch: wrap at the end, handle end if / block headers ------------
        mem = st.mem[b]
        wrap = fetch & (mem >= n)
        mem = jnp.where(wrap, 0, mem)
        nl = jnp.where(wrap, 0, st.nest_level[b])
        ph = st.patch_line[b] == mem
        local = jnp.where(jnp.any(ph), st.patch_row[b, jnp.argmax(ph)], mem)
        gl_f = T["line0"][prog] + local
        kind_f = T["line_kind"][gl_f]
        is_endif = fetch & (kind_f == C.K_ENDIF)
        is_header = fetch & (kind_f == C.K_NORMAL) & T["line_then"][gl_f] & (~T["line_short"][gl_f] | (nl > 0))
        new_exec = fetch & ~is_endif & ~is_header
        ni = jnp.clip(nl, 0, NEST_DEPTH - 1)
        nest = st.nest.at[b, ni].set(jnp.where(is_header & (nl < NEST_DEPTH), gl_f, st.nest[b, ni]))
        nl = jnp.where(is_endif, jnp.maximum(nl - 1, 0), jnp.where(is_header, nl + 1, nl))
        st = st._replace(nest=nest, nest_level=st.nest_level.at[b].set(nl),
                         mem=st.mem.at[b].set(jnp.where(is_endif | is_header, mem + 1, mem)))
        pend = c["pend"] | new_exec
        gl = jnp.where(new_exec, gl_f, c["gl"])
        lvl = jnp.where(new_exec, 0, c["lvl"])
        ci = jnp.where(new_exec, 0, c["ci"])
        ok = jnp.where(new_exec, True, c["ok"])
        cnt = jnp.where(new_exec, 0, c["cnt"])
        epc = jnp.where(new_exec, 0, c["epc"])
        sp = jnp.where(new_exec, 0, c["sp"])
        kind = T["line_kind"][gl]
        depth = jnp.where(kind == C.K_NORMAL, nl, 0)

        vals, sids = self.values(st, b)

        # --- one open block header (outermost first) ------------------------------
        in_stack = pend & ok & (lvl < depth)
        hdr = st.nest[b, jnp.clip(lvl, 0, NEST_DEPTH - 1)]
        hv, hfound = self.variant(T, vals, sids, hdr)
        hok = T["acts"][hv][C.ACT["header_ok"]] > 0
        hc, hn = self.conds_ok(T, vals, hv, ci)
        ok = jnp.where(in_stack, ok & hfound & hok & hc & (lvl < NEST_DEPTH), ok)
        hdone = ci + CONDS_PER_ITER >= hn
        cnt = jnp.where(in_stack & hdone, cnt + hn, cnt)
        lvl = jnp.where(in_stack & hdone, lvl + 1, lvl)
        ci = jnp.where(in_stack, jnp.where(hdone, 0, ci + CONDS_PER_ITER), ci)
        ready = pend & ((lvl >= depth) | ~ok)

        # --- the line itself ----------------------------------------------------
        v, found = self.variant(T, vals, sids, gl)
        row = T["acts"][v]
        lc, cn = self.conds_ok(T, vals, v, ci)
        ok = jnp.where(ready & ok, ok & lc, ok)
        ldone = (ci + CONDS_PER_ITER >= cn) | ~ok
        decided = ready & ldone
        ci = jnp.where(ready & ~ldone, ci + CONDS_PER_ITER, ci)
        disp = row[C.ACT["disp"]].astype(i32)
        atype = jnp.where(found, row[C.ACT["atype"]].astype(i32), C.A_NOP)
        alive_ok = (st.dead[b] == 0) & (st.damage[b] < self.rules.max_damage)
        fire = decided & alive_ok & ok & (cnt + cn + disp <= 20)
        is_expr = fire & ((atype == C.A_MATH) | (atype == C.A_CMATH))

        # --- expression chunk ---------------------------------------------------
        first = epc == 0
        stk = jnp.where(first, jnp.zeros(EXPR_STACK, f32), c["stk"])
        err = jnp.where(first, row[C.ACT["rerr"]] > 0, c["err"])
        nd = c["nd"]
        stk, sp, err, nd, epc2, edone = self.expr_chunk(T, vals, row, draws, nd, stk, sp, err, epc,
                                                        atype, is_expr)
        epc = jnp.where(is_expr, epc2, epc)
        complete = decided & (~is_expr | edone)
        do = complete & fire
        res = jnp.trunc(stk[0])

        # --- execute -------------------------------------------------------------
        st = st._replace(queen=self.set1(st.queen, b, 1, complete & T["line_queen"][gl] & (nl == 0)),
                         mem=self.set1(st.mem, b, st.mem[b] + 1, complete & alive_ok))
        at = jnp.where(do, atype, C.A_NOP)
        st, nd = self.apply(T, st, b, vals, sids, row, at, draws, nd, res, err)
        mega = jnp.isin(at, jnp.array(sorted(C.MEGA_ACTIONS), i32))
        if self.rules.smart_clicks_disabled:
            mega = jnp.bool_(True)
        sub2 = jnp.where(mega, 10, c["sub"] + 1 + jnp.where(at == C.A_SCAN, 2, 0))
        sub = jnp.where(complete, sub2, c["sub"])
        return dict(st=st, sub=sub, nd=nd, done=complete & (sub >= 10), iters=c["iters"],
                    pend=pend & ~complete, gl=gl, lvl=lvl, ci=ci, ok=ok, cnt=cnt, epc=epc, stk=stk,
                    sp=sp, err=err)

    def expr_chunk(self, T, vals, row, draws, nd, stk, sp, err, epc, atype, active):
        """Evaluate up to EXPR_PER_ITER more steps of a math (left-to-right terms) or cmath
        (RPN) expression; the value accumulates in stk[0] (math) or the RPN stack.

        The costly operations (^, atn, cos, exp, log, sin, tan; all rare in practice) are
        computed once per call, for the first step only: one met later ends the chunk."""
        is_math = atype == C.A_MATH
        base = jnp.where(is_math, row[C.ACT["t0"]], row[C.ACT["r0"]]).astype(i32)
        total = jnp.where(is_math, row[C.ACT["tn"]], row[C.ACT["rn"]]).astype(i32)
        SD = EXPR_STACK
        rpn = T["rpn"]
        nr = rpn.shape[0]

        def rare(r):
            op, fn = r[0].astype(i32), r[1].astype(i32)
            return (op == C.R_POW) | ((op == C.R_FN) & ((fn == 1) | (fn == 2) | (fn == 3) | (fn == 6)
                                                        | (fn == 9) | (fn == 11)))

        r0 = rpn[jnp.clip(base + epc, 0, nr - 1)]
        a0 = stk[jnp.clip(sp - 2, 0, SD - 1)]
        b0 = stk[jnp.clip(sp - 1, 0, SD - 1)]
        fn0 = r0[1].astype(i32)
        slow0 = jnp.where(r0[0].astype(i32) == C.R_POW, jnp.power(a0, b0),
                jnp.select([fn0 == 1, fn0 == 2, fn0 == 3, fn0 == 6, fn0 == 9, fn0 == 11],
                           [jnp.arctan(b0), jnp.cos(b0), jnp.exp(b0), jnp.log(jnp.maximum(b0, 1e-30)),
                            jnp.sin(b0), jnp.tan(b0)], 0.0))
        stopped = jnp.bool_(False)
        done_n = i32(0)
        for k in range(EXPR_PER_ITER):
            i = epc + k
            r = rpn[jnp.clip(base + i, 0, nr - 1)]
            if k > 0:
                stopped = stopped | (~is_math & rare(r))
            valid = active & (i < total) & ~stopped
            done_n = done_n + valid.astype(i32)
            # math term: (op, operand)
            tr = T["terms"][jnp.clip(base + i, 0, T["terms"].shape[0] - 1)]
            top = tr[0].astype(i32)
            tv = self.operand(vals, tr, 1)
            mt = stk[0]
            mv = jnp.select([top == C.M_ADD, top == C.M_SUB, top == C.M_MUL,
                             (top == C.M_DIV) & (tv != 0)], [mt + tv, mt - tv, mt * tv, mt / tv], mt)
            # cmath token: (op, fn, operand)
            op = r[0].astype(i32)
            fn = r[1].astype(i32)
            a = stk[jnp.clip(sp - 2, 0, SD - 1)]
            bb = stk[jnp.clip(sp - 1, 0, SD - 1)]
            push_v = self.operand(vals, r, 2)
            ia, ib = a.astype(i32), bb.astype(i32)
            tf = lambda c_: jnp.where(c_, -1.0, 0.0)  # noqa: E731
            safe = jnp.where(bb == 0, 1.0, bb)
            bin_v = jnp.select(
                [op == C.R_MUL, op == C.R_DIV, op == C.R_IDIV, op == C.R_MOD,
                 op == C.R_ADD, op == C.R_SUB, op == C.R_EQ, op == C.R_LT, op == C.R_GT,
                 op == C.R_NE, op == C.R_LE, op == C.R_GE, op == C.R_AND, op == C.R_OR,
                 op == C.R_XOR],
                [a * bb, a / safe, jnp.trunc(a / safe), jnp.fmod(a, bb), a + bb,
                 a - bb, tf(a == bb), tf(a < bb), tf(a > bb), tf(a != bb), tf(a <= bb), tf(a >= bb),
                 (ia & ib).astype(f32), (ia | ib).astype(f32), (ia ^ ib).astype(f32)], 0.0)
            is_fn = op == C.R_FN
            is_rnd = valid & ~is_math & is_fn & (fn == 7)
            u = draws[nd % DRAWS_PER_TURN]
            nd = jnp.where(is_rnd, nd + 1, nd)
            fv = jnp.select([fn == 0, fn == 4, fn == 5, fn == 7, fn == 8, fn == 10],
                            [jnp.abs(bb), jnp.trunc(bb), jnp.floor(bb), u.astype(f32) / 16777216.0,
                             jnp.sign(bb), jnp.sqrt(jnp.maximum(bb, 0.0))], 0.0)
            if k == 0:
                fv = jnp.where(rare(r), slow0, fv)
                bin_v = jnp.where(op == C.R_POW, slow0, bin_v)
            is_push = op == C.R_PUSH
            is_un = (op == C.R_NEG) | (op == C.R_NOT) | is_fn
            un_v = jnp.where(op == C.R_NEG, -bb, jnp.where(op == C.R_NOT, (~ib).astype(f32), fv))
            e_bin = ((op == C.R_DIV) | (op == C.R_IDIV)) & (bb == 0) & ~is_push & ~is_un
            e_fn = is_fn & (((fn == 6) & (bb <= 0)) | ((fn == 10) & (bb < 0)))
            err = err | (valid & ~is_math & (e_bin | e_fn))
            pos = jnp.where(is_push, sp, jnp.where(is_un, sp - 1, sp - 2))
            val = jnp.where(is_push, push_v, jnp.where(is_un, un_v, bin_v))
            new_sp = jnp.where(is_push, sp + 1, jnp.where(is_un, sp, sp - 1))
            pos = jnp.where(is_math, 0, jnp.clip(pos, 0, SD - 1))
            val = jnp.where(is_math, mv, val)
            stk = jnp.where(valid, stk.at[pos].set(val), stk)
            sp = jnp.where(valid & ~is_math, jnp.clip(new_sp, 0, SD), sp)
        epc = epc + done_n
        return stk, sp, err, nd, epc, epc >= total

    # ------------------------------------------------------------- actions
    def apply(self, T, st, b, vals, sids, row, at, draws, nd, res, res_err):
        """Execute action `at` (A_NOP for none) of variant `row` for bug b. `res` is the
        value of a math/cmath expression, `res_err` whether evaluating it failed."""
        r = self.rules
        S = self.S
        A = lambda base: self.operand(vals, row, C.ACT[base])  # noqa: E731
        x, y, h = st.x[b], st.y[b], st.h[b]
        fuel, ammo, shield = st.fuel[b], st.ammo[b], st.shield[b]
        slot = b + 1

        def draw(nd):
            return draws[nd % DRAWS_PER_TURN], nd + 1

        # --- movement -----------------------------------------------------
        is_move = ((at == C.A_MOVE_FWD) | (at == C.A_MOVE_BACK)) & (fuel > 0)
        sgn = jnp.where(at == C.A_MOVE_BACK, -1, 1)
        nx = x + sgn * jnp.asarray(DX)[h]
        ny = y + sgn * jnp.asarray(DY)[h]
        border = (nx <= 0) | (nx >= COLS - 1) | (ny <= 0) | (ny >= ROWS - 1)
        cell = self.mg(st, nx, ny)
        can = is_move & ~border & ((cell == OPEN) | (cell == FLAG) | (cell == MINE))
        bump = is_move & ~can
        lay = can & (ammo > 1) & (st.mine[b] == 1)
        move_fuel = fuel - jnp.where(is_move, st.burn[b], 0)
        flag_full = can & (cell == FLAG) & (st.damage[b] == 0)
        flag_fix = can & (cell == FLAG) & (st.damage[b] != 0)
        on_mine = can & (cell == MINE)

        # --- turns ---------------------------------------------------------
        is_turn = ((at == C.A_TURN_R) | (at == C.A_TURN_L)) & (fuel > 0)
        new_h = jnp.where(at == C.A_TURN_R, h % 4 + 1, (h + 2) % 4 + 1)

        # --- fire weapon ---------------------------------------------------
        is_fire = (at == C.A_FIRE) & (shield == 0) & (ammo >= 1)
        rng5 = jnp.arange(1, 6)
        fx = x + jnp.asarray(DX)[h] * rng5
        fy = y + jnp.asarray(DY)[h] * rng5
        fc = self.mg(st, fx, fy)
        stop = (fc == WALL) | ((fc >= 1) & (fc <= 10))
        first = jnp.argmax(stop)
        hit_slot = jnp.where(jnp.any(stop) & (fc[first] != WALL), fc[first], 0)
        tgt = hit_slot - 1
        tshield = st.shield[jnp.clip(tgt, 0, S - 1)] == 1
        shell_dmg = jnp.where(tshield, jnp.array([2, 1, 0, 0, 0])[first], jnp.array([5, 4, 3, 2, 1])[first])
        shell_pts = jnp.where(tshield, jnp.array([40, 20, 0, 0, 0])[first],
                              jnp.array([100, 80, 60, 40, 20])[first])
        fire_hit = is_fire & (hit_slot > 0)

        # --- discharge / self destruct (8 neighbours) ----------------------
        is_dis = at == C.A_DISCHARGE
        is_sd = at == C.A_SELF_DESTRUCT
        ngx = x + jnp.asarray(NBX)[:8]
        ngy = y + jnp.asarray(NBY)[:8]
        ngc = self.mg(st, ngx, ngy)
        ng_bug = (ngc >= 1) & (ngc <= 10)
        clear = is_dis & ((ngc == MINE) | (ngc == FLAG))
        # every grid write of this action in one scatter (the actions are exclusive)
        wx = jnp.concatenate([jnp.stack([x, nx, x]), ngx])
        wy = jnp.concatenate([jnp.stack([y, ny, y]), ngy])
        wv = jnp.concatenate([jnp.stack([jnp.where(lay, MINE, OPEN), slot, OPEN]), jnp.zeros(8, i32)])
        wm = jnp.concatenate([jnp.stack([can, can, is_sd]), clear])
        wm = wm & (wx >= 0) & (wx < COLS) & (wy >= 0) & (wy < ROWS)
        flat = jnp.where(wm, (wy + 1) * (COLS + 2) + wx + 1, (ROWS + 2) * (COLS + 2))
        grid = st.grid.reshape(-1).at[flat].set(wv, mode="drop").reshape(ROWS + 2, COLS + 2)
        sd_blast = jnp.maximum(0, r.max_damage - st.damage[b])
        ng_hits = jnp.zeros(S, i32).at[jnp.clip(ngc - 1, 0, S - 1)].add(ng_bug.astype(i32))
        n_ng = jnp.sum(ng_bug)

        # --- other-bug deltas ------------------------------------------------
        oth_dmg = (jnp.where(fire_hit, jnp.zeros(S, i32).at[jnp.clip(tgt, 0, S - 1)].add(shell_dmg), 0)
                   + jnp.where(is_dis, ng_hits * 2, 0) + jnp.where(is_sd, ng_hits * sd_blast, 0))
        oth_burn = (jnp.where(fire_hit, jnp.zeros(S, i32).at[jnp.clip(tgt, 0, S - 1)].add(shell_dmg), 0)
                    + jnp.where(is_dis, ng_hits * 3, 0) + jnp.where(is_sd, ng_hits * 10, 0))

        # --- launches --------------------------------------------------------
        is_mis = (at == C.A_LAUNCH_MISSILE) & (st.mis_on[b] == 0) & (ammo >= 10) & (fuel >= 300)
        is_gr = (at == C.A_LAUNCH_GRENADE) & (st.gr_on[b] == 0) & (ammo >= 5) & (fuel >= 200)
        u_mis, nd_m = draw(nd)
        nd = jnp.where((is_mis | is_gr) & (shield == 1), nd_m, nd)
        misfire = is_mis & (shield == 1) & (u_mis < 15099495)  # u < 0.9
        grfire = is_gr & (shield == 1) & (u_mis < 11744052)  # u < 0.7

        # --- repairs ---------------------------------------------------------
        is_rep = (at == C.A_REPAIR) & (shield == 0)
        rep_nodmg = is_rep & (st.damage[b] <= 0)
        rep_try = is_rep & (st.damage[b] > 0)
        u1, nd1 = draw(nd)
        nd = jnp.where(rep_try, nd1, nd)
        rep_ok = rep_try & (u1 < 1677722)  # u < 0.1
        rep_fail = rep_try & ~rep_ok
        u2, nd2 = draw(nd)
        nd = jnp.where(rep_fail, nd2, nd)
        rep_rate = rep_nodmg | (rep_fail & (u2 < 838861))  # u < 0.05

        # --- random ----------------------------------------------------------
        is_rand = at == C.A_RANDOM
        u3, nd3 = draw(nd)
        nd = jnp.where(is_rand, nd3, nd)

        # --- variables -------------------------------------------------------
        is_assign = (at == C.A_ASSIGN) | (at == C.A_MATH) | (at == C.A_CMATH)
        vtype = row[C.ACT["vtype"]].astype(i32)
        csrc = row[C.ACT["a0_src"]].astype(i32)
        cidx = row[C.ACT["a0_idx"]].astype(i32)
        cnum, csid = self.source_value(vals, sids, csrc, cidx)
        res = jnp.where(jnp.isfinite(res), res, 0.0)
        new_num = jnp.select([vtype == 0, vtype == 1, vtype == 3], [row[C.ACT["vnum"]], cnum, cnum], res)
        new_sid = jnp.select([vtype == 0, vtype == 1, vtype == 3],
                             [row[C.ACT["vsid"]].astype(i32), csid, row[C.ACT["vsid"]].astype(i32)], -1)
        do_write = is_assign & ~((at == C.A_CMATH) & res_err) & ((at != C.A_CMATH) | (vtype == 2))
        ttype = row[C.ACT["ttype"]].astype(i32)
        tidx = row[C.ACT["tidx"]].astype(i32)
        wv = do_write & (ttype == 1)
        wh = do_write & (ttype == 2)
        vi = jnp.clip(tidx, 0, self.NV - 1)
        hi = jnp.clip(tidx, 0, self.NH - 1)
        hr = self.hive_row(st, b)
        vnum = st.vnum.at[b, vi].set(jnp.where(wv, new_num, st.vnum[b, vi]))
        vsid = st.vsid.at[b, vi].set(jnp.where(wv, new_sid, st.vsid[b, vi]))
        hnum = st.hnum.at[hr, hi].set(jnp.where(wh, new_num, st.hnum[hr, hi]))
        hsid = st.hsid.at[hr, hi].set(jnp.where(wh, new_sid, st.hsid[hr, hi]))

        # --- jumps -----------------------------------------------------------
        target = self.jump_target(T, vals, row)
        mem = st.mem[b]  # already advanced past this line
        is_goto = (at == C.A_GOTO) & (target >= 0)
        is_gosub = (at == C.A_GOSUB) & (target >= 0) & (st.curnest[b] < GOSUB_DEPTH)
        is_ret = (at == C.A_RETURN) & (st.curnest[b] > 0)
        cn_b = st.curnest[b]
        subnest = st.subnest.at[b, jnp.clip(cn_b, 0, GOSUB_DEPTH - 1)].set(
            jnp.where(is_gosub, mem, st.subnest[b, jnp.clip(cn_b, 0, GOSUB_DEPTH - 1)]))
        ret_to = st.subnest[b, jnp.clip(cn_b - 1, 0, GOSUB_DEPTH - 1)]
        new_mem = jnp.where(is_goto | is_gosub, target, jnp.where(is_ret, ret_to, mem))
        new_cn = cn_b + is_gosub.astype(i32) - is_ret.astype(i32)

        # --- iff ---------------------------------------------------------------
        is_iff = (at == C.A_IFF) & (not r.iff_disabled) & (st.team[b] == 0)
        new_iff = row[C.ACT["iparam"]].astype(i32)
        present = jnp.any(st.hkey == new_iff)
        free = jnp.argmax(st.hkey < 0)
        alloc = is_iff & ~present & jnp.any(st.hkey < 0)
        hkey = st.hkey.at[free].set(jnp.where(alloc, new_iff, st.hkey[free]))
        hnum = hnum.at[free].set(jnp.where(alloc, 0.0, hnum[free]))
        hsid = hsid.at[free].set(jnp.where(alloc, -1, hsid[free]))

        # --- data link ---------------------------------------------------------
        opp = jnp.array([0, 3, 4, 1, 2], i32)[h]
        rx, ry = x + jnp.asarray(DX)[opp], y + jnp.asarray(DY)[opp]
        rc = self.mg(st, rx, ry)
        rt = jnp.clip(rc - 1, 0, S - 1)
        linked = (rc >= 1) & (rc <= 10) & (st.h[rt] == opp)
        is_inj = (at == C.A_DLINK_INJECT) & linked
        tmem = st.mem[rt]
        tprog = st.prog[rt]
        inj_row = T["n_lines"][tprog] + row[C.ACT["iparam"]].astype(i32)
        inj_ok = is_inj & (tmem < T["n_lines"][tprog])
        same = st.patch_line[rt] == tmem
        pslot = jnp.where(jnp.any(same), jnp.argmax(same), st.patch_n[rt] % PATCHES)
        patch_line = st.patch_line.at[rt, pslot].set(jnp.where(inj_ok, tmem, st.patch_line[rt, pslot]))
        patch_row = st.patch_row.at[rt, pslot].set(jnp.where(inj_ok, inj_row, st.patch_row[rt, pslot]))
        patch_n = st.patch_n.at[rt].add(jnp.where(inj_ok & ~jnp.any(same), 1, 0))
        is_res = (at == C.A_DLINK_RES) & (shield == 1) & linked
        amount = A("a0_src")
        res_kind = row[C.ACT["iparam"]].astype(i32)
        xf_fuel = jnp.where(is_res & (res_kind == 1) & (amount > 0),
                            jnp.minimum(amount.astype(i32), fuel), 0)
        xf_ammo = jnp.where(is_res & (res_kind == 2) & (amount > 0),
                            jnp.minimum(amount.astype(i32), ammo), 0)

        # --- scans ---------------------------------------------------------------
        scan_res = self.scan(st, b, row, vals)

        # --- assemble the acting bug ---------------------------------------------
        def upd(arr, v, cond=True):
            return arr.at[b].set(jnp.where(cond, v, arr[b]))

        cloak_break = (is_fire | is_mis | is_gr) & (st.cloak[b] == 1)
        n_dmg = (st.damage[b] + jnp.where(on_mine, 5, 0) + jnp.where(misfire, 6, 0)
                 + jnp.where(grfire, 4, 0) + jnp.where(is_dis, 1, 0))
        n_dmg = jnp.where(flag_full, 5, jnp.where(flag_fix, 0, n_dmg))
        n_dmg = jnp.where(rep_ok, jnp.maximum(0, st.damage[b] - 1), n_dmg)
        n_burn = (st.burn[b] + jnp.where(flag_full | on_mine, 5, 0)
                  + jnp.where(is_mis, jnp.where(shield == 1, 9, 6), 0)
                  + jnp.where(is_gr, jnp.where(shield == 1, 18, 12), 0)
                  + jnp.where(rep_nodmg | rep_fail, 6, 0))
        n_fuel = (jnp.where(is_move, move_fuel, jnp.where(is_turn, fuel - st.burn[b], fuel))
                  + jnp.where(flag_fix, 350, 0) - jnp.where(is_mis, 300, 0)
                  - jnp.where(is_gr, 200, 0) - xf_fuel)
        n_ammo = (ammo - jnp.where(lay, 2, 0) + jnp.where(flag_fix, 30, 0) - jnp.where(is_fire, 1, 0)
                  - jnp.where(is_mis, 10, 0) - jnp.where(is_gr, 5, 0) - xf_ammo)
        n_score = (st.score[b] + jnp.where(flag_fix, 50, 0) + jnp.where(fire_hit, shell_pts, 0)
                   + jnp.where(is_dis, 100 * n_ng, 0) + jnp.where(is_sd, 400 * n_ng, 0))

        damage = upd(st.damage + oth_dmg, n_dmg + oth_dmg[b])
        burn = upd(st.burn + oth_burn, n_burn + oth_burn[b])
        fuel_v = upd(st.fuel, n_fuel).at[rt].add(xf_fuel)
        ammo_v = upd(st.ammo, n_ammo).at[rt].add(xf_ammo)
        st = st._replace(
            grid=grid,
            x=upd(st.x, nx, can), y=upd(st.y, ny, can),
            h=upd(st.h, new_h, is_turn),
            movecount=jnp.where(can, 0, st.movecount),
            barrier=upd(st.barrier, jnp.where(bump, 1, 0), is_move),
            damage=damage, burn=burn, fuel=fuel_v, ammo=ammo_v,
            score=upd(st.score, n_score),
            fuel_rate=upd(st.fuel_rate, st.fuel_rate[b] + 1, rep_rate),
            shield=upd(st.shield, jnp.where(at == C.A_SHIELD_UP, 1, 0),
                       (at == C.A_SHIELD_UP) | (at == C.A_SHIELD_DOWN)),
            cloak=upd(st.cloak, jnp.where((at == C.A_CLOAK_ON) & (ammo >= 2) & (fuel >= 10), 1,
                                          jnp.where(at == C.A_CLOAK_ON, st.cloak[b], 0)),
                      (at == C.A_CLOAK_ON) | (at == C.A_CLOAK_OFF) | cloak_break),
            mine=upd(st.mine, jnp.where(at == C.A_MINES_ON, 1, 0),
                     (at == C.A_MINES_ON) | (at == C.A_MINES_OFF)),
            random=upd(st.random, (u3 >> 22) + 1, is_rand),
            scan=upd(st.scan, scan_res, at == C.A_SCAN),
            mis_on=upd(st.mis_on, 1, is_mis), mis_x=upd(st.mis_x, x, is_mis),
            mis_y=upd(st.mis_y, y, is_mis), mis_h=upd(st.mis_h, h, is_mis),
            gr_on=upd(st.gr_on, 1, is_gr), gr_x=upd(st.gr_x, x, is_gr),
            gr_y=upd(st.gr_y, y, is_gr), gr_h=upd(st.gr_h, h, is_gr),
            gr_d=upd(st.gr_d, jnp.where(is_gr, 3, jnp.maximum(2, A("a0_src").astype(i32))),
                     is_gr | (at == C.A_SET_FUSE)),
            gr_t=upd(st.gr_t, 0, is_gr),
            dead=upd(st.dead, 1, is_sd), fstat=upd(st.fstat, 1, is_sd),
            vnum=vnum, vsid=vsid, hkey=hkey, hnum=hnum, hsid=hsid,
            mem=upd(st.mem, new_mem), curnest=upd(st.curnest, new_cn), subnest=subnest,
            iff=upd(st.iff, new_iff, is_iff),
            overheat=st.overheat.at[rt].add(jnp.where(is_inj, 1, 0)),
            patch_line=patch_line, patch_row=patch_row, patch_n=patch_n,
        )
        st = st._replace(fuel=upd(st.fuel, 0, is_sd))
        return st, nd

    def jump_target(self, T, vals, row):
        dyn = row[C.ACT["dyn"]] > 0
        static = row[C.ACT["iparam"]].astype(i32)
        lab0 = row[C.ACT["lab0"]].astype(i32)
        labn = row[C.ACT["labn"]].astype(i32)
        want = jnp.stack([self.operand(vals, row, C.ACT[f"l{j}_src"]) for j in range(C.LABEL_HOLES)])
        # binary search over rows sorted lexicographically by their values
        tab = T["labels"]
        nrow = tab.shape[0]

        def less(rowv):  # rowv < vals lexicographically
            lt = jnp.bool_(False)
            eq = jnp.bool_(True)
            for j in range(C.LABEL_HOLES):
                lt = lt | (eq & (rowv[j] < want[j]))
                eq = eq & (rowv[j] == want[j])
            return lt

        def body(_, c):
            lo, hi = c
            mid = (lo + hi) // 2
            go = less(tab[jnp.clip(lab0 + mid, 0, nrow - 1)])
            lo = jnp.where((lo < hi) & go, mid + 1, lo)
            hi = jnp.where((lo < hi) & ~go, mid, hi)
            return lo, hi

        steps = max(1, int(np.ceil(np.log2(self.ML + 1))) + 1)
        lo, hi = i32(0), labn
        for k in range(steps):
            lo, hi = body(k, (lo, hi))
        cand = tab[jnp.clip(lab0 + lo, 0, nrow - 1)]
        hit = (lo < labn) & jnp.all(cand[:C.LABEL_HOLES] == want)
        dyn_t = jnp.where(hit, cand[C.LABEL_HOLES].astype(i32), -1)
        return jnp.where(dyn, dyn_t, static)

    def scan(self, st, b, row, vals):
        x, y, h = st.x[b], st.y[b], st.h[b]
        kind = row[C.ACT["iparam"]].astype(i32)
        a = self.operand(vals, row, C.ACT["a0_src"]).astype(i32)
        bb = self.operand(vals, row, C.ACT["b0_src"]).astype(i32)
        my_iff = st.iff[b]
        S = self.S

        def result(v):
            bug = (v >= 1) & (v <= 10)
            fr = st.iff[jnp.clip(v - 1, 0, S - 1)] == my_iff
            return jnp.select([v == WALL, v == FLAG, v == MINE, bug & fr, bug],
                              [1, 5, 3, 4, 2], 0)  # barrier, flag, mine, friend, enemy

        def resolve(vals, mask):
            rs = jnp.where(mask, jax.vmap(result)(vals), 0)
            return jnp.select([jnp.any(rs == 2), jnp.any(rs == 4), jnp.any(rs == 3),
                               jnp.any(rs == 5), jnp.any(rs == 1)], [2, 4, 3, 5, 1], 0)

        # sets of cells
        per = self.mg(st, x + jnp.asarray(PX)[1:], y + jnp.asarray(PY)[1:])
        rel = a + (h - 1) * 2
        rel = jnp.where(rel > 8, (rel - 1) % 8 + 1, rel)
        pos_d = jnp.where(kind == C.SC_RELATIVE, rel, a)
        pos_ok = (pos_d >= 1) & (pos_d <= 8)
        pc = jnp.clip(pos_d, 0, 8)
        pos_v = self.mg(st, x + jnp.asarray(PX)[pc], y + jnp.asarray(PY)[pc])
        gps_v = self.mg(st, a, bb)
        cross = self.mg(st, x + jnp.array([0, 1, 0, -1]), y + jnp.array([-1, 0, 1, 0]))
        corner = self.mg(st, x + jnp.array([1, 1, -1, -1]), y + jnp.array([-1, 1, 1, -1]))
        # rays
        ddx = jnp.select([kind == C.SC_LEFT, kind == C.SC_RIGHT], [jnp.asarray(LX)[h], jnp.asarray(RX)[h]],
                         jnp.asarray(DX)[h])
        ddy = jnp.select([kind == C.SC_LEFT, kind == C.SC_RIGHT], [jnp.asarray(LY)[h], jnp.asarray(RY)[h]],
                         jnp.asarray(DY)[h])
        n50 = jnp.arange(1, 51)
        ray = self.mg(st, x + ddx * n50, y + ddy * n50)
        nz = ray != 0
        fi = jnp.argmax(nz)
        any_nz = jnp.any(nz)
        line_res = jnp.where(any_nz & (fi < 5), result(ray[fi]), 0)
        long_v = jnp.where(any_nz, ray[fi], 0)
        long_res = jnp.where(long_v == b + 1, 6, result(long_v))

        one = jnp.ones(1, jnp.bool_)
        return jnp.select(
            [kind == C.SC_PERIMETER, (kind == C.SC_POSITION) | (kind == C.SC_RELATIVE),
             kind == C.SC_GPS, kind == C.SC_CROSS, kind == C.SC_CORNER,
             (kind == C.SC_FORWARD) | (kind == C.SC_LEFT) | (kind == C.SC_RIGHT),
             kind == C.SC_LONG],
            [resolve(per, jnp.ones(8, jnp.bool_)), jnp.where(pos_ok, result(pos_v), 0),
             resolve(gps_v[None], one), resolve(cross, jnp.ones(4, jnp.bool_)),
             resolve(corner, jnp.ones(4, jnp.bool_)), line_res, long_res],
            0)

    # ---------------------------------------------------------- per-slot phases
    def pre(self, st, b):
        """slotPreProcess: returns (state, whether the script runs)."""
        r = self.rules
        alive = st.dead[b] == 0
        grid = self.mgset(st.grid, st.x[b], st.y[b], b + 1, alive)
        st = st._replace(grid=grid)
        if r.shield_overheat_disabled:
            return st, alive
        up = st.shield[b] == 1
        oh = jnp.where(up, st.overheat[b] + jnp.where(st.cloak[b] == 1, 2, 1),
                       jnp.maximum(st.overheat[b] - 1, 0))
        pen = jnp.where(up, st.penalty[b], jnp.maximum(st.penalty[b] - 1, 0))
        ok = (oh < r.overheat_limit) & (pen == 0)
        trip = ~ok & (pen == 0)
        pen = jnp.where(trip, 10, pen)
        sh = jnp.where(trip, 0, st.shield[b])
        st = st._replace(
            overheat=self.set1(st.overheat, b, oh, alive),
            penalty=self.set1(st.penalty, b, pen, alive),
            shield=self.set1(st.shield, b, sh, alive))
        return st, alive & ok

    def check_dead(self, st, b):
        r = self.rules
        die = (st.dead[b] == 0) & (st.damage[b] >= r.max_damage)
        st = st._replace(
            dead=self.set1(st.dead, b, 1, die), fuel=self.set1(st.fuel, b, 0, die),
            fstat=self.set1(st.fstat, b, 1, die),
            grid=self.mgset(st.grid, st.x[b], st.y[b], FLAG, die),
            score=self.set1(st.score, b, st.score[b] - 1000, die & (st.queen[b] == 1)))
        if r.battle_ends_queen_death:
            st = st._replace(running=st.running & ~(die & (st.queen[b] == 1)))
        return st

    def post(self, st, b):
        """slotPostProcess: strategy node, cloak upkeep, missile and grenade flight."""
        on_node = st.node_on & (st.x[b] == st.node_x) & (st.y[b] == st.node_y)
        st = st._replace(score=self.set1(st.score, b, st.score[b] + 10, on_node),
                         fuel=self.set1(st.fuel, b, st.fuel[b] + 1, on_node))
        cl = st.cloak[b] == 1
        f = st.fuel[b] - jnp.where(cl, 10, 0)
        a = st.ammo[b] - jnp.where(cl, 2, 0)
        off = cl & ((f <= 0) | (a <= 0))
        st = st._replace(fuel=self.set1(st.fuel, b, f), ammo=self.set1(st.ammo, b, jnp.where(off, jnp.maximum(a, 0), a)),
                         cloak=self.set1(st.cloak, b, 0, off))
        for _ in range(4):
            st = self.missile_step(st, b)
        for _ in range(2):
            st = self.grenade_step(st, b)
        return st

    def blast(self, st, owner, cx, cy, active, kind):
        """3x3 blast of `owner`'s missile (kind 0) or grenade (kind 1)."""
        S = self.S
        # (direct shield-up, direct shield-down, splash up, splash down)
        dmg = [(6, 9, 4, 7), (4, 7, 2, 5)][kind]
        pts = [(300, 500, 200, 400), (600, 1000, 100, 200)][kind]
        burn = [(6, 9, 4, 7), (12, 18, 2, 4)][kind]
        grid = st.grid
        damage, burn_v, score = st.damage, st.burn, st.score
        for k in range(9):
            bx, by = cx + int(NBX[k]), cy + int(NBY[k])
            v = self.mg(st._replace(grid=grid), bx, by)
            bug = active & (v >= 1) & (v <= 10)
            t = jnp.clip(v - 1, 0, S - 1)
            up = st.shield[t] == 1
            direct = k == 8
            dd = jnp.where(up, dmg[0] if direct else dmg[2], dmg[1] if direct else dmg[3])
            pp = jnp.where(up, pts[0] if direct else pts[2], pts[1] if direct else pts[3])
            bb = jnp.where(up, burn[0] if direct else burn[2], burn[1] if direct else burn[3])
            damage = damage.at[t].add(jnp.where(bug, dd, 0))
            burn_v = burn_v.at[t].add(jnp.where(bug, bb, 0))
            score = score.at[owner].add(jnp.where(bug & (t != owner), pp, 0))
            grid = self.mgset(grid, bx, by, OPEN, active & ((v == FLAG) | (v == MINE)))
        return st._replace(grid=grid, damage=damage, burn=burn_v, score=score)

    def missile_step(self, st, b):
        S = self.S
        on = st.mis_on[b] == 1
        mh = st.mis_h[b]
        nx = st.mis_x[b] + jnp.asarray(DX)[mh]
        ny = st.mis_y[b] + jnp.asarray(DY)[mh]
        cell = self.mg(st, nx, ny)
        st = st._replace(mis_x=self.set1(st.mis_x, b, nx, on), mis_y=self.set1(st.mis_y, b, ny, on))
        others = jnp.arange(S) != b
        # open cell: mid-air intercept with another missile, else any grenade
        m_at = others & (st.mis_on == 1) & (st.mis_x == nx) & (st.mis_y == ny)
        g_at = (st.gr_on == 1) & (st.gr_x == nx) & (st.gr_y == ny)
        open_ = on & (cell == OPEN)
        xm = jnp.argmax(m_at)
        xg = jnp.argmax(g_at)
        inter_m = open_ & jnp.any(m_at)
        inter_g = open_ & ~jnp.any(m_at) & jnp.any(g_at)
        hit = on & (cell != OPEN)
        # hitting something: other missiles there vanish, a flag or mine there is destroyed
        kill_m = hit & m_at
        grid = self.mgset(st.grid, nx, ny, OPEN, hit & ((cell == FLAG) | (cell == MINE)))
        st = st._replace(grid=grid, mis_on=jnp.where(kill_m, 0, st.mis_on),
                         mis_x=jnp.where(kill_m, 0, st.mis_x), mis_y=jnp.where(kill_m, 0, st.mis_y))
        det = hit | inter_m | inter_g
        st = self.blast(st, b, nx, ny, det, 0)
        st = self.blast(st, xm, nx, ny, inter_m, 0)
        st = self.blast(st, xg, nx, ny, inter_g, 1)
        st = st._replace(
            mis_on=st.mis_on.at[xm].set(jnp.where(inter_m, 0, st.mis_on[xm])),
            mis_x=st.mis_x.at[xm].set(jnp.where(inter_m, 0, st.mis_x[xm])),
            mis_y=st.mis_y.at[xm].set(jnp.where(inter_m, 0, st.mis_y[xm])),
            gr_on=st.gr_on.at[xg].set(jnp.where(inter_g, 0, st.gr_on[xg])),
            gr_x=st.gr_x.at[xg].set(jnp.where(inter_g, 0, st.gr_x[xg])),
            gr_y=st.gr_y.at[xg].set(jnp.where(inter_g, 0, st.gr_y[xg])))
        st = st._replace(mis_on=self.set1(st.mis_on, b, 0, det), mis_x=self.set1(st.mis_x, b, 0, det),
                         mis_y=self.set1(st.mis_y, b, 0, det))
        return st

    def grenade_step(self, st, b):
        S = self.S
        on = st.gr_on[b] == 1
        gh = st.gr_h[b]
        adv = on & (st.gr_d[b] > st.gr_t[b])
        nx = st.gr_x[b] + jnp.asarray(DX)[gh]
        ny = st.gr_y[b] + jnp.asarray(DY)[gh]
        cell = self.mg(st, nx, ny)
        gx = jnp.where(adv, nx, st.gr_x[b])
        gy = jnp.where(adv, ny, st.gr_y[b])
        st = st._replace(gr_t=self.set1(st.gr_t, b, st.gr_t[b] + 1, adv),
                         gr_x=self.set1(st.gr_x, b, gx, on), gr_y=self.set1(st.gr_y, b, gy, on))
        hit = adv & (cell != OPEN)
        st = st._replace(grid=self.mgset(st.grid, nx, ny, OPEN, hit & ((cell == FLAG) | (cell == MINE))))
        det = hit | (on & ~adv)
        others = jnp.arange(S) != b
        g_at = on & ~det & others & (st.gr_on == 1) & (st.gr_x == gx) & (st.gr_y == gy)
        m_at = on & ~det & (st.mis_on == 1) & (st.mis_x == gx) & (st.mis_y == gy)
        det = det | jnp.any(g_at) | jnp.any(m_at)
        st = st._replace(gr_on=jnp.where(g_at, 0, st.gr_on), gr_x=jnp.where(g_at, 0, st.gr_x),
                         gr_y=jnp.where(g_at, 0, st.gr_y),
                         mis_on=jnp.where(m_at, 0, st.mis_on), mis_x=jnp.where(m_at, 0, st.mis_x),
                         mis_y=jnp.where(m_at, 0, st.mis_y))
        st = self.blast(st, b, gx, gy, det, 1)
        for o in range(S):
            st = self.blast(st, o, gx, gy, m_at[o], 0)
        st = st._replace(gr_on=self.set1(st.gr_on, b, 0, det), gr_x=self.set1(st.gr_x, b, 0, det),
                         gr_y=self.set1(st.gr_y, b, 0, det))
        return st

    # ------------------------------------------------------------- the tick
    def tick(self, T, st):
        """One click: every bug's turn in slot order, then end-of-turn bookkeeping."""
        tkey = jax.random.fold_in(st.key, st.turn)
        for b in range(self.S):
            st, run = self.pre(st, b)
            draws = (jax.random.bits(jax.random.fold_in(tkey, b), (DRAWS_PER_TURN,), jnp.uint32) >> 8).astype(i32)
            # Masked rather than lax.cond: under vmap a cond around the turn's while loop
            # was 30x slower than running it and discarding the result.
            ran = self.run_turn(T, st, b, draws)
            st = jax.tree.map(lambda a, c: jnp.where(run, a, c), ran, st)
            st = self.check_dead(st, b)
            st = self.post(st, b)
            st = self.check_dead(st, b)
        return self.end_turn(st)

    def end_turn(self, st):
        r = self.rules
        S = self.S
        running = st.running
        movecount = st.movecount
        fuel = st.fuel
        no_move = jnp.bool_(False)
        if r.end_if_no_move:
            movecount = movecount + 1
            no_move = movecount >= 101
            fuel = jnp.where(no_move, 0, fuel)
            running = running & ~no_move
        turn = st.turn + 1
        running = running & (turn < r.max_turns)
        tcnt = st.tcnt + 1
        tenth = tcnt >= 10
        fuel = jnp.where(tenth, fuel - st.fuel_rate, fuel)
        tcnt = jnp.where(tenth, 0, tcnt)
        running = running & (jnp.sum(st.dead) < S - 1)
        fuel = jnp.maximum(fuel, 0)
        fstat = jnp.where((st.fstat == 0) & (fuel <= 0), 1, st.fstat)
        if not r.iff_disabled:
            alive = st.dead == 0
            changes = i32(0)
            tiff = i32(-12345)
            for k in range(S):
                ch = alive[k] & (st.iff[k] != tiff)
                changes = changes + ch.astype(i32)
                tiff = jnp.where(ch, st.iff[k], tiff)
            running = running & (changes != 1)
        if S == MAX_SLOTS:
            running = running & ~jnp.all(fuel <= 0)
        st = st._replace(running=running, movecount=movecount, fuel=fuel, turn=turn, tcnt=tcnt,
                         fstat=fstat)
        ended = st.running == jnp.bool_(False)
        winner = jnp.where(ended & (st.winner < 0), self.resolve_winner(st), st.winner)
        return st._replace(winner=winner)

    def resolve_winner(self, st):
        alive = st.dead == 0
        sc = jnp.where(alive & (st.score > -1), st.score, -2**30)
        j = jnp.argmax(sc)
        any_alive = jnp.any(alive & (st.score > -1))
        sc2 = jnp.where(st.score > -1, st.score, -2**30)
        j2 = jnp.argmax(sc2)
        any2 = jnp.any(st.score > -1)
        return jnp.where(any_alive, j, jnp.where(any2, j2, -1)).astype(i32)

    def run(self, T, st):
        return jax.lax.while_loop(lambda s: s.running, lambda s: self.tick(T, s), st)

    # ------------------------------------------------------- many battles
    def run_queue(self, T, jobs, maps, lanes: int, rounds_iters: int = 16):
        """Play every battle in `jobs`, `lanes` at a time. Jit this; do not vmap it.

        jobs: dict of arrays with leading axis N, one entry per battle:
          key [N, 2] uint32, prog [N, S], map [N] (index into `maps`),
          starts [N, S, 3] (x, y, heading), node [N, 3] (x, y, enabled), iff [N, S].
        maps: [M, ROWS, COLS] cell values with every start cell open.

        Lanes run independently: each round gives every lane `rounds_iters` interpreter
        iterations of its current bug's turn, then moves the lanes whose turn ended on to
        the next bug or tick; a lane whose battle ended takes the next unplayed one. So no
        lane waits for another's long turn or long battle, as the lanes of a plain
        vmap(run) would.

        Returns dict of [N] arrays: winner (bug index or -1), ticks, score [N, S],
        dead [N, S], damage [N, S].
        """
        c = self.queue_init(T, jobs, maps, lanes)
        c = jax.lax.while_loop(lambda c: jnp.any(c[0]["active"]) | (c[2] < jobs["prog"].shape[0]),
                               lambda c: self.queue_round(T, jobs, maps, c, rounds_iters), c)
        return c[1]

    def _load(self, T, jobs, maps, n):
        j = jax.tree.map(lambda a: a[n], jobs)
        return self.init(j["key"], T, j["prog"], jnp.asarray(maps, i32)[j["map"]], j["starts"],
                         (j["node"][0], j["node"][1], j["node"][2] > 0), j["iff"])

    def queue_init(self, T, jobs, maps, lanes):
        """(lane registers [lanes], results [N], next unplayed job). Lanes start idle."""
        S = self.S
        N = jobs["prog"].shape[0]
        st0 = self._load(T, jobs, maps, 0)
        L = dict(st=st0, tc=self.turn_carry(), b=i32(-1), in_turn=jnp.bool_(False), job=i32(-1),
                 active=jnp.bool_(False), draws=jnp.zeros(DRAWS_PER_TURN, i32))
        L = jax.tree.map(lambda a: jnp.broadcast_to(a, (lanes,) + jnp.shape(a)), L)
        res = dict(winner=jnp.full(N, -1, i32), ticks=jnp.zeros(N, i32), score=jnp.zeros((N, S), i32),
                   dead=jnp.zeros((N, S), i32), damage=jnp.zeros((N, S), i32))
        return L, res, i32(0)

    def queue_round(self, T, jobs, maps, c, rounds_iters):
        S = self.S
        N = jobs["prog"].shape[0]
        lanes, res, nxt = c
        B = lanes["b"].shape[0]

        def lane_steps(L):
            def body(_, L):
                go = L["active"] & L["in_turn"]
                tc = dict(L["tc"], st=L["st"])
                tc2 = self.turn_step(T, jnp.maximum(L["b"], 0), L["draws"], tc)
                tc2["iters"] = tc2["iters"] + 1
                tc2["done"] = tc2["done"] | (tc2["iters"] >= MAX_ITERS_PER_TURN)
                new = jax.tree.map(lambda a, c_: jnp.where(go, a, c_), tc2, tc)
                st = new.pop("st")
                return dict(L, st=st, tc=new, in_turn=L["in_turn"] & ~new["done"])
            return jax.lax.fori_loop(0, rounds_iters, body, L)

        def finish(L):
            """End the current bug's turn; maybe the tick; maybe the battle."""
            act = L["active"] & ~L["in_turn"]
            st, b = L["st"], L["b"]
            bb = jnp.maximum(b, 0)
            fin = act & (b >= 0)
            st2 = self.check_dead(self.post(self.check_dead(st, bb), bb), bb)
            st = jax.tree.map(lambda a, c_: jnp.where(fin, a, c_), st2, st)
            nb = b + 1
            tick_end = act & (nb >= S)
            st = jax.tree.map(lambda a, c_: jnp.where(tick_end, a, c_), self.end_turn(st), st)
            over = tick_end & ~st.running
            out = dict(winner=st.winner, ticks=st.turn, score=st.score, dead=st.dead, damage=st.damage)
            return dict(L, st=st, b=jnp.where(act, jnp.where(nb >= S, 0, nb), b)), act & ~over, over, out

        def start(L, cont, new_job):
            """Load a new battle if given one, then begin the next bug's turn."""
            got = new_job >= 0
            st = jax.tree.map(lambda a, c_: jnp.where(got, a, c_),
                              self._load(T, jobs, maps, jnp.maximum(new_job, 0)), L["st"])
            b = jnp.where(got, 0, L["b"])
            go = cont | got
            stp, run = self.pre(st, b)
            st = jax.tree.map(lambda a, c_: jnp.where(go, a, c_), stp, st)
            tkey = jax.random.fold_in(st.key, st.turn)
            draws = (jax.random.bits(jax.random.fold_in(tkey, b), (DRAWS_PER_TURN,), jnp.uint32) >> 8).astype(i32)
            tc = jax.tree.map(lambda a, c_: jnp.where(go, a, c_), self.turn_carry(), L["tc"])
            active = (L["active"] & cont) | got | (L["active"] & L["in_turn"])
            return dict(st=st, tc=tc, b=b, in_turn=jnp.where(go, run, L["in_turn"]),
                        job=jnp.where(got, new_job, L["job"]), active=active,
                        draws=jnp.where(go, draws, L["draws"]))

        lanes = jax.vmap(lane_steps)(lanes)
        old_job = lanes["job"]
        lanes, cont, over, out = jax.vmap(finish)(lanes)
        # record finished battles, then hand out unplayed ones to idle lanes
        dst = jnp.where(over, old_job, N)  # out of range: dropped
        res = {k: res[k].at[dst].set(out[k], mode="drop") for k in res}
        idle = over | ~lanes["active"]
        rank = jnp.cumsum(idle) - 1
        cand = nxt + rank
        new_job = jnp.where(idle & (cand < N), cand, -1)
        nxt = jnp.minimum(nxt + jnp.sum(idle), N)
        lanes = jax.vmap(start)(lanes, cont, new_job)
        lanes = dict(lanes, active=lanes["active"] & ~(over & (new_job < 0)))
        del B
        return lanes, res, nxt


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

SLOT_CHARS = "IJKLMNOPQR"
WALL_CHARS = "ZIJKLMNOPQRSHV123456789F"


class MapInfo(NamedTuple):
    rows: list  # the 32 map rows (45 characters each)
    fixed: dict  # slot (1-based) -> (x, y) from its I..R character
    z_cells: list  # (x, y) of the random-start 'Z' cells
    name: str = ""


def parse_map(text: str, name: str = "") -> MapInfo:
    """Read a .map file: the rows of exactly 45 characters are the map."""
    rows = [ln for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if len(ln) == COLS]
    if len(rows) < ROWS:
        raise ValueError(f"map has {len(rows)} rows of {COLS} characters, need {ROWS}")
    rows = rows[:ROWS]
    fixed, z = {}, []
    for i, r in enumerate(rows):
        for j, ch in enumerate(r):
            if ch in SLOT_CHARS:
                fixed.setdefault(SLOT_CHARS.index(ch) + 1, (j, i))
            elif ch == "Z":
                z.append((j, i))
    return MapInfo(rows, fixed, z, name)


def map_grid(rows) -> np.ndarray:
    """Cell values for map rows whose 'Z' cells were already resolved (engine map())."""
    g = np.zeros((ROWS, COLS), np.int32)
    for i in range(ROWS):
        for j in range(COLS):
            ch = rows[i][j]
            v = WALL if ch in WALL_CHARS else OPEN
            border = i in (0, ROWS - 1) or j in (0, COLS - 1)
            if ch == "0":
                v = WALL if border else OPEN
            if ch == "F":
                v = FLAG
            if ch == "9":
                v = MINE
            g[i, j] = v
    return g


def place(info: MapInfo, n_bugs: int, rng: np.random.Generator):
    """The Battle Setup's placement: bugs without a fixed start character take a random
    'Z' cell (unused ones become open floor); bugs with neither use the engine's default
    positions. Returns (rows with Z resolved, starts [(x, y)])."""
    z = list(info.z_cells)
    order = rng.permutation(len(z)) if z else []
    z = [z[k] for k in order]
    starts = []
    used = set()
    for s in range(1, n_bugs + 1):
        if s in info.fixed:
            starts.append(info.fixed[s])
        elif z:
            starts.append(z.pop(0))
        else:
            starts.append(DEFAULT_POS[s - 1])
        used.add(starts[-1])
    rows = ["".join(("0" if ch == "Z" and (j, i) not in used else ch) for j, ch in enumerate(r))
            for i, r in enumerate(info.rows)]
    return rows, starts


def node_candidates(rows) -> list:
    """Cells the strategy node may be placed on: interior '0' cells."""
    return [(j, i) for i in range(1, ROWS - 1) for j in range(1, COLS - 1) if rows[i][j] == "0"]
