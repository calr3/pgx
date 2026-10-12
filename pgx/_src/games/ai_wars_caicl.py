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

"""Compiler from CAICL, the Cybug scripting language of A.I. Wars, to fixed tables.

The reference interpreter (`engine_caicl.js` in the web edition, itself a port of the
VB6 original) works on text. Every line is rebuilt as a string on every execution:
the enclosing block-`if` headers are prefixed to it, `~vN`/`~hN` variables and
`#token`s are substituted word by word, and the result is matched against command
prefixes, often at fixed character columns. A variable can hold any text, including a
whole command (`assign vm turn left` ... `~vm`) or part of a label (`goto l #scan`).

To run that under JAX, each line is compiled ahead of time into structured
*variants*. A substituted word is a *hole*. A hole whose value can only ever be a
number stays symbolic: the compiled command reads it at run time (as `parseFloat`,
`parseInt`, or the leading digits that a fixed-column slice cuts out of it). A hole
that may hold a string literal is expanded: the line is compiled once per string it
can hold, plus once for "a number". Where a fixed-column parser makes the parse depend
on how many characters a number prints as (`if x coordinate is < #enemy_x then ...`
only works for two-digit values), the number case is further split by printed width.
At run time the interpreter computes each hole's class (string id, or width) and picks
the matching variant.

Which strings a variable may hold is found by a fixpoint over all `assign`s in the
library, which is why programs that may meet in battle are compiled together: they
share the string table, and hive (team) variables are shared between programs.

Approximations (each reported in `Program.warnings` where it applies):

- A value assembled from literal text and a number (`assign v0 math ~v0 + 1`) is not
  representable; it is stored as an unknown word (numerically 0).
- An `iff code` or injected `data link command` whose text contains a variable is
  unsupported and ignored.
- A block header that cannot be parsed as pure conditions consuming exactly its own
  text (for example a malformed fixed-column condition) disables its body lines.
- Numbers wider than `MAX_WIDTH` characters, or non-integers, in a width-sensitive
  position are approximated.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAX_LINES = 2001  # the engine reads lines 1..2001; line 2002 wraps to 1
MAX_WIDTH = 8  # printed widths enumerated for width-sensitive numeric holes
MAX_VARIANTS_PER_LINE = 512
LABEL_HOLES = 3  # numbers a computed goto/gosub label may contain

# ---------------------------------------------------------------------------
# Enumerations shared with the JAX interpreter
# ---------------------------------------------------------------------------

# Operand sources
SRC_LIT, SRC_VAR, SRC_HIVE, SRC_TOK, SRC_RESULT = 0, 1, 2, 3, 4
# Operand read modes
MODE_FLOAT, MODE_INT, MODE_LEAD = 0, 1, 2

TOKENS = [
    "cur_fuel", "cur_ammo", "cur_life", "cur_head", "cur_score", "strat_x", "strat_y",
    "random", "burn", "alive", "scan", "shield", "heatlevel", "cloakstatus", "x_pos",
    "y_pos", "enemy_d", "enemy_h", "enemy_x", "enemy_y", "friend_d", "friend_h",
    "friend_x", "friend_y", "inbound",
]
TOK = {t: i for i, t in enumerate(TOKENS)}
# Engine state that conditions read but scripts cannot name as #tokens.
INTERNAL = ["barrier", "mis_ready", "gr_ready", "scan_raw"]
ITOK = {t: len(TOKENS) + i for i, t in enumerate(INTERNAL)}
N_TOKENS = len(TOKENS) + len(INTERNAL)

SCAN_RESULTS = ["nothing", "barrier", "enemy", "mine", "friend", "flag", "self"]
SCAN_ID = {s: i for i, s in enumerate(SCAN_RESULTS)}

# Condition types
(C_FALSE, C_VALUE, C_BUMP, C_NOT_BUMP, C_SHIELD_UP, C_SHIELD_DOWN, C_FACING, C_NOT_FACING,
 C_XCOORD, C_YCOORD, C_DAMAGE, C_FUEL, C_AMMO, C_RANDOM, C_NO_AMMO, C_SCAN_FOUND,
 C_MISSILE_READY, C_MISSILE_NOT_READY, C_GRENADE_READY, C_GRENADE_NOT_READY,
 C_DAMAGE_I, C_FUEL_I) = range(22)
# C_DAMAGE / C_FUEL compare against max_damage * pct / 100 computed at run time (pct read
# from a variable); C_DAMAGE_I / C_FUEL_I are the same with a literal pct, folded into an
# exact integer bound (OP_GE, OP_LE or OP_EQ against `a.lit`), because the engine's double
# arithmetic (10 * 0.7 = 7.000000000000001) does not survive float32.
# Comparison ops for C_VALUE / C_XCOORD / ...
OP_NONE, OP_GT, OP_LT, OP_EQ, OP_NE, OP_GE, OP_LE = range(7)
VALUE_OPS = {">": OP_GT, "<": OP_LT, "=": OP_EQ, "<>": OP_NE, ">=": OP_GE, "=>": OP_GE,
             "<=": OP_LE, "=<": OP_LE}
COL_OPS = {">": OP_GT, "<": OP_LT, "=": OP_EQ}

# Action types
(A_NOP, A_MOVE_FWD, A_MOVE_BACK, A_TURN_R, A_TURN_L, A_MINES_ON, A_MINES_OFF,
 A_LAUNCH_MISSILE, A_LAUNCH_GRENADE, A_SET_FUSE, A_ASSIGN, A_MATH, A_CMATH, A_GOTO,
 A_GOSUB, A_RETURN, A_IFF, A_DLINK_INJECT, A_DLINK_RES, A_DLINK_OTHER, A_RANDOM,
 A_SHIELD_UP, A_SHIELD_DOWN, A_CLOAK_ON, A_CLOAK_OFF, A_REPAIR, A_SCAN, A_FIRE,
 A_DISCHARGE, A_SELF_DESTRUCT) = range(30)
ACTION_NAMES = [
    "nop", "move_fwd", "move_back", "turn_r", "turn_l", "mines_on", "mines_off",
    "launch_missile", "launch_grenade", "set_fuse", "assign", "math", "cmath", "goto",
    "gosub", "return", "iff", "dlink_inject", "dlink_res", "dlink_other", "random",
    "shield_up", "shield_down", "cloak_on", "cloak_off", "repair", "scan", "fire",
    "discharge", "self_destruct",
]
# Commands that end the bug's turn ("mega" commands in the engine).
MEGA_ACTIONS = {A_TURN_R, A_TURN_L, A_LAUNCH_MISSILE, A_LAUNCH_GRENADE, A_DLINK_INJECT,
                A_DLINK_RES, A_SHIELD_UP, A_SHIELD_DOWN, A_CLOAK_ON, A_CLOAK_OFF, A_REPAIR,
                A_FIRE, A_DISCHARGE, A_SELF_DESTRUCT}

# Scan kinds
(SC_NONE, SC_PERIMETER, SC_POSITION, SC_RELATIVE, SC_GPS, SC_CROSS, SC_CORNER, SC_FORWARD,
 SC_LEFT, SC_RIGHT, SC_LONG) = range(11)

# Math (left-to-right) ops
M_NONE, M_ADD, M_SUB, M_MUL, M_DIV = range(5)

# cmath RPN ops
(R_PUSH, R_NEG, R_NOT, R_POW, R_MUL, R_DIV, R_IDIV, R_MOD, R_ADD, R_SUB, R_EQ, R_LT, R_GT,
 R_NE, R_LE, R_GE, R_AND, R_OR, R_XOR, R_FN) = range(20)
FUNCS = ["abs", "atn", "cos", "exp", "fix", "int", "log", "rnd", "sgn", "sin", "sqr", "tan"]
FN_ID = {f: i for i, f in enumerate(FUNCS)}

# Line kinds
K_NORMAL, K_COLON, K_ENDIF = 0, 1, 2

# Value class keys
CLASS_NUM = -1  # a number whose printed width does not matter


def width_class(w: int) -> int:
    return -1 - w  # -2 .. -(MAX_WIDTH+1)


# ---------------------------------------------------------------------------
# JS helpers
# ---------------------------------------------------------------------------

_JS_FLOAT = re.compile(r"[+-]?(Infinity|(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)")
_JS_WS = " \t\n\r\x0b\x0c ﻿"


def js_parse_float(s: str) -> float:
    s = s.lstrip(_JS_WS)
    m = _JS_FLOAT.match(s)
    if not m:
        return math.nan
    return float(m.group(0).replace("Infinity", "inf"))


def js_parse_int(s: str) -> float:
    s = s.lstrip(_JS_WS)
    m = re.match(r"[+-]?\d+", s)
    if not m:
        return math.nan
    return float(int(m.group(0)))


def or0(x: float) -> float:
    return 0.0 if (x != x or x == 0) else x


def js_num_str(x: float) -> str:
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == int(x) and abs(x) < 1e21:
        return str(int(x))
    r = repr(x)
    if "e" in r:
        return r  # close enough; such values are approximations anyway
    return r


def is_canonical_number(s: str) -> bool:
    v = js_parse_float(s)
    return v == v and not math.isinf(v) and js_num_str(v) == s


def collapse_spaces(s: str) -> str:
    out = []
    prev = ""
    for c in s:
        c = " " if c == "\t" else c
        if c == " " and prev == " ":
            continue
        out.append(c)
        prev = c
    return "".join(out)


def pad_math_chars(s: str) -> str:
    return "".join((" " + c) if c in "#~^()*+-/" else c for c in s)


def var_idx(name: str) -> int:
    ch = name[1]
    if "1" <= ch <= "9":
        return int(ch)
    if ch == "0":
        return 10
    return 11 + (ord(ch) - 97)


def hive_idx(name: str) -> int:
    ch = name[1]
    if "1" <= ch <= "9":
        return int(ch) + 1
    if ch == "0":
        return 11
    return 12 + (ord(ch) - 97)


# ---------------------------------------------------------------------------
# Symbolic text: holes
# ---------------------------------------------------------------------------

OPAQUE0 = 0xF000  # chr(OPAQUE0 + k): numeric hole k, printed width unknown
DIGIT0 = 0xE000  # chr(DIGIT0 + 16 k + i): character i of numeric hole k at a known width
RESULT_HOLE = 0xEFFF  # the computed result of math/cmath, re-parsed by `assign`


def opaque(k: int) -> str:
    return chr(OPAQUE0 + k)


def is_hole_char(c: str) -> bool:
    o = ord(c)
    return 0xE000 <= o <= 0xF0FF


def hole_of(c: str) -> Tuple[int, int]:
    """(hole, digit index) for a hole character; digit -1 for opaque holes."""
    o = ord(c)
    if o == RESULT_HOLE:
        return (-2, -1)
    if o >= OPAQUE0:
        return (o - OPAQUE0, -1)
    return ((o - DIGIT0) // 16, (o - DIGIT0) % 16)


class NeedWidth(Exception):
    def __init__(self, hole: int):
        super().__init__(hole)
        self.hole = hole


class Unsupported(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Operand:
    src: int = SRC_LIT
    idx: int = 0
    mode: int = MODE_FLOAT
    m: int = 0  # leading chars for MODE_LEAD
    lit: float = 0.0

    @staticmethod
    def literal(v: float) -> "Operand":
        return Operand(SRC_LIT, 0, MODE_FLOAT, 0, float(v))


@dataclasses.dataclass(frozen=True)
class Cond:
    ctype: int
    op: int = OP_NONE
    a: Operand = Operand()
    b: Operand = Operand()


@dataclasses.dataclass(frozen=True)
class Action:
    atype: int = A_NOP
    ttype: int = 0  # assign target: 0 none, 1 var, 2 hive
    tidx: int = 0
    vtype: int = 0  # assign value: 0 literal(num, sid), 1 operand copy, 2 result, 3 template
    vnum: float = 0.0
    vsid: int = -1
    a: Operand = Operand()
    b: Operand = Operand()
    iparam: int = 0  # scan kind / iff id / inject id / resource kind / static target
    terms: Tuple[Tuple[int, Operand], ...] = ()  # math: ((op, operand), ...)
    rpn: Tuple[Tuple[int, int, Operand], ...] = ()  # cmath: ((op, fn, operand), ...)
    rpn_err: bool = False
    labels: Tuple[Tuple[Tuple[float, ...], int], ...] = ()  # computed goto: ((values, line), ...)
    label_ops: Tuple[Operand, ...] = ()  # the numbers a computed label is built from
    dynamic_label: bool = False
    hole_copy: Tuple[int, int] = (-1, -1)  # assign from a hole: (hole id, -) resolved later
    dispatches: int = 1


@dataclasses.dataclass
class Variant:
    key: Tuple[int, ...]
    conds: List[Cond]
    action: Action
    header_ok: bool  # as a block header: pure conditions consuming exactly its text


@dataclasses.dataclass
class Line:
    number: int  # 1-based source line
    text: str
    kind: int
    then: bool
    short: bool
    queen: bool
    holes: List[Tuple[int, int]]  # (src, idx) per hole
    key_holes: List[int]  # holes that select variants
    key_width: List[bool]
    variants: List[Variant]  # mixed-radix order over key_choices (first hole outermost)
    key_choices: List[List[int]]  # per key hole, the value classes it is compiled for


# ---------------------------------------------------------------------------
# Library-wide state
# ---------------------------------------------------------------------------


class StringTable:
    def __init__(self):
        self.strings: List[str] = []
        self.ids: Dict[str, int] = {}

    def intern(self, s: str) -> int:
        if s not in self.ids:
            self.ids[s] = len(self.strings)
            self.strings.append(s)
        return self.ids[s]


@dataclasses.dataclass
class Settings:
    max_damage: int = 10
    start_fuel: int = 2500
    start_ammo: int = 30
    overheat_limit: int = 30
    shield_overheat_disabled: bool = False
    smart_clicks_disabled: bool = False
    strategy_node_disabled: bool = False
    blood_mode: bool = False
    last_cybug_bonus: bool = True


# Engine defaults (engine_core.js DEF); `_D` in the interpreter always reads these.
DEF_DAMSET = [0, 6, 9, 4, 7, 4, 7, 2, 5, 2, 5, 1, 4, 0, 3, 0, 2, 0, 1, 5, 5, 2, 1]
DEF_START_FUEL = 2500
DEF_START_AMMO = 30


class Context:
    """Per-line compile context: hole sources and the current variant assignment."""

    def __init__(self, prog: "ProgramCompiler", holes: List[Tuple[int, int]]):
        self.prog = prog
        self.holes = holes
        self.width: Dict[int, int] = {}

    def hole_operand(self, k: int, mode: int = MODE_FLOAT, m: int = 0) -> Operand:
        if k == -2:
            return Operand(SRC_RESULT, 0, mode, m, 0.0)
        src, idx = self.holes[k]
        return Operand(src, idx, mode, m, 0.0)


# ---------------------------------------------------------------------------
# Symbolic numeric parsing
# ---------------------------------------------------------------------------


def _hole_run(s: str, i: int) -> Tuple[int, int, int]:
    """At s[i] (a hole char): (hole, first digit index, run length)."""
    k, d = hole_of(s[i])
    if d < 0:
        return k, -1, 1
    n = 1
    while i + n < len(s) and is_hole_char(s[i + n]) and hole_of(s[i + n]) == (k, d + n):
        n += 1
    return k, d, n


def sym_parse_float_word(ctx: Context, w: str) -> Operand:
    """parseFloat(word) || 0 for a whole word."""
    if w and is_hole_char(w[0]):
        k, d, n = _hole_run(w, 0)
        if d <= 0 and (d < 0 or n == ctx.width.get(k, 0)) and n == len(w):
            return ctx.hole_operand(k, MODE_FLOAT)
        raise Unsupported("partial hole in word")
    for c in w:
        if is_hole_char(c):
            raise Unsupported("hole inside word")
    return Operand.literal(or0(js_parse_float(w)))


def sym_parse_int(ctx: Context, s: str) -> Operand:
    """parseInt(s, 10) || 0 on a symbolic string."""
    i = 0
    while i < len(s) and s[i] in _JS_WS:
        i += 1
    if i < len(s) and is_hole_char(s[i]):
        k, d, n = _hole_run(s, i)
        if d < 0:
            return ctx.hole_operand(k, MODE_INT)
        if d != 0:
            raise Unsupported("slice starts inside a number")
        w = ctx.width[k]
        if n >= w:
            return ctx.hole_operand(k, MODE_INT)
        return ctx.hole_operand(k, MODE_LEAD, n)
    # literal: sign then digits; a hole right after a sign cannot occur (holes are words)
    lit = []
    j = i
    if j < len(s) and s[j] in "+-":
        lit.append(s[j])
        j += 1
    while j < len(s) and s[j].isdigit() and not is_hole_char(s[j]):
        lit.append(s[j])
        j += 1
    return Operand.literal(or0(js_parse_int("".join(lit))))


def check_opaque_before(s: str, end: int):
    for c in s[:end]:
        if is_hole_char(c):
            k, d = hole_of(c)
            if d < 0 and k >= 0:
                raise NeedWidth(k)


def has_hole(s: str) -> bool:
    return any(is_hole_char(c) for c in s)


# ---------------------------------------------------------------------------
# cmath: symbolic port of the MODULE1.BAS evaluator, emitting RPN
# ---------------------------------------------------------------------------


class _Expr:
    def __init__(self, ctx: Context, s: str):
        self.ctx = ctx
        self.ei = s
        self.err = False
        self.tok = ""
        self.sp = ""
        self.hole: Optional[Operand] = None
        self.rpn: List[Tuple[int, int, Operand]] = []

    def nxt(self):
        self.tok = ""
        self.sp = ""
        self.hole = None
        ei = self.ei
        while ei and ei[0] in " \t\r\n":
            ei = ei[1:]
        if not ei:
            self.ei = ei
            return
        c0 = ei[0]
        if is_hole_char(c0):
            k, d, n = _hole_run(ei, 0)
            self.tok = "num"
            self.hole = self.ctx.hole_operand(k, MODE_FLOAT)
            self.sp = "0"
            self.ei = ei[n:]
            return
        c = c0.lower()
        if ("0" <= c <= "9") or c == ".":
            self.tok = "num"
            while ei and (("0" <= ei[0] <= "9") or ei[0] == "."):
                self.sp += ei[0]
                ei = ei[1:]
        elif ("a" <= c <= "z") or c == "_":
            self.tok = "id"
            while ei:
                ch2 = ei[0].lower()
                if ("a" <= ch2 <= "z") or ch2 == "_" or ("0" <= ch2 <= "9"):
                    self.sp += ei[0]
                    ei = ei[1:]
                else:
                    break
            kw = self.sp.lower()
            if "and eqv imp mod not or xor".find(kw) >= 0:
                self.tok = kw
        elif c in "<>":
            ei = ei[1:]
            self.tok = c
            if ei and ei[0] in "=>":
                self.tok += ei[0]
                ei = ei[1:]
        else:
            self.tok = c
            ei = ei[1:]
        if not self.sp:
            self.sp = self.tok
        self.ei = ei

    def match(self, t: str):
        if not self.err and self.tok != t:
            self.err = True
        self.nxt()

    def emit(self, op: int, fn: int = 0, operand: Operand = Operand()):
        self.rpn.append((op, fn, operand))

    def prs(self, p: int):
        if self.tok == "num":
            if self.hole is not None:
                self.emit(R_PUSH, 0, self.hole)
            else:
                self.emit(R_PUSH, 0, Operand.literal(or0(js_parse_float(self.sp))))
            self.nxt()
        elif self.tok == "-":
            self.nxt()
            self.prs(11)
            self.emit(R_NEG)
        elif self.tok == "not":
            self.nxt()
            self.prs(6)
            self.emit(R_NOT)
        elif self.tok == "(":
            self.nxt()
            self.prs(1)
            self.match(")")
        elif self.tok == "id":
            fun = self.sp.lower()
            self.nxt()
            self.match("(")
            self.prs(1)
            self.match(")")
            if fun in FN_ID:
                self.emit(R_FN, FN_ID[fun])
            else:
                self.err = True
        else:
            self.err = True
            self.emit(R_PUSH, 0, Operand.literal(0.0))
        binops = [
            (11, "^", R_POW, 12), (10, "*", R_MUL, 11), (10, "/", R_DIV, 11),
            (9, "\\", R_IDIV, 10), (8, "mod", R_MOD, 9), (7, "+", R_ADD, 8),
            (7, "-", R_SUB, 8), (6, "=", R_EQ, 7), (6, "<", R_LT, 7), (6, ">", R_GT, 7),
            (6, "<>", R_NE, 7), (6, "<=", R_LE, 7), (6, ">=", R_GE, 7), (5, "and", R_AND, 6),
            (4, "or", R_OR, 5), (3, "xor", R_XOR, 4),
        ]
        while not self.err:
            for lvl, t, op, q in binops:
                if p <= lvl and self.tok == t:
                    self.nxt()
                    self.prs(q)
                    self.emit(op)
                    break
            else:
                break

    def run(self):
        self.nxt()
        self.prs(1)
        if self.tok != "" and not self.err:
            self.err = True
        return self.rpn, self.err


# ---------------------------------------------------------------------------
# The dispatcher: symbolic port of engine_caicl.js dispatch()
# ---------------------------------------------------------------------------

FACING = {"north": 1, "east": 2, "south": 3, "west": 4}
_HW = r"[\w-]"
_RE_FACING = re.compile(r"^if facing (" + _HW + r"+) then\s*(.*)", re.S)
_RE_NOT_FACING = re.compile(r"^if not facing (" + _HW + r"+) then\s*(.*)", re.S)


def _col_cond(ctx, cmd, ctype, op_at, val_lo, val_hi, cont_at):
    check_opaque_before(cmd, cont_at)
    op = COL_OPS.get(cmd[op_at] if op_at < len(cmd) else "", OP_NONE)
    val = sym_parse_int(ctx, cmd[val_lo:val_hi])
    if op == OP_NONE:
        return ("cond", Cond(C_FALSE), "")
    cont = cmd[cont_at:].strip()
    if ctype in (C_DAMAGE, C_FUEL) and val.src == SRC_LIT:
        base = ctx.prog.lib.settings.max_damage if ctype == C_DAMAGE else DEF_START_FUEL
        th = base * (val.lit / 100)  # double, as in the engine
        itype = C_DAMAGE_I if ctype == C_DAMAGE else C_FUEL_I
        if op == OP_GT:
            return ("cond", Cond(itype, OP_GE, Operand.literal(math.floor(th) + 1)), cont)
        if op == OP_LT:
            return ("cond", Cond(itype, OP_LE, Operand.literal(math.ceil(th) - 1)), cont)
        if th != int(th):
            return ("cond", Cond(C_FALSE), "")
        return ("cond", Cond(itype, OP_EQ, Operand.literal(th)), cont)
    return ("cond", Cond(ctype, op, val), cont)


DYNAMIC = "\x00dynamic"  # an assigned value mixing text and several numbers (unrepresentable)
TEMPLATE_MARK = "\x03"  # where a template string's number goes
MAX_TEMPLATE = 80
FALSE_END = "\x02false\x02"  # a chain ended by a statically false condition


class ProgramCompiler:
    def __init__(self, lib: "Library", source: str, name: str, pid: int):
        self.lib = lib
        self.name = name
        self.pid = pid
        raw = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")[:MAX_LINES]
        self.raw = raw
        self.labels: Dict[str, int] = {}
        for i, ln in enumerate(raw):
            key = ln.strip().lower()
            if key not in self.labels:
                self.labels[key] = i  # raw index
        self.lines: List[Line] = []
        self.compact: Dict[int, int] = {}  # raw index -> compact index
        self.warnings: List[str] = []
        self.injected: List[Line] = []
        self.probing = False

    # -- string results reported to the library fixpoint ------------------
    def note_assign(self, ttype, slot, value):
        if not self.probing:
            self.lib.note_assign(self.pid, ttype, slot, value)

    def intern(self, text: str) -> int:
        return -1 if self.probing else self.lib.strings.intern(text)

    # -- label lookup ------------------------------------------------------
    def label_target(self, label_text: str) -> int:
        """Raw line index for a static label, or -1."""
        return self.labels.get(label_text, -1)

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, ctx: Context, cmd: str):
        """One call of engine dispatch(slot, cmd). Returns ("act", Action) |
        ("cond", Cond, continuation) | ("nop",)."""
        if cmd == "move forward":
            return ("act", Action(A_MOVE_FWD))
        if cmd == "move backward":
            return ("act", Action(A_MOVE_BACK))
        if cmd == "turn right":
            return ("act", Action(A_TURN_R))
        if cmd == "turn left":
            return ("act", Action(A_TURN_L))
        if cmd == "lay mines on":
            return ("act", Action(A_MINES_ON))
        if cmd == "lay mines off":
            return ("act", Action(A_MINES_OFF))
        if cmd == "launch missile":
            return ("act", Action(A_LAUNCH_MISSILE))
        if cmd == "launch grenade":
            return ("act", Action(A_LAUNCH_GRENADE))
        if cmd[:16] == "set grenade fuse":
            return ("act", Action(A_SET_FUSE, a=sym_parse_int(ctx, cmd[16:].strip())))
        if cmd[:21] == "if missile ready then":
            return ("cond", Cond(C_MISSILE_READY), cmd[22:].strip())
        if cmd[:25] == "if missile not ready then":
            return ("cond", Cond(C_MISSILE_NOT_READY), cmd[26:].strip())
        if cmd[:21] == "if grenade ready then":
            return ("cond", Cond(C_GRENADE_READY), cmd[22:].strip())
        if cmd[:25] == "if grenade not ready then":
            return ("cond", Cond(C_GRENADE_NOT_READY), cmd[26:].strip())
        if cmd[:6] == "assign":
            return self.parse_assign(ctx, cmd)
        if cmd[:4] == "math":
            parts = [p for p in cmd.split(" ") if p]
            if len(parts) >= 4:
                first = sym_parse_float_word(ctx, parts[3])
                terms = [(M_ADD, first)]
                xi = 4
                while xi + 1 < len(parts):
                    opw = parts[xi]
                    op = {"+": M_ADD, "-": M_SUB, "*": M_MUL, "/": M_DIV}.get(opw, M_NONE)
                    terms.append((op, sym_parse_float_word(ctx, parts[xi + 1])))
                    xi += 2
                kind, act = self.parse_assign(ctx, "assign " + parts[1] + " " + chr(RESULT_HOLE))
                if kind == "act" and act.atype == A_ASSIGN:
                    return ("act", dataclasses.replace(act, atype=A_MATH, terms=tuple(terms), dispatches=2))
                return ("act", Action(A_NOP, dispatches=2))
            return ("nop",)
        if cmd[:5] == "cmath":
            vn2 = cmd[6:8].strip()
            eq = cmd.find("=")
            expr = cmd[eq + 1:].strip()
            rpn, err = _Expr(ctx, expr).run()
            kind, act = self.parse_assign(ctx, "assign " + vn2 + " " + chr(RESULT_HOLE))
            if kind == "act" and act.atype == A_ASSIGN:
                act = dataclasses.replace(act, atype=A_CMATH, rpn=tuple(rpn), rpn_err=err, dispatches=2)
            else:
                act = Action(A_CMATH, rpn=tuple(rpn), rpn_err=err, dispatches=2)  # draws only
            if err and not any(r[0] == R_FN and r[1] == FN_ID["rnd"] for r in rpn):
                return ("nop",)
            return ("act", act)
        if cmd[:4] == "goto":
            return self.parse_jump(ctx, A_GOTO, cmd[5:].strip() + ":")
        if cmd[:5] == "gosub":
            return self.parse_jump(ctx, A_GOSUB, cmd[6:].strip() + ":")
        if cmd[:6] == "return":
            return ("act", Action(A_RETURN))
        if cmd[:8] == "iff code":
            txt = cmd[9:].strip()
            if has_hole(txt):
                self.warn(f"iff code with a variable ({txt!r}) ignored")
                return ("nop",)
            return ("act", Action(A_IFF, iparam=self.intern(txt)))
        if cmd[:9] == "data link":
            if "password" in cmd or "debug on" in cmd or "debug off" in cmd:
                return ("nop",)
            if cmd[:17] == "data link command":
                inject = cmd.strip()[17:].strip()
                if has_hole(inject):
                    self.warn(f"data link command with a variable ({inject!r}) ignored")
                    return ("act", Action(A_DLINK_OTHER))
                return ("act", Action(A_DLINK_INJECT, iparam=(-1 if self.probing else self.lib.inject_id(inject))))
            if cmd[:19] == "data link resource ":
                parts = cmd.strip().split(" ")
                kind = parts[3] if len(parts) > 3 else ""
                amount = sym_parse_int(ctx, parts[4]) if len(parts) > 4 else Operand.literal(0)
                res = {"fuel": 1, "ammo": 2}.get(kind, 0)
                return ("act", Action(A_DLINK_RES, iparam=res, a=amount))
            return ("act", Action(A_DLINK_OTHER))
        if cmd == "generate random":
            return ("act", Action(A_RANDOM))
        if cmd == "raise shield":
            return ("act", Action(A_SHIELD_UP))
        if cmd == "lower shield":
            return ("act", Action(A_SHIELD_DOWN))
        if cmd == "cloak on":
            return ("act", Action(A_CLOAK_ON))
        if cmd == "cloak off":
            return ("act", Action(A_CLOAK_OFF))
        if cmd == "attempt repairs":
            return ("act", Action(A_REPAIR))
        if cmd in ("debug on", "debug off", "beep"):
            return ("nop",)
        if cmd[:8] == "if value":
            wds = [w for w in cmd.split(" ") if w]
            if len(wds) >= 6:
                op = VALUE_OPS.get(wds[3], OP_NONE)
                if op == OP_NONE or has_hole(wds[3]):
                    return ("cond", Cond(C_FALSE), "")
                a = sym_parse_float_word(ctx, wds[2])
                b = sym_parse_float_word(ctx, wds[4])
                return ("cond", Cond(C_VALUE, op, a, b), " ".join(wds[6:]))
            return ("cond", Cond(C_FALSE), "")
        if cmd[:20] == "if bump barrier then":
            return ("cond", Cond(C_BUMP), cmd[21:].strip())
        if cmd[:24] == "if not bump barrier then":
            return ("cond", Cond(C_NOT_BUMP), cmd[25:].strip())
        if cmd[:20] == "if shield is up then":
            return ("cond", Cond(C_SHIELD_UP), cmd[21:].strip())
        if cmd[:22] == "if shield is down then":
            return ("cond", Cond(C_SHIELD_DOWN), cmd[23:].strip())
        fm = _RE_FACING.match(cmd)
        if fm:
            d = FACING.get(fm.group(1)) if not has_hole(fm.group(1)) else None
            if d is None:
                return ("cond", Cond(C_FALSE), "")
            return ("cond", Cond(C_FACING, d), fm.group(2).strip())
        nfm = _RE_NOT_FACING.match(cmd)
        if nfm:
            d = FACING.get(nfm.group(1)) if not has_hole(nfm.group(1)) else None
            if d is None:
                return ("cond", Cond(C_FALSE), "")
            return ("cond", Cond(C_NOT_FACING, d), nfm.group(2).strip())
        if cmd[:18] == "if x coordinate is":
            return _col_cond(ctx, cmd, C_XCOORD, 19, 21, 23, 29)
        if cmd[:18] == "if y coordinate is":
            return _col_cond(ctx, cmd, C_YCOORD, 19, 21, 23, 29)
        if cmd[:12] == "if damage is":
            return _col_cond(ctx, cmd, C_DAMAGE, 13, 15, 17, 23)
        if cmd[:10] == "if fuel is":
            return _col_cond(ctx, cmd, C_FUEL, 11, 13, 15, 21)
        if cmd[:10] == "if ammo is":
            return _col_cond(ctx, cmd, C_AMMO, 11, 13, 15, 21)
        if cmd[:12] == "if random is":
            check_opaque_before(cmd, 20)
            rn = sym_parse_int(ctx, cmd[13:14])
            return ("cond", Cond(C_RANDOM, OP_EQ, rn), cmd[20:].strip())
        if cmd[:15] == "if no ammo then":
            return ("cond", Cond(C_NO_AMMO), cmd[16:].strip())
        if cmd[:14] == "if scan found ":
            rest2 = cmd[14:]
            then_pos = rest2.find(" then")
            if then_pos >= 0:
                what = rest2[:then_pos].strip()
                sid = SCAN_ID.get(what, -1) if not has_hole(what) else -1
                if sid < 0:
                    return ("cond", Cond(C_FALSE), "")
                return ("cond", Cond(C_SCAN_FOUND, sid), rest2[then_pos + 5:].strip())
            return ("nop",)
        if "scan" in cmd:
            return self.parse_scan(ctx, cmd)
        if cmd == "fire weapon":
            return ("act", Action(A_FIRE))
        if cmd == "discharge energy":
            return ("act", Action(A_DISCHARGE))
        if cmd == "self destruct":
            return ("act", Action(A_SELF_DESTRUCT))
        return ("nop",)

    def parse_assign(self, ctx: Context, cmd: str):
        rest = cmd[7:].strip()
        vn = rest[:2]
        val = rest[3:].strip()
        if not vn or vn[0] not in "vh" or len(vn) < 2:
            if vn[:1] in ("v", "h") and len(vn) == 1:
                pass  # varIdx(name) with name[1] undefined: JS throws; treat as no-op
            return ("act", Action(A_NOP))
        if is_hole_char(vn[1]):
            return ("act", Action(A_NOP))
        ttype = 1 if vn[0] == "v" else 2
        tidx = self.lib.var_slot(var_idx(vn)) if ttype == 1 else self.lib.hive_slot(hive_idx(vn))
        if val and all(is_hole_char(c) for c in val):
            k, d, n = _hole_run(val, 0)
            if n == len(val) and d <= 0 and (d < 0 or n == ctx.width.get(k, 0)):
                if k == -2:
                    act = Action(A_ASSIGN, ttype, tidx, vtype=2)
                else:
                    src, idx = ctx.holes[k]
                    act = Action(A_ASSIGN, ttype, tidx, vtype=1, a=Operand(src, idx))
                    self.note_assign(ttype, tidx, ("copy", src, idx))
                return ("act", act)
        if has_hole(val) and DYNAMIC not in val and len(val) <= MAX_TEMPLATE:
            runs = []
            i = 0
            while i < len(val):
                if is_hole_char(val[i]):
                    k, d, n = _hole_run(val, i)
                    runs.append((i, k, d, n))
                    i += n
                else:
                    i += 1
            if len(runs) == 1:
                i, k, d, n = runs[0]
                if k >= 0 and d <= 0 and (d < 0 or n == ctx.width.get(k, 0)):
                    # text with one number in it: a template; the number lives in vnum
                    sid = self.intern(val[:i] + TEMPLATE_MARK + val[i + n:])
                    self.note_assign(ttype, tidx, ("sid", sid))
                    src, idx = ctx.holes[k]
                    return ("act", Action(A_ASSIGN, ttype, tidx, vtype=3, vsid=sid, a=Operand(src, idx)))
        if has_hole(val) or DYNAMIC in val:
            if DYNAMIC not in val:
                self.warn(f"assign of text with several numbers ({rest!r}) stored as an unknown word")
            sid = self.intern(DYNAMIC)
            self.note_assign(ttype, tidx, ("sid", sid))
            return ("act", Action(A_ASSIGN, ttype, tidx, vtype=0, vnum=0.0, vsid=sid))
        if is_canonical_number(val):
            return ("act", Action(A_ASSIGN, ttype, tidx, vtype=0, vnum=js_parse_float(val), vsid=-1))
        if val == "":
            # '' reads back as '0' (Cybug_var || '0')
            return ("act", Action(A_ASSIGN, ttype, tidx, vtype=0, vnum=0.0, vsid=-1))
        sid = self.intern(val)
        self.note_assign(ttype, tidx, ("sid", sid))
        return ("act", Action(A_ASSIGN, ttype, tidx, vtype=0, vnum=or0(js_parse_float(val)), vsid=sid))

    def parse_jump(self, ctx: Context, atype: int, label: str):
        if not has_hole(label):
            raw_target = self.label_target(label)
            return ("act", Action(atype, iparam=raw_target))
        # numeric holes: a table from their values to the labels that fit the pattern
        parts, ops = [], []
        i = 0
        while i < len(label):
            if is_hole_char(label[i]):
                k, d, n = _hole_run(label, i)
                if d > 0:
                    raise Unsupported("label cuts into a number")
                ops.append(ctx.hole_operand(k))
                parts.append(r"(-?\d+(?:\.\d+)?)")
                i += n
            else:
                parts.append(re.escape(label[i]))
                i += 1
        if len(ops) > LABEL_HOLES:
            self.warn(f"computed label with more than {LABEL_HOLES} variables ({label!r}) unsupported")
            return ("act", Action(atype, iparam=-1))
        pat = re.compile("".join(parts) + r"\Z", re.S)
        table = {}
        for key, raw_i in self.labels.items():
            m = pat.match(key)
            if m and all(is_canonical_number(g) for g in m.groups()):
                vals = tuple(js_parse_float(g) for g in m.groups())
                vals = vals + (0.0,) * (LABEL_HOLES - len(vals))
                if vals not in table:
                    table[vals] = raw_i
        ops = ops + [Operand.literal(0.0)] * (LABEL_HOLES - len(ops))
        return ("act", Action(atype, labels=tuple(sorted(table.items())), label_ops=tuple(ops),
                              dynamic_label=True, iparam=-1))

    def parse_scan(self, ctx: Context, cmd: str):
        tl = cmd.strip().lower()
        if tl == "scan perimeter":
            return ("act", Action(A_SCAN, iparam=SC_PERIMETER))
        if tl[:13] == "scan position":
            return ("act", Action(A_SCAN, iparam=SC_POSITION, a=sym_parse_int(ctx, tl[14:])))
        if tl[:13] == "scan relative":
            return ("act", Action(A_SCAN, iparam=SC_RELATIVE, a=sym_parse_int(ctx, tl[14:])))
        if tl[:11] == "gps scan x ":
            xl = sym_parse_int(ctx, tl[11:])
            yp = tl.find(" y ")
            yl = sym_parse_int(ctx, tl[yp + 3:]) if yp >= 0 else Operand.literal(0)
            return ("act", Action(A_SCAN, iparam=SC_GPS, a=xl, b=yl))
        simple = {"cross scan": SC_CROSS, "corner scan": SC_CORNER, "scan forward": SC_FORWARD,
                  "scan left": SC_LEFT, "scan right": SC_RIGHT, "long range scan": SC_LONG}
        if tl in simple:
            return ("act", Action(A_SCAN, iparam=simple[tl]))
        return ("nop",)

    def warn(self, msg: str):
        if not self.probing and msg not in self.warnings:
            self.warnings.append(msg)

    # -- chains ------------------------------------------------------------
    def chain(self, ctx: Context, text: str):
        """doAction's redirect loop over `text`: (conds, action, leftover-or-None).

        leftover is the continuation when the chain ends on a condition."""
        conds: List[Cond] = []
        cmd = text
        for _ in range(64):
            r = self.dispatch(ctx, cmd.strip().lower())
            if r[0] == "cond":
                conds.append(r[1])
                if r[1].ctype == C_FALSE:
                    return conds, Action(A_NOP), FALSE_END
                cmd = r[2]
                continue
            if r[0] == "act":
                return conds, r[1], None
            return conds, Action(A_NOP), cmd
        return conds, Action(A_NOP), None

    # -- substitution ------------------------------------------------------
    def substitute(self, s: str, holes: List[Tuple[int, int]], hole_texts: Dict[int, str]):
        out = []
        for w in s.split(" "):
            wl = w.lower()
            rep = None
            if len(wl) == 3 and wl[0] == "~":
                src = None
                if wl[1] == "h":
                    src = (SRC_HIVE, self.lib.hive_slot(hive_idx(wl[1:])))
                elif wl[1] == "v":
                    src = (SRC_VAR, self.lib.var_slot(var_idx(wl[1:])))
                if src is not None:
                    k = len(holes)
                    holes.append(src)
                    rep = hole_texts.get(k, opaque(k))
            elif len(wl) > 1 and wl[0] == "#":
                rep = self.lib.token_text(wl[1:])
                if rep is None:
                    k = len(holes)
                    holes.append((SRC_TOK, TOK[wl[1:]]))
                    rep = hole_texts.get(k, opaque(k))
            out.append(rep if rep is not None else w)
        return " ".join(out)

    def compile_text(self, s: str, number: int) -> Line:
        """Compile one non-empty, non-comment line (already space-collapsed)."""
        colon = ":" in s
        kind = K_NORMAL
        if not colon and s.strip().lower() in ("end if", "endif"):
            kind = K_ENDIF
        if colon:
            kind = K_COLON
        stl = s.strip().lower()
        then = (not colon) and stl.endswith("then")
        short = len(stl) <= 4
        body = s
        if "math" in body.lower():
            body = collapse_spaces(pad_math_chars(body))

        holes: List[Tuple[int, int]] = []
        self.substitute(body, holes, {})
        n_holes = len(holes)
        string_dom = [self.lib.domain(self.pid, holes[k]) for k in range(n_holes)]

        def compile_variant(assign: Dict[int, object]):
            """assign: hole -> sid (string), ('w', w) width, or None (opaque number)."""
            texts = {}
            ctx = Context(self, [])
            for k, a in assign.items():
                if isinstance(a, int):
                    texts[k] = self.lib.strings.strings[a].replace(TEMPLATE_MARK, opaque(k))
                elif isinstance(a, tuple):
                    texts[k] = "".join(chr(DIGIT0 + 16 * k + i) for i in range(a[1]))
                    ctx.width[k] = a[1]
            hs: List[Tuple[int, int]] = []
            txt = self.substitute(body, hs, texts)
            ctx.holes = hs
            slc = txt.strip().lower()
            conds, act, _ = self.chain(ctx, slc)
            # As a block header the line is followed by " <body>"; it is usable as one
            # when it is pure conditions whose continuation is exactly that body.
            sentinel = "\x01body\x01"
            self.probing = True  # the probe's own actions are not real: record nothing
            try:
                h_conds, h_act, left = self.chain(ctx, slc + " " + sentinel)
            finally:
                self.probing = False
            header_ok = h_act.atype == A_NOP and (left is FALSE_END or (
                left is not None and left.strip() == sentinel))
            if header_ok and (len(h_conds) != len(conds) or h_conds != conds):
                header_ok = False
            return conds, act, header_ok, h_conds

        # Pass 1: which holes need a printed width (only probed for the number case)
        width_holes = set()
        str_choices = [[None] + sorted(dom) for dom in string_dom]

        def combos(choices):
            out = [{}]
            for k, ch in enumerate(choices):
                out = [{**o, k: c} for o in out for c in ch]
                if len(out) > MAX_VARIANTS_PER_LINE:
                    raise Unsupported("too many variants")
            return out

        try:
            base = combos(str_choices)
        except Unsupported:
            self.warn(f"line {number}: too many string combinations; compiled as a no-op")
            return Line(number, s, kind, then, short, False, holes, [], [],
                        [Variant((), [], Action(), False)], [])

        def resolve(assign):
            a = dict(assign)
            for _ in range(n_holes + 1):
                try:
                    return compile_variant(a), a
                except NeedWidth as e:
                    width_holes.add(e.hole)
                    a[e.hole] = ("w", 2)
            raise Unsupported("width resolution")

        for a in base:
            try:
                resolve(a)
            except Unsupported:
                pass

        key_holes = sorted(set(k for k in range(n_holes) if string_dom[k]) | width_holes)
        choices = []
        for k in range(n_holes):
            ch: List[object] = []
            if k in width_holes:
                ch += [("w", w) for w in range(1, MAX_WIDTH + 1)]
            else:
                ch.append(None)
            ch += sorted(string_dom[k])
            choices.append(ch)
        key_choices = []
        for k in key_holes:
            key_choices.append([c if isinstance(c, int) else
                                (width_class(c[1]) if isinstance(c, tuple) else CLASS_NUM)
                                for c in choices[k]])
        variants = []
        try:
            all_combos = combos(choices)
        except Unsupported:
            self.warn(f"line {number}: too many variants; compiled as a no-op")
            all_combos = []
        for a in all_combos:
            key = []
            for k in key_holes:
                v = a[k]
                if isinstance(v, int):
                    key.append(v)
                elif isinstance(v, tuple):
                    key.append(width_class(v[1]))
                else:
                    key.append(CLASS_NUM)
            try:
                conds, act, header_ok, _ = compile_variant(a)
            except NeedWidth:
                conds, act, header_ok = [], Action(), False
                self.warn(f"line {number}: unresolved width")
            except Unsupported as e:
                conds, act, header_ok = [], Action(), False
                self.warn(f"line {number}: {e}")
            variants.append(Variant(tuple(key), conds, act, header_ok))
        if not variants:
            variants = [Variant(tuple(CLASS_NUM for _ in key_holes), [], Action(), False)]
        if then and not all(v.header_ok for v in variants):
            self.warn(f"line {number}: block header is not a plain condition; its body may misbehave")
        queen = stl == "queen"
        return Line(number, s, kind, then, short, queen, holes, key_holes,
                    [k in width_holes for k in key_holes], variants, key_choices)

    def compile(self):
        self.lines = []
        self.compact = {}
        self.warnings = []
        for i, raw in enumerate(self.raw):
            tr = raw.strip()
            if tr == "" or tr[0] == ";":
                continue
            self.compact[i] = len(self.lines)
            self.lines.append(self.compile_text(collapse_spaces(raw), i + 1))
        self.injected = [self.compile_text(collapse_spaces(t), 0) for t in self.lib.injects]

    def target(self, raw_index: int) -> int:
        """Compact index of the first executable line at or after a raw line."""
        if raw_index < 0:
            return -1
        for j in range(raw_index, len(self.raw)):
            if j in self.compact:
                return self.compact[j]
        return len(self.lines)  # past the end: wraps


class Library:
    """A set of CAICL programs compiled together (shared strings and hive domains)."""

    def __init__(self, sources: Sequence[str], names: Optional[Sequence[str]] = None,
                 settings: Optional[Settings] = None, default_iffs: Sequence[str] = ()):
        self.settings = settings or Settings()
        self.strings = StringTable()
        self.strings.intern(DYNAMIC)
        for s in default_iffs:
            self.strings.intern(s)
        self.var_slots: Dict[int, int] = {}
        self.hive_slots: Dict[int, int] = {}
        self.var_dom: Dict[int, set] = {}
        self.hive_dom: Dict[int, set] = {}
        self.copies: set = set()
        self.injects: List[str] = []
        self._inject_ids: Dict[str, int] = {}
        names = list(names) if names is not None else [f"prog{i}" for i in range(len(sources))]
        self.programs = [ProgramCompiler(self, src, nm, i)
                         for i, (src, nm) in enumerate(zip(sources, names))]
        for _ in range(20):
            before = self._signature()
            for p in self.programs:
                p.compile()
            self._propagate()
            if self._signature() == before:
                break
        else:
            raise RuntimeError("string-domain fixpoint did not converge")

    def _signature(self):
        return (tuple(sorted((k, tuple(sorted(v))) for k, v in self.var_dom.items())),
                tuple(sorted((k, tuple(sorted(v))) for k, v in self.hive_dom.items())),
                len(self.injects), len(self.strings.strings))

    def var_slot(self, idx: int) -> int:
        return self.var_slots.setdefault(idx, len(self.var_slots))

    def hive_slot(self, idx: int) -> int:
        return self.hive_slots.setdefault(idx, len(self.hive_slots))

    # Variables are private to a bug, so their domains are per program; hive variables
    # are shared by every bug with the same IFF, so theirs span the library.
    def _dom_key(self, pid: int, src_t: int, slot: int):
        return (pid, slot) if src_t == SRC_VAR else slot

    def domain(self, pid: int, src: Tuple[int, int]) -> set:
        st, i = src
        if st == SRC_VAR:
            return set(self.var_dom.get((pid, i), ()))
        if st == SRC_HIVE:
            return set(self.hive_dom.get(i, ()))
        return set()

    def note_assign(self, pid: int, ttype: int, slot: int, value):
        dst_t = SRC_VAR if ttype == 1 else SRC_HIVE
        dom = self.var_dom if ttype == 1 else self.hive_dom
        dst = self._dom_key(pid, dst_t, slot)
        dom.setdefault(dst, set())
        if value[0] == "sid":
            dom[dst].add(value[1])
        elif value[1] in (SRC_VAR, SRC_HIVE):
            self.copies.add((dst_t, dst, value[1], self._dom_key(pid, value[1], value[2])))

    def _propagate(self):
        changed = True
        while changed:
            changed = False
            for dst_t, dst, src_t, src in self.copies:
                s = (self.var_dom if src_t == SRC_VAR else self.hive_dom).get(src, set())
                d = (self.var_dom if dst_t == SRC_VAR else self.hive_dom).setdefault(dst, set())
                if not s <= d:
                    d |= s
                    changed = True

    def inject_id(self, text: str) -> int:
        if text not in self._inject_ids:
            self._inject_ids[text] = len(self.injects)
            self.injects.append(text)
        return self._inject_ids[text]

    def token_text(self, tok: str) -> Optional[str]:
        """Literal text for constant tokens; None for run-time tokens."""
        st = self.settings
        consts = {
            "max_fuel": DEF_START_FUEL, "set_ammo": DEF_START_AMMO, "max_life": st.max_damage,
            "overheatoff": 0, "snodeoff": 0, "smartclickoff": 0, "bloodmodeon": 0,
            "lastcybugon": 1, "overheatset": 30,
        }
        if tok in consts:
            return js_num_str(consts[tok])
        if tok in TOK:
            return None
        if tok[:6] == "damset":
            n = js_parse_int(tok[6:])
            v = DEF_DAMSET[int(n)] if n == n and 1 <= n <= 22 else 0
            return js_num_str(v)
        return "#" + tok


# ---------------------------------------------------------------------------
# Lowering to arrays
# ---------------------------------------------------------------------------

OPF = ["src", "idx", "mode", "m", "lit"]  # operand fields


def _cols(*groups):
    cols, i = {}, 0
    for g in groups:
        if isinstance(g, tuple):  # (prefix, n operands)
            pre, n = g
            for j in range(n):
                for f in OPF:
                    cols[f"{pre}{j}_{f}"] = i
                    i += 1
        else:
            cols[g] = i
            i += 1
    return cols


# Columns of the per-variant action table and the per-condition table.
ACT = _cols("atype", "ttype", "tidx", "vtype", "vnum", "vsid", ("a", 1), ("b", 1),
            ("l", LABEL_HOLES), "iparam", "t0", "tn", "r0", "rn", "rerr", "lab0", "labn",
            "dyn", "disp", "header_ok", "c0", "cn")
COND = _cols("op", ("a", 1), ("b", 1), "scale")  # holds iff a <op> b * scale
TERM = _cols("op", ("a", 1))
RPN = _cols("op", "fn", ("a", 1))


def normalize_cond(c: Cond, st: Settings):
    """Every condition as `a <op> b * scale` over operands (engine state via tokens)."""
    tok = lambda name: Operand(SRC_TOK, TOK[name] if name in TOK else ITOK[name])  # noqa: E731
    lit = Operand.literal
    t = c.ctype
    if t == C_VALUE:
        return c.op, c.a, c.b, 1.0
    if t in (C_BUMP, C_NOT_BUMP):
        return OP_EQ, tok("barrier"), lit(1 if t == C_BUMP else 0), 1.0
    if t in (C_SHIELD_UP, C_SHIELD_DOWN):
        return OP_EQ, tok("shield"), lit(1 if t == C_SHIELD_UP else 0), 1.0
    if t in (C_FACING, C_NOT_FACING):
        return (OP_EQ if t == C_FACING else OP_NE), tok("cur_head"), lit(c.op), 1.0
    if t == C_XCOORD:
        return c.op, tok("x_pos"), c.a, 1.0
    if t == C_YCOORD:
        return c.op, tok("y_pos"), c.a, 1.0
    if t == C_DAMAGE:
        return c.op, tok("cur_life"), c.a, st.max_damage / 100
    if t == C_FUEL:
        return c.op, tok("cur_fuel"), c.a, DEF_START_FUEL / 100
    if t == C_DAMAGE_I:
        return c.op, tok("cur_life"), c.a, 1.0
    if t == C_FUEL_I:
        return c.op, tok("cur_fuel"), c.a, 1.0
    if t == C_AMMO:
        return c.op, tok("cur_ammo"), c.a, 1.0
    if t == C_RANDOM:
        return OP_EQ, tok("random"), c.a, 1.0
    if t == C_NO_AMMO:
        return OP_LE, tok("cur_ammo"), lit(0), 1.0
    if t == C_SCAN_FOUND:
        return OP_EQ, tok("scan_raw"), lit(c.op), 1.0
    if t in (C_MISSILE_READY, C_MISSILE_NOT_READY):
        return OP_EQ, tok("mis_ready"), lit(1 if t == C_MISSILE_READY else 0), 1.0
    if t in (C_GRENADE_READY, C_GRENADE_NOT_READY):
        return OP_EQ, tok("gr_ready"), lit(1 if t == C_GRENADE_READY else 0), 1.0
    return OP_NONE, lit(0), lit(0), 1.0  # C_FALSE


def _op_vals(o: Operand):
    return [o.src, o.idx, o.mode, o.m, o.lit]


@dataclasses.dataclass
class Tables:
    """Compiled programs as flat arrays.

    Programs are concatenated: program p's compact lines are rows
    `line0[p] .. line0[p] + n_rows[p]` of the `line_*` arrays (its `n_lines[p]` real
    lines, then one injected `data link command` line per library inject). Pointers
    into `acts`, `conds`, `terms`, `rpn` and `labels` are absolute; line numbers held in
    actions (jump targets) are relative to the program."""

    arrays: Dict[str, np.ndarray]
    dims: Dict[str, int]
    names: List[str]
    warnings: List[List[str]]
    strings: List[str]


_PTR = {"acts": [("c0", "conds"), ("t0", "terms"), ("r0", "rpn"), ("lab0", "labels")]}


def lower(lib: Library, select: Optional[Sequence[int]] = None) -> Tables:
    idx = list(select) if select is not None else list(range(len(lib.programs)))
    progs = [lib.programs[i] for i in idx]
    per = [_lower_program(p) for p in progs]
    K = max(1, max(d["line_keysrc"].shape[1] for d in per))
    CH = max(1, max(d["line_choices"].shape[2] for d in per))
    off = {k: 0 for k in ("lines", "acts", "conds", "terms", "rpn", "labels")}
    parts: Dict[str, list] = {}
    line0, n_lines, n_rows = [], [], []
    for d in per:
        L = d["line_kind"].shape[0]
        line0.append(off["lines"])
        n_lines.append(int(d["n_lines"]))
        n_rows.append(L)
        d = dict(d)
        d["line_var0"] = d["line_var0"] + off["acts"]
        acts = d["acts"].copy()
        for col, tab in _PTR["acts"]:
            acts[:, ACT[col]] += off[tab]
        d["acts"] = acts
        for k in ("line_keysrc", "line_keywidth", "line_nchoice", "line_choices"):
            v = d[k]
            shape = list(v.shape)
            shape[1] = K
            if k == "line_choices":
                shape[2] = CH
            fill = {"line_choices": -99, "line_nchoice": 1}.get(k, 0)
            w = np.full(shape, fill, v.dtype)
            w[tuple(slice(0, n) for n in v.shape)] = v
            d[k] = w
        for k, v in d.items():
            if v.ndim >= 1:
                parts.setdefault(k, []).append(v)
        off["lines"] += L
        for tab in ("acts", "conds", "terms", "rpn", "labels"):
            off[tab] += d[tab].shape[0]
    arrays = {}
    for k, vs in parts.items():
        a = np.concatenate(vs)
        if a.shape[0] == 0:  # keep one row so gathers have something to read
            a = np.zeros((1,) + a.shape[1:], a.dtype)
        arrays[k] = a
    arrays["line0"] = np.array(line0, np.int32)
    arrays["n_lines"] = np.array(n_lines, np.int32)
    arrays["n_rows"] = np.array(n_rows, np.int32)
    dims = {
        "rows": int(max(n_rows)),
        "keys": K,
        "choices": CH,
        "max_conds": max(1, max(int(d["max_nconds"]) for d in per)),
        "max_terms": max(1, max(int(d["max_nterms"]) for d in per)),
        "max_rpn": max(1, max(int(d["max_nrpn"]) for d in per)),
        "max_labels": max(1, max(int(d["max_nlabels"]) for d in per)),
        "vars": max(1, len(lib.var_slots)),
        "hives": max(1, len(lib.hive_slots)),
    }
    for k in ("max_nconds", "max_nterms", "max_nrpn", "max_nlabels"):
        arrays.pop(k, None)
    return Tables(arrays, dims, [p.name for p in progs], [list(p.warnings) for p in progs],
                  list(lib.strings.strings))


def _lower_program(p: ProgramCompiler) -> Dict[str, np.ndarray]:
    lines = p.lines + p.injected
    L = len(lines)
    K = max([len(ln.key_holes) for ln in lines] + [1])
    CH = max([len(c) for ln in lines for c in ln.key_choices] + [1])
    out = {
        "line_kind": np.zeros(L, np.int32),
        "line_then": np.zeros(L, np.bool_),
        "line_short": np.zeros(L, np.bool_),
        "line_queen": np.zeros(L, np.bool_),
        "line_number": np.zeros(L, np.int32),
        "line_keysrc": np.zeros((L, K, 2), np.int32),
        "line_keywidth": np.zeros((L, K), np.bool_),
        "line_nchoice": np.ones((L, K), np.int32),
        "line_choices": np.full((L, K, CH), -99, np.int32),
        "line_var0": np.zeros(L, np.int32),
    }
    acts, conds, terms, rpns, labels = [], [], [], [], []
    mx = dict(nconds=0, nterms=0, nrpn=0, nlabels=0)
    for li, ln in enumerate(lines):
        out["line_kind"][li] = ln.kind
        out["line_then"][li] = ln.then
        out["line_short"][li] = ln.short
        out["line_queen"][li] = ln.queen
        out["line_number"][li] = ln.number
        for j, k in enumerate(ln.key_holes):
            out["line_keysrc"][li, j] = ln.holes[k]
            out["line_keywidth"][li, j] = ln.key_width[j]
            ch = ln.key_choices[j]
            out["line_nchoice"][li, j] = len(ch)
            out["line_choices"][li, j, :len(ch)] = ch
        out["line_var0"][li] = len(acts)
        for v in ln.variants:
            a = v.action
            row = np.zeros(len(ACT))
            row[ACT["atype"]] = a.atype
            row[ACT["ttype"]] = a.ttype
            row[ACT["tidx"]] = a.tidx
            row[ACT["vtype"]] = a.vtype
            row[ACT["vnum"]] = a.vnum
            row[ACT["vsid"]] = a.vsid
            row[ACT["a0_src"]:ACT["a0_src"] + 5] = _op_vals(a.a)
            row[ACT["b0_src"]:ACT["b0_src"] + 5] = _op_vals(a.b)
            for j, o in enumerate(a.label_ops):
                row[ACT[f"l{j}_src"]:ACT[f"l{j}_src"] + 5] = _op_vals(o)
            iparam = a.iparam
            if a.atype in (A_GOTO, A_GOSUB) and not a.dynamic_label:
                iparam = p.target(iparam)
            row[ACT["iparam"]] = iparam
            row[ACT["t0"]], row[ACT["tn"]] = len(terms), len(a.terms)
            row[ACT["r0"]], row[ACT["rn"]] = len(rpns), len(a.rpn)
            row[ACT["rerr"]] = a.rpn_err
            row[ACT["lab0"]], row[ACT["labn"]] = len(labels), len(a.labels)
            row[ACT["dyn"]] = a.dynamic_label
            row[ACT["disp"]] = a.dispatches
            row[ACT["header_ok"]] = v.header_ok
            row[ACT["c0"]], row[ACT["cn"]] = len(conds), len(v.conds)
            acts.append(row)
            for c in v.conds:
                op, ca, cb, scale = normalize_cond(c, p.lib.settings)
                conds.append([op] + _op_vals(ca) + _op_vals(cb) + [scale])
            for op, o in a.terms:
                terms.append([op] + _op_vals(o))
            for op, fn, o in a.rpn:
                rpns.append([op, fn] + _op_vals(o))
            for vals, raw_i in a.labels:
                labels.append(list(vals) + [p.target(raw_i)])
            mx["nconds"] = max(mx["nconds"], len(v.conds))
            mx["nterms"] = max(mx["nterms"], len(a.terms))
            mx["nrpn"] = max(mx["nrpn"], len(a.rpn))
            mx["nlabels"] = max(mx["nlabels"], len(a.labels))

    def arr(rows, width):
        return np.array(rows, np.float64).reshape(-1, width) if rows else np.zeros((0, width))

    out.update({
        "n_lines": np.array(len(p.lines), np.int32),
        "acts": arr(acts, len(ACT)),
        "conds": arr(conds, len(COND)),
        "terms": arr(terms, len(TERM)),
        "rpn": arr(rpns, len(RPN)),
        "labels": arr(labels, LABEL_HOLES + 1),
        "max_nconds": np.int32(mx["nconds"]), "max_nterms": np.int32(mx["nterms"]),
        "max_nrpn": np.int32(mx["nrpn"]), "max_nlabels": np.int32(mx["nlabels"]),
    })
    return out
