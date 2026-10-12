"""Check the pgx A.I. Wars engine against the original web engine, tick by tick.

Runs random battles between the bundled Cybugs in both the reference JS engine (via
run_ref.js under Node) and pgx, feeding the JS engine pgx's random draws, and compares
every bug's position, heading, stats, projectiles, line pointer and nesting, plus the
whole map, after every tick.

    tools/ai_wars/fetch.sh                      # once: engine, maps and Cybugs
    JAX_PLATFORMS=cpu python tools/ai_wars/conformance.py --n 100 [--seed 1] [--bugs a.ai,b.ai]

AIWARS_SRC (default ~/calr3gh/aiwars_src) is fetch.sh's directory; NODE the node binary.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pgx._src.games import ai_wars as W  # noqa: E402
from pgx._src.games import ai_wars_caicl as C  # noqa: E402
from pgx.ai_wars import DEFAULT_IFFS, device_tables  # noqa: E402

SRC = os.environ.get("AIWARS_SRC", os.path.expanduser("~/calr3gh/aiwars_src"))
NODE = os.environ.get("NODE", "node")
HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_ref.js")
FIELDS = ["x", "y", "h", "damage", "fuel", "ammo", "score", "shield", "overheat", "penalty",
          "cloak", "mine", "dead", "burn", "fuel_rate", "queen", ("mistat", "mis_on"),
          ("misx", "mis_x"), ("misy", "mis_y"), ("grstat", "gr_on"), ("grax", "gr_x"),
          ("gray", "gr_y"), ("grad", "gr_d"), ("grat", "gr_t"), "random", "barrier",
          ("nest", "nest_level"), ("sub", "curnest")]
SCAN = {s: i for i, s in enumerate(C.SCAN_RESULTS)}


def draw_tape(key, n_bugs, max_turns):
    """The draws pgx makes: tape[t, b, k] for the k-th draw of bug b's turn in tick t."""
    def tk(t):
        tkey = jax.random.fold_in(key, t)
        return jax.vmap(lambda b: jax.random.bits(jax.random.fold_in(tkey, b), (W.DRAWS_PER_TURN,),
                                                  jnp.uint32) >> 8)(jnp.arange(n_bugs))
    return np.asarray(jax.jit(jax.vmap(tk))(jnp.arange(max_turns)), np.uint32)


def run_reference(srcs, rows, starts, heads, node, key, workdir, max_turns=2000):
    S = len(srcs)
    tape = draw_tape(key, S, max_turns)
    tape.tofile(os.path.join(workdir, "tape.bin"))
    spec = dict(map_text="\n".join(rows), node=dict(x=int(node[0]), y=int(node[1])), tape_file="tape.bin",
                n_draws=W.DRAWS_PER_TURN, max_turns=max_turns, trace_vars=False,
                cybugs=[dict(slot=s + 1, script_text=srcs[s], iff=DEFAULT_IFFS[s], team=0,
                             x=int(starts[s][0]), y=int(starts[s][1]), h=int(heads[s])) for s in range(S)])
    with open(os.path.join(workdir, "battle.json"), "w") as f:
        json.dump(spec, f)
    out = os.path.join(workdir, "out.json")
    subprocess.run([NODE, HARNESS, os.path.join(workdir, "battle.json"), out], check=True,
                   env=dict(os.environ, AIWARS_SRC=SRC))
    with open(out) as f:
        return json.load(f)


def compare(lib, eng, TT, tick, pidx, rows, starts, heads, node, key, ref):
    """First tick where pgx and the reference disagree, with the differences (None, [] if never)."""
    S = len(pidx)
    progs = [lib.programs[i] for i in pidx]
    iffs = [lib.strings.ids[DEFAULT_IFFS[s]] for s in range(S)]
    st = eng.init(key, TT, pidx, W.map_grid(rows), np.array([[x, y, h] for (x, y), h in zip(starts, heads)]),
                  (node[0], node[1], True), iffs)

    def check(st, r):
        errs = []
        for s in range(S):
            rb = r["bugs"][s]
            for f in FIELDS:
                jf, wf = (f, f) if isinstance(f, str) else f
                wv, jv = int(np.asarray(getattr(st, wf))[s]), int(rb[jf])
                if wv != jv:
                    errs.append(f"bug{s} {jf}: js={jv} pgx={wv}")
            if int(st.scan[s]) != SCAN.get(rb["scan"], -9):
                errs.append(f"bug{s} scan: js={rb['scan']} pgx={C.SCAN_RESULTS[int(st.scan[s])]}")
            raw = rb["mem"] - 1  # the reference counts raw lines from 1
            jm = progs[s].target(raw) if raw < len(progs[s].raw) else len(progs[s].lines)
            if int(st.mem[s]) != jm:
                errs.append(f"bug{s} line: js={rb['mem']} (compact {jm}) pgx={int(st.mem[s])}")
        g = np.asarray(eng.board(st)).reshape(-1)
        gs = "".join("." if v == 0 else "#" if v == 20 else "F" if v == 30 else "9" if v == 40 else chr(48 + v)
                     for v in g)
        if gs != r["grid"]:
            diff = [(i % W.COLS, i // W.COLS, r["grid"][i], gs[i]) for i in range(len(gs)) if gs[i] != r["grid"][i]]
            errs.append(f"map differs at (x, y, js, pgx) {diff[:6]}")
        if bool(st.running) != (r["mode"] == 1):
            errs.append(f"running js={r['mode']} pgx={bool(st.running)}")
        return errs

    trace = ref["trace"]
    errs = check(st, trace[0])
    if errs:
        return 0, errs
    for t in range(1, len(trace)):
        st = tick(TT, st)
        errs = check(st, trace[t])
        if errs:
            return t, errs
    if int(st.winner) != ref["winner"] - 1:
        return len(trace), [f"winner js={ref['winner'] - 1} pgx={int(st.winner)}"]
    return None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bugs", default="", help="comma-separated subset of Cybug files")
    a = ap.parse_args()
    names = a.bugs.split(",") if a.bugs else sorted(os.listdir(f"{SRC}/plain"))
    maps = sorted(m for m in os.listdir(f"{SRC}/Maps") if m.endswith(".map"))
    r = random.Random(a.seed)
    battles = [(r.sample(names, 2), r.choice(maps), a.seed * 100000 + i) for i in range(a.n)]
    used = sorted(set(p for pr, _, _ in battles for p in pr))
    srcs = [open(f"{SRC}/plain/{n}", encoding="latin1").read() for n in used]
    lib = C.Library(srcs, used, default_iffs=DEFAULT_IFFS)
    T = C.lower(lib)
    eng = W.Engine(T.dims, 2)
    TT = device_tables(T)
    tick = jax.jit(eng.tick)
    ok = 0
    with tempfile.TemporaryDirectory() as wd:
        for pair, mp, seed in battles:
            rng = np.random.default_rng(seed)
            info = W.parse_map(open(f"{SRC}/Maps/{mp}", encoding="latin1").read(), mp)
            rows, starts = W.place(info, 2, rng)
            heads = [int(rng.integers(1, 5)) for _ in range(2)]
            cands = W.node_candidates(rows)
            node = cands[int(rng.integers(len(cands)))]
            key = jax.random.PRNGKey(seed)
            pidx = [used.index(p) for p in pair]
            t0 = time.time()
            ref = run_reference([srcs[i] for i in pidx], rows, starts, heads, node, key, wd)
            bad, errs = compare(lib, eng, TT, tick, pidx, rows, starts, heads, node, key, ref)
            tag = f"{seed} {pair} {mp} ticks={ref['ticks']}"
            if bad is None:
                ok += 1
                print("OK  ", tag, f"{time.time() - t0:.1f}s", flush=True)
            else:
                print("FAIL", tag, "at tick", bad, errs[:6], flush=True)
    print(f"{ok}/{a.n} identical")
    sys.exit(0 if ok == a.n else 1)


if __name__ == "__main__":
    main()
