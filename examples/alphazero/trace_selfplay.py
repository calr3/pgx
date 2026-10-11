"""Profile a few self-play steps on the GPU and summarise the trace.

  python examples/alphazero/trace_selfplay.py <ckpt> <trace_dir> [train.py config overrides...]

Runs train.selfplay once to compile, then once more under jax.profiler.trace, and reports
from the Perfetto trace: wall time, the share of it the GPU was busy (union of kernel
intervals: the rest is launch gaps, loop-condition syncs and host work), and the kernels
taking the most GPU time. Run from pgx's root with the run's overrides, resume_as_new_run=False
and a short max_num_steps (e.g. 4).
"""

import collections
import glob
import gzip
import json
import pickle
import sys
import time

ckpt_path, trace_dir = sys.argv[1:3]
sys.argv = ["train.py"] + sys.argv[3:]
sys.path.insert(0, "examples/alphazero")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import __main__  # noqa: E402

import train  # noqa: E402

__main__.Config = train.Config
ckpt = pickle.load(open(ckpt_path, "rb"))
B = train.config.selfplay_batch_size
s = jax.vmap(train.env.init)(jax.random.split(jax.random.PRNGKey(0), B))
m_rep, s_rep = jax.tree_util.tree_map(lambda x: x[None], (ckpt["model"], s))
sims = train.config.num_simulations


def run():
    out = train.selfplay(m_rep, s_rep, jax.random.PRNGKey(2)[None], sims)
    jax.block_until_ready(out)


run()  # compile
t = time.perf_counter()
with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
    run()
wall = time.perf_counter() - t

path = sorted(glob.glob(f"{trace_dir}/**/*perfetto_trace.json.gz", recursive=True))[-1]
events = json.load(gzip.open(path))
events = events["traceEvents"] if isinstance(events, dict) else events
gpu_pids = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
            and "gpu" in str(e.get("args", {}).get("name", "")).lower()}
kernels = [e for e in events if e.get("ph") == "X" and e.get("pid") in gpu_pids and e.get("dur")]
intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
busy, end = 0.0, -1.0
for a, b in intervals:
    if a > end:
        busy += b - a
        end = b
    elif b > end:
        busy += b - end
        end = b
span = (intervals[-1][1] - intervals[0][0]) if intervals else 0.0
by_name = collections.Counter()
count = collections.Counter()
for e in kernels:
    by_name[e["name"][:70]] += e["dur"]
    count[e["name"][:70]] += 1
steps = train.config.max_num_steps
print(f"{steps} self-play steps at {sims} sims, batch {B}: wall {wall:.2f} s; GPU kernel span "
      f"{span / 1e6:.2f} s, busy {busy / 1e6:.2f} s ({100 * busy / max(span, 1):.0f}% of the span); "
      f"{len(kernels)} kernels ({len(kernels) / steps / sims:.0f} a simulation)")
for name, us in by_name.most_common(12):
    print(f"  {us / 1e3:9.1f} ms  {100 * us / max(busy, 1):5.1f}%  x{count[name]:<6} {name}")
