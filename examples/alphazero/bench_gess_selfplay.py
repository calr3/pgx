"""GPU benchmark for the gess_joint move generator and self-play.

  python examples/alphazero/bench_gess_selfplay.py <new|old> <positions.npz> <ckpt|init> [train.py config overrides...]

Run from pgx's root. positions.npz holds `boards` (n, 400) int8 and `colors` (n,) int32 of
real positions (at least the self-play batch); the config overrides are the run's own, with
resume_as_new_run=False and a short max_num_steps (e.g. 16).

old: gess_joint.legal_moves is replaced by legal_moves_reference (the per-centre version)
before anything is traced. Prints: the generator alone on 1024 real positions; the network's
forward pass at the self-play batch; one self-play step at full and at fast simulations.
"""
import pickle
import sys
import time

mode, pos_path, ckpt_path = sys.argv[1:4]
sys.argv = ["train.py"] + sys.argv[4:]
sys.path.insert(0, "examples/alphazero")

import jax
import jax.numpy as jnp
import numpy as np

import pgx.gess_joint as gj

if mode == "old":
    gj.legal_moves = gj.legal_moves_reference

import __main__

import train  # noqa: E402  (builds env, forward, selfplay from argv)
from pgx._src.games.gess import GameState

__main__.Config = train.Config


def timed(label, f, *args, reps=5):
    jax.block_until_ready(f(*args))  # compile
    t = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(f(*args))
    ms = (time.perf_counter() - t) / reps * 1000
    print(f"{mode:>3}  {label:<46} {ms:9.1f} ms", flush=True)
    return ms


B = train.config.selfplay_batch_size
d = np.load(pos_path)
boards = jnp.asarray(d["boards"][:B])
colors = jnp.asarray(d["colors"][:B])
gen = jax.jit(jax.vmap(lambda b, c: gj.legal_moves(GameState(color=c, board=b))))
timed(f"legal_moves x{B}", gen, boards, colors)

if ckpt_path == "init":  # freshly initialised weights for the configured network (timing only)
    _s = jax.vmap(train.env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    ckpt = {"model": train.forward.init(jax.random.PRNGKey(0), _s.observation, moves=train.moves_of(_s))}
else:
    ckpt = pickle.load(open(ckpt_path, "rb"))
model = ckpt["model"]
s0 = jax.vmap(train.env.init)(jax.random.split(jax.random.PRNGKey(0), B))
fwd = jax.jit(lambda m, s: train.selfplay_forward.apply(
    m[0], m[1], s.observation, is_eval=True, moves=train.moves_of(s))[0])
mb = train.cast_floating(model, train.selfplay_dtype)
timed(f"network forward x{B} (selfplay dtype)", fwd, mb, s0)

# Self-play: a few steps from positions some way into games.
rep = jax.tree_util.tree_map(lambda x: x[None], (model, s0))
step = jax.jit(jax.vmap(train.env.step))
s = s0
key = jax.random.PRNGKey(1)
for i in range(16):  # move the games on a little with random moves
    key, k = jax.random.split(key)
    a = jax.random.categorical(k, jnp.where(s.legal_action_mask, 0.0, -jnp.inf))
    n = step(s, a)
    s = jax.tree_util.tree_map(lambda x, y: jnp.where(n.terminated.reshape((-1,) + (1,) * (x.ndim - 1)), y, x), n, s)
m_rep, s_rep = jax.tree_util.tree_map(lambda x: x[None], (model, s))
steps = train.config.max_num_steps
for sims in (train.config.num_simulations, train.config.fast_num_simulations):
    ms = timed(f"selfplay {steps} steps, {sims} sims (playout cap off)",
               lambda: train.selfplay(m_rep, s_rep, jax.random.PRNGKey(2)[None], sims), reps=2)
    print(f"{mode:>3}  -> {ms / steps:.1f} ms a step at {sims} sims", flush=True)
