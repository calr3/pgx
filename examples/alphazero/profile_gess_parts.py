"""Where a gess_joint training iteration's time goes, on the GPU: device time of the network's
forward pass and of the env step at the self-play batch (in a compiled loop, so no Python
dispatch is counted), and of one training update as configured.

  python examples/alphazero/profile_gess_parts.py <ckpt> [train.py config overrides...]

Run from pgx's root with the run's own config overrides and resume_as_new_run=False. Compare
with bench_gess_selfplay.py's time per self-play step: per simulation, the rest is mctx.
"""
import pickle
import sys
import time

ckpt_path = sys.argv[1]
sys.argv = ["train.py"] + sys.argv[2:]
sys.path.insert(0, "examples/alphazero")

import jax
import jax.numpy as jnp

import __main__

import train  # noqa: E402
from trajectories import Sample

__main__.Config = train.Config
cfg = train.config
B, T = cfg.selfplay_batch_size, cfg.training_batch_size
REPS = 20


def device_ms(label, body, carry):
    """ms per call of body (carry -> carry), run REPS times in one compiled loop."""
    loop = jax.jit(lambda c: jax.lax.fori_loop(0, REPS, lambda i, c: body(c), c))
    jax.block_until_ready(loop(carry))
    t = time.perf_counter()
    jax.block_until_ready(loop(carry))
    ms = (time.perf_counter() - t) / REPS * 1000
    print(f"{label:<52} {ms:8.1f} ms", flush=True)
    return ms


ckpt = pickle.load(open(ckpt_path, "rb"))
model = train.cast_floating(ckpt["model"], train.selfplay_dtype)

# Positions some way into games.
s = jax.vmap(train.env.init)(jax.random.split(jax.random.PRNGKey(0), B))
step = jax.jit(jax.vmap(train.env.step))
key = jax.random.PRNGKey(1)
for i in range(16):
    key, k = jax.random.split(key)
    a = jax.random.categorical(k, jnp.where(s.legal_action_mask, 0.0, -jnp.inf))
    n = step(s, a)
    s = jax.tree_util.tree_map(lambda x, y: jnp.where(n.terminated.reshape((-1,) + (1,) * (x.ndim - 1)), y, x), n, s)


def fwd(c):
    obs, acc = c
    (logits, value), _ = train.selfplay_forward.apply(model[0], model[1], obs, is_eval=True, moves=s._moves)
    return obs, acc + value.sum() + logits[:, 0].sum()  # keep the result live


f_ms = device_ms(f"network forward, batch {B} ({train.selfplay_dtype.__name__})", fwd, (s.observation, 0.0))
a0 = jnp.argmax(s.legal_action_mask, axis=1)  # first legal move of each position


def env_step(c):
    st, acc = c
    nxt = jax.vmap(train.env.step)(st, a0)
    return st, acc + nxt._moves[:, 0].sum() + nxt.observation[:, 0, 0, 0].sum()


e_ms = device_ms(f"env step (moves + observation), batch {B}", env_step, (s, 0))

# One training update, as configured: a batch of real observations and move lists, uniform
# policy targets over the legal moves.
reps = T // B
obs = jnp.tile(s.observation, (reps, 1, 1, 1))
moves = jnp.tile(s._moves, (reps, 1))
legal = (moves >= 0).astype(jnp.float32)
data = Sample(
    obs=obs,
    policy_tgt=legal / legal.sum(axis=1, keepdims=True),
    value_tgt=jnp.zeros(T),
    mask=jnp.ones(T, jnp.bool_),
    policy_mask=jnp.ones(T, jnp.bool_),
    moves=moves,
)
full = ckpt["model"]
opt_state = train.optimizer.init(full[0])
put = lambda x: jax.tree_util.tree_map(lambda y: y[None], x)
m, o, d = put(full), put(opt_state), put(data)
k = jax.random.PRNGKey(3)[None]
for _ in range(3):  # compile, and again for the device-resident outputs fed back
    m, o = train.train(m, o, d, k)[:2]
jax.block_until_ready(m)
t = time.perf_counter()
for _ in range(10):
    m, o, p_loss = train.train(m, o, d, k)[:3]
    p_loss.mean().item()  # as train.py's loop does after each update
u_ms = (time.perf_counter() - t) / 10 * 1000
print(f"training update, batch {T} (micro_batches={cfg.train_micro_batches}, remat={cfg.gf_remat}) "
      f"{u_ms:8.1f} ms", flush=True)
print(f"-> training per iteration ({cfg.num_updates_per_iter} updates): {u_ms * cfg.num_updates_per_iter / 60000:.1f} min")
