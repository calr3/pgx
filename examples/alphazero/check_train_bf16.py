"""Is bfloat16 training (train_bf16) as good as float32? Run once per precision, same seed:

  python examples/alphazero/check_train_bf16.py <ckpt> <out.json> [train.py config overrides...]

Loads the checkpoint's weights and optimizer state and its run's replay buffer
(data_state.pkl), then (1) on 8 fixed held-out batches compares this precision's loss and
gradients with float32's (and float32 gradients across batches, as a noise yardstick), and
(2) makes 240 updates on a fixed sequence of disjoint minibatches and augmentation keys, scoring the
held-out loss in float32 every 40 updates.
"""
import json
import os
import pickle
import sys
import time

ckpt_path, out_path = sys.argv[1:3]
sys.argv = ["train.py"] + sys.argv[3:]
sys.path.insert(0, "examples/alphazero")

import jax
import jax.numpy as jnp
import numpy as np

import __main__

import train  # noqa: E402
from replay_buffer import ReplayBuffer

__main__.Config = train.Config
cfg = train.config
T = cfg.training_batch_size
put = lambda x: jax.tree_util.tree_map(lambda y: jnp.asarray(y)[None], x)

ckpt = pickle.load(open(ckpt_path, "rb"))
t = time.time()
with open(os.path.join(os.path.dirname(ckpt_path), "data_state.pkl"), "rb") as f:
    state = pickle.load(f)
buf = ReplayBuffer(state["replay_buffer"]["capacity"] if "capacity" in state["replay_buffer"] else 4 * 2**18)
buf.load_state_dict(state["replay_buffer"])
del state
print(f"buffer: {buf.num_samples} samples, loaded in {time.time() - t:.0f} s", flush=True)

rng = np.random.default_rng(0)
perm = rng.permutation(buf.num_samples)
held = [buf.gather(perm[i * T:(i + 1) * T]) for i in range(8)]
train_idx = perm[8 * T:]


def loss_and_grad(params, data, dtype):
    def f(p):
        q = train.cast_floating(p, dtype)
        fwd = train.train_forward if dtype == train.train_dtype else train.forward
        (logits, value), _ = fwd.apply(q, ckpt["model"][1], data.obs, is_eval=False, moves=data.moves)
        pl = jnp.sum(train.optax.softmax_cross_entropy(logits, data.policy_tgt) * data.policy_mask) / jnp.maximum(
            jnp.sum(data.policy_mask), 1)
        vl = jnp.mean(train.optax.l2_loss(value, data.value_tgt) * data.mask)
        return pl + vl, (pl, vl)
    (total, (pl, vl)), g = jax.value_and_grad(f, has_aux=True)(params)
    return total, pl, vl, g


_lg32 = jax.jit(lambda p, d: loss_and_grad(p, d, jnp.float32))
_lgx = jax.jit(lambda p, d: loss_and_grad(p, d, train.train_dtype))


def chunked(fn, chunks=8):
    """fn over `chunks` equal slices of the batch, averaged (memory: no remat here)."""
    def run(p, d):
        n = d.obs.shape[0] // chunks
        outs = [fn(p, jax.tree_util.tree_map(lambda x: x[i * n:(i + 1) * n], d)) for i in range(chunks)]
        return jax.tree_util.tree_map(lambda *xs: sum(xs) / chunks, *outs)
    return run


lg32, lgx = chunked(_lg32), chunked(_lgx)
flat = lambda g: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g)])
cos = lambda a, b: float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))

params = jax.tree_util.tree_map(jnp.asarray, ckpt["model"][0])
report = {"dtype": train.train_dtype.__name__, "same_batch": []}
prev32 = None
for i, d in enumerate(held):
    d = jax.tree_util.tree_map(jnp.asarray, d)
    t32, p32, v32, g32 = lg32(params, d)
    tx, px, vx, gx = lgx(params, d)
    a, b = flat(g32), flat(gx)
    row = {"policy32": float(p32), "policyX": float(px), "value32": float(v32), "valueX": float(vx),
           "grad_cos": cos(a, b), "grad_norm_ratio": float(jnp.linalg.norm(b) / jnp.linalg.norm(a)),
           "grad_cos_vs_other_batch32": cos(a, prev32) if prev32 is not None else None}
    prev32 = a
    report["same_batch"].append(row)
    print(row, flush=True)


def held_loss(params):
    tot = []
    for d in held:
        d = jax.tree_util.tree_map(jnp.asarray, d)
        t32, p32, v32, _ = lg32(params, d)
        tot.append((float(p32), float(v32)))
    return np.mean(tot, axis=0).tolist()


m, o = put(ckpt["model"]), put(ckpt["opt_state"])
key = jax.random.PRNGKey(5)
report["race"] = [(0, held_loss(params))]
t = time.time()
for u in range(1, 241):  # 248 disjoint batches fit in a full buffer after the held-out 8
    d = put(buf.gather(train_idx[(u - 1) * T:u * T]))
    key, k = jax.random.split(key)
    m, o, pl, vl = train.train(m, o, d, k[None])[:4]
    if u % 40 == 0:
        h = held_loss(jax.tree_util.tree_map(lambda x: x[0], m[0]))
        report["race"].append((u, h))
        print(f"update {u}: train {float(pl[0]):.4f}/{float(vl[0]):.4f}  held-out {h[0]:.4f}/{h[1]:.4f}  "
              f"({time.time() - t:.0f} s)", flush=True)
json.dump(report, open(out_path, "w"), indent=1)
