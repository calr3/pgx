"""mctx_fast.backward_fast against mctx's own backward pass: identical trees and actions,
and the time of the search bookkeeping (a dummy network, so the search is all bookkeeping).

  python examples/alphazero/check_mctx_backward.py [batch] [num_actions] [simulations]
"""

import sys
import time

import jax
import jax.numpy as jnp
import mctx
import numpy as np

sys.path.insert(0, "examples/alphazero")
import mctx_fast  # noqa: E402

B, A, SIMS = (int(x) for x in (sys.argv[1:4] if len(sys.argv) > 3 else (256, 768, 64)))


def recurrent_fn(params, key, action, emb):
    # Cheap and deterministic in (embedding, action), so trees grow realistically.
    h = (emb * 31 + action[:, None] * 7 + 3) % 1009
    logits = jnp.sin(h.astype(jnp.float32) * 0.37 + jnp.arange(A)[None] * 0.11) * 2.0
    value = jnp.cos(h[:, 0].astype(jnp.float32) * 0.13) * 0.9
    reward = jnp.where(h[:, 0] % 97 == 0, 1.0, 0.0)
    out = mctx.RecurrentFnOutput(reward=reward, discount=-jnp.ones(B), prior_logits=logits, value=value)
    return out, h


def run(fast):
    mctx_fast.install(fast)
    root = mctx.RootFnOutput(
        prior_logits=jnp.sin(jnp.arange(B * A).reshape(B, A) * 0.01), value=jnp.zeros(B),
        embedding=jnp.arange(B)[:, None])
    f = jax.jit(lambda k: mctx.gumbel_muzero_policy(None, k, root, recurrent_fn, SIMS, gumbel_scale=1.0))
    out = f(jax.random.PRNGKey(1))
    jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(3):
        jax.block_until_ready(f(jax.random.PRNGKey(1)))
    return out, (time.perf_counter() - t) / 3


stock, t_stock = run(False)
fast, t_fast = run(True)
same = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
    jax.tree_util.tree_leaves(stock.search_tree), jax.tree_util.tree_leaves(fast.search_tree)))
print(f"batch {B}, {A} actions, {SIMS} sims [{jax.default_backend()}]: stock {t_stock * 1000:.0f} ms, "
      f"fast {t_fast * 1000:.0f} ms ({t_stock / t_fast:.1f}x); trees identical {same}, "
      f"actions identical {bool((stock.action == fast.action).all())}")
