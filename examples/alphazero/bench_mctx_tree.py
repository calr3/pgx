"""Measure mctx tree overhead (no real network) and test a batched backward.

Usage: python examples/alphazero/bench_mctx_tree.py B NUM_ACTIONS EMB_SIZE SIMS[,SIMS...]
See MCTX_PERF_REVIEW.md for results.
"""
import functools, sys, time
import jax, jax.numpy as jnp, numpy as np
import mctx
from mctx._src import search as S
from mctx._src import tree as tree_lib

Tree = tree_lib.Tree


def backward_batched(tree, leaf_index):
  """Same as search.backward, but batched by hand instead of vmap(while_loop).

  vmap of a while_loop with a batched predicate selects the *whole* carry
  (including [B, N, A] arrays) every iteration; here each iteration is a
  B-element gather + B-element scatter, and finished rows write nothing.
  """
  B, N = tree.node_values.shape
  br = jnp.arange(B)
  leaf_index = jnp.asarray(leaf_index, jnp.int32)

  def cond(s):
    return jnp.any(s[2] != Tree.ROOT_INDEX)

  def body(s):
    tree, leaf_value, index = s
    active = index != Tree.ROOT_INDEX
    parent = jnp.where(active, tree.parents[br, index], 0)
    action = jnp.where(active, tree.action_from_parent[br, index], 0)
    count = tree.node_visits[br, parent]
    reward = tree.children_rewards[br, parent, action]
    new_leaf = reward + tree.children_discounts[br, parent, action] * leaf_value
    parent_value = (tree.node_values[br, parent] * count + new_leaf) / (count + 1.0)
    child_value = tree.node_values[br, index]
    child_count = tree.children_visits[br, parent, action] + 1
    p = jnp.where(active, parent, N)  # out of bounds -> dropped
    tree = tree.replace(
        node_values=tree.node_values.at[br, p].set(parent_value, mode="drop"),
        node_visits=tree.node_visits.at[br, p].set(count + 1, mode="drop"),
        children_values=tree.children_values.at[br, p, action].set(child_value, mode="drop"),
        children_visits=tree.children_visits.at[br, p, action].set(child_count, mode="drop"))
    return tree, jnp.where(active, new_leaf, leaf_value), jnp.where(active, parent, index)

  tree, _, _ = jax.lax.while_loop(
      cond, body, (tree, tree.node_values[br, leaf_index], leaf_index))
  return tree


def make(B, A, emb, policy):
  def recurrent_fn(params, key, action, e):
    # Cheap pseudo-network: deterministic function of the embedding.
    h = (e["h"] * 31 + action[:, None] + 7) % 1000
    logits = jnp.sin(h[:, :1].astype(jnp.float32) + jnp.arange(A) * 0.37)
    value = jnp.cos(h[:, 0].astype(jnp.float32))
    out = mctx.RecurrentFnOutput(
        reward=jnp.zeros(B), discount=-jnp.ones(B), prior_logits=logits, value=value * 0.5)
    return out, {"h": h}

  def run(key):
    root = mctx.RootFnOutput(
        prior_logits=jax.random.normal(key, (B, A)), value=jnp.zeros(B),
        embedding={"h": jnp.zeros((B, emb), jnp.int32)})
    fn = mctx.gumbel_muzero_policy if policy == "gumbel" else mctx.muzero_policy
    out = fn(params=(), rng_key=key, root=root, recurrent_fn=recurrent_fn,
             num_simulations=SIMS)
    return out.action_weights, out.search_tree
  return jax.jit(run)


def bench(f, key, reps=3):
  jax.block_until_ready(f(key))
  ts = []
  for i in range(reps):
    t = time.perf_counter(); jax.block_until_ready(f(jax.random.fold_in(key, i))); ts.append(time.perf_counter() - t)
  return min(ts)


if __name__ == "__main__":
  B, A, EMB = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
  key = jax.random.PRNGKey(0)
  for policy in ["gumbel", "muzero"]:
    for SIMS in [int(s) for s in sys.argv[4].split(",")]:
      orig = S.backward
      f0 = make(B, A, EMB, policy); t0 = bench(f0, key); r0 = f0(key)
      S.backward = backward_batched
      f1 = make(B, A, EMB, policy); t1 = bench(f1, key); r1 = f1(key)
      S.backward = orig
      same = all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves(r0), jax.tree.leaves(r1)))
      print(f"{policy:6s} B={B} A={A} emb={EMB} sims={SIMS:4d}: orig {t0*1e3:8.1f} ms  batched-backward {t1*1e3:8.1f} ms  x{t0/t1:4.2f}  identical={same}", flush=True)
