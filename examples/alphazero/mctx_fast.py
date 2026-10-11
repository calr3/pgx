"""A faster, exactly equivalent replacement for mctx's backward pass (mctx 0.0.6).

mctx.search.backward walks each leaf-to-root path in a while_loop vmapped over the batch,
carrying the whole tree. With a batched predicate, JAX runs the loop for the longest path and
selects between the old and new carry on every pass, so every pass copies every [B, N, A]
array (children_values, children_visits: ~200 MB each at B=1024, N=65, A=768). This version
walks the paths first with a small carry, then writes all updates with one scatter per array
(a path never repeats a node, so the scatters cannot collide). Trees and actions are
identical (check_mctx_backward.py); search bookkeeping was 8.4x faster at B=256 on CPU, and a
64-simulation self-play step (E28's network, batch 1024) 20% faster on the GPU.

Use: `import mctx_fast; mctx_fast.install()` before any search is traced (train.py does
so with mctx_fast_backward=true).
"""

import jax
import jax.numpy as jnp
from mctx._src import search as _search
from mctx._src.tree import Tree

def backward_fast(tree, leaf_index):
    """Same updates as mctx.search.backward, without carrying the tree through a
    batched while_loop: walk each leaf-to-root path first (small carry), then
    write every update with one scatter per array (paths never repeat a node)."""
    B, N = tree.node_values.shape
    L = N  # a path has at most N - 1 edges

    def walk(t, leaf):  # one batch element; t is the unbatched tree (closure, not carried)
        def cond(c):
            return c[0] != Tree.ROOT_INDEX
        def body(c):
            index, leaf_value, child_value, k, P, A, PV, PC, CV, CC = c
            parent = t.parents[index]
            count = t.node_visits[parent]
            action = t.action_from_parent[index]
            leaf_value = t.children_rewards[parent, action] + t.children_discounts[parent, action] * leaf_value
            parent_value = (t.node_values[parent] * count + leaf_value) / (count + 1.0)
            P, A = P.at[k].set(parent), A.at[k].set(action)
            PV, PC = PV.at[k].set(parent_value), PC.at[k].set(count + 1)
            CV, CC = CV.at[k].set(child_value), CC.at[k].set(t.children_visits[parent, action] + 1)
            return parent, leaf_value, parent_value, k + 1, P, A, PV, PC, CV, CC
        leaf = jnp.asarray(leaf, jnp.int32)
        v0 = t.node_values[leaf]
        init = (leaf, v0, v0, jnp.int32(0), jnp.full(L, N, jnp.int32), jnp.zeros(L, jnp.int32),
                jnp.zeros(L, t.node_values.dtype), jnp.zeros(L, t.node_visits.dtype),
                jnp.zeros(L, t.children_values.dtype), jnp.zeros(L, t.children_visits.dtype))
        out = jax.lax.while_loop(cond, body, init)
        return out[4:]

    P, A, PV, PC, CV, CC = jax.vmap(walk)(tree, leaf_index)
    b = jnp.arange(B)[:, None]
    return tree.replace(
        node_values=tree.node_values.at[b, P].set(PV, mode="drop"),
        node_visits=tree.node_visits.at[b, P].set(PC, mode="drop"),
        children_values=tree.children_values.at[b, P, A].set(CV, mode="drop"),
        children_visits=tree.children_visits.at[b, P, A].set(CC, mode="drop"),
    )


_STOCK = _search.backward


def install(enabled: bool = True) -> None:
    """Route mctx's searches through backward_fast (or back to the stock version)."""
    _search.backward = backward_fast if enabled else _STOCK
