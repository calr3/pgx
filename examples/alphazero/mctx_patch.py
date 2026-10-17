"""Faster drop-in for mctx's backward pass; importing this module applies it.

mctx's `search.backward` is `jax.vmap` over a `lax.while_loop`. With a batched
loop predicate JAX turns every iteration into `select(pred, new, old)` over the
whole carry - the tree - so `children_values` and `children_visits`
([B, N, num_actions] each) are fully rewritten on every step up the tree. That
makes the search quadratic in `num_simulations`.

`batched_backward` loops over the batch by hand instead: each step gathers and
scatters B elements, and rows that have already reached the root write nothing.
The resulting tree is bit-identical to mctx's (tests/test_mctx_patch.py). On CPU
it cut tree time 1.5x at 32 simulations and 4x at 128 (Gumbel, B=128, 400
actions; see MCTX_PERF_REVIEW.md).

`search.search` looks `backward` up as a module global at call time, so
replacing that attribute is enough. Import this before anything is jitted.
"""

import jax
import jax.numpy as jnp
from mctx._src import search as _search

Tree = _search.Tree

# Kept for the equivalence test, and so the patch can be undone.
stock_backward = getattr(_search, "stock_backward", _search.backward)


def batched_backward(tree: Tree, leaf_index: jnp.ndarray) -> Tree:
    """Same contract as `mctx._src.search.backward`, for a batched tree."""
    batch_size, num_nodes = tree.node_values.shape
    rows = jnp.arange(batch_size)
    leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)

    def cond_fun(loop_state):
        _, _, index = loop_state
        return jnp.any(index != Tree.ROOT_INDEX)

    def body_fun(loop_state):
        tree, leaf_value, index = loop_state
        active = index != Tree.ROOT_INDEX
        parent = jnp.where(active, tree.parents[rows, index], Tree.ROOT_INDEX)
        action = jnp.where(active, tree.action_from_parent[rows, index], 0)
        count = tree.node_visits[rows, parent]
        reward = tree.children_rewards[rows, parent, action]
        new_leaf_value = reward + tree.children_discounts[rows, parent, action] * leaf_value
        parent_value = (tree.node_values[rows, parent] * count + new_leaf_value) / (count + 1.0)
        children_values = tree.node_values[rows, index]
        children_counts = tree.children_visits[rows, parent, action] + 1
        # Finished rows get an out-of-bounds node index, so their writes are dropped.
        target = jnp.where(active, parent, num_nodes)
        tree = tree.replace(
            node_values=tree.node_values.at[rows, target].set(parent_value, mode="drop"),
            node_visits=tree.node_visits.at[rows, target].set(count + 1, mode="drop"),
            children_values=tree.children_values.at[rows, target, action].set(
                children_values, mode="drop"
            ),
            children_visits=tree.children_visits.at[rows, target, action].set(
                children_counts, mode="drop"
            ),
        )
        leaf_value = jnp.where(active, new_leaf_value, leaf_value)
        return tree, leaf_value, jnp.where(active, parent, index)

    loop_state = (tree, tree.node_values[rows, leaf_index], leaf_index)
    tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)
    return tree


_search.stock_backward = stock_backward
_search.backward = batched_backward
