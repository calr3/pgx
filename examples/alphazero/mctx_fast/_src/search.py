# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
#
# Modified for pgx's AlphaZero examples (see mctx_fast/__init__.py): the root
# step of each simulation runs outside the traversal loop, the backward pass
# collects its writes and applies them once, and new nodes are written with a
# slice update instead of a scatter. The resulting tree is bit-identical to
# upstream mctx 0.0.71 (tests/test_mctx_fast.py).
"""A JAX implementation of batched MCTS."""
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from mctx_fast._src import base
from mctx_fast._src import tree as tree_lib

Tree = tree_lib.Tree
T = TypeVar("T")


def search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)
  # Simulation `sim` runs on a tree of sim + 1 nodes, so a path through visited
  # nodes has at most sim < num_simulations edges. With max_depth at least
  # num_simulations no simulation stops on an already-expanded node, and every
  # batch element expands the same node index, sim + 1. That write can then be
  # one slice update along the node axis instead of a scatter.
  every_sim_adds_node = (
      isinstance(max_depth, int) and max_depth >= num_simulations)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = simulate(
        simulate_keys, tree, root_action_selection_fn,
        interior_action_selection_fn, max_depth)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    if every_sim_adds_node:
      new_node_index = jnp.asarray(sim + 1, dtype=jnp.int32)
      next_node_index = jnp.full([batch_size], new_node_index)
    else:
      new_node_index = None
      next_node_index = tree.children_index[batch_range, parent_index, action]
      next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                  sim + 1, next_node_index)
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index, new_node_index=new_node_index)
    tree = backward(tree, next_node_index)
    loop_state = rng_key, tree
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree))

  return tree


class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool


def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Batched traversal; see `_simulate` (the functions are not vmapped)."""
  return jax.vmap(
      lambda key, t: _simulate(key, t, root_action_selection_fn,
                               interior_action_selection_fn, max_depth)
  )(rng_key, tree)


def _simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Upstream runs the root step inside the loop and picks the root or interior
  rule with `lax.cond` on the depth. Under `vmap` that cond becomes a select,
  so both rules ran at every depth. The root step is always the first one, so
  it runs here once, before the loop, consuming the same key as upstream.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: _unbatched_ MCTS tree state.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def step(rng_key, node_index, action, depth):
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)

  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    rng_key, action_selection_key = jax.random.split(state.rng_key)
    action = interior_action_selection_fn(action_selection_key, tree,
                                          node_index, state.depth)
    return step(rng_key, node_index, action, state.depth)

  root_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  rng_key, action_selection_key = jax.random.split(rng_key)
  action = root_action_selection_fn(action_selection_key, tree, root_index)
  initial_state = step(rng_key, root_index, action, depth)
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


def expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array,
    new_node_index: Optional[chex.Array] = None) -> Tree[T]:
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.
    new_node_index: optional scalar. When given, every element of
      `next_node_index` equals it, and the node-indexed writes become slice
      updates.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

  # Retrieve states for nodes to be evaluated.
  embedding = jax.tree.map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  tree = update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, embedding,
      new_node_index=new_node_index)

  if new_node_index is None:
    parents = batch_update(tree.parents, parent_index, next_node_index)
    action_from_parent = batch_update(
        tree.action_from_parent, action, next_node_index)
  else:
    parents = _set_node(tree.parents, parent_index, new_node_index)
    action_from_parent = _set_node(
        tree.action_from_parent, action, new_node_index)

  # Return updated tree topology.
  return tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=parents,
      action_from_parent=action_from_parent)


class _BackwardWrites(NamedTuple):
  """Values the backward pass assigns, one row per edge on the path."""
  parent: chex.Array  # Out of range (= num_nodes) in unused rows.
  action: chex.Array
  parent_value: chex.Array
  parent_visits: chex.Array
  child_value: chex.Array
  child_visits: chex.Array


class _BackwardState(NamedTuple):
  writes: _BackwardWrites
  step: chex.Array
  index: chex.Array
  index_value: chex.Array  # Current (already updated) node_values[index].
  leaf_value: chex.Array


@jax.vmap
def backward(
    tree: Tree[T],
    leaf_index: chex.Numeric) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Upstream updated the tree inside the loop. Under `vmap` a while_loop with a
  batched predicate selects over its whole carry on every iteration, so each
  level of each simulation copied every `[B, N, num_actions]` array. Here the
  loop only reads the tree: the nodes on a path are distinct, so each read sees
  the value upstream would have seen, provided the child's freshly updated
  value is carried along. The writes are applied once at the end, with the same
  arithmetic, so the result is identical.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """
  num_nodes = tree.node_values.shape[0]
  max_path = num_nodes - 1
  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)

  def cond_fun(state):
    return state.index != Tree.ROOT_INDEX

  def body_fun(state):
    # Here we update the value of our parent, so we start by reversing.
    index = state.index
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * (
        state.leaf_value)
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_counts = tree.children_visits[parent, action] + 1
    w, k = state.writes, state.step
    writes = _BackwardWrites(
        parent=w.parent.at[k].set(parent),
        action=w.action.at[k].set(action),
        parent_value=w.parent_value.at[k].set(parent_value),
        parent_visits=w.parent_visits.at[k].set(count + 1),
        child_value=w.child_value.at[k].set(state.index_value),
        child_visits=w.child_visits.at[k].set(children_counts))
    return _BackwardState(
        writes=writes, step=k + 1, index=parent,
        index_value=parent_value.astype(tree.node_values.dtype),
        leaf_value=leaf_value)

  value_dtype = tree.node_values.dtype
  leaf_value = tree.node_values[leaf_index]
  writes = _BackwardWrites(
      parent=jnp.full([max_path], num_nodes, dtype=jnp.int32),
      action=jnp.zeros([max_path], dtype=jnp.int32),
      parent_value=jnp.zeros([max_path], dtype=value_dtype),
      parent_visits=jnp.zeros([max_path], dtype=tree.node_visits.dtype),
      child_value=jnp.zeros([max_path], dtype=tree.children_values.dtype),
      child_visits=jnp.zeros([max_path], dtype=tree.children_visits.dtype))
  init = _BackwardState(
      writes=writes, step=jnp.zeros((), jnp.int32), index=leaf_index,
      index_value=leaf_value, leaf_value=leaf_value)
  w = jax.lax.while_loop(cond_fun, body_fun, init).writes

  # Unused rows have parent == num_nodes and are dropped.
  return tree.replace(
      node_values=tree.node_values.at[w.parent].set(
          w.parent_value, mode="drop"),
      node_visits=tree.node_visits.at[w.parent].set(
          w.parent_visits, mode="drop"),
      children_values=tree.children_values.at[w.parent, w.action].set(
          w.child_value, mode="drop"),
      children_visits=tree.children_visits.at[w.parent, w.action].set(
          w.child_visits, mode="drop"))


# Utility function to set the values of certain indices to prescribed values.
# This is vmapped to operate seamlessly on batches.
def update(x, vals, *indices):
  return x.at[indices].set(vals)


batch_update = jax.vmap(update)


def _set_node(x, vals, node_index):
  """`x[b, node_index] = vals[b]` for every b, as one slice update.

  The batched equivalent of `batch_update(x, vals, node_indices)` when all
  node indices are the same scalar; `.at[].set` casts to `x.dtype`, so this
  does too.
  """
  return jax.lax.dynamic_update_index_in_dim(
      x, jnp.asarray(vals).astype(x.dtype), node_index, axis=1)


def update_tree_node(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array,
    new_node_index: Optional[chex.Array] = None) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.
    new_node_index: optional scalar equal to every element of `node_index`;
      when given, the writes are slice updates along the node axis.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1
  if new_node_index is not None:
    write = lambda x, vals: _set_node(x, vals, new_node_index)
    return tree.replace(
        children_prior_logits=write(tree.children_prior_logits, prior_logits),
        raw_values=write(tree.raw_values, value),
        node_values=write(tree.node_values, value),
        node_visits=write(tree.node_visits, new_visit),
        embeddings=jax.tree.map(write, tree.embeddings, embedding))
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(
          tree.raw_values, value, node_index),
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree.map(
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any) -> Tree:
  """Initializes tree state at search root."""
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)

  # Create a new empty tree state and fill its root.
  tree = Tree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(
          batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(
          batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree.map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data)

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding)
  return tree
