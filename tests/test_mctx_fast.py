"""mctx_fast (examples/alphazero/mctx_fast) must return the same search as mctx.

mctx_fast is mctx 0.0.71 with a faster search loop: the root step runs outside
the traversal loop, the backward pass applies its writes once, and new nodes
are written with a slice update. None of that may change a single bit of the
result, so every test compares the whole search tree, the action and the
action weights exactly against the upstream package.

Run on CPU (`JAX_PLATFORMS=cpu`): on the GPU two runs of the same code can
already differ, so only a CPU comparison can show the outputs are identical.
"""

import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pgx

mctx = pytest.importorskip("mctx")

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "alphazero")),
)

import mctx_fast  # noqa: E402
from model_tournament import tree_embedding, untree_embedding  # noqa: E402

NUM_ACTIONS = 9
NUM_STATES = 997


def _toy_problem(seed: int, batch_size: int):
    """A deterministic random game: the embedding is a state id, and each
    (state, action) leads to a fixed successor with fixed logits, value, reward
    and discount. Discounts include 0 (terminal) and +1 (same player again)."""
    k = jax.random.split(jax.random.PRNGKey(seed), 6)
    logits_table = jax.random.normal(k[0], (NUM_STATES, NUM_ACTIONS)) * 2
    value_table = jax.random.uniform(k[1], (NUM_STATES,), minval=-1, maxval=1)
    reward_table = jax.random.choice(k[2], jnp.array([-1.0, 0.0, 0.0, 0.0, 1.0]), (NUM_STATES,))
    discount_table = jax.random.choice(k[3], jnp.array([-1.0, -1.0, 1.0, 0.0]), (NUM_STATES,))
    root_states = jax.random.randint(k[4], (batch_size,), 0, NUM_STATES)
    invalid = jax.random.bernoulli(k[5], 0.3, (batch_size, NUM_ACTIONS))
    invalid = invalid.at[:, 0].set(False)  # at least one legal action

    def recurrent_fn(m, params, rng_key, action, embedding):
        del rng_key
        nxt = (embedding * 31 + action * 7 + 1 + params) % NUM_STATES
        out = m.RecurrentFnOutput(
            reward=reward_table[nxt],
            discount=discount_table[nxt],
            prior_logits=logits_table[nxt],
            value=value_table[nxt],
        )
        return out, nxt

    root = (logits_table[root_states], value_table[root_states], root_states)
    return root, recurrent_fn, invalid


def _assert_trees_equal(a, b, skip=""):
    # The two packages define their own Tree classes, so compare the leaves by
    # path (field names) rather than the tree structures.
    leaves_a = jax.tree_util.tree_flatten_with_path(a)[0]
    leaves_b = jax.tree_util.tree_flatten_with_path(b)[0]
    paths_a = [jax.tree_util.keystr(p) for p, _ in leaves_a]
    assert paths_a == [jax.tree_util.keystr(p) for p, _ in leaves_b]
    for path, (_, x), (_, y) in zip(paths_a, leaves_a, leaves_b):
        if skip and skip in path:
            continue
        assert x.dtype == y.dtype, path
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=path)


def _run_toy(m, policy_name, seed, batch_size, num_simulations, policy_kwargs):
    (logits, value, states), recurrent_fn, invalid = _toy_problem(seed, batch_size)
    root = m.RootFnOutput(prior_logits=logits, value=value, embedding=states)

    @jax.jit
    def run(key):
        return getattr(m, policy_name)(
            params=jnp.int32(3),
            rng_key=key,
            root=root,
            recurrent_fn=functools.partial(recurrent_fn, m),
            num_simulations=num_simulations,
            invalid_actions=invalid,
            **policy_kwargs,
        )

    return run(jax.random.PRNGKey(seed + 100))


TOY_CASES = [
    ("gumbel_muzero_policy", {}),
    ("gumbel_muzero_policy", {"max_num_considered_actions": 4, "gumbel_scale": 0.5}),
    ("gumbel_muzero_policy", {"qtransform": mctx.qtransform_by_parent_and_siblings}),
    ("muzero_policy", {}),
    ("muzero_policy", {"temperature": 0.25}),
    # max_depth < num_simulations keeps the scatter path for new nodes, since a
    # simulation can then stop on an existing node.
    ("gumbel_muzero_policy", {"max_depth": 3}),
    ("muzero_policy", {"max_depth": 2}),
    ("gumbel_muzero_policy", {"max_depth": 64}),
]


@pytest.mark.parametrize("policy_name,kwargs", TOY_CASES)
@pytest.mark.parametrize("num_simulations", [1, 7, 32])
def test_toy_search_matches_mctx(policy_name, kwargs, num_simulations):
    for seed in range(3):
        fast_kwargs = dict(kwargs)
        if fast_kwargs.get("qtransform") is mctx.qtransform_by_parent_and_siblings:
            fast_kwargs["qtransform"] = mctx_fast.qtransform_by_parent_and_siblings
        ref = _run_toy(mctx, policy_name, seed, 16, num_simulations, kwargs)
        new = _run_toy(mctx_fast, policy_name, seed, 16, num_simulations, fast_kwargs)
        _assert_trees_equal(ref.search_tree, new.search_tree)
        np.testing.assert_array_equal(ref.action, new.action)
        np.testing.assert_array_equal(ref.action_weights, new.action_weights)


def _run_pgx(m, env_id, batch_size, num_simulations, num_moves, strip_observation=False):
    """Searches with real pgx states as embeddings and a random linear "network"
    on the observation, playing a few moves so the trees start from varied
    positions."""
    env = pgx.make(env_id)
    obs_size = int(np.prod(env.observation_shape))
    w = jax.random.normal(jax.random.PRNGKey(1), (obs_size, env.num_actions + 1)) * 0.1

    def evaluate(state):
        out = state.observation.reshape(state.observation.shape[0], -1).astype(jnp.float32) @ w
        return out[:, :-1], jnp.tanh(out[:, -1])

    def recurrent_fn(params, rng_key, action, state):
        if strip_observation:
            state = untree_embedding(env, state)
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action, jax.random.split(rng_key, action.shape[0]))
        logits, value = evaluate(state)
        value = jnp.where(state.terminated, 0.0, value)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        discount = jnp.where(state.current_player == current_player, 1.0, -1.0)
        discount = jnp.where(state.terminated, 0.0, discount)
        out = m.RecurrentFnOutput(
            reward=state.rewards[jnp.arange(action.shape[0]), current_player],
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return out, (tree_embedding(state) if strip_observation else state)

    @jax.jit
    def play(key):
        state = jax.vmap(env.init)(jax.random.split(key, batch_size))
        trees = []
        for _ in range(num_moves):
            key, k1, k2 = jax.random.split(key, 3)
            logits, value = evaluate(state)
            out = m.gumbel_muzero_policy(
                params=None,
                rng_key=k1,
                root=m.RootFnOutput(
                    prior_logits=logits,
                    value=value,
                    embedding=tree_embedding(state) if strip_observation else state,
                ),
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=~state.legal_action_mask,
                qtransform=m.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )
            trees.append(out)
            state = jax.vmap(env.step)(state, out.action, jax.random.split(k2, batch_size))
        return trees

    return play(jax.random.PRNGKey(0))


@pytest.mark.parametrize("env_id", ["gess", "epaminondas", "pig", "tic_tac_toe"])
def test_pgx_search_matches_mctx(env_id):
    ref = _run_pgx(mctx, env_id, batch_size=4, num_simulations=16, num_moves=3)
    new = _run_pgx(mctx_fast, env_id, batch_size=4, num_simulations=16, num_moves=3)
    for r, n in zip(ref, new):
        _assert_trees_equal(r.search_tree, n.search_tree)
        np.testing.assert_array_equal(r.action, n.action)
        np.testing.assert_array_equal(r.action_weights, n.action_weights)


@pytest.mark.parametrize("env_id", ["gess", "epaminondas", "pig", "dots_and_boxes"])
def test_observation_free_tree_embedding_matches(env_id):
    """train.py stores states in the tree without their observation
    (model_tournament.tree_embedding). The search must not change."""
    ref = _run_pgx(mctx, env_id, batch_size=4, num_simulations=16, num_moves=3)
    new = _run_pgx(mctx_fast, env_id, batch_size=4, num_simulations=16, num_moves=3, strip_observation=True)
    for r, n in zip(ref, new):
        assert n.search_tree.embeddings.observation.shape[-1] == 0
        _assert_trees_equal(r.search_tree, n.search_tree, skip=".observation")
        np.testing.assert_array_equal(r.action, n.action)
        np.testing.assert_array_equal(r.action_weights, n.action_weights)
