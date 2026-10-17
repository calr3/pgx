"""The batched mctx backward pass in examples/alphazero/mctx_patch.py must give
exactly the tree stock mctx gives."""

import os
import sys

import jax
import jax.numpy as jnp
import mctx
import pytest
from mctx._src import search

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "alphazero")
    ),
)

import mctx_patch  # noqa: E402

B, A = 16, 12


def recurrent_fn(params, rng_key, action, embedding):
    # A deterministic pseudo-network over a hashed path, with rewards, terminal
    # edges (discount 0) and multi-stage turns (discount +1) mixed in.
    h = (embedding * 31 + action[:, None] + 7) % 1009
    x = h[:, 0].astype(jnp.float32)
    logits = jnp.sin(x[:, None] * 0.13 + jnp.arange(A) * 0.37)
    discount = jnp.select([h[:, 0] % 7 == 0, h[:, 0] % 3 == 0], [0.0, 1.0], -1.0)
    out = mctx.RecurrentFnOutput(
        reward=jnp.where(discount == 0.0, jnp.sign(jnp.cos(x)), 0.0),
        discount=discount,
        prior_logits=logits,
        value=0.5 * jnp.cos(x * 0.71),
    )
    return out, h


def run(policy, num_simulations, max_depth):
    key = jax.random.PRNGKey(3)
    root = mctx.RootFnOutput(
        prior_logits=jax.random.normal(key, (B, A)),
        value=jnp.zeros(B),
        embedding=jnp.arange(B * 2, dtype=jnp.int32).reshape(B, 2),
    )
    invalid = jax.random.bernoulli(jax.random.PRNGKey(4), 0.3, (B, A))
    invalid = invalid.at[:, 0].set(False)
    fn = mctx.gumbel_muzero_policy if policy == "gumbel" else mctx.muzero_policy
    out = jax.jit(
        lambda k: fn(
            params=(),
            rng_key=k,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=invalid,
            max_depth=max_depth,
        )
    )(key)
    return jax.tree.leaves((out.action, out.action_weights, out.search_tree))


@pytest.mark.parametrize("policy", ["gumbel", "muzero"])
@pytest.mark.parametrize("num_simulations,max_depth", [(32, None), (24, 3)])
def test_batched_backward_matches_stock(policy, num_simulations, max_depth):
    assert search.backward is mctx_patch.batched_backward
    try:
        search.backward = mctx_patch.stock_backward
        expected = run(policy, num_simulations, max_depth)
    finally:
        search.backward = mctx_patch.batched_backward
    got = run(policy, num_simulations, max_depth)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g.dtype == e.dtype and g.shape == e.shape
        assert bool(jnp.array_equal(g, e)), "trees differ"
