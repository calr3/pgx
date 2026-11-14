"""The calr3/mctx fork must search exactly like upstream mctx on pgx games.

The fork (examples/alphazero/requirements.txt) changes how the search loop
runs and how the tree is stored, but not its results. It carries the original
upstream code as `mctx._src.tests.reference`; its own tests compare the two on
a synthetic game, and these do the same with real pgx states as embeddings and
the multi-stage turns, chance and terminal states of the games trained here.

Run on CPU (`JAX_PLATFORMS=cpu`): on the GPU two runs of the same code can
already differ, so only a CPU comparison can show the outputs are identical.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pgx

mctx = pytest.importorskip("mctx")
reference = pytest.importorskip(
    "mctx._src.tests.reference", reason="mctx is not the calr3/mctx fork"
)

# The original Tree's fields. The fork's Tree provides the same names, some as
# properties computed from per-node arrays.
TREE_FIELDS = (
    "node_visits", "raw_values", "node_values", "parents", "action_from_parent",
    "children_index", "children_prior_logits", "children_visits", "children_rewards",
    "children_discounts", "children_values", "embeddings", "root_invalid_actions",
    "extra_data",
)


def _assert_identical(ref, new):
    np.testing.assert_array_equal(ref.action, new.action)
    np.testing.assert_array_equal(ref.action_weights, new.action_weights)
    for field in TREE_FIELDS:
        ref_leaves = jax.tree_util.tree_flatten_with_path(getattr(ref.search_tree, field))[0]
        new_leaves = jax.tree_util.tree_flatten_with_path(getattr(new.search_tree, field))[0]
        assert len(ref_leaves) == len(new_leaves), field
        for (path, x), (_, y) in zip(ref_leaves, new_leaves):
            name = field + jax.tree_util.keystr(path)
            if field == "children_index":
                y = y.astype(x.dtype)  # the fork stores int16 when it fits
            assert x.dtype == y.dtype, name
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def _run_pgx(m, env_id, batch_size, num_simulations, num_moves, **policy_kwargs):
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
        return out, state

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
                root=m.RootFnOutput(prior_logits=logits, value=value, embedding=state),
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=~state.legal_action_mask,
                qtransform=m.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
                **policy_kwargs,
            )
            trees.append(out)
            state = jax.vmap(env.step)(state, out.action, jax.random.split(k2, batch_size))
        return trees

    return play(jax.random.PRNGKey(0))


GAMES = ["gess", "epaminondas", "pig", "dots_and_boxes", "tic_tac_toe"]


@pytest.mark.parametrize("env_id", GAMES)
def test_fork_matches_reference(env_id):
    ref = _run_pgx(reference, env_id, batch_size=4, num_simulations=16, num_moves=3)
    new = _run_pgx(mctx, env_id, batch_size=4, num_simulations=16, num_moves=3)
    for r, n in zip(ref, new):
        _assert_identical(r, n)


def test_search_tree_bf16_stays_close():
    """search_tree_bf16 changes results slightly; check it stays sane."""
    full = _run_pgx(mctx, "gess", batch_size=4, num_simulations=16, num_moves=2)
    low = _run_pgx(
        mctx, "gess", batch_size=4, num_simulations=16, num_moves=2,
        prior_logits_dtype=jnp.bfloat16,
    )
    for f, lo in zip(full, low):
        assert lo.search_tree.children_prior_logits.dtype == jnp.bfloat16
        assert np.all(np.isfinite(lo.action_weights))
        np.testing.assert_allclose(lo.action_weights, f.action_weights, atol=0.1)
