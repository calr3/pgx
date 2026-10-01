import jax
import jax.numpy as jnp
import numpy as np

import pgx
from pgx.gess import Gess
from pgx.gess_joint import MAX_MOVES, GessJoint, legal_moves
from pgx._src.games.gess import N

joint = GessJoint()
two = Gess()
_j_init, _j_step = jax.jit(joint.init), jax.jit(joint.step)
_t_init, _t_step = jax.jit(two.init), jax.jit(two.step)


def _two_step_moves(s):
    """All (source, destination) moves of a two-step Gess state at stage 0."""
    out = set()
    for src in np.nonzero(np.asarray(s.legal_action_mask))[0]:
        s1 = _t_step(s, jnp.int32(src))
        for dst in np.nonzero(np.asarray(s1.legal_action_mask))[0]:
            out.add(int(src) * N + int(dst))
    return out


def test_api():
    pgx.api_test(joint, 3, use_key=False)


def test_move_list_is_sorted_padded_and_complete_at_the_start():
    s = _j_init(jax.random.PRNGKey(0))
    moves = np.asarray(s._moves)
    legal = moves[moves >= 0]
    assert np.all(np.diff(legal) > 0)
    assert np.all(moves[len(legal):] == -1)
    assert np.array_equal(np.asarray(s.legal_action_mask), moves >= 0)
    assert not bool(s._overflow)
    assert set(legal.tolist()) == _two_step_moves(_t_init(jax.random.PRNGKey(0)))
    assert len(legal) == 478  # measured; well under MAX_MOVES
    assert MAX_MOVES >= 528


def test_plays_like_two_step_gess():
    # Random games in both envs, the joint one choosing a move and the two-step
    # one playing its piece then its destination: same states, observations,
    # rewards and move lists throughout.
    rng = np.random.default_rng(0)
    for g in range(3):
        sj = _j_init(jax.random.PRNGKey(g))
        st = _t_init(jax.random.PRNGKey(g))
        for _ in range(25):
            if bool(sj.terminated):
                break
            assert set(np.asarray(sj._moves)[np.asarray(sj.legal_action_mask)].tolist()) == _two_step_moves(st)
            assert np.array_equal(np.asarray(sj.observation), np.asarray(st.observation))
            assert int(sj.current_player) == int(st.current_player)
            a = int(rng.choice(np.nonzero(np.asarray(sj.legal_action_mask))[0]))
            move = int(sj._moves[a])
            sj = _j_step(sj, jnp.int32(a))
            st = _t_step(_t_step(st, jnp.int32(move // N)), jnp.int32(move % N))
            assert np.array_equal(np.asarray(sj._x.board), np.asarray(st._x.board))
            assert bool(sj.terminated) == bool(st.terminated)
            assert np.array_equal(np.asarray(sj.rewards), np.asarray(st.rewards))


def test_legal_moves_matches_the_state():
    s = _j_init(jax.random.PRNGKey(0))
    moves, overflow = legal_moves(s._x)
    assert np.array_equal(np.asarray(moves), np.asarray(s._moves))
    assert not bool(overflow)
