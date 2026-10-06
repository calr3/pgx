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


def _random_boards(key, count, density):
    """Random 20x20 boards (stones on the 18x18 playing area only) and colours."""
    k1, k2, k3 = jax.random.split(key, 3)
    cells = jax.random.uniform(k1, (count, 20, 20))
    stones = jnp.where(cells < density / 2, 1, jnp.where(cells < density, 2, 0)).astype(jnp.int8)
    inner = jnp.zeros((20, 20), jnp.bool_).at[1:19, 1:19].set(True)
    boards = jnp.where(inner, stones, 0).reshape(count, 400)
    colors = jax.random.bernoulli(k2, 0.5, (count,)).astype(jnp.int32)
    return boards, colors


def test_shift_move_generator_equals_the_gather_version():
    """legal_moves (whole-board shifts) against legal_moves_reference (per-centre
    gathers): identical lists and overflow flags on random boards from sparse to
    dense, and on positions from random play."""
    from pgx._src.games.gess import GameState
    from pgx.gess_joint import legal_moves_reference

    both = jax.jit(jax.vmap(lambda b, c: (legal_moves(GameState(color=c, board=b)),
                                          legal_moves_reference(GameState(color=c, board=b)))))
    key = jax.random.PRNGKey(7)
    checked = 0
    for density in (0.05, 0.15, 0.3, 0.45, 0.6, 0.8):
        key, k = jax.random.split(key)
        boards, colors = _random_boards(k, 64, density)
        (m, o), (rm, ro) = both(boards, colors)
        assert np.array_equal(np.asarray(m), np.asarray(rm)), density
        assert np.array_equal(np.asarray(o), np.asarray(ro)), density
        checked += 64
    # Positions from random play (all stage 0, both colours).
    states = jax.vmap(_j_init)(jax.random.split(key, 64))
    step = jax.jit(jax.vmap(_j_step))
    for i in range(30):
        key, k = jax.random.split(key)
        a = jax.random.categorical(k, jnp.where(states.legal_action_mask, 0.0, -jnp.inf))
        nxt = step(states, a)
        states = jax.tree_util.tree_map(
            lambda n, s: jnp.where(nxt.terminated.reshape((-1,) + (1,) * (n.ndim - 1)), s, n), nxt, states
        )
        (m, o), (rm, ro) = both(states._x.board, states._x.color)
        assert np.array_equal(np.asarray(m), np.asarray(rm)), i
        checked += 64
    assert checked == 64 * 36
