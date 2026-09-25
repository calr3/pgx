"""Score a trained pig model against exact optimal play.

Pig is solved: Neller & Presser, "Optimal Play of the Dice Game Pig" (2004),
https://cupola.gettysburg.edu/csfac/4/ . Let P(i, j, k) be the probability that
the player to move wins, having banked i, with the opponent on j and a turn
total of k. Holding hands over the move, rolling either busts or adds to k:

    hold(i, j, k) = 1                     if i + k >= 100
                  = 1 - P(j, i + k, 0)    otherwise
    roll(i, j, k) = 1/6 * (1 - P(j, i, 0)) + 1/6 * sum_{r=2..6} P(i, j, k + r)
    P(i, j, k)    = max(hold, roll)       (k = 0 must roll)

P depends on itself through P(., ., 0), so this is solved by value iteration;
for fixed (i, j) the turn total only ever grows, so each sweep is an exact
backward recursion in k. The table is computed here rather than transcribed, so
it can be checked against the paper's published first-player win probability.

This is a standalone script (no search, no mctx): it evaluates the raw policy
head, which is what AlphaZero's search is amplifying, plus the value head
against the true win probabilities.

    python3 examples/alphazero/pig_optimal.py checkpoints/pig_*/0000*.ckpt
"""

import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pgx

sys.path.insert(0, "examples/alphazero")
from config import Config  # noqa: E402
from network import make_forward, mlp_input_features  # noqa: E402

TARGET = 100
KMAX = 121  # turn totals above this are never reached in play


def solve(tol: float = 1e-12, max_sweeps: int = 1000):
    """Value-iterate P(i, j, k). Returns (win_prob, roll_is_optimal)."""
    i = np.arange(TARGET)[:, None]
    j = np.arange(TARGET)[None, :]
    swap = (j * TARGET + i).ravel()  # index of (j, i) in a flattened table
    w = np.zeros((TARGET, TARGET))  # P(i, j, 0)

    for sweep in range(max_sweeps):
        v = np.ones((KMAX + 7, TARGET, TARGET))  # v[k] = P(i, j, k); 1 once i+k >= 100
        roll_opt = np.zeros((KMAX, TARGET, TARGET), dtype=bool)
        # Busting hands the opponent the move at (j, i) with nothing banked.
        bust = 1.0 - w.ravel()[swap].reshape(TARGET, TARGET)
        for k in range(KMAX - 1, -1, -1):
            banked = i + k
            # 1 - P(j, i+k, 0), or a certain win once the target is reached.
            safe = np.clip(banked, 0, TARGET - 1)
            hold = np.where(banked >= TARGET, 1.0, 1.0 - w[j, safe])
            roll = (bust + v[k + 2] + v[k + 3] + v[k + 4] + v[k + 5] + v[k + 6]) / 6.0
            if k == 0:
                v[k] = roll  # a turn always starts with a roll
            else:
                v[k] = np.maximum(hold, roll)
                roll_opt[k] = roll > hold
        delta = np.abs(v[0] - w).max()
        w = v[0]
        if delta < tol:
            break
    return v[:KMAX].copy(), roll_opt, sweep, delta


def sustained_hold(actions):
    """Lowest turn total from which the policy holds and never rolls again.

    The first hold on its own is misleading: a policy can hold at a turn total
    of 1 and still roll at 10, which is not a threshold at all.
    """
    holds = np.asarray(actions) == 0
    for k in range(len(holds)):
        if holds[k:].all():
            return k + 1
    return -1


def observation(own, opp, turn, roll):
    """pgx pig's observation (v1) for the mover, from its four counters. Older
    checkpoints take the leading features they were built for."""
    xp = jnp if any(isinstance(a, jax.Array) for a in (own, opp, turn, roll)) else np
    own, opp, turn, roll = (xp.asarray(a) for a in xp.broadcast_arrays(own, opp, turn, roll))
    banked = xp.minimum(own + turn, TARGET)
    return xp.stack([own, opp, turn, turn, roll, roll, banked], axis=-1)


def optimal_action(table_roll, obs):
    """Action (0 = hold, 1 = roll) of the optimal player, from a pgx pig obs."""
    own = jnp.clip(obs[:, 0].astype(jnp.int32), 0, TARGET - 1)
    opp = jnp.clip(obs[:, 1].astype(jnp.int32), 0, TARGET - 1)
    turn = jnp.clip(obs[:, 2].astype(jnp.int32), 0, KMAX - 1)
    roll = table_roll[turn, own, opp] & (obs[:, 4] > 1)
    return roll.astype(jnp.int32)


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)

    win_prob, roll_opt, sweeps, delta = solve()
    print(f"Solved pig in {sweeps + 1} sweeps (max change {delta:.2e}).")
    print(
        f"  first player's win probability with optimal play: {win_prob[0, 0, 0]:.4f}"
        "   [Neller & Presser 2004: 0.5306]"
    )
    thresholds = [int(np.argmin(roll_opt[1:, i, j]) + 1) for i, j in [(0, 0), (50, 50), (70, 0)]]
    print(f"  optimal hold thresholds at (0,0), (50,50), (70,0): {thresholds}")

    table_roll = jnp.asarray(roll_opt)
    table_p = jnp.asarray(win_prob)

    # Every state a player can actually face: turn total at least 1 (the die is
    # rolled before the decision) and the game not already won.
    ii, jj, kk = np.meshgrid(
        np.arange(TARGET), np.arange(TARGET), np.arange(1, KMAX), indexing="ij"
    )
    live = (ii + kk) < TARGET + 6  # unreachable far above the target
    ii, jj, kk = ii[live], jj[live], kk[live]
    obs_all = observation(ii, jj, kk, np.full(ii.shape, 5)).astype(jnp.float32)
    opt_all = jnp.asarray(roll_opt[kk, ii, jj]).astype(jnp.int32)
    true_v = 2.0 * jnp.asarray(win_prob[kk, ii, jj]) - 1.0  # win prob -> [-1, 1]

    env = pgx.make("pig")
    num_games = 4096
    match = make_match(env, table_roll, num_games)
    hold_at_20 = pgx.make_baseline_model("pig_v0")
    reference = {
        "hold-at-20": lambda obs, legal: (
            jnp.argmax(jnp.where(legal, hold_at_20(obs)[0], -jnp.inf), axis=-1),
            jnp.zeros(obs.shape[0]),
        ),
        "always roll": lambda obs, legal: (legal[:, 1].astype(jnp.int32), jnp.zeros(obs.shape[0])),
    }

    print()
    print(
        f"{'checkpoint':<44} {'agree':>7} {'played':>7} {'v_rmse':>7} "
        f"{'vs optimal':>11} {'hold@(0,0)':>11}"
    )
    for path in paths:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        config = Config(**ckpt["config"].__dict__)
        params, model_state = ckpt["model"]
        forward = make_forward(env.num_actions, config)
        # The leading features this checkpoint was built for: six before pig v1.
        features = mlp_input_features(params, config)

        def value_of(obs):
            (_, value), _ = forward.apply(params, model_state, obs[:, :features], is_eval=True)
            return value

        @jax.jit
        def expectimax(obs, legal):
            """Act on the value head alone, expanding the six die faces exactly.

            Pig's chance is enumerable, so the value of an action is an average
            over faces rather than a sample of one. This isolates how much of the
            model's strength is in the value head and how much the policy head
            throws away.
            """
            own, opp, turn = obs[:, 0], obs[:, 1], obs[:, 2]
            faces = jnp.arange(1, 7, dtype=obs.dtype)[None, :]  # (1, 6)

            def mean_value(o, p, t, r):
                """Mean value to the player to move over the six faces r."""
                s = observation(*jnp.broadcast_arrays(o, p, t, r))
                return value_of(s.reshape(-1, s.shape[-1])).reshape(-1, 6).mean(axis=-1)

            # Rolling keeps the turn: the face lands on the turn total.
            q_roll = mean_value(own[:, None], opp[:, None], turn[:, None] + faces, faces)
            # Holding banks the turn total and hands over the move, which the
            # opponent starts by rolling - unless banking wins outright.
            banked = own + turn
            q_hold = jnp.where(
                banked >= TARGET,
                1.0,
                -mean_value(opp[:, None], banked[:, None], faces, faces),
            )
            q = jnp.stack([q_hold, q_roll], axis=-1)
            return jnp.argmax(jnp.where(legal, q, -jnp.inf), axis=-1), value_of(obs)

        @jax.jit
        def policy(obs, legal):
            (logits, value), _ = forward.apply(params, model_state, obs[:, :features], is_eval=True)
            logits = jnp.where(legal, logits, jnp.finfo(logits.dtype).min)
            return jnp.argmax(logits, axis=-1), value

        # Decision agreement and value error over every reachable state.
        agree_n = err = count = 0.0
        for start in range(0, obs_all.shape[0], 65536):
            sl = slice(start, start + 65536)
            legal = jnp.ones((obs_all[sl].shape[0], 2), dtype=bool)
            act, value = policy(obs_all[sl], legal)
            agree_n += float((act == opt_all[sl]).sum())
            err += float(((value - true_v[sl]) ** 2).sum())
            count += act.shape[0]
        agree, v_rmse = agree_n / count, (err / count) ** 0.5

        R, played_agree = match(jax.random.PRNGKey(0), policy)
        # Optimal play wins 53.06% as the first player, so a perfect model would
        # score 0.5 here (it gives away the first move in half the games).
        wins = float((R > 0).mean())
        # The turn total at which the model first prefers holding from 0-0.
        probe = observation(jnp.zeros(100), jnp.zeros(100), jnp.arange(1, 101), jnp.full((100,), 5.0))
        act, _ = policy(probe, jnp.ones((100, 2), dtype=bool))
        thr = sustained_hold(act)
        print(
            f"{path:<44} {agree:7.3f} {float(played_agree):7.3f} {v_rmse:7.3f} "
            f"{wins:11.3f} {thr:11d}"
        )

        # The same value head, played by one-ply expectimax instead of by its
        # own policy head.
        R, played_agree = match(jax.random.PRNGKey(0), expectimax)
        act, _ = expectimax(probe, jnp.ones((100, 2), dtype=bool))
        thr = sustained_hold(act)
        print(
            f"{'  ^ value head, 1-ply expectimax':<44} {'':>7} {float(played_agree):7.3f} "
            f"{'':>7} {float((R > 0).mean()):11.3f} {thr:11d}"
        )

    for name, policy in reference.items():
        R, played_agree = match(jax.random.PRNGKey(0), policy)
        print(
            f"{'[' + name + ']':<44} {'':>7} {float(played_agree):7.3f} {'':>7} "
            f"{float((R > 0).mean()):11.3f}"
        )


def make_match(env, table_roll, num_games):
    """Play `policy` against the optimal player, seats balanced."""

    def match(key, policy):
        key, subkey = jax.random.split(key)
        state = jax.vmap(env.init)(jax.random.split(subkey, num_games))
        me = (jnp.arange(num_games) >= num_games // 2).astype(jnp.int32)

        def body(val):
            key, state, R, hit, n = val
            key, step_key = jax.random.split(key)
            obs = state.observation.astype(jnp.float32)
            mine, _ = policy(obs, state.legal_action_mask)
            theirs = optimal_action(table_roll, obs)
            action = jnp.where(state.current_player == me, mine, theirs)
            # Agreement restricted to live states the model actually faces,
            # where it has a real choice (rolling is forced off after a 1).
            scored = (state.current_player == me) & ~state.terminated & (obs[:, 4] > 1)
            hit = hit + jnp.sum(scored & (mine == theirs))
            n = n + jnp.sum(scored)
            state = jax.vmap(env.step)(
                state, action, jax.random.split(step_key, num_games)
            )
            return key, state, R + state.rewards[jnp.arange(num_games), me], hit, n

        _, _, R, hit, n = jax.lax.while_loop(
            lambda v: ~v[1].terminated.all(),
            body,
            (key, state, jnp.zeros(num_games), jnp.int32(0), jnp.int32(0)),
        )
        return R, hit / n

    return match


if __name__ == "__main__":
    main()

