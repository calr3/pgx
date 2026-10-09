"""Deep analysis of the Gess (gess_joint) starting position with one network.

  python examples/alphazero/analyse_opening.py <ckpt> <out.json> [options]

Three parts, run in order:

1. Root search: one Gumbel MuZero search from the start at each budget in --root-sims
   (deterministic: gumbel_scale 0, --considered moves), reporting each move's visits, value
   and prior, and whether the preferred move changes as the search grows.
2. Playouts of black's first moves: the policy's --first-moves most likely moves plus any the
   root search rates higher, each followed by complete games with the network searching
   every move on both sides (--play-sims, Gumbel noise on, so games vary). Game slots are
   refilled as games finish; at the time limit no new game starts and the running ones
   finish, so short games are not over-represented.
3. Replies: for black's --reply-lines best first moves (by part 2), white's --replies most
   likely replies, played out the same way.

Results are reported for black (who moves first): score = (wins + draws / 2) / games, with
its standard error. Moves are written as tdgauntlet does: the piece's centre, then its
destination, as column letter (a-t) and row number (20 at the top, 1 at the bottom).
"""

import argparse
import json
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import mctx
import numpy as np

import pgx

sys.path.insert(0, "examples/alphazero")
import __main__  # noqa: E402

from config import Config  # noqa: E402
from network import cast_floating, make_forward  # noqa: E402

__main__.Config = Config  # checkpoints pickle their Config as __main__.Config

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("out")
ap.add_argument("--root-sims", default="1000,4000,16000,64000")
ap.add_argument("--considered", type=int, default=32)
ap.add_argument("--root-budget-min", type=float, default=10.0)
ap.add_argument("--first-moves", type=int, default=12)
ap.add_argument("--play-sims", type=int, default=128)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--play-budget-min", type=float, default=40.0)
ap.add_argument("--reply-lines", type=int, default=3)
ap.add_argument("--replies", type=int, default=6)
ap.add_argument("--reply-budget-min", type=float, default=10.0)
ap.add_argument("--max-plies", type=int, default=400)
args = ap.parse_args()

ck = pickle.load(open(args.ckpt, "rb"))
cfg = ck["config"]
env = pgx.make("gess_joint")
dtype = jnp.bfloat16 if jax.default_backend() == "gpu" else jnp.float32
forward = make_forward(env.num_actions, cfg, dtype=dtype)
model = cast_floating(ck["model"], dtype)
qtransform = mctx.qtransform_completed_by_mix_value
step_v = jax.vmap(env.step)
COLS = "abcdefghijklmnopqrst"


def san(m):
    sq = lambda c: f"{COLS[c % 20]}{20 - c // 20}"
    return f"{sq(m // 400)}-{sq(m % 400)}"


def evaluate(state):
    (logits, value), _ = forward.apply(model[0], model[1], state.observation, is_eval=True, moves=state._moves)
    return logits.astype(jnp.float32), jnp.where(state.terminated, 0.0, value.astype(jnp.float32))


def recurrent_fn(params, rng_key, action, state):
    del params, rng_key
    player = state.current_player
    state = step_v(state, action)
    logits, value = evaluate(state)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    reward = state.rewards[jnp.arange(action.shape[0]), player]
    discount = jnp.where(state.terminated, 0.0, jnp.where(state.current_player == player, 1.0, -1.0))
    return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits, value=value), state


def search(state, key, sims, gumbel_scale, considered=16):
    logits, value = evaluate(state)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
    return mctx.gumbel_muzero_policy(
        params=None, rng_key=key, root=root, recurrent_fn=recurrent_fn, num_simulations=sims,
        invalid_actions=~state.legal_action_mask, qtransform=qtransform,
        max_num_considered_actions=considered, gumbel_scale=gumbel_scale,
    )


def batched(state, n):
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n, *x.shape)), state)


def play_move(state, move):
    """state after `move` (a move code), for an unbatched state."""
    hit = np.nonzero(np.asarray(state._moves) == move)[0]
    assert len(hit) == 1, f"{san(move)} is not legal here"
    return env.step(state, jnp.int32(hit[0]))


start = env.init(jax.random.PRNGKey(0))
report = {"checkpoint": args.ckpt, "side_to_move": "black"}
logits0, value0 = jax.jit(evaluate)(batched(start, 1))
moves0 = np.asarray(start._moves)
legal0 = moves0 >= 0
prior0 = np.where(legal0, np.asarray(logits0[0]), -np.inf)
prior0 = np.exp(prior0 - prior0.max())
prior0 /= prior0.sum()
report["network_value"] = float(value0[0])
print(f"start: {legal0.sum()} legal moves, network value for black {float(value0[0]):+.3f}", flush=True)

# 1. Root search.
report["root_search"] = []
t_root = time.time()
for sims in [int(s) for s in args.root_sims.split(",")]:
    if time.time() - t_root > args.root_budget_min * 60:
        print(f"root search: skipping {sims} sims (time budget)", flush=True)
        break
    t = time.time()
    out = jax.jit(lambda s, k: search(s, k, sims, 0.0, args.considered))(batched(start, 1), jax.random.PRNGKey(1))
    tree = out.search_tree
    visits = np.asarray(tree.children_visits[0, 0])
    q = np.asarray(tree.qvalues(jnp.zeros(1, jnp.int32))[0])
    order = np.argsort(-visits)[:10]
    rows = [{"move": san(int(moves0[i])), "visits": int(visits[i]), "value": float(q[i]),
             "prior": float(prior0[i]), "weight": float(out.action_weights[0, i])} for i in order]
    best = san(int(moves0[int(out.action[0])]))
    report["root_search"].append({"sims": sims, "seconds": time.time() - t, "chosen": best, "top": rows})
    print(f"root search, {sims} sims ({time.time() - t:.0f} s): chooses {best}", flush=True)
    for r in rows[:6]:
        print(f"   {r['move']:<9} visits {r['visits']:>6}  value {r['value']:+.3f}  prior {r['prior'] * 100:5.1f}%", flush=True)


def playouts(lines, budget_min, label):
    """Complete games from each line (a list of move codes from the start); black's score per line."""
    prefix = []
    for line in lines:
        s = start
        for m in line:
            s = play_move(s, m)
        prefix.append(s)
    prefix = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *prefix)
    nlines = len(lines)

    @jax.jit
    def step(state, key):
        out = search(state, key, args.play_sims, 1.0)
        nxt = step_v(state, out.action)
        return jax.tree_util.tree_map(
            lambda a, b: jnp.where(state.terminated.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), nxt, state)

    B = args.batch
    slot_line = np.arange(B) % nlines
    state = jax.tree_util.tree_map(lambda x: x[jnp.asarray(slot_line)], prefix)
    plies = np.zeros(B, int)
    active = np.ones(B, bool)
    next_line = B % nlines
    results = [[0, 0, 0, 0] for _ in lines]  # black wins, draws, black losses, cut off at max_plies
    lengths = [[] for _ in lines]
    key = jax.random.PRNGKey(2)
    t0 = time.time()
    while active.any():
        key, k = jax.random.split(key)
        state = step(state, k)
        plies += active
        done = (np.asarray(state.terminated) | (plies >= args.max_plies)) & active
        if done.any():
            black = np.asarray(state.rewards[:, 0])
            ended = np.asarray(state.terminated)
            for i in np.flatnonzero(done):
                r = results[slot_line[i]]
                r[3 if not ended[i] else 0 if black[i] > 0 else 2 if black[i] < 0 else 1] += 1
                lengths[slot_line[i]].append(int(plies[i]) + len(lines[slot_line[i]]))
            refill = done & (time.time() - t0 < budget_min * 60)
            active &= ~(done & ~refill)
            if refill.any():
                idx = np.flatnonzero(refill)
                new_lines = (next_line + np.arange(len(idx))) % nlines
                next_line = int((next_line + len(idx)) % nlines)
                slot_line[idx] = new_lines
                plies[idx] = 0
                fresh = jax.tree_util.tree_map(lambda x: x[jnp.asarray(new_lines)], prefix)
                state = _refill(state, fresh, jnp.asarray(idx))
    rows = []
    for line, (w, d, l_, cut), ln in zip(lines, results, lengths):
        n = w + d + l_
        score = (w + d / 2) / max(n, 1)
        se = (score * (1 - score) / max(n, 1)) ** 0.5
        rows.append({"line": [san(m) for m in line], "games": n, "black_wins": w, "draws": d,
                     "black_losses": l_, "cut_off": cut, "black_score": score, "se": se,
                     "mean_plies": float(np.mean(ln)) if ln else None})
    rows.sort(key=lambda r: -r["black_score"])
    print(f"{label} ({(time.time() - t0) / 60:.1f} min, {sum(r['games'] for r in rows)} games):", flush=True)
    for r in rows:
        print(f"   {' '.join(r['line']):<20} black {r['black_score']:.3f} +/- {r['se']:.3f}  "
              f"({r['black_wins']}-{r['draws']}-{r['black_losses']}, {r['games']} games, ~{r['mean_plies']:.0f} plies"
              + (f", {r['cut_off']} cut off unfinished" if r["cut_off"] else "") + ")",
              flush=True)
    return rows


def _refill(state, fresh, idx):
    """state with the slots `idx` replaced by `fresh` (one state per slot, in order)."""
    return jax.tree_util.tree_map(lambda a, b: a.at[idx].set(b), state, fresh)


# 2. Black's first moves.
candidates = [int(moves0[i]) for i in np.argsort(-prior0)[: args.first_moves]]
for part in report["root_search"]:
    for r in part["top"][:4]:
        code = next(int(m) for m in moves0 if m >= 0 and san(int(m)) == r["move"])
        if code not in candidates:
            candidates.append(code)
report["first_moves"] = playouts([[m] for m in candidates], args.play_budget_min, "black's first moves")

# 3. White's replies to black's best first moves.
lines = []
for r in report["first_moves"][: args.reply_lines]:
    first = next(int(m) for m in moves0 if m >= 0 and san(int(m)) == r["line"][0])
    s1 = play_move(start, first)
    lg, _ = jax.jit(evaluate)(batched(s1, 1))
    mv = np.asarray(s1._moves)
    p = np.where(mv >= 0, np.asarray(lg[0]), -np.inf)
    lines += [[first, int(mv[i])] for i in np.argsort(-p)[: args.replies]]
report["replies"] = playouts(lines, args.reply_budget_min, "white's replies")

json.dump(report, open(args.out, "w"), indent=1)
print(f"wrote {args.out}", flush=True)
