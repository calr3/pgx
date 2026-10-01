"""Does the policy underrate long sliding captures?

For positions from a tdgauntlet match: every legal capturing move (opponent stones
taken, own ring kept) and the move actually played are scored by a short search
of the resulting position; for captures at least as good as the played move, we
record the raw network's rank of the piece (stage 0) and of the destination given
the piece (stage 1), by slide distance.

Run from the repo root, on the CPU (it needs no GPU and takes ~20 min for 2000
positions at 64 sims); summarise the output with gess_slide_report.py:

  JAX_PLATFORMS=cpu python examples/alphazero/gess_slide_diag.py \
      ../tdgauntlet/results/gess_e18_v_e17.json checkpoints/<run>/<step>.ckpt 2000 64 out.json
  python examples/alphazero/gess_slide_report.py out.json

The checkpoint's network is the one measured and the one whose search scores
the moves. Positions are sampled with a fixed seed, so two checkpoints run on
the same match file see the same positions.
"""
import json, sys, time, pickle
import numpy as np

match_path, ckpt_path, num_positions, sims, out_path = sys.argv[1:6]
num_positions, sims = int(num_positions), int(sims)
import pickle as _p
sys.path.insert(0, "examples/alphazero")
from config import Config as _C
import __main__; __main__.Config = _C
_cfg = _p.load(open(ckpt_path, "rb"))["config"]
sys.argv = ["x"] + [f"{k}={int(v) if k == 'gf_joint_head' else v}" for k, v in _cfg.model_dump().items()
                    if k.startswith(("env_id", "architecture", "gf_", "obs_planes"))] + ["selfplay_bf16=false"]
sys.path.insert(0, "examples/alphazero")
import jax, jax.numpy as jnp, mctx
from config import Config
import __main__; __main__.Config = Config
import train
from pgx._src.games import gess as G

env, game = train.env, G.Game()
ckpt = pickle.load(open(ckpt_path, "rb"))
model = ckpt["model"]
t0 = time.time()

# ─── positions ──────────────────────────────────────────────────────────────
D = json.load(open(match_path))
OPEN = {o["id"]: o["state"] for o in D["openings"]}
gstep = jax.jit(game.step)


def start(g):
    st = OPEN[g["opening"]]
    x = game.init()
    b = np.array([".BW".index(c) for c in st["board"]], dtype=np.asarray(x.board).dtype)
    return x._replace(board=jnp.asarray(b.reshape(np.asarray(x.board).shape)),
                      color=jnp.int32(st["to_move"]), no_capture_turns=jnp.int32(st["quiet_moves"]))


rng = np.random.default_rng(0)
cands = []  # (x, played (src, dst))
for g in D["games"]:
    x = start(g)
    for i, m in enumerate(g["moves"]):
        if i >= 6:
            cands.append((x, (m["move"]["from"], m["move"]["to"])))
        x = gstep(x, m["move"]["from"]); x = gstep(x, m["move"]["to"])
pick = rng.choice(len(cands), size=min(num_positions, len(cands)), replace=False)
positions = [cands[i] for i in pick]
print(f"{len(positions)} positions from {len(cands)} ({time.time()-t0:.0f}s)", flush=True)


def to_state(x):
    s = env.init(jax.random.PRNGKey(0))
    s = s.replace(_x=x, current_player=x.color, legal_action_mask=game.legal_action_mask(x))
    return s.replace(observation=env.observe(s, s.current_player))


to_state_j = jax.jit(to_state)

# ─── move enumeration ───────────────────────────────────────────────────────
N = 400
cells = jnp.arange(N)


@jax.jit
def all_moves(x):
    src_mask = game.legal_action_mask(x)
    mover, opp = jnp.int8(x.color + 1), jnp.int8(2 - x.color)
    n_opp, n_own = (x.board == opp).sum(), (x.board == mover).sum()

    def per_src(s):
        x1 = game.step(x, s)
        dm = game.legal_action_mask(x1) & src_mask[s]

        def per_dst(d):
            y = game.step(x1, d)
            return (n_opp - (y.board == opp).sum(), n_own - (y.board == mover).sum(),
                    G._has_ring(y.board, mover), y.winner)
        return (dm,) + jax.vmap(per_dst)(cells)
    return jax.vmap(per_src)(cells)  # each (400 src, 400 dst)


def dist(s, d):
    return max(abs(s // 20 - d // 20), abs(s % 20 - d % 20))


# ─── network priors ─────────────────────────────────────────────────────────
@jax.jit
def logits_of(obs):
    (l, v), _ = train.forward.apply(model[0], model[1], obs, is_eval=True)
    return l, v


def ranks(logits, mask):
    l = np.where(mask, logits, -np.inf)
    p = np.exp(l - l.max()); p /= p.sum()
    order = np.argsort(-l)
    r = np.empty(N, int); r[order] = np.arange(N)
    return p, r


# ─── short search of the position after a move (value for the mover) ────────
@jax.jit
def search_value(states, key):
    (l, v), _ = train.forward.apply(model[0], model[1], states.observation, is_eval=True)
    root = mctx.RootFnOutput(prior_logits=l, value=v, embedding=states)
    out = mctx.gumbel_muzero_policy(
        params=model, rng_key=key, root=root, recurrent_fn=train.recurrent_fn,
        num_simulations=sims, invalid_actions=~states.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value, gumbel_scale=1.0)
    return out.search_tree.summary().value  # for the player to move (the opponent)


B = 64
step_state = jax.jit(jax.vmap(env.step))
records, jobs = [], []
t1 = time.time()
for pi, (x, played) in enumerate(positions):
    dm, opp_cap, own_lost, ring, winner = (np.asarray(a) for a in all_moves(x))
    st = to_state_j(x)
    l0, _ = logits_of(st.observation[None])
    p0, r0 = ranks(np.asarray(l0[0]), np.asarray(st.legal_action_mask))
    color = int(x.color)
    legal_src = np.nonzero(np.asarray(st.legal_action_mask))[0]
    moves = []
    for s in legal_src:
        for d in np.nonzero(dm[s])[0]:
            win = winner[s, d] == color
            lose = winner[s, d] == 1 - color
            if (opp_cap[s, d] > 0 and ring[s, d] and not lose) or win or (int(s), int(d)) == played:
                moves.append((int(s), int(d)))
    if played not in moves:
        moves.append(played)
    # stage-1 priors for each source involved
    srcs = sorted({s for s, _ in moves})
    st1 = step_state(jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (len(srcs),) + a.shape), st),
                     jnp.array(srcs), jax.random.split(jax.random.PRNGKey(0), len(srcs)))
    l1, _ = logits_of(st1.observation)
    l1, m1 = np.asarray(l1), np.asarray(st1.legal_action_mask)
    p1r = {s: ranks(l1[i], m1[i]) for i, s in enumerate(srcs)}
    for s, d in moves:
        rec = dict(pos=pi, src=s, dst=d, dist=dist(s, d), opp_cap=int(opp_cap[s, d]),
                   own_lost=int(own_lost[s, d]), win=bool(winner[s, d] == color),
                   played=(s, d) == played, n_src=int(len(legal_src)),
                   p_src=float(p0[s]), r_src=int(r0[s]), p_dst=float(p1r[s][0][d]),
                   r_dst=int(p1r[s][1][d]), n_dst=int(m1[srcs.index(s)].sum()))
        records.append(rec)
        jobs.append((len(records) - 1, st, s, d))
    if pi % 50 == 0:
        print(f"enumerated {pi+1}/{len(positions)}, {len(records)} moves ({time.time()-t1:.0f}s)", flush=True)

print(f"enumeration done: {len(records)} moves in {time.time()-t1:.0f}s", flush=True)

# post-move states, searched in batches of B
t2 = time.time()
key = jax.random.PRNGKey(1)
two = jax.jit(lambda st, s, d: env.step(env.step(st, s), d))
for b0 in range(0, len(jobs), B):
    chunk = jobs[b0:b0 + B]
    posts = [two(st, s, d) for _, st, s, d in chunk]
    while len(posts) < B:
        posts.append(posts[-1])
    batch = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *posts)
    key, sub = jax.random.split(key)
    val = np.asarray(search_value(batch, sub))
    term = np.asarray(batch.terminated)
    for j, (ri, st, s, d) in enumerate(chunk):
        r = records[ri]
        if term[j]:
            mover = int(np.asarray(st.current_player))
            r["value"] = float(np.asarray(batch.rewards)[j, mover])
        else:
            r["value"] = -float(val[j])
    if (b0 // B) % 20 == 0:
        done = min(b0 + B, len(jobs))
        el = time.time() - t2
        print(f"searched {done}/{len(jobs)} ({el:.0f}s, eta {el/done*(len(jobs)-done):.0f}s)", flush=True)

json.dump(records, open(out_path, "w"))
print(f"wrote {len(records)} records to {out_path}; total {time.time()-t0:.0f}s", flush=True)
