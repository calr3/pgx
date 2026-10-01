"""Convert a ``gess`` GessFormer checkpoint to ``gess_joint`` (one action per move).

The trunk, value head and piece head carry over; the move-scoring part of the
whole-move policy head (network._gess_move_policy) starts at zero, so the
converted network chooses pieces exactly as the original did and spreads each
piece's probability evenly over its destinations. The two-step destination head
and any auxiliary heads are dropped. The output holds a config and model only,
for ``init_from`` and as an MCTS evaluation opponent.

  JAX_PLATFORMS=cpu python examples/alphazero/convert_gess_joint.py \\
      checkpoints/<gess run>/<step>.ckpt checkpoints/<name>/000000.ckpt
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402

import pgx  # noqa: E402
from config import Config  # noqa: E402
from network import make_forward  # noqa: E402

import __main__  # noqa: E402

__main__.Config = Config  # checkpoints pickle their Config as __main__.Config

src, out = sys.argv[1:3]
ckpt = pickle.load(open(src, "rb"))
cfg = ckpt["config"].model_copy(update={"env_id": "gess_joint", "aux_win_weight": 0.0, "gf_joint_head": 0})
env = pgx.make("gess_joint")
state = env.init(jax.random.PRNGKey(0))
forward = make_forward(env.num_actions, cfg)
fresh, net_state = forward.init(jax.random.PRNGKey(0), state.observation[None], moves=state._moves[None])

params = {}
for module, entries in fresh.items():
    params[module] = {}
    for name, value in entries.items():
        old = ckpt["model"][0].get(module, {}).get(name)
        if old is not None and old.shape == value.shape:
            params[module][name] = old
        else:
            params[module][name] = value
            print(f"new: {module}/{name} {value.shape}")
for module, entries in ckpt["model"][0].items():
    for name in entries:
        if name not in fresh.get(module, {}):
            print(f"dropped: {module}/{name}")

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(
        {
            "config": cfg,
            "model": (jax.device_get(params), jax.device_get(net_state)),
            "iteration": 0,
            "frames": 0,
            "hours": 0.0,
            "evaluated": True,
            "converted_from": src,
            "env_id": env.id,
            "env_version": env.version,
            "pgx.__version__": pgx.__version__,
        },
        f,
    )
print(f"wrote {out}")
