# Export a trained AlphaZero network to ONNX, for running in a browser (or
# anywhere ONNX Runtime goes). The policy/value network only: the game rules and
# the MCTS search still have to exist on the target platform.
#
# Two steps, because the conversion needs the `jax2onnx` toolchain, which pulls
# in its own JAX/flax versions and should live in a throwaway virtualenv:
#
#   # 1. in this repo's environment: weights, config and reference outputs
#   python examples/alphazero/export_onnx.py dump checkpoints/<run>/000100.ckpt /tmp/export
#
#   # 2. in a scratch venv: python -m venv /tmp/onnxenv
#   #    /tmp/onnxenv/bin/pip install jax2onnx onnx onnxruntime dm-haiku "jax==0.8.2" "jaxlib==0.8.2"
#   /tmp/onnxenv/bin/python examples/alphazero/export_onnx.py convert /tmp/export
#   /tmp/onnxenv/bin/python examples/alphazero/export_onnx.py verify /tmp/export
#
# `verify` replays the reference positions through ONNX Runtime and reports the
# difference from JAX. Use opset 20: it is what onnxruntime-web supports, and
# jax2onnx's opset 17 output does not validate.

import json
import os
import pickle
import sys
import time

import numpy as np

OPSET = 20


def dump(checkpoint: str, out_dir: str) -> None:
    """Write params.npz, config.json and reference.npz from a checkpoint."""
    import jax
    import jax.numpy as jnp
    import pgx

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    from network import make_forward

    sys.modules["__main__"].Config = Config  # older checkpoints pickled it there
    os.makedirs(out_dir, exist_ok=True)
    with open(checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    cfg = Config(**ckpt["config"].__dict__)
    params, state = ckpt["model"]
    assert not state, "only networks without batch-norm state are supported"

    flat = {f"{m}|{n}": np.asarray(v) for m, d in params.items() for n, v in d.items()}
    np.savez(os.path.join(out_dir, "params.npz"), **flat)
    json.dump(cfg.model_dump(), open(os.path.join(out_dir, "config.json"), "w"))

    env = pgx.make(cfg.env_id)
    state0 = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 3))
    state1 = jax.vmap(env.step)(state0, jnp.argmax(state0.legal_action_mask, axis=-1))
    obs = np.concatenate([np.asarray(state0.observation), np.asarray(state1.observation)])
    (logits, value), _ = make_forward(env.num_actions, cfg).apply(
        params, state, jnp.asarray(obs), is_eval=True
    )
    np.savez(
        os.path.join(out_dir, "reference.npz"),
        obs=obs, logits=np.asarray(logits), value=np.asarray(value),
    )
    print(f"dumped {len(flat)} tensors, obs {obs.shape}, actions {env.num_actions} -> {out_dir}")


def _build(out_dir: str):
    """Rebuild the network from the dump; returns (apply_fn, obs_shape)."""
    import jax.numpy as jnp

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import network

    cfg_dict = json.load(open(os.path.join(out_dir, "config.json")))
    # Rematerialisation is a training-time memory trick and would only clutter
    # the exported graph.
    cfg_dict.update(gf_remat=False, rf_remat=False, bf_remat=False)
    cfg = type("Cfg", (), cfg_dict)  # attribute access is all make_forward needs

    flat = np.load(os.path.join(out_dir, "params.npz"))
    params = {}
    for key in flat.files:
        module, name = key.split("|")
        params.setdefault(module, {})[name] = jnp.asarray(flat[key])

    ref = np.load(os.path.join(out_dir, "reference.npz"))
    transformed = network.make_forward(ref["logits"].shape[-1], cfg)
    return lambda x: transformed.apply(params, {}, x, is_eval=True)[0], ref["obs"].shape[1:]


def convert(out_dir: str) -> None:
    import jax
    import jax.numpy as jnp
    import jax2onnx

    apply_fn, obs_shape = _build(out_dir)
    ref = np.load(os.path.join(out_dir, "reference.npz"))
    logits, value = jax.jit(apply_fn)(jnp.asarray(ref["obs"]))
    print(
        "jax re-run vs reference: logits max diff",
        float(np.abs(np.asarray(logits) - ref["logits"]).max()),
        "| value max diff", float(np.abs(np.asarray(value) - ref["value"]).max()),
    )

    path = os.path.join(out_dir, "model.onnx")
    t0 = time.time()
    jax2onnx.to_onnx(
        apply_fn, [("B", *obs_shape)], model_name="alphazero", export_mode="web",
        opset=OPSET, return_mode="file", output_path=path,
    )
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


def verify(out_dir: str) -> None:
    import onnx
    import onnxruntime as ort

    ref = np.load(os.path.join(out_dir, "reference.npz"))
    path = os.path.join(out_dir, "model.onnx")
    model = onnx.load(path)
    onnx.checker.check_model(model)
    print(f"opset {model.opset_import[0].version}, {len(model.graph.node)} nodes")

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: ref["obs"].astype(np.float32)})
    logits, value = (out[0], out[1]) if out[0].ndim == 2 else (out[1], out[0])
    print(
        "onnx vs jax: logits max diff", float(np.abs(logits - ref["logits"]).max()),
        f"(scale {float(np.abs(ref['logits']).max()):.1f})",
        "| value max diff", float(np.abs(value.reshape(-1) - ref["value"]).max()),
        "| best-move agreement", float((logits.argmax(-1) == ref["logits"].argmax(-1)).mean()),
    )
    one = ref["obs"][:1].astype(np.float32)
    sess.run(None, {name: one})
    t0 = time.time()
    for _ in range(20):
        sess.run(None, {name: one})
    print(f"CPU latency, batch 1: {(time.time() - t0) / 20 * 1000:.1f} ms")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "dump":
        dump(sys.argv[2], sys.argv[3])
    elif command == "convert":
        convert(sys.argv[2])
    elif command == "verify":
        verify(sys.argv[2])
    else:
        print(__doc__ or "usage: export_onnx.py {dump <ckpt> <dir> | convert <dir> | verify <dir>}")
        sys.exit(1)
