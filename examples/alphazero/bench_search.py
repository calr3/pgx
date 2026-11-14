"""Time one self-play search per mctx implementation, on whatever device is present.

Runs the same batched Gumbel MuZero search as train.py's self-play with each
implementation in turn and reports compile time, run time and XLA's temp-buffer
size for the compiled search:

  reference   - the original upstream mctx (carried by the fork)
  fork        - the calr3/mctx fork: faster loop, compact tree
  fork_bf16   - the fork with the tree's prior logits stored in bfloat16
                (search_tree_bf16; changes results slightly)

reference and fork must give the same actions; the script checks this on every
run and reports any disagreement. On a GPU, two runs of identical code can
already differ slightly, so a mismatch there is only a warning;
tests/test_mctx_fork.py is the exactness check, and it runs on CPU.

Usage (from the repo root; any train.py config key is accepted):

  python -u examples/alphazero/bench_search.py env_id=gess architecture=gessformer \\
      selfplay_batch_size=1024 num_simulations=32 selfplay_bf16=true

  # The network from a checkpoint (its config, then any overrides):
  python -u examples/alphazero/bench_search.py checkpoint=checkpoints/.../000400.ckpt

Bench-only keys:
  checkpoint=PATH     use this checkpoint's params and config (else random init)
  impls=a,b,...       implementations to time (default reference,fork,fork_bf16)
  repeats=N           timed runs per implementation (default 10)
  start_plies=N       random plies played before searching (default 20), so
                      the roots are mid-game positions rather than the opening
  profile_dir=DIR     also write a jax.profiler trace of one run per impl
  out_json=PATH       append the results as one JSON line, for collecting runs

The batch is selfplay_batch_size, all on one device (train.py splits it across
devices, so pass the per-device share when comparing with a multi-device run).
Network inference dominates real self-play; with a trained-size network the
search overhead this measures is a smaller share than with a tiny one.
"""

import json
import os
import platform
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pgx
from omegaconf import OmegaConf

import search_impl
from config import Config
from model_tournament import load_from_checkpoint
from network import cast_floating, make_forward

BENCH_KEYS = {
    "checkpoint": "",
    "impls": "reference,fork,fork_bf16",
    "repeats": 10,
    "start_plies": 20,
    "profile_dir": "",
    "out_json": "",
}
IMPLS = {
    # name: (mctx_impl, search_tree_bf16)
    "reference": ("reference", False),
    "fork": ("fork", False),
    "fork_bf16": ("fork", True),
}


def parse_args():
    cli = OmegaConf.to_container(OmegaConf.from_cli())
    bench = {k: cli.pop(k, v) for k, v in BENCH_KEYS.items()}
    if bench["checkpoint"]:
        config, model = load_from_checkpoint(bench["checkpoint"])
        config = Config(**{**config.model_dump(), **cli})
    else:
        config, model = Config(**cli), None
    return config, model, bench


def random_positions(env, batch_size: int, plies: int, key):
    """Initial states advanced by `plies` uniformly random legal moves."""

    def step(state, key):
        k1, k2 = jax.random.split(key)
        logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
        action = jax.random.categorical(k1, logits)
        nxt = jax.vmap(env.step)(state, action, jax.random.split(k2, batch_size))
        # Keep finished games at their last state; restart is not needed here.
        keep = state.terminated | nxt.terminated
        nxt = jax.tree_util.tree_map(
            lambda a, b: jnp.where(keep.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), state, nxt
        )
        return nxt, None

    key, init_key = jax.random.split(key)
    state = jax.vmap(env.init)(jax.random.split(init_key, batch_size))
    state, _ = jax.lax.scan(step, state, jax.random.split(key, plies))
    return state


def make_search(m, env, forward, num_simulations, policy_kwargs):
    """train.py's self-play search (deterministic-env branch), for package m."""

    def recurrent_fn(model, rng_key, action, state):
        params, model_state = model
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action, jax.random.split(rng_key, action.shape[0]))
        (logits, value), _ = forward.apply(params, model_state, state.observation, is_eval=True)
        value = jnp.where(state.terminated, 0.0, value)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        discount = jnp.where(state.current_player == current_player, 1.0, -1.0)
        discount = jnp.where(state.terminated, 0.0, discount)
        reward = state.rewards[jnp.arange(action.shape[0]), current_player]
        return m.RecurrentFnOutput(
            reward=reward, discount=discount, prior_logits=logits, value=value
        ), state

    @jax.jit
    def search(model, state, key):
        params, model_state = model
        (logits, value), _ = forward.apply(params, model_state, state.observation, is_eval=True)
        out = m.gumbel_muzero_policy(
            params=model,
            rng_key=key,
            root=m.RootFnOutput(prior_logits=logits, value=value, embedding=state),
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=m.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
            **policy_kwargs,
        )
        return out.action, out.action_weights

    return search


def main():
    config, model, bench = parse_args()
    env = pgx.make(config.env_id)
    device = jax.devices()[0]
    dtype = jnp.bfloat16 if config.selfplay_bf16 else jnp.float32
    forward = make_forward(env.num_actions, config, dtype=dtype)
    key = jax.random.PRNGKey(0)
    batch = config.selfplay_batch_size
    state = random_positions(env, batch, bench["start_plies"], key)
    if model is None:
        model = make_forward(env.num_actions, config).init(key, state.observation[:2])
    model = cast_floating(model, dtype)
    sims = config.num_simulations

    print(
        f"{config.env_id} {config.architecture} batch={batch} sims={sims} "
        f"selfplay_bf16={config.selfplay_bf16} device={device.device_kind} "
        f"({jax.default_backend()}, {jax.device_count()} devices) jax={jax.__version__}",
        flush=True,
    )
    results, actions = [], {}
    for name in bench["impls"].split(","):
        impl, tree_bf16 = IMPLS[name]
        m = search_impl.load(impl)
        search = make_search(m, env, forward, sims, search_impl.policy_kwargs(tree_bf16))

        t = time.perf_counter()
        compiled = search.lower(model, state, key).compile()
        compile_s = time.perf_counter() - t
        mem = compiled.memory_analysis()
        temp_mib = getattr(mem, "temp_size_in_bytes", float("nan")) / 2**20

        jax.block_until_ready(search(model, state, key))  # warm-up
        times = []
        for i in range(bench["repeats"]):
            k = jax.random.fold_in(key, i)
            t = time.perf_counter()
            out = jax.block_until_ready(search(model, state, k))
            times.append(time.perf_counter() - t)
        actions[name] = np.asarray(search(model, state, key)[0])
        if bench["profile_dir"]:
            with jax.profiler.trace(os.path.join(bench["profile_dir"], name)):
                jax.block_until_ready(search(model, state, key))
        del out

        median = statistics.median(times)
        row = {
            "impl": name,
            "median_ms": median * 1e3,
            "min_ms": min(times) * 1e3,
            "positions_per_s": batch / median,
            "compile_s": compile_s,
            "temp_mib": temp_mib,
        }
        results.append(row)
        print(
            f"{name:10s} median {row['median_ms']:9.1f} ms  min {row['min_ms']:9.1f} ms  "
            f"{row['positions_per_s']:9.0f} pos/s  compile {compile_s:6.1f} s  "
            f"temp {temp_mib:8.1f} MiB",
            flush=True,
        )

    base = next((r for r in results if r["impl"] == "reference"), None)
    if base:
        for r in results:
            r["speedup_vs_reference"] = base["median_ms"] / r["median_ms"]
        print("speedup vs reference: " + ", ".join(
            f"{r['impl']} {r['speedup_vs_reference']:.2f}x" for r in results
        ))
    if "reference" in actions and "fork" in actions:
        same = float(np.mean(actions["reference"] == actions["fork"]))
        note = "" if same == 1.0 else (
            "  (expected 1.0; on GPU small run-to-run differences are possible, on CPU it is a bug)"
        )
        print(f"fork vs reference: same action for {same:.1%} of positions{note}")
    if "fork_bf16" in actions and "fork" in actions:
        same = float(np.mean(actions["fork_bf16"] == actions["fork"]))
        print(f"fork_bf16 vs fork: same action for {same:.1%} of positions")

    if bench["out_json"]:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": platform.node(),
            "device": device.device_kind,
            "backend": jax.default_backend(),
            "jax": jax.__version__,
            "argv": sys.argv[1:],
            "env_id": config.env_id,
            "architecture": config.architecture,
            "batch": batch,
            "num_simulations": sims,
            "selfplay_bf16": config.selfplay_bf16,
            "results": results,
        }
        with open(bench["out_json"], "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
