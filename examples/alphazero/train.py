# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import pickle
import signal
import time
from functools import partial
from types import FrameType
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import pgx
import wandb
from omegaconf import OmegaConf
from pgx.experimental import auto_reset

from config import Config
from network import make_forward, make_optimizer
from replay_buffer import ReplayBuffer
from trajectories import PendingTrajectories, Sample

# A Haiku model is a (params, state) pair, as returned by forward.init.
Model = tuple[hk.Params, hk.State]

devices = jax.local_devices()
num_devices: int = len(devices)


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)


def _validate_config(config: Config, num_devices: int) -> None:
    # Each iteration produces selfplay_batch_size * max_num_steps samples. With
    # the default num_updates_per_iter=0 they are split into whole minibatches of
    # training_batch_size, which should not silently drop a remainder. Each
    # minibatch is split across devices, which requires the second check.
    total_samples = config.selfplay_batch_size * config.max_num_steps
    if config.num_updates_per_iter == 0 and total_samples % config.training_batch_size != 0:
        num_updates = total_samples // config.training_batch_size
        remainder = total_samples - num_updates * config.training_batch_size
        raise ValueError(
            "training_batch_size must evenly divide selfplay_batch_size * max_num_steps. "
            f"Got selfplay_batch_size={config.selfplay_batch_size} * "
            f"max_num_steps={config.max_num_steps} = {total_samples} samples, "
            f"training_batch_size={config.training_batch_size} -> "
            f"{num_updates} full minibatches with {remainder} samples left over. "
            f"Pick a training_batch_size that divides {total_samples} "
            "(e.g. a power of two)."
        )
    if config.training_batch_size % num_devices != 0:
        raise ValueError(
            "training_batch_size must be divisible by the number of devices. "
            f"Got training_batch_size={config.training_batch_size}, num_devices={num_devices}."
        )


_validate_config(config, num_devices)

env = pgx.make(config.env_id)
baseline = pgx.make_baseline_model(config.env_id + "_v0")


forward = make_forward(env.num_actions, config)
optimizer = make_optimizer(config)


def recurrent_fn(
    model: Model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State
) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    # +1 when the same player is still to move (multi-stage turn), -1 when the
    # opponent is now to move (normal alternating case), 0 at terminal.
    discount = jnp.where(state.current_player == current_player, 1.0, -1.0)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


# num_simulations is a static (broadcast, not mapped) argument so it can be
# varied across iterations via config.sim_schedule. Each distinct value triggers
# one XLA recompile of this self-play step; the schedule changes it only a
# handful of times over a run.
@partial(jax.pmap, static_broadcasted_argnums=(3,))
def selfplay(
    model: Model, state: pgx.State, rng_key: jnp.ndarray, num_simulations: int
) -> tuple[pgx.State, SelfplayOutput]:
    """Play max_num_steps from `state`, returning the final state and the steps.

    Unless config.continue_games, `state` is ignored and every game slot
    restarts from the initial position.
    """
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state: pgx.State, key: jnp.ndarray) -> tuple[pgx.State, SelfplayOutput]:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,  # 0.0 for perfect information games
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        # +1 when the same player is still to move (multi-stage turn), -1 when
        # the opponent is now to move (normal alternating case), 0 at terminal.
        discount = jnp.where(state.current_player == actor, 1.0, -1.0)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    if not config.continue_games:
        keys = jax.random.split(sub_key, batch_size)
        state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    state, data = jax.lax.scan(step_fn, state, key_seq)

    return state, data


@jax.pmap
def init_selfplay_state(rng_key: jnp.ndarray) -> pgx.State:
    return jax.vmap(env.init)(jax.random.split(rng_key, config.selfplay_batch_size // num_devices))


def loss_fn(
    model_params: hk.Params, model_state: hk.State, samples: Sample
) -> tuple[jnp.ndarray, tuple[hk.State, jnp.ndarray, jnp.ndarray]]:
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)  # mask if the episode is truncated

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(
    model: Model, opt_state: optax.OptState, data: Sample
) -> tuple[Model, optax.OptState, jnp.ndarray, jnp.ndarray]:
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state, model_params)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss


@jax.pmap
def evaluate(rng_key: jnp.ndarray, my_model: Model) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A simplified evaluation by sampling. Only for debugging.
    Please use MCTS and run tournaments for serious evaluation.

    Returns (R, value_sse, value_n): the per-game outcome from the model's
    perspective, and the summed squared error / count of the model's value-head
    predictions against the eventual game outcome (see below).
    """
    my_model_params, my_model_state = my_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    # Seat-balance the eval: the model plays seat 0 in the first half of the
    # batch and seat 1 in the second half, so any first-player advantage cancels
    # out instead of biasing avg_R. Per-game seat assignment (shape (batch,)).
    my_player = (jnp.arange(batch_size) >= batch_size // 2).astype(jnp.int32)

    def body_fn(
        val: tuple[jnp.ndarray, pgx.State, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, pgx.State, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        key, state, R, sum_v2, sum_sv, n = val
        # Only score states that are still in play; already-finished games keep
        # getting stepped (until every game terminates) but must not contribute.
        alive = (~state.terminated).astype(jnp.float32)
        (my_logits, my_value), _ = forward.apply(
            my_model_params, my_model_state, state.observation, is_eval=True
        )
        opp_logits, _ = baseline(state.observation)
        is_my_turn = state.current_player == my_player
        logits = jnp.where(is_my_turn.reshape((-1, 1)), my_logits, opp_logits)

        # The value head predicts the outcome for the player to move. The true
        # target is that player's final reward = s * R, where R is my_player's
        # final reward and s = +1 on the model's turns, -1 on the opponent's
        # (the game is zero-sum). R is unknown until the game ends, so instead of
        # the squared error we accumulate its expansion
        #   (v - s*R)^2 = v^2 - 2*R*(s*v) + R^2,
        # combined with R after the loop. (s^2 = 1, so the last term is just R^2.)
        s = jnp.where(is_my_turn, 1.0, -1.0)
        sum_v2 = sum_v2 + alive * my_value**2
        sum_sv = sum_sv + alive * s * my_value
        n = n + alive

        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R, sum_v2, sum_sv, n)

    zeros = jnp.zeros(batch_size)
    _, _, R, sum_v2, sum_sv, n = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        body_fn,
        (key, state, zeros, zeros, zeros, zeros),
    )
    # Per-game sum of squared value errors, recombined with the final outcome R.
    sse = sum_v2 - 2.0 * R * sum_sv + n * R**2
    return R, sse.sum(), n.sum()


# Set by the SIGINT handler to request a clean shutdown: the training loop
# finishes the current iteration, writes a checkpoint, then exits. A second
# Ctrl+C forces an immediate exit (no final checkpoint).
_stop_requested = False


def _request_stop(signum: int, frame: FrameType | None) -> None:
    global _stop_requested
    if _stop_requested:
        print("\nSecond interrupt received -- exiting immediately (no checkpoint).")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt
    _stop_requested = True
    print(
        "\nInterrupt received -- will finish the current iteration, write a "
        "checkpoint, then exit. Press Ctrl+C again to exit immediately."
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _request_stop)

    # Load checkpoint up front (if resuming) so we can also recover the wandb run id
    ckpt = None
    if config.resume_from:
        print(f"Resuming from checkpoint: {config.resume_from}")
        with open(config.resume_from, "rb") as f:
            ckpt = pickle.load(f)

    # Resume the original wandb run when possible
    wandb_run_id = config.wandb_run_id
    if not wandb_run_id and ckpt is not None:
        wandb_run_id = ckpt.get("wandb_run_id", "")
    wandb.init(
        project=f"pgx-az-{config.env_id}",
        config=config.model_dump(),
        id=wandb_run_id or None,
        resume="allow" if wandb_run_id else None,
    )

    # Initialize model and opt_state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
    opt_state = optimizer.init(params=model[0])

    # Logging/training state (may be overwritten when resuming from a checkpoint)
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    rng_key = jax.random.PRNGKey(config.seed)

    # Optionally resume from a previous checkpoint
    if ckpt is not None:
        model = ckpt["model"]
        opt_state = ckpt["opt_state"]
        iteration = ckpt["iteration"]
        frames = ckpt["frames"]
        hours = ckpt["hours"]
        if config.reseed_on_resume:
            # Keep rng_key = PRNGKey(config.seed) (set above) so self-play
            # generates a fresh game stream rather than replaying the checkpoint's.
            print(f"Reseeding RNG from seed={config.seed} (fresh self-play games)")
        else:
            rng_key = ckpt["rng_key"]
        print(f"Resumed at iteration {iteration} ({frames} frames, {hours:.2f} hours)")

    # Replicate to all devices: add a leading device axis (size num_devices)
    # that pmap maps onto the local devices.
    model, opt_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices, *x.shape)), (model, opt_state)
    )

    # Prepare checkpoint dir. When resuming, keep writing into the original
    # checkpoint's directory; otherwise create a fresh timestamped one.
    if config.resume_from:
        ckpt_dir = os.path.dirname(config.resume_from)
    else:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        now = now.strftime("%Y%m%d%H%M%S")
        ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save_checkpoint(
        iteration: int,
        rng_key: jnp.ndarray,
        model: Model,
        opt_state: optax.OptState,
        frames: int,
        hours: float,
    ) -> None:
        model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
        ckpt_path = os.path.join(ckpt_dir, f"{iteration:06d}.ckpt")
        print(f"Saving checkpoint: {os.path.relpath(ckpt_path)}")
        with open(ckpt_path, "wb") as f:
            dic = {
                "config": config,
                "rng_key": rng_key,
                "model": jax.device_get(model_0),
                "opt_state": jax.device_get(opt_state_0),
                "iteration": iteration,
                "frames": frames,
                "hours": hours,
                "wandb_run_id": wandb.run.id,
                "pgx.__version__": pgx.__version__,
                "env_id": env.id,
                "env_version": env.version,
            }
            pickle.dump(dic, f)

    samples_per_iter = config.selfplay_batch_size * config.max_num_steps
    replay_buffer = ReplayBuffer(config.replay_buffer_iters * samples_per_iter)
    max_num_updates = config.updates_per_iter()
    replay_buffer_announced = False
    trajectories = PendingTrajectories(config.max_pending_steps)
    # Self-play games in progress; fresh games on start and on resume. Keyed off
    # the seed without consuming rng_key, so its stream is unchanged.
    selfplay_state = init_selfplay_state(
        jax.random.split(jax.random.fold_in(jax.random.PRNGKey(config.seed), 1), num_devices)
    )

    # Initialize logging dict
    log = {"iteration": iteration, "hours": hours, "frames": frames}

    while True:
        if iteration % config.eval_interval == 0:
            # Evaluation
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            R, value_sse, value_n = evaluate(keys, model)
            # value_sse / value_n are summed per device; aggregate across devices
            # by re-dividing the totals (avoids weighting devices unequally).
            value_mse = (value_sse.sum() / value_n.sum()).item()
            log.update(
                {
                    f"eval/vs_baseline/avg_R": R.mean().item(),
                    f"eval/vs_baseline/win_rate": ((R == 1).sum() / R.size).item(),
                    f"eval/vs_baseline/draw_rate": ((R == 0).sum() / R.size).item(),
                    f"eval/vs_baseline/lose_rate": ((R == -1).sum() / R.size).item(),
                    f"eval/vs_baseline/value_mse": value_mse,
                }
            )

            # Store checkpoints
            save_checkpoint(iteration, rng_key, model, opt_state, frames, hours)

        print(log)
        wandb.log(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log = {"iteration": iteration}
        st = time.time()

        # Selfplay
        num_simulations = config.num_simulations_at(iteration)
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        selfplay_state, data = selfplay(model, selfplay_state, keys, num_simulations)
        # Fraction of game slots that reached a terminal state within
        # max_num_steps (i.e. finished at least one game this iteration).
        terminate_rate = data.terminated.any(axis=1).mean().item()

        # Compute value targets and fill the replay buffer on the host, so only
        # one minibatch at a time occupies device memory during training.
        # (#devices, max_num_steps, batch, ...) -> (max_num_steps, #devices * batch, ...)
        data = jax.tree_util.tree_map(
            lambda x: np.moveaxis(x, 0, 1).reshape((x.shape[1], -1, *x.shape[3:])),
            jax.device_get(data),
        )
        frames += data.terminated.size
        samples: Sample = trajectories.process(data, carry=config.continue_games)
        del data
        new_samples = samples.mask.size
        value_target_fraction = samples.mask.mean().item() if new_samples else 0.0
        if new_samples > 0:
            replay_buffer.add(samples)
        del samples
        if not replay_buffer_announced and replay_buffer.nbytes > 0:
            replay_buffer_announced = True
            print(f"Replay buffer: {replay_buffer.nbytes / 2**30:.2f} GiB allocated for "
                  f"{config.replay_buffer_iters} iterations")

        # Never make more passes than the buffer holds while it is filling up
        # (e.g. early on with continue_games, when few games have finished).
        num_updates = min(max_num_updates, replay_buffer.num_samples // config.training_batch_size)
        rng_key, subkey = jax.random.split(rng_key)
        if num_updates > 0:
            minibatches = replay_buffer.sample(subkey, num_updates * config.training_batch_size)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((num_updates, num_devices, -1) + x.shape[1:]), minibatches
            )

        # Training
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
        if num_updates > 0:
            log["train/policy_loss"] = sum(policy_losses) / num_updates
            log["train/value_loss"] = sum(value_losses) / num_updates

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/terminate_rate": terminate_rate,
                "train/replay_buffer_samples": replay_buffer.num_samples,
                "train/num_updates": num_updates,
                "train/sample_reuse": num_updates * config.training_batch_size / samples_per_iter,
                "train/new_samples": new_samples,
                "train/value_target_fraction": value_target_fraction,
                "train/pending_steps": trajectories.num_pending,
                "selfplay/num_simulations": num_simulations,
                "hours": hours,
                "frames": frames,
            }
        )

        if _stop_requested:
            print(log)
            wandb.log(log)
            print(f"Saving checkpoint at iteration {iteration} before exiting...")
            save_checkpoint(iteration, rng_key, model, opt_state, frames, hours)
            break
