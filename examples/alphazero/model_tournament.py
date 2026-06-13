# Copyright 2026 The Pgx Authors. All Rights Reserved.
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

"""TPU-native model-vs-model tournament.

run_tournament.py steps one game at a time and round-trips to the host every
move (board rendering, keyboard input, per-move prints). This script instead
plays a whole batch of games entirely on device: the per-move Gumbel MuZero
search, the env stepping, and the result accounting all run inside a single
pmapped while_loop, and only the final per-game results are copied back to the
host once per round. Only model seats are supported — exactly two checkpoints,
which may have different architectures and different per-model search budgets
(or be the same path, as a ~50% sanity check).

Design notes:
- Per-game model selection: at each move, every game in the batch needs the
  search of whichever model owns the player to move, and games drift out of
  sync (terminations, multi-action turns). Rather than partitioning the
  batch dynamically, each step runs BOTH models' searches over the full batch
  and each game keeps the action from its owner. Per game-step that costs
  sims_a + sims_b network evaluations — the same as evaluating both networks
  at every node of one shared search when the budgets are equal — and it
  supports mismatched architectures and per-model num_simulations.
- Within one game's search, every node is evaluated by the searching player's
  network (it models the opponent as itself).
- Games run in parallel because the network is hopelessly underutilizing a TPU
  at batch 1; throughput scales until HBM runs out (each MCTS tree stores
  `num_simulations` embedded states per game — lower batch_size if you OOM).
- Each random opening is played twice with the seats swapped: game i and game
  i + batch/2 share an opening, so first-move advantage cancels within pairs.
  With gumbel_scale=0.0 the search is deterministic and the random openings
  are the only thing that makes games distinct. Opening actions that would end
  the game on the spot are excluded, so the opponent always gets to actually
  play.
- Already-terminated games keep being stepped until the whole batch finishes
  (pgx returns the state unchanged with zero rewards), so the search work for
  them is wasted; the tail of a round drains at full per-step cost. Stats are
  unaffected.
"""

import pickle
import time
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import pgx
from omegaconf import OmegaConf
from pydantic import BaseModel

from config import Config
from network import AZNet

# A Haiku model is a (params, state) pair, as returned by forward.init.
Model = tuple[hk.Params, hk.State]


class TourneyConfig(BaseModel):
    env_id: pgx.EnvId = "go_9x9"
    seed: int = 49064405
    # Comma-separated paths of exactly two checkpoints: "model A,model B".
    # The same path twice is allowed (expect ~50% after enough games, unless
    # the search budgets below differ).
    models: str = ""
    # Total games to play. Must be a multiple of batch_size.
    games: int = 256
    # Games played in parallel per round, across all devices. Must be divisible
    # by num_devices, with an even per-device share (for the seat-swapped
    # pairing). Bigger is faster until the MCTS tree no longer fits in HBM.
    batch_size: int = 256
    # MCTS budget per move. Either a single value shared by both models, or a
    # comma-separated pair "sims A,sims B" in the same order as `models` —
    # e.g. num_simulations=64,512 pits a shallow A against a deep B.
    num_simulations: int | str = 256
    max_num_considered_actions: int = 16
    # 0.0 = deterministic argmax play (perfect-information default); raise it
    # to add Gumbel exploration noise to move selection.
    gumbel_scale: float = 0.0
    # Uniformly-random legal actions played before the models take over, giving
    # each game pair a distinct starting position.
    random_opening_plies: int = 2
    # Hard cap on total plies per game (including the opening); games still
    # running at the cap are truncated and scored as draws.
    max_num_steps: int = 256

    class Config:
        extra = "forbid"

    def num_simulations_pair(self) -> tuple[int, int]:
        """Parse num_simulations into (sims for model A, sims for model B)."""
        parts = [p.strip() for p in str(self.num_simulations).split(",") if p.strip()]
        if len(parts) not in (1, 2):
            raise ValueError(
                f"num_simulations must be one value or an 'A,B' pair, got "
                f"{self.num_simulations!r}"
            )
        sims = [int(p) for p in parts]
        if any(s <= 0 for s in sims):
            raise ValueError(f"num_simulations must be positive, got {self.num_simulations!r}")
        return (sims[0], sims[-1])


class RoundResult(NamedTuple):
    """Per-game results of one round, shape (num_devices, batch_per_device)."""
    r: jnp.ndarray           # final reward from model A's perspective (+1/0/-1)
    terminated: jnp.ndarray  # bool; False = truncated at max_num_steps
    length: jnp.ndarray      # plies played (including the random opening)


def load_from_checkpoint(path: str) -> tuple[Config, Model]:
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    # Rehydrate through the current Config so checkpoints predating newer fields
    # come back with those fields populated to their defaults.
    config = Config(**ckpt["config"].__dict__)
    return config, ckpt["model"]


def make_forward(num_actions: int, config: Config) -> hk.TransformedWithState:
    def forward_fn(x: jnp.ndarray, is_eval: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        net = AZNet(
            num_actions=num_actions,
            num_channels=config.num_channels,
            num_blocks=config.num_layers,
            resnet_v2=config.resnet_v2,
            num_heads=config.num_heads,
            num_attention_layers=config.num_attention_layers,
        )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def make_recurrent_fn(env: pgx.Env, forward: hk.TransformedWithState) -> mctx.RecurrentFn:
    def recurrent_fn(
        model: Model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State
    ) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
        del rng_key
        model_params, model_state = model

        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)

        (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        # +1 when the same player is still to move, -1 when the opponent is now to move, 0 at terminal.
        discount = jnp.where(state.current_player == current_player, 1.0, -1.0)
        discount = jnp.where(state.terminated, 0.0, discount)

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        ), state

    return recurrent_fn


def build_round_runner(
    env: pgx.Env,
    forward_a: hk.TransformedWithState,
    forward_b: hk.TransformedWithState,
    tcfg: TourneyConfig,
    num_devices: int,
):
    """Build the pmapped function that plays one round of batched games.

    Returns run_round(model_a, model_b, rng_keys) -> RoundResult, where the
    models are device-replicated (leading axis num_devices) and rng_keys has
    one key per device.
    """
    bs = tcfg.batch_size // num_devices
    half = bs // 2
    sims_a, sims_b = tcfg.num_simulations_pair()
    # Model A's seat per game: player 0 in the first half of each device's
    # batch, player 1 in the second half (the seat-swapped mirror games).
    a_seat = jnp.where(jnp.arange(bs) < half, 0, 1).astype(jnp.int32)

    recurrent_a = make_recurrent_fn(env, forward_a)
    recurrent_b = make_recurrent_fn(env, forward_b)

    def search(
        model: Model,
        forward: hk.TransformedWithState,
        recurrent_fn: mctx.RecurrentFn,
        num_simulations: int,
        key: jnp.ndarray,
        state: pgx.State,
    ) -> jnp.ndarray:
        """One model's full-batch search; returns the chosen action per game."""
        model_params, model_state = model
        (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
            max_num_considered_actions=tcfg.max_num_considered_actions,
            gumbel_scale=tcfg.gumbel_scale,
        )
        return policy_output.action

    @jax.pmap
    def run_round(model_a: Model, model_b: Model, rng_key: jnp.ndarray) -> RoundResult:
        key_init, key_open, key_loop = jax.random.split(rng_key, 3)

        # Tile init keys across the two halves so each mirror pair starts from
        # the same position even in envs with stochastic initialization.
        init_keys = jnp.tile(jax.random.split(key_init, half), (2, 1))
        state = jax.vmap(env.init)(init_keys)
        R = jnp.zeros(bs)

        # ── Random opening ─────────────────────────────────────────────────
        # Sample uniformly-random legal actions for the first half and replay
        # them verbatim in the second half, so each pair shares its opening.
        # Actions that would terminate the game immediately can crush the
        # mover's own last ring) are excluded so that no opening decides a
        # game before the opponent ever chooses a move; this is found by brute
        # force, stepping every (game, action) pair. (For a stage-0 source
        # choice nothing terminates, so nothing is excluded there.)
        def opening_ply(carry, key):
            state, R = carry
            key_act, key_step = jax.random.split(key)
            half_state = jax.tree_util.tree_map(lambda x: x[:half], state)
            all_actions = jnp.arange(state.legal_action_mask.shape[-1])
            ends = jax.vmap(  # over games
                jax.vmap(lambda s, a: env.step(s, a).terminated, in_axes=(None, 0)),
                in_axes=(0, None),
            )(half_state, all_actions)  # (half, num_actions)
            legal = state.legal_action_mask[:half]
            safe = legal & ~ends
            # Fall back to plain legality in the (pathological) games where
            # every legal action ends the game.
            mask = jnp.where(safe.any(axis=-1, keepdims=True), safe, legal)
            logits_half = jnp.where(mask, 0.0, -jnp.inf)
            action_half = jax.random.categorical(key_act, logits_half, axis=-1)
            action = jnp.concatenate([action_half, action_half])
            step_keys = jnp.tile(jax.random.split(key_step, half), (2, 1))
            state = jax.vmap(env.step)(state, action, step_keys)
            R = R + state.rewards[jnp.arange(bs), a_seat]
            return (state, R), None

        if tcfg.random_opening_plies > 0:
            (state, R), _ = jax.lax.scan(
                opening_ply, (state, R),
                jax.random.split(key_open, tcfg.random_opening_plies),
            )

        # ── Main loop: full MCTS for every live game, until all finish ─────
        def cond_fn(carry) -> jnp.ndarray:
            _, state, _, _, ply = carry
            return ~state.terminated.all() & (ply < tcfg.max_num_steps)

        def body_fn(carry):
            key, state, R, length, ply = carry
            key, key_a, key_b, key_step = jax.random.split(key, 4)
            alive = ~state.terminated

            # Both models search every game; each game then keeps the action
            # from whichever model owns the player to move.
            action_a = search(model_a, forward_a, recurrent_a, sims_a, key_a, state)
            action_b = search(model_b, forward_b, recurrent_b, sims_b, key_b, state)
            use_a = state.current_player == a_seat
            action = jnp.where(use_a, action_a, action_b)

            # Terminated games are stepped too (with whatever action the search
            # picked); pgx leaves them unchanged with zero rewards.
            step_keys = jax.random.split(key_step, bs)
            state = jax.vmap(env.step)(state, action, step_keys)
            R = R + state.rewards[jnp.arange(bs), a_seat]
            length = length + alive.astype(jnp.int32)
            return key, state, R, length, ply + 1

        length = jnp.full(bs, tcfg.random_opening_plies, dtype=jnp.int32)
        _, state, R, length, _ = jax.lax.while_loop(
            cond_fn, body_fn,
            (key_loop, state, R, length, jnp.int32(tcfg.random_opening_plies)),
        )
        return RoundResult(r=R, terminated=state.terminated, length=length)

    return run_round


def _elo(score: float) -> float:
    """Logistic Elo difference implied by an average score in [0, 1]."""
    if score <= 0.0 or score >= 1.0:
        return float("inf") if score >= 1.0 else float("-inf")
    return 400.0 * np.log10(score / (1.0 - score))


def _print_summary(
    r: np.ndarray,
    terminated: np.ndarray,
    length: np.ndarray,
    a_seat: np.ndarray,
    sims: tuple[int, int],
) -> None:
    n = r.size
    a_wins, b_wins = int((r == 1).sum()), int((r == -1).sum())
    draws = n - a_wins - b_wins
    truncated = int((~terminated).sum())
    # Per-game score for A in {0, 0.5, 1}; mirror pairs are correlated, so the
    # standard error below (which assumes independent games) is indicative only.
    score = (r + 1.0) / 2.0
    se = score.std() / np.sqrt(n) if n > 1 else 0.0
    print(
        f"  A wins {a_wins} ({100 * a_wins / n:.1f}%) | "
        f"B wins {b_wins} ({100 * b_wins / n:.1f}%) | "
        f"draws {draws} (of which truncated: {truncated})"
    )
    elo_ab = _elo(float(score.mean()))
    print(f"  A score: {score.mean():.3f} ± {se:.3f}  (Elo diff A−B: {elo_ab:+.0f})")
    sims_a, sims_b = sims
    if sims_a != sims_b and np.isfinite(elo_ab):
        # Elo gained per doubling of search budget, from the deeper side's
        # perspective (positive = more search helps).
        doublings = abs(np.log2(sims_b / sims_a))
        gain = (elo_ab if sims_a > sims_b else -elo_ab) / doublings
        print(
            f"  Elo gain per search doubling: {gain:+.1f} "
            f"(sims {min(sims_a, sims_b)}→{max(sims_a, sims_b)}: {doublings:.1f} doublings)"
        )
    for seat in (0, 1):
        m = a_seat == seat
        if m.any():
            print(
                f"  A as P{seat}: score {score[m].mean():.3f} over {int(m.sum())} games"
            )
    print(f"  avg game length: {length.mean():.1f} plies")


if __name__ == "__main__":
    tcfg = TourneyConfig(**OmegaConf.from_cli())
    print(tcfg)

    model_paths = [p.strip() for p in tcfg.models.split(",") if p.strip()]
    if len(model_paths) != 2:
        raise ValueError(
            f"models must list exactly two checkpoint paths, got {len(model_paths)}: "
            f"models={tcfg.models!r}"
        )
    sims_a, sims_b = tcfg.num_simulations_pair()

    devices = jax.local_devices()
    num_devices = len(devices)
    if tcfg.batch_size % num_devices != 0 or (tcfg.batch_size // num_devices) % 2 != 0:
        raise ValueError(
            f"batch_size ({tcfg.batch_size}) must be divisible by num_devices "
            f"({num_devices}) with an even per-device share (for seat-swapped pairs)."
        )
    if tcfg.games % tcfg.batch_size != 0:
        raise ValueError(
            f"games ({tcfg.games}) must be a multiple of batch_size ({tcfg.batch_size})."
        )

    env = pgx.make(tcfg.env_id)
    config_a, model_a = load_from_checkpoint(model_paths[0])
    config_b, model_b = load_from_checkpoint(model_paths[1])
    print(f"Model A: {model_paths[0]} (sims={sims_a})")
    print(f"Model B: {model_paths[1]} (sims={sims_b})")

    forward_a = make_forward(env.num_actions, config_a)
    forward_b = make_forward(env.num_actions, config_b)

    # Replicate both models to all devices (leading axis mapped by pmap).
    model_a, model_b = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices, *x.shape)), (model_a, model_b)
    )

    run_round = build_round_runner(env, forward_a, forward_b, tcfg, num_devices)

    bs = tcfg.batch_size // num_devices
    half = bs // 2
    # Host-side mirror of the per-device seat assignment, flattened across devices.
    a_seat_round = np.tile(np.concatenate([np.zeros(half), np.ones(half)]), num_devices)

    num_rounds = tcfg.games // tcfg.batch_size
    all_r: list[np.ndarray] = []
    all_terminated: list[np.ndarray] = []
    all_length: list[np.ndarray] = []
    rng_key = jax.random.PRNGKey(tcfg.seed)
    for round_i in range(num_rounds):
        rng_key, subkey = jax.random.split(rng_key)
        st = time.perf_counter()
        result = jax.block_until_ready(
            run_round(model_a, model_b, jax.random.split(subkey, num_devices))
        )
        elapsed = time.perf_counter() - st
        all_r.append(np.asarray(result.r).ravel())
        all_terminated.append(np.asarray(result.terminated).ravel())
        all_length.append(np.asarray(result.length).ravel())

        note = " (includes XLA compile)" if round_i == 0 else ""
        print(
            f"Round {round_i + 1}/{num_rounds}: {tcfg.batch_size} games in "
            f"{elapsed:.1f}s ({tcfg.batch_size / elapsed:.2f} games/s){note}"
        )
        _print_summary(
            np.concatenate(all_r),
            np.concatenate(all_terminated),
            np.concatenate(all_length),
            np.tile(a_seat_round, round_i + 1),
            (sims_a, sims_b),
        )
        print("", flush=True)
