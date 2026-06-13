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

import os
import pickle
import random
import re
import time
from functools import partial
from typing import NamedTuple
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import mctx
import pgx
import pgx.gess
import pgx.heckmeck as Heckmeck
from omegaconf import OmegaConf
from pgx.g_hex import black
from pgx.g_hex2 import black2
from pydantic import BaseModel
from config import Config
from network import AZNet
from abc import ABC, abstractmethod

# A Haiku model is a (params, state) pair, as returned by forward.init.
Model = tuple[hk.Params, hk.State]

# ANSI colors for terminal output.
_GREEN = "\033[32m"
_ORANGE = "\033[38;5;208m"  # 256-color orange for major game-event lines
_GREY = "\033[38;5;245m"   # dim grey for verbose debug info (action weights, value)
_BLUE_BRIGHT = "\033[38;5;39m"  # most likely move
_BLUE_DIM = "\033[38;5;67m"     # next few likely moves
_RESET = "\033[0m"


class TourneyConfig(BaseModel):
    env_id: pgx.EnvId = "g_hex"
    seed: int = 49064405
    games: int = 256
    # Comma-separated list of player types, one per seat. Valid types:
    #   random  -> RandomAgent
    #   me      -> KeyboardAgent (human at the keyboard)
    #   model   -> a trained ModelAgent; each successive 'model' gets the next
    #              index passed to load_model_based_agent (0, 1, 2, ...).
    # e.g. players="model,model,me,model" -> load_model_based_agent(0), KeyboardAgent,
    #      load_model_based_agent(1), load_model_based_agent(2).
    players: str = "random,model"
    # Comma-separated, fully-qualified checkpoint paths for the 'model' seats.
    # They are applied round-robin: the n-th model seat uses paths[n % len(paths)],
    # so one path is shared by all models, two paths alternate, etc.
    models: str = ""


class MctsConfig(NamedTuple):
    num_simulations: int = 6144
    max_num_considered_actions: int = 16


class Cli(ABC):
    @abstractmethod
    def get_action_id(self) -> int | None:
        pass

    @abstractmethod
    def display(self, state: pgx.State) -> None:
        pass

    @abstractmethod
    def describe_action(self, action: jnp.ndarray) -> str:
        pass

    def display_action_weights(self, action_weights: jnp.ndarray) -> None:
        """Print a game-specific view of the MCTS policy output.

        Called by ModelAgent after each move. The default is a no-op; games
        with a useful board-shaped debug view override it.
        """
        pass


_GESS_SIZE = 20   # total grid side
_GESS_COLS = 'abcdefghijklmnopqrst'


def _gess_idx_to_label(idx: int) -> str:
    """Flat board index (row_idx*20+col) → label like 'p3'.

    Uses the standard Gess notation: label = 20 − row_index, so row_index 0
    (the top border) = label 20, row_index 19 (bottom border) = label 1.
    Column a–t corresponds to column index 0–19.
    """
    col     = idx % _GESS_SIZE
    row_idx = idx // _GESS_SIZE
    return f"{_GESS_COLS[col]}{_GESS_SIZE - row_idx}"


def _gess_label_to_idx(s: str) -> int | None:
    """Parse a label like 'p3' → flat index, or None on failure.

    Standard Gess notation: row_index = 20 − label_number.
    """
    m = re.fullmatch(r'([a-t])(20|1[0-9]|[1-9])', s.strip().lower())
    if not m:
        return None
    col     = ord(m.group(1)) - ord('a')
    row_idx = _GESS_SIZE - int(m.group(2))
    return row_idx * _GESS_SIZE + col


class GessCli(Cli):
    """CLI for Gess.

    Each full move is two actions: pick source centre, then destination.
    The user may enter either a single square label per prompt, or the
    complete move 'a2-e6' at the source prompt (the destination is stashed
    and used automatically at the next stage).
    """

    def __init__(self) -> None:
        self._stage = 0          # updated by display()
        self._src_label: str | None = None
        self._stashed_dst: int | None = None

    def display(self, state: pgx.State) -> None:
        board = np.array(state._x.board[0]).reshape(_GESS_SIZE, _GESS_SIZE)
        stage  = int(state._x.stage[0])
        source = int(state._x.source[0])
        self._stage = stage
        if stage == 0:
            self._src_label   = None
            self._stashed_dst = None
        else:
            # The chosen source square is recorded in the state, so derive its
            # label from there. This keeps describe_action's "moved <src> -> ..."
            # correct for model players, which never go through get_action_id.
            self._src_label = _gess_idx_to_label(source)

        src_r, src_c = source // _GESS_SIZE, source % _GESS_SIZE

        # The outer ring (row/col index 0 and 19) is an unplayable border that can
        # never be occupied, so we omit it and render only the inner 18x18 area.
        inner = range(1, _GESS_SIZE - 1)
        header = "     " + " ".join(_GESS_COLS[1:_GESS_SIZE - 1])
        black_stones = int((board == pgx.gess.BLACK).sum())
        white_stones = int((board == pgx.gess.WHITE).sum())
        print(f"{header}   Stones[Black={black_stones} White={white_stones}]")
        # Row index 1 = label 19 (top); row index 18 = label 2 (bottom).
        for r in inner:
            row_label = str(_GESS_SIZE - r).rjust(3)
            cells = []
            for c in inner:
                val = int(board[r, c])
                in_src_fp = (stage == 1 and
                             abs(r - src_r) <= 1 and abs(c - src_c) <= 1)
                if val == 1:
                    ch = 'X' if in_src_fp else 'x'   # black
                elif val == 2:
                    ch = 'O' if in_src_fp else 'o'   # white
                elif in_src_fp:
                    ch = '□'                          # empty source-fp cell
                else:
                    ch = '·'                          # empty playing area
                cells.append(ch)
            print(f"{row_label} |{' '.join(cells)}|")
        print(header)

        if stage == 1:
            print(f"  Source selected: {self._src_label}  — enter destination.")
        else:
            player = "Black (x)" if int(state.current_player[0]) == 0 else "White (o)"
            print(f"  {player} to move.")
        print()

    def get_action_id(self) -> int | None:
        if self._stage == 0:
            raw = input("Move (e.g. c7-e9) or source (e.g. c7): ").strip().lower()
            if '-' in raw:
                parts = raw.split('-', 1)
                src_idx = _gess_label_to_idx(parts[0])
                dst_idx = _gess_label_to_idx(parts[1])
                if src_idx is not None and dst_idx is not None:
                    self._src_label   = parts[0].strip()
                    self._stashed_dst = dst_idx
                    return src_idx
                return None
            idx = _gess_label_to_idx(raw)
            if idx is not None:
                self._src_label = raw
            return idx
        else:
            # Use stashed destination if the user entered a full move in stage 0.
            if self._stashed_dst is not None:
                dst = self._stashed_dst
                self._stashed_dst = None
                return dst
            raw = input(f"Destination (source={self._src_label}, e.g. e9): ").strip().lower()
            return _gess_label_to_idx(raw)

    def describe_action(self, action: jnp.ndarray) -> str:
        label = _gess_idx_to_label(int(action[0]))
        if self._stage == 0:
            return f"chose source {label}"
        else:
            return f"moved {self._src_label} -> {label}"

    def display_action_weights(self, action_weights: jnp.ndarray) -> None:
        # Actions are flat indices into the full 20x20 board (row_idx*20 + col),
        # which is the whole action space, so render the entire grid with file/row
        # labels matching the board so moves are easy to eyeball. Invalid actions
        # are masked to exactly-zero weight, so show those cells blank rather than
        # as 0.00%. The most likely move is highlighted in bright blue and the
        # next five in a dimmer blue.
        grid = np.asarray(action_weights).reshape(_GESS_SIZE, _GESS_SIZE)
        # Rank legal (nonzero) actions high->low; top one is bright, next five dim.
        flat = grid.ravel()
        ranked = [i for i in np.argsort(flat)[::-1] if flat[i] != 0]
        highlight = {i: _BLUE_DIM for i in ranked[1:6]}
        if ranked:
            highlight[ranked[0]] = _BLUE_BRIGHT
        cell_w = 9
        header = " " * 4 + "".join(f"{_GESS_COLS[c]:^{cell_w}}" for c in range(_GESS_SIZE))

        def fmt(flat_idx: int, w: float) -> str:
            if w == 0:
                return " " * cell_w
            text = f"{100*w:6.2f}%  "
            color = highlight.get(flat_idx)
            # Restore grey (the surrounding debug colour) after a highlighted cell.
            return f"{color}{text}{_GREY}" if color else text

        print(header)
        for r in range(_GESS_SIZE):
            cells = "".join(fmt(r * _GESS_SIZE + c, w) for c, w in enumerate(grid[r]))
            print(f"{_GESS_SIZE - r:>3} {cells}")
        print(header)
        print("")


class DomineeringCli(Cli):
    def get_action_id(self) -> int | None:
        square_code = input("Move: ")
        col = ord(square_code[0]) - ord('a')
        row = int(square_code[1]) - 1
        if col < 0 or col >= 8 or row < 0 or row >= 8:
            return None
        return row * 7 + col

    def display(self, state: pgx.State) -> None:
        print("   abcdefgh")
        print("\n".join(
              f"{idx+1} |" + "".join("·" if cell else "■" for cell in row) + "|"
              for idx, row in enumerate(readable_domineering_board(state))
        ))
        print("")


    def describe_action(self, action: jnp.ndarray) -> str:
        return f"played {'abcdefgh'[action[0] % 7]}{1 + (action[0] // 7)}"

    def display_action_weights(self, action_weights: jnp.ndarray) -> None:
        action_weights = jnp.hstack([
            action_weights.reshape(8, 7),
            jnp.zeros((8, 1), dtype=jnp.float32),
        ])
        print("\n".join(
            "".join(f"{100*w:6.2f}%  " if w != 0 else " " * 9 for w in w_row)
            for w_row in action_weights
        ))
        print("")


class GHexCli(Cli):
    def get_action_id(self) -> int | None:
        name = input("Move: ")
        m = re.fullmatch("([0-9]+) on ([0-9]+)", name)
        if m is None:
            return None
        tile = int(m.group(1))
        triangle = int(m.group(2))
        return black(tile, triangle)

    def display(self, state: pgx.State) -> None:
        print(env.pretty_game(state))
        print("")
        print(f"Black tiles remaining: {self._pretty_tiles(state._x.tiles[0][0])}")
        print(f"White tiles remaining: {self._pretty_tiles(state._x.tiles[0][1])}")
        print("")

    def _pretty_tiles(self, tiles: jnp.ndarray) -> str:
      return "  ".join([(f"{val:2}" if tiles[i] else "  ")
                       for i, val in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])])

    def describe_action(self, action: jnp.ndarray) -> str:
        return f"played the {1 + (action[0] % 10)} on triangle {action[0] // 10}"

    def display_action_weights(self, action_weights: jnp.ndarray) -> None:
        action_weights = action_weights.reshape(21, 10)
        print("\n".join([f"Tri {tri_i:2}: " + "  ".join([f"{100*w:6.2f}%" if w != 0 else " " * 7 for w in tri_row])
                         for tri_i, tri_row in enumerate(action_weights)]))
        print("")


class GHex2Cli(Cli):
    def get_action_id(self) -> int | None:
        name = input("Move: ")
        m = re.fullmatch("([0-9]+) on ([0-9]+)", name)  # TODO: allow * as tile
        if m is None:
            return None
        tile = int(m.group(1))
        triangle = int(m.group(2))
        return black2(tile, triangle)

    def display(self, state: pgx.State) -> None:
        print(env.pretty_game(state))
        print("")
        print(f"Black tiles remaining: {self._pretty_tiles2(state._x.tiles[0][0])}")
        print(f"White tiles remaining: {self._pretty_tiles2(state._x.tiles[0][1])}")
        print("")

    def _pretty_tiles2(self, tiles: jnp.ndarray) -> str:
      return "  ".join([(f"{val:2}" if tiles[i] else "  ")
                       for i, val in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])])

    def describe_action(self, action: jnp.ndarray) -> str:
        return f"played the {1 + (action[0] % 11)} on triangle {action[0] // 11}"  # TODO: *

    def display_action_weights(self, action_weights: jnp.ndarray) -> None:
        action_weights = action_weights.reshape(23, 11)
        print("\n".join([f"Tri {tri_i:2}: " + "  ".join([f"{100*w:6.2f}%" if w != 0 else " " * 7 for w in tri_row])
                         for tri_i, tri_row in enumerate(action_weights)]))
        print("")


class PigCli(Cli):
    def get_action_id(self) -> int | None:
        choice = input("Roll again or stop? [r/s]: ").strip().lower()
        if choice in ("r", "roll", "continue"):
            return 1
        if choice in ("s", "stop"):
            return 0
        return None

    def display(self, state: pgx.State) -> None:
        totals = state._x.totals[0]
        last_roll = int(state._x.last_roll[0])
        turn_total = int(state._x.turn_total[0])
        current = int(state.current_player[0])
        pig_out = last_roll == 1
        print(f"Scores: P0={int(totals[0])}  P1={int(totals[1])}")
        if pig_out:
            print(f"Player {current}'s turn: rolled a 1 — pig out! (must stop, turn total {turn_total} is lost)")
        else:
            print(f"Player {current}'s turn: last rolled {last_roll}, turn total = {turn_total}")
        print("")

    def describe_action(self, action: jnp.ndarray) -> str:
        return "stopped" if action[0] == 0 else "rolled again"


#class HeckmeckCli(Cli):
#    def get_action_id(self) -> int | None:
#
#    def display(self, state: pgx.State) -> None:
#        print(f"To play: P{state.current_player[0]}")
#        print(f"Grill: {state._x.grill[0] * HECKMECK_TILE_VALS}")
#        print(f"Player stacks: P0={state._x.stacks[0][0]}")
#        print(f"               P1={state._x.stacks[0][1]}")
#        print(f"               P2={state._x.stacks[0][2]}")
#        print(f"Dice taken:  {state._x.dice_taken[0]}")
#        print(f"Dice rolled: {state._x.dice_rolled[0]}")
#
#    def describe_action(self, action: jnp.ndarray) -> str:
#        return f"took action {action[0]}: {_HECKMECK_ACTION_NAMES[action[0]]}"


def get_cli(env_id: pgx.EnvId) -> Cli:
    match env_id:
        case "gess":
            return GessCli()
        case "domineering":
            return DomineeringCli()
        case "g_hex":
            return GHexCli()
        case "g_hex2":
            return GHex2Cli()
        case "pig":
            return PigCli()
        #case "heckmeck":
        #    return HeckmeckCli()
        case _:
            raise ValueError(f"No CLI support for {env_id}")


class Agent(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_action(self, key: jnp.ndarray, state: pgx.State) -> jnp.ndarray:
        pass


class KeyboardAgent(Agent):
    def __init__(self, cli: Cli) -> None:
        self.cli = cli

    def get_name(self) -> str:
        return "Human"

    def get_action(self, key: jnp.ndarray, state: pgx.State) -> jnp.ndarray:
        legal = state.legal_action_mask[0]
        forced = jnp.where(legal)[0]
        if len(forced) == 1:
            return jnp.int32([int(forced[0])])
        action_i = None
        while action_i is None or not legal[action_i]:
            try:
                action_i = self.cli.get_action_id()
            except Exception as e:
                print(f"fail: {e}")
                action_i = None
        return jnp.int32([action_i])


class RandomAgent(Agent):
    def get_name(self) -> str:
        return "Rando"

    def get_action(self, key: jnp.ndarray, state: pgx.State) -> jnp.ndarray:
        action_i = None
        while action_i is None or not state.legal_action_mask[0][action_i]:
            action_i = random.randint(0, len(state.legal_action_mask[0]) - 1)
        print(f"  Picking randomly! Selected action index {action_i} for no particular reason.")
        return jnp.int32([action_i])



class ModelAgent(Agent):
    def __init__(
        self,
        name_prefix: str,
        mcts_config: MctsConfig,
        config: Config,
        model: Model,
        cli: Cli,
    ) -> None:
        def forward_fn(x: jnp.ndarray, is_eval: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
            net = AZNet(
                num_actions=env.num_actions,
                num_channels=config.num_channels,
                num_blocks=config.num_layers,
                resnet_v2=config.resnet_v2,
                num_heads=config.num_heads,
                num_attention_layers=config.num_attention_layers,
            )
            policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
            return policy_out, value_out

        forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

        def recurrent_fn(
            model: Model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State
        ) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
            del rng_key
            model_params, model_state = model

            current_player = state.current_player
            state = jax.vmap(env.step)(state, action)

            (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
            # Mask first, then scale for numerical stability.
            logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
            #logits = logits - jnp.max(logits, axis=-1, keepdims=True)

            reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
            value = jnp.where(state.terminated, 0.0, value)
            # +1 when the same player is still to move (a Gess turn is two actions
            # -- pick source, then destination -- by one player), -1 when the
            # opponent is now to move, 0 at terminal. Hardcoding -1 assumed strict
            # alternation, which inverts the value across the intra-turn stage
            # boundary and makes MCTS prefer sources from which every move loses.
            discount = jnp.where(state.current_player == current_player, 1.0, -1.0)
            discount = jnp.where(state.terminated, 0.0, discount)

            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            )
            return recurrent_fn_output, state

        self.name_prefix = name_prefix
        self.cli = cli
        self.mcts_config = mcts_config
        self.mcts = partial(ModelAgent._run_mcts, forward, recurrent_fn, mcts_config, model, self.cli) # Unjitted, for debugging.
        self.mcts_jit = jax.jit(self.mcts)

    def get_name(self) -> str:
        return f"Model[{self.name_prefix}:sims={self.mcts_config.num_simulations}]"

    def get_action(self, key: jnp.ndarray, state: pgx.State) -> jnp.ndarray:
        if int(state.legal_action_mask.sum()) == 1:
            print("[short-circuit] Only one legal move; skipping search")
            return jnp.argmax(state.legal_action_mask)   # the sole legal action

        # Debug view into the policy evaluation: (slow)
        self.mcts(key, state, print_debug_info=True)

        start_time = time.perf_counter()
        policy_output, value = self.mcts_jit(key, state)
        print(f"Thought for {time.perf_counter() - start_time:.1f} seconds.")

        print(_GREY, end="")
        self.cli.display_action_weights(policy_output.action_weights)
        print(f"value={value}{_RESET}")

        return policy_output.action

    @staticmethod  # Static for JITting.
    def _run_mcts(
        forward: hk.TransformedWithState,
        recurrent_fn: mctx.RecurrentFn,
        mcts_config: MctsConfig,
        model: Model,
        cli: Cli,
        key: jnp.ndarray,
        state: pgx.State,
        print_debug_info: bool = False,
    ) -> tuple[mctx.PolicyOutput, jnp.ndarray] | None:
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, 1)  # Batch size must be 1
        key, subkey = jax.random.split(key)

        model_params, model_state = model
        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )

        if print_debug_info:
            # Inspect the raw network prior (before MCTS): render the masked,
            # softmaxed policy with the game-specific board view, plus the value.
            logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
            logits = logits - jnp.max(logits, axis=-1, keepdims=True)
            action_weights = jax.scipy.special.softmax(logits, axis=-1)
            print(f"{_GREY}First look:\n", end="")
            cli.display_action_weights(action_weights)
            print(f"value={value}{_RESET}")
            return None

        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=subkey,
            root=root,
            invalid_actions=~state.legal_action_mask,
            recurrent_fn=recurrent_fn,
            num_simulations=mcts_config.num_simulations,
            max_num_considered_actions=mcts_config.max_num_considered_actions,
            #qtransform=mctx.qtransform_completed_by_mix_value, # TODO: optimize? https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py
            gumbel_scale=0.0,
        )
        return policy_output, value


def load_from_checkpoint(path: str) -> tuple[Config, Model]:
  with open(path, "rb") as f:
      ckpt = pickle.load(f)
  # Rehydrate through the current Config so checkpoints predating newer fields
  # (e.g. num_heads) come back with those fields populated to their defaults.
  config = Config(**ckpt["config"].__dict__)
  return config, ckpt["model"]

def readable_domineering_board(state: pgx.State) -> jnp.ndarray:
  board = state._x.board[0]
  return jax.lax.select(state.current_player[0] == 0, board, board.transpose())


_HECKMECK_ACTION_NAMES = [
    "Took worms and kept rolling",
    "Took 1s and kept rolling",
    "Took 2s and kept rolling",
    "Took 3s and kept rolling",
    "Took 4s and kept rolling",
    "Took 5s and kept rolling",
    "Took worms and ended turn",
    "Took 1s and ended turn",
    "Took 2s and ended turn",
    "Took 3s and ended turn",
    "Took 4s and ended turn",
    "Took 5s and ended turn",
    "Busted",
]


def _load_model_based_agent(model_index: int, path: str, cli: Cli) -> ModelAgent:
    config, model = load_from_checkpoint(path)
    return ModelAgent(f"m{model_index}:ckpt={path}", MctsConfig(), config, model, cli)


def build_agents(players: str, model_paths: list[str], cli: Cli) -> list[Agent]:
    """Parse a players spec like "random,me,model,model" into a list of agents.

    Each successive 'model' token is given the next integer index and a
    checkpoint path chosen round-robin from model_paths, so one path is shared
    by every model seat, two paths alternate between seats, and so on.
    """
    agents: list[Agent] = []
    model_index = 0
    for token in players.split(","):
        kind = token.strip().lower()
        if kind in ("random", "rando", "rand"):
            agents.append(RandomAgent())
        elif kind in ("me", "human", "keyboard", "kb"):
            agents.append(KeyboardAgent(cli))
        elif kind in ("model", "ai", "nn"):
            if not model_paths:
                raise ValueError(
                    "players includes a 'model' seat but no models paths were given. "
                    "Pass models=<path1>[,<path2>,...]."
                )
            path = model_paths[model_index % len(model_paths)]
            agents.append(_load_model_based_agent(model_index, path, cli))
            model_index += 1
        else:
            raise ValueError(
                f"Unknown player type {token!r} in players={players!r}. "
                "Valid types are: random, me, model."
            )
    if len(agents) < 2:
        raise ValueError(
            f"Need at least 2 players, got {len(agents)} from players={players!r}."
        )
    return agents


if __name__ == "__main__":
    tourney_conf_dict = OmegaConf.from_cli()
    tourney_config: TourneyConfig = TourneyConfig(**tourney_conf_dict)
    print(tourney_config)

    cli = get_cli(tourney_config.env_id)

    devices = jax.local_devices()

    env = pgx.make(tourney_config.env_id)
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    print("\n\nLet's play!\n\n\n")

    def run_game(game_num: int, agents: list[Agent], start_depth: int) -> jnp.ndarray:
        key = jax.random.PRNGKey(tourney_config.seed ^ game_num)
        key, subkey = jax.random.split(key)
        state: pgx.State = init_fn(jax.random.split(subkey, 1))

        # Play random moves up to start_depth to determine the initial position.
        # These must be the same for each minimatch so we divide by the number of agents.
        random.seed(77659 ^ (game_num // len(agents)))
        for move_i in range(0, start_depth):
            action_i = None
            while action_i is None or not state.legal_action_mask[0][action_i]:
                action_i = random.randint(0, len(state.legal_action_mask[0]) - 1)
            state = step_fn(state, jnp.int32([action_i]), jax.random.split(subkey, 1))

        turn_num = 1 + start_depth
        while True:
            cli.display(state)

            if state.terminated.all():
                print(f"{_ORANGE}Game over! winner={state._x.winner} rewards={state.rewards}{_RESET}")
                return state._x.winner

            agent = agents[state.current_player[0]]
            print(f"{_ORANGE}Game {game_num}, turn {turn_num}, player={state.current_player[0]}: {agent.get_name()} to select action...{_RESET}", flush=True)
            action = agent.get_action(key, state)
            print(f"{agent.get_name()} {_GREEN}{cli.describe_action(action)}{_RESET}\n")

            key, subkey = jax.random.split(key)
            state = step_fn(state, action, jax.random.split(subkey, 1))
            turn_num += 1


    model_paths = [p.strip() for p in tourney_config.models.split(",") if p.strip()]
    agents = build_agents(tourney_config.players, model_paths, cli)
    wins = np.zeros_like(agents)
    for game_num in range(0, tourney_config.games):
        rotation_pos = game_num % len(agents)
        game_agents = agents[rotation_pos:] + agents[:rotation_pos]
        print(f"{_ORANGE}Game {game_num} of {tourney_config.games}: {" vs ".join([agent.get_name() for agent in game_agents])}{_RESET}")

        winner = run_game(game_num, game_agents, 0 if game_num < 2 else 1)  # TODO: more interesting start depths
        for w in range(0, len(agents)):
            if (winner + rotation_pos) % len(agents) == w:
                wins[w] += 1
        win_rates = 100 * wins / (1 + game_num)
        print(f"""#######################################################################
                  Win rates after {1 + game_num} {'game' if game_num == 0 else 'games'}:""")
        for w in range(0, len(agents)):
            print(f"       {agents[w].get_name():>20}: {win_rates[w]:6.2f}% ({wins[w]})", flush=True)
