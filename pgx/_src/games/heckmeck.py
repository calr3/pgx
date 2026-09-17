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

"""Heckmeck am Bratwurmeck (Pickomino), Reiner Knizia / Zoch 2005.

Deviations from the published rules, all deliberate:

- **A reduced game.** 4 dice and 8 tiles valued 11-18, with 1,1,2,2,3,3,4,4
  worms, against the published 8 dice and 16 tiles valued 21-36 with
  1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4 worms. The full-size constants are kept
  commented out below. The reduced worm ladder is steeper than a scaled-down
  version of the original.
- **Exactly 3 players**, where the published game takes 2 to 7.

The rest follows the rulebook, including the two cases it spells out that are
easy to get wrong: a player who busts with nothing to return turns no tile
over, and a returned tile that is the highest on the grill means no tile is
turned over either.

An action is one of 6 "set aside this face and roll on", 6 "set aside this face
and claim from the grill", a bust, and 6 "set aside this face and snatch an
opponent's topmost tile". Snatching needs an exact match, and tile values are
unique, so the face alone identifies which opponent is robbed.
"""

from typing import NamedTuple, Optional
import functools
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey


_PLAYER_COUNT = 3

#_DICE_COUNT = 8

#_TILE_VALS = jnp.int32([
#    0,
#    21, 22, 23, 24,
#    25, 26, 27, 28,
#    29, 30, 31, 32,
#    33, 34, 35, 36,
#])

#_WORM_VALS = jnp.int32([
#    0,
#    1, 1, 1, 1,
#    2, 2, 2, 2,
#    3, 3, 3, 3,
#    4, 4, 4, 4,
#])

_TILE_VALS = jnp.int32([
    0,
    11, 12, 13, 14,
    15, 16, 17, 18,
])

_WORM_VALS = jnp.int32([
    0,
    1, 1, 2, 2,
    3, 3, 4, 4,
])

_DICE_COUNT = 4
_TILE_COUNT = len(_TILE_VALS) - 1

# Value of each dice index, for calculation of turn total.
_DICE_VALS = jnp.int32([5, 1, 2, 3, 4, 5])

# Which dice a player gets to keep, based on how many they already rolled.
_DICE_MASK = jnp.array([jnp.concatenate([jnp.ones(_DICE_COUNT - die_count), jnp.zeros(die_count)])
                        for die_count in jnp.arange(0, _DICE_COUNT+1)],
                       dtype=jnp.int32)


class GameState(NamedTuple):
    """Internal state for the game Heckmeck."""

    color: Array = jnp.int32(0)
    # True (available), False (taken or turned over). Index 0 is not a tile - it
    # is the "no tile" slot that _TILE_VALS and _WORM_VALS pad with a zero - so
    # it is never available, or the game could not end while it stood open.
    grill: Array = jnp.ones_like(_TILE_VALS, jnp.bool_).at[0].set(False)
    stacks: Array = jnp.zeros((_PLAYER_COUNT, len(_TILE_VALS) - 1), jnp.int32)
    dice_rolled: Array = jnp.zeros(6, jnp.int32)
    dice_taken: Array = jnp.zeros(6, jnp.int32)
    winner: Array = jnp.int32(-1)


class Game:
    """The game representation of Heckmeck."""

    def init(self, key: PRNGKey) -> GameState:
        state = GameState()
        return state._replace(
            dice_rolled = _roll(state.dice_taken, key),
        )


    def step(self, state: GameState, action: Array, key: PRNGKey) -> GameState:
        return jax.lax.switch(action,
          [
             # Actions 0 - 5: Take and roll
             functools.partial(_step_take_and_roll, 0),
             functools.partial(_step_take_and_roll, 1),
             functools.partial(_step_take_and_roll, 2),
             functools.partial(_step_take_and_roll, 3),
             functools.partial(_step_take_and_roll, 4),
             functools.partial(_step_take_and_roll, 5),
             # Actions 6 - 11: Take and stop
             functools.partial(_step_take_and_stop, 0),
             functools.partial(_step_take_and_stop, 1),
             functools.partial(_step_take_and_stop, 2),
             functools.partial(_step_take_and_stop, 3),
             functools.partial(_step_take_and_stop, 4),
             functools.partial(_step_take_and_stop, 5),
             # Action 12: Bust
             _step_bust,
             # Actions 13 - 18: Take and snatch an opponent's topmost tile
             functools.partial(_step_take_and_steal, 0),
             functools.partial(_step_take_and_steal, 1),
             functools.partial(_step_take_and_steal, 2),
             functools.partial(_step_take_and_steal, 3),
             functools.partial(_step_take_and_steal, 4),
             functools.partial(_step_take_and_steal, 5),
          ],
          state, action, key,
        )


    def observe(self, state: GameState) -> Array:
        stack_state = jnp.array([
            state.stacks[state.color],
            state.stacks[(state.color + 1) % _PLAYER_COUNT],
            state.stacks[(state.color + 2) % _PLAYER_COUNT],
        ])

        global_state = jnp.hstack([
            jnp.ones(len(_TILE_VALS) - 1, dtype=jnp.int32) * state.grill[1:],
            state.dice_rolled,
            state.dice_taken,
        ])

        return jnp.dstack([
            stack_state,
            jnp.tile(global_state, _PLAYER_COUNT * (len(_TILE_VALS) - 1)).reshape(_PLAYER_COUNT, len(_TILE_VALS) - 1, -1),
        ])


    def legal_action_mask(self, state: GameState) -> Array:
        has_worm_already = state.dice_taken[0] > 0
        dice_taken_already = state.dice_taken.sum()
        total_already = (_DICE_VALS * state.dice_taken).sum()

        # A face may only be set aside if it was just rolled and has not been
        # set aside already; taking it takes every die showing it.
        can_take = (state.dice_taken == 0) & (state.dice_rolled > 0)
        total_after = total_already + _DICE_VALS * state.dice_rolled
        # Any claim needs a worm among the dice set aside - taking the worms
        # themselves supplies one.
        would_have_worm = has_worm_already | (jnp.arange(6) == 0)

        grill_values = state.grill * _TILE_VALS
        smallest_grill_value = jnp.min(jnp.where(grill_values == 0, 999, grill_values))

        # A tile can also be snatched off the top of an opponent's stack, but
        # only on an exact match. Tile values are unique, so a total matches at
        # most one opponent, and only their topmost tile is exposed.
        tops = state.stacks[:, 0]
        exposed = (tops > 0) & (jnp.arange(_PLAYER_COUNT) != state.color)
        can_steal = (exposed[:, None] & (_TILE_VALS[tops][:, None] == total_after[None, :])).any(axis=0)

        # Rolling on needs a die left to roll.
        take_and_roll = can_take & (dice_taken_already + state.dice_rolled < _DICE_COUNT)
        take_and_stop = can_take & would_have_worm & (total_after >= smallest_grill_value)
        take_and_steal = can_take & would_have_worm & can_steal

        # Busting is legal exactly when all other moves are illegal.
        any_legal = jnp.concatenate([take_and_roll, take_and_stop, take_and_steal]).any()
        return jnp.hstack([take_and_roll, take_and_stop, ~any_legal, take_and_steal])


    def is_terminal(self, state: GameState) -> Array:
        return state.grill.sum() == 0


    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1, -1]).at[state.winner].set(1.0),
            jnp.zeros(_PLAYER_COUNT, jnp.float32),
        )


def _winner(grill: Array, stacks: Array) -> Array:
    worm_totals = _WORM_VALS[stacks].sum(axis=1)
    best_tiles = stacks.max(axis=1)

    return jax.lax.select(grill.sum() == 0,
                          jnp.argmax(worm_totals + best_tiles / 64),
                          jnp.int32(-1))


def _roll(dice_taken: Array, key: PRNGKey) -> Array:
    roll = jax.random.randint(key, shape=(_DICE_COUNT), minval=1, maxval=7, dtype=jnp.int32)
    roll_kept = roll * _DICE_MASK[dice_taken.sum()]
    return jnp.int32([
        (roll_kept == 6).sum(),
        (roll_kept == 1).sum(),
        (roll_kept == 2).sum(),
        (roll_kept == 3).sum(),
        (roll_kept == 4).sum(),
        (roll_kept == 5).sum(),
    ])


def _step_take_and_roll(die, state: GameState, action: Array, key: PRNGKey) -> GameState:
    dice_taken = state.dice_taken.at[action].set(state.dice_rolled[action])
    return state._replace(
        dice_rolled = _roll(dice_taken, key),
        dice_taken = dice_taken,
    )


def _step_take_and_stop(die, state: GameState, action: Array, key: PRNGKey) -> GameState:
    dice_taken = state.dice_taken.at[action - 6].set(state.dice_rolled[action - 6])
    total_made = (_DICE_VALS * dice_taken).sum()
    tile_index_taken = jnp.argmax(state.grill * _TILE_VALS * (_TILE_VALS <= total_made))

    new_grill = state.grill.at[tile_index_taken].set(False)
    new_stacks = state.stacks.at[state.color].set(
        jnp.roll(state.stacks[state.color], 1).at[0].set(tile_index_taken))
    fresh_dice_taken = jnp.zeros(6, dtype=jnp.int32)

    return state._replace(
        color = (state.color + 1) % _PLAYER_COUNT,
        grill = new_grill,
        stacks = new_stacks,
        dice_rolled = _roll(fresh_dice_taken, key),
        dice_taken = fresh_dice_taken,
        winner = _winner(new_grill, new_stacks),
    )


def _step_take_and_steal(die, state: GameState, action: Array, key: PRNGKey) -> GameState:
    dice_taken = state.dice_taken.at[die].set(state.dice_rolled[die])
    total_made = (_DICE_VALS * dice_taken).sum()

    # Legality has already established that exactly one opponent's topmost tile
    # matches the total, so the match identifies the victim.
    tops = state.stacks[:, 0]
    victim = jnp.argmax(
        (tops > 0) & (jnp.arange(_PLAYER_COUNT) != state.color) & (_TILE_VALS[tops] == total_made)
    )
    tile_index_taken = state.stacks[victim, 0]

    new_stacks = state.stacks.at[victim].set(jnp.pad(state.stacks[victim][1:], (0, 1)))
    new_stacks = new_stacks.at[state.color].set(
        jnp.roll(new_stacks[state.color], 1).at[0].set(tile_index_taken))
    fresh_dice_taken = jnp.zeros(6, dtype=jnp.int32)

    # The grill is untouched: the tile moves from one stack to another.
    return state._replace(
        color = (state.color + 1) % _PLAYER_COUNT,
        stacks = new_stacks,
        dice_rolled = _roll(fresh_dice_taken, key),
        dice_taken = fresh_dice_taken,
        winner = _winner(state.grill, new_stacks),
    )


def _step_bust(state: GameState, action: Array, key: PRNGKey) -> GameState:
    assert isinstance(key, Array)

    returned_tile = state.stacks[state.color][0]

    # The tile goes back on the grill first, and only then is the highest tile
    # still on the grill turned over. So if the returned tile is itself the
    # highest, nothing is turned over - and a player with no tile to return
    # turns nothing over either.
    has_tile = returned_tile > 0
    grill_returned = jax.lax.select(has_tile, state.grill.at[returned_tile].set(True), state.grill)
    highest = len(_TILE_VALS) - 1 - jnp.argmax(grill_returned[::-1])
    new_grill = jax.lax.select(has_tile & (highest != returned_tile),
                               grill_returned.at[highest].set(False),
                               grill_returned)
    new_stacks = state.stacks.at[state.color].set(jnp.pad(state.stacks[state.color][1:], (0,1)))
    fresh_dice_taken = jnp.zeros(6, jnp.int32)

    return state._replace(
        color = (state.color + 1) % _PLAYER_COUNT,
        grill = new_grill,
        stacks = new_stacks,
        dice_rolled = _roll(fresh_dice_taken, key),
        dice_taken = fresh_dice_taken,
        winner = _winner(new_grill, new_stacks)
    )

