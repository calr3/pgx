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

"""A plain alpha-beta negamax engine for pgx environments.

This is a deliberately conventional opponent: no network, no rollouts, just a
hand-written evaluation function and a full-width search. It exists so model
strength can be read against something whose behaviour is understood, rather
than only against other checkpoints or against random play (which a trained
Epaminondas model already beats on raw policy argmax, with no search at all).

Two things about pgx make this less standard than it looks.

**Multi-stage turns.** One Epaminondas move is three pgx actions - lead, rear,
destination - and `current_player` only flips on the third. A negamax that
negated at every ply would invert the value across the intra-move boundary and
have the mover minimising against itself. So the sign flip here is driven by
whether `current_player` actually changed, not by depth parity, and search
depth is counted in *full moves*: a node is only a leaf when the move boundary
and the depth limit coincide. (The same bug, in the same shape, is commented on
in interactive_tournament's MCTS `recurrent_fn`.)

**Stepping is a JAX call.** Per-node dispatch would dominate the search, so
children are expanded in one batched `vmap(step)` per node and scored in one
batched evaluation. Alpha-beta is still sequential over the ordered children;
only the expansion is vectorised.

The Epaminondas evaluation follows King and Peterson, "Epaminondas: Exploring
Combat Tactics", ICGA Journal 37(3), which describes the six heuristics of their
LEONIDAS agent: mobility, material dominance, crossings, center of mass, home
row defense, and territory. See EpaminondasEvaluator for what the paper does and
does not pin down - notably it never publishes the weights that combine them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import jax
import jax.numpy as jnp
import numpy as np

import pgx
from pgx._src.games.epaminondas import (
    HEIGHT,
    MAX_STEP,
    N,
    NUM_DIRS,
    RAY_IDX,
    RAY_OK,
    WIDTH,
    _OPP_DIR,
)

# NumPy copies of the ray geometry. The evaluation runs on the host, on batches
# of children, so it wants NumPy rather than the jnp tables the rules use.
_RAY_IDX_NP = np.asarray(RAY_IDX)  # (NUM_DIRS, N, MAX_STEP + 1)
_RAY_OK_NP = np.asarray(RAY_OK)
_OPP_DIR_NP = np.asarray(_OPP_DIR)

# A win is worth more than any heuristic term can reach, with depth subtracted
# so a forced mate is preferred sooner rather than later.
WIN_SCORE = 1_000_000.0


class Evaluator(Protocol):
    """Scores a batch of states, each from its own side-to-move's perspective."""

    def __call__(self, states: pgx.State) -> np.ndarray:  # pragma: no cover
        ...


@dataclass
class EpaminondasWeights:
    """Linear weights over the six LEONIDAS heuristics.

    **These weights are not from the paper.** King and Peterson define all six
    terms but never publish the coefficients that combine them; they describe
    their own function as "unrefined" and list tuning it as future work. So
    these are chosen here, scaled so that no single term swamps the others given
    its natural range on a 14x12 board, and they are the obvious thing to tune
    if this engine is ever used as a serious benchmark rather than a sanity
    check. A head-to-head between two weight sets is the only way to rank them.
    """

    material: float = 30.0
    crossing: float = 250.0
    home_row_defense: float = 8.0
    territory: float = 0.5
    centrality: float = 4.0
    mobility: float = 2.0


class EpaminondasEvaluator:
    """The six LEONIDAS heuristics, vectorised over a batch of boards.

    Every term is computed for both sides and subtracted, so the result is
    already from the perspective of the side to move and needs no further sign
    handling. Where the paper leaves something ambiguous it is flagged in the
    method that implements it.
    """

    def __init__(self, weights: Optional[EpaminondasWeights] = None) -> None:
        self.weights = weights or EpaminondasWeights()

    def __call__(self, states: pgx.State) -> np.ndarray:
        boards = np.asarray(states._x.board)  # (B, N), 0 empty / 1 white / 2 black
        colors = np.asarray(states._x.color)  # (B,), 0 white / 1 black
        return self.score(boards, colors)

    def score(self, boards: np.ndarray, colors: np.ndarray) -> np.ndarray:
        boards = np.atleast_2d(boards)
        colors = np.atleast_1d(colors)
        own_stone = (colors + 1).astype(boards.dtype)[:, None]
        opp_stone = (2 - colors).astype(boards.dtype)[:, None]
        own = boards == own_stone
        opp = boards == opp_stone
        empty = boards == 0

        w = self.weights
        score = np.zeros(boards.shape[0], dtype=np.float64)
        score += w.material * self._material(own, opp)
        score += w.crossing * self._crossing(own, opp, colors)
        score += w.home_row_defense * self._home_row_defense(own, opp, colors)
        score += w.territory * self._territory(own, opp, empty)
        score += w.centrality * self._centrality(own, opp)
        if w.mobility:
            # The run tables are the expensive part of the evaluation and both
            # sides share the empty-square one, so build all three once here
            # rather than twice inside _mobility.
            score += w.mobility * self._mobility_from_runs(
                _runs_all_dirs(own), _runs_all_dirs(opp), _runs_all_dirs(empty)
            )
        return score

    # -- 4.2 Material dominance ------------------------------------------
    def _material(self, own: np.ndarray, opp: np.ndarray) -> np.ndarray:
        """"the difference of the sums of opposing pieces"."""
        return own.sum(1).astype(np.float64) - opp.sum(1)

    # -- 4.3 Crossing ----------------------------------------------------
    def _crossing(self, own: np.ndarray, opp: np.ndarray, colors: np.ndarray) -> np.ndarray:
        """"sums the number of pieces on the opponent's back row", one point each.

        White's home rank is row 0 and black's is row HEIGHT - 1, so the row a
        player is trying to reach is the opponent's home rank.
        """
        own_g = own.reshape(-1, HEIGHT, WIDTH)
        opp_g = opp.reshape(-1, HEIGHT, WIDTH)
        white = colors == 0
        own_target = np.where(white, HEIGHT - 1, 0)
        opp_target = np.where(white, 0, HEIGHT - 1)
        rows = np.arange(own_g.shape[0])
        return (
            own_g[rows, own_target].sum(1).astype(np.float64)
            - opp_g[rows, opp_target].sum(1)
        )

    # -- 4.5 Home row defense --------------------------------------------
    def _home_row_defense(
        self, own: np.ndarray, opp: np.ndarray, colors: np.ndarray
    ) -> np.ndarray:
        """"the largest contiguous phalanx on the home row"."""
        own_g = own.reshape(-1, HEIGHT, WIDTH)
        opp_g = opp.reshape(-1, HEIGHT, WIDTH)
        white = colors == 0
        rows = np.arange(own_g.shape[0])
        own_home = own_g[rows, np.where(white, 0, HEIGHT - 1)]
        opp_home = opp_g[rows, np.where(white, HEIGHT - 1, 0)]
        return _longest_run(own_home).astype(np.float64) - _longest_run(opp_home)

    # -- 4.6 Territory ----------------------------------------------------
    def _territory(self, own: np.ndarray, opp: np.ndarray, empty: np.ndarray) -> np.ndarray:
        """Per piece, the surrounding squares "empty, or occupied by a friendly piece".

        Off-board neighbours score nothing, so a piece on an edge is worth less
        than one in the open - which is the intent of the term.
        """
        nb = _RAY_IDX_NP[:, :, 1]  # (NUM_DIRS, N) the adjacent square each way
        on_board = _RAY_OK_NP[:, :, 1]  # (NUM_DIRS, N)

        def side(mine: np.ndarray, theirs: np.ndarray) -> np.ndarray:
            # A neighbour counts when it is on the board and not an enemy piece.
            good = ~theirs[:, nb] & on_board[None, :, :]  # (B, NUM_DIRS, N)
            return (good & mine[:, None, :]).sum(axis=(1, 2))

        return side(own, opp).astype(np.float64) - side(opp, own)

    # -- 4.4 Center of mass ------------------------------------------------
    def _centrality(self, own: np.ndarray, opp: np.ndarray) -> np.ndarray:
        """"the Euclidean distance for all pieces from the center of the board".

        The paper says the two sides' scores are subtracted and that "a positive
        value indicates a higher center of mass", without fixing which direction
        is good. Being nearer the centre is the property the term is motivated
        by (it cites chess opening theory on controlling the middle), so this
        returns mean *enemy* distance minus mean *own* distance: positive means
        our pieces are the more central. Mean rather than sum, so that the term
        measures shape and does not silently restate material.
        """
        rows, cols = np.divmod(np.arange(N), WIDTH)
        cr, cc = (HEIGHT - 1) / 2.0, (WIDTH - 1) / 2.0
        dist = np.sqrt((rows - cr) ** 2 + (cols - cc) ** 2)  # (N,)

        def mean_dist(mask: np.ndarray) -> np.ndarray:
            n = mask.sum(1)
            total = (mask * dist[None, :]).sum(1)
            return np.divide(total, n, out=np.zeros_like(total), where=n > 0)

        return mean_dist(opp) - mean_dist(own)

    # -- 4.1 Mobility ------------------------------------------------------
    def _mobility(self, own: np.ndarray, opp: np.ndarray, empty: np.ndarray) -> np.ndarray:
        """"averages the number of spaces that can be traversed by all the
        player's phalanxes [then] adds the greatest distance any of those
        phalanxes can travel".

        A phalanx is enumerated as (lead square, direction): a group of g pieces
        led by square s along d can advance up to min(g, empty squares ahead),
        which is the movement rule. Each group is therefore counted once per
        direction it could lead in, which is what "all the player's phalanxes"
        has to mean for the average to be well defined.

        **This does not reproduce the paper's worked example**, and the paper is
        not self-consistent enough to reproduce. It says a lone piece free to
        move in all eight directions scores 1/8 = 0.125 for the average part,
        plus a greatest distance of 1, totalling 1.125. No reading of "averages
        the number of spaces that can be traversed" gives 1/8 here: the piece
        can traverse one space in each of eight directions, so the average
        traversable distance is 1 and the total is 2.0, which is what this
        returns. Their 1/8 looks like one phalanx divided by eight directions,
        which is a count, not a distance.

        The consequence is only one of scale - their average term is tiny next
        to their max term, ours is comparable - and since the weights are ours
        rather than theirs anyway, the difference is absorbed by w.mobility.
        """

        return self._mobility_from_runs(
            _runs_all_dirs(own), _runs_all_dirs(opp), _runs_all_dirs(empty)
        )

    def _mobility_from_runs(
        self, runs_own: np.ndarray, runs_opp: np.ndarray, runs_empty: np.ndarray
    ) -> np.ndarray:
        """`_mobility` given the precomputed run tables for both sides."""
        # Empty squares ahead start at the next square along d, which is a fixed
        # gather, so index the empty runs once and reuse for both sides.
        nxt = _RAY_IDX_NP[:, :, 1]  # (NUM_DIRS, N)
        ahead = runs_empty[:, np.arange(NUM_DIRS)[:, None], nxt]  # (B, NUM_DIRS, N)
        ahead = np.where(_RAY_OK_NP[None, :, :, 1], ahead, 0)

        def side(runs_mine: np.ndarray) -> np.ndarray:
            # Pieces extending *back* against d are the group s can lead along d.
            group = runs_mine[:, _OPP_DIR_NP, :]  # (B, NUM_DIRS, N)
            travel = np.where(group >= 1, np.minimum(group, ahead), 0)
            count = (group >= 1).sum(axis=(1, 2))
            total = travel.sum(axis=(1, 2)).astype(np.float64)
            mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            return mean + travel.max(axis=(1, 2))

        return side(runs_own) - side(runs_opp)


def _runs_all_dirs(mask: np.ndarray) -> np.ndarray:
    """Per batch, direction and square: consecutive True cells including itself.

    The NumPy twin of the rules module's `_all_runs`.
    """
    vals = mask[:, _RAY_IDX_NP] & _RAY_OK_NP[None, :, :, :]  # (B, DIRS, N, STEP+1)
    # accumulate-and-count is the same as cumprod-and-sum on booleans, and
    # stays in bool rather than promoting to int64 for every cell.
    return np.logical_and.accumulate(vals, axis=3).sum(axis=3)


def _longest_run(rows: np.ndarray) -> np.ndarray:
    """Longest run of True in each row of a (B, L) boolean array."""
    best = np.zeros(rows.shape[0], dtype=np.int64)
    cur = np.zeros(rows.shape[0], dtype=np.int64)
    for i in range(rows.shape[1]):
        cur = np.where(rows[:, i], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


@dataclass
class SearchStats:
    nodes: int = 0
    evaluations: int = 0
    depth_reached: int = 0
    elapsed: float = 0.0
    score: float = 0.0
    timed_out: bool = False


class _Expansion:
    """Every child of one node: states on device, decision data on the host.

    Reading `state.terminated` or `state.current_player` off a device array is a
    synchronisation, and doing it per child is what makes a naive tree search
    over a JAX env slow. The whole batch is fetched once here instead, so the
    recursion below touches the host arrays only.
    """

    __slots__ = ("actions", "states", "terminated", "player", "reward", "legal", "value")

    def __init__(
        self,
        actions: np.ndarray,
        states: pgx.State,
        root_mover: int,
        evaluator: Evaluator,
    ) -> None:
        self.actions = actions
        self.states = states
        # One device_get for the fields the search itself reads. These are the
        # generic pgx ones; anything game-specific is the evaluator's business,
        # which is what keeps this class independent of the game.
        self.terminated, self.player, rewards, self.legal = jax.device_get(
            (
                states.terminated,
                states.current_player,
                states.rewards,
                states.legal_action_mask,
            )
        )
        self.reward = rewards[:, root_mover]
        raw = np.asarray(evaluator(states), dtype=np.float64)
        assert len(raw) == len(actions), "evaluator must score every child"
        # The evaluator scores from each state's own side to move; express it
        # from the root player's perspective instead, once, here.
        self.value = np.where(self.player == root_mover, raw, -raw)

    def __len__(self) -> int:
        return len(self.actions)

    def state(self, i: int) -> pgx.State:
        return jax.tree_util.tree_map(lambda x: x[i], self.states)


class NegamaxEngine:
    """Iterative-deepening alpha-beta over a pgx environment.

    `depth` counts full moves, not pgx actions. Multi-stage games expand every
    stage of a move before the depth counter moves, so depth 1 means "all my
    moves, then evaluate" for Epaminondas just as it does for a one-action game.
    """

    def __init__(
        self,
        env: pgx.Env,
        evaluator: Evaluator,
        max_depth: int = 64,
        time_limit_s: Optional[float] = 2.0,
        seed: int = 0,
        warm_up: bool = True,
    ) -> None:
        self.env = env
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.time_limit_s = time_limit_s
        self._step = jax.jit(jax.vmap(env.step))
        self._key = jax.random.PRNGKey(seed)
        self._key_cache: dict[int, jnp.ndarray] = {}
        self._deadline: Optional[float] = None
        self._root_mover = 0
        self.stats = SearchStats()
        if warm_up:
            self.warm_up()

    def warm_up(self) -> None:
        """Compile `vmap(step)` for every batch size the search can ask for.

        Each distinct batch shape costs a full XLA compilation - seconds, for
        Epaminondas - and the search would otherwise pay them mid-move, out of
        the move's own time budget. Doing them all here means a one-off cost at
        construction and honest timings thereafter.
        """
        state = self.env.init(self._key)
        b = 1
        while b <= _bucket(int(np.asarray(state.legal_action_mask).size)):
            parents = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (b,) + x.shape), state
            )
            actions = jnp.zeros(b, jnp.int32)
            jax.block_until_ready(self._step(parents, actions, self._keys(b)))
            b *= 2

    # -- expansion ---------------------------------------------------------
    def _keys(self, b: int) -> jnp.ndarray:
        """Step keys for a batch of `b`, cached by size.

        Epaminondas ignores the key, but `Env.step` takes one, and re-splitting
        per node would put an avoidable device op in the hot path.
        """
        if b not in self._key_cache:
            self._key_cache[b] = jax.random.split(self._key, b)
        return self._key_cache[b]

    def _expand(self, state: pgx.State, legal: np.ndarray) -> _Expansion:
        """Every legal successor of `state`, stepped in one batched call.

        The batch is padded up to the next power of two. `vmap(step)` is
        retraced and recompiled for every distinct batch shape it sees, and the
        number of legal actions changes at nearly every node, so an exactly
        sized batch means a fresh XLA compilation per node - which measured at
        roughly five nodes per second, entirely compilation. Bucketing caps that
        at one compile per bucket. The padding repeats a real action and its
        results are sliced off before anything looks at them.
        """
        actions = np.flatnonzero(legal)
        n = len(actions)
        b = _bucket(n)
        padded = np.concatenate([actions, np.full(b - n, actions[0])]) if b > n else actions
        parents = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (b,) + x.shape), state
        )
        children = self._step(parents, jnp.asarray(padded), self._keys(b))
        if b > n:
            children = jax.tree_util.tree_map(lambda x: x[:n], children)
        exp = _Expansion(actions, children, self._root_mover, self.evaluator)
        self.stats.nodes += n
        self.stats.evaluations += n
        return exp

    # -- search ------------------------------------------------------------
    def select_action(self, state: pgx.State) -> tuple[int, SearchStats]:
        """Best action for `state`, by iterative deepening within the budget.

        Each completed depth replaces the previous answer; a depth abandoned on
        the clock is discarded entirely, so the move returned is always the
        recommendation of a fully searched depth.
        """
        self.stats = SearchStats()
        start = time.perf_counter()
        self._deadline = start + self.time_limit_s if self.time_limit_s else None
        self._root_mover = int(state.current_player)

        legal = np.asarray(state.legal_action_mask)
        actions = np.flatnonzero(legal)
        if len(actions) == 1:
            self.stats.elapsed = time.perf_counter() - start
            return int(actions[0]), self.stats

        # Seed with the best child by static evaluation, so that a budget too
        # small to finish even depth 1 still produces a considered move rather
        # than whichever action happens to have the lowest index.
        root = self._expand(state, legal)
        best_action = int(root.actions[_ordered(root, maximising=True)[0]])
        best_score = -np.inf

        for depth in range(1, self.max_depth + 1):
            try:
                score, action = self._search_root(root, depth)
            except _Timeout:
                self.stats.timed_out = True
                break
            best_action, best_score = action, score
            self.stats.depth_reached = depth
            # A forced result is final; deeper search cannot improve on it.
            if abs(best_score) >= WIN_SCORE - 1000:
                break
            if self._out_of_time():
                break

        self.stats.elapsed = time.perf_counter() - start
        self.stats.score = float(best_score)
        return int(best_action), self.stats

    def _search_root(self, exp: _Expansion, depth: int) -> tuple[float, int]:
        """One full-width iteration over the root's children.

        The root expansion is made once by the caller and reused at every depth:
        re-stepping the root's children each iteration would repeat the single
        most expensive batch of the search for no reason.
        """
        mover = self._root_mover
        best_score, best_action = -np.inf, int(exp.actions[0])
        alpha = -np.inf
        for i in _ordered(exp, maximising=True):
            spent = 1 if exp.player[i] != mover else 0
            score = self._value(exp, i, depth - spent, 1, alpha, np.inf)
            if score > best_score:
                best_score, best_action = score, int(exp.actions[i])
            alpha = max(alpha, best_score)
        return best_score, best_action

    def _value(
        self,
        exp: _Expansion,
        i: int,
        depth: int,
        ply: int,
        alpha: float,
        beta: float,
    ) -> float:
        """Value of child `i` of `exp`, always from the root player's seat.

        Using one fixed perspective instead of negamax's alternating sign is
        what makes multi-stage turns fall out for free: whether this node
        maximises or minimises is read from who is actually to move, so the two
        intra-move stages of an Epaminondas move stay on the same side of the
        comparison rather than flipping with depth parity.
        """
        if exp.terminated[i]:
            # Prefer a win that arrives sooner, and a loss that arrives later.
            return float(exp.reward[i]) * (WIN_SCORE - ply)
        if depth <= 0:
            return float(exp.value[i])
        if self._out_of_time():
            raise _Timeout

        child = self._expand(exp.state(i), exp.legal[i])
        mover = int(exp.player[i])
        maximising = mover == self._root_mover

        if maximising:
            value = -np.inf
            for j in _ordered(child, maximising=True):
                spent = 1 if child.player[j] != mover else 0
                value = max(value, self._value(child, j, depth - spent, ply + 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # the minimising parent already has a better option
            return value

        value = np.inf
        for j in _ordered(child, maximising=False):
            spent = 1 if child.player[j] != mover else 0
            value = min(value, self._value(child, j, depth - spent, ply + 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _out_of_time(self) -> bool:
        return self._deadline is not None and time.perf_counter() > self._deadline


def _ordered(exp: _Expansion, maximising: bool) -> np.ndarray:
    """Child indices, most promising first, so alpha-beta prunes early.

    The expansion already evaluated every child in one batched call, so this
    ordering is effectively free and is worth far more than it costs.
    """
    vals = np.where(exp.terminated, exp.reward * WIN_SCORE, exp.value)
    return np.argsort(-vals if maximising else vals)


class _Timeout(Exception):
    """Raised to unwind the search when the move budget is spent."""


def _bucket(n: int) -> int:
    """The batch size to compile for, given `n` real children."""
    return 1 << (n - 1).bit_length() if n > 1 else 1


def make_evaluator(env_id: str) -> Evaluator:
    """The evaluation function for `env_id`.

    Only Epaminondas has one so far; this is the seam where another game's
    heuristic would be registered.
    """
    if env_id == "epaminondas":
        return EpaminondasEvaluator()
    raise ValueError(
        f"No negamax evaluation function for {env_id!r}. "
        "Add one to negamax.make_evaluator."
    )
