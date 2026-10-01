# Host-side conversion of self-play steps into training samples.
#
# Kept in its own module (like config.py) so it can be imported and tested
# without running train.py's module-level setup.

from typing import Any, NamedTuple

import numpy as np


class Sample(NamedTuple):
    obs: np.ndarray
    policy_tgt: np.ndarray
    value_tgt: np.ndarray
    mask: np.ndarray
    # Whether policy_tgt came from a full search (playout cap randomization).
    policy_mask: np.ndarray
    # Auxiliary immediate-win target (Config.aux_win_weight): which actions win
    # at once, and whether that label exists (False for steps generated before
    # the target was turned on). None when the target is off; defaults keep
    # samples pickled before these fields existed loadable.
    aux_win: np.ndarray | None = None
    aux_valid: np.ndarray | None = None


_FIELDS = ("obs", "action_weights", "reward", "discount", "terminated", "policy_mask")
# Present only when the auxiliary immediate-win target is on.
_AUX_FIELDS = ("aux_win", "aux_valid")


class PendingTrajectories:
    """Turns self-play steps into samples with final-outcome value targets.

    Each call to `process` receives the next max_num_steps of self-play as
    host arrays shaped (T, B, ...) with fields obs, action_weights, reward,
    discount, terminated and (optionally) policy_mask. A step's value target is the discounted return to
    the end of its game, so it is only known ("resolved") once a terminal step
    for that game slot has been seen.

    With carry=True (self-play games continue across iterations), unresolved
    steps are held back and emitted in a later call once their game ends, so
    every step eventually gets a real value target. A step held back for more
    than max_pending_steps is emitted with mask=False (policy target only).

    With carry=False (every iteration restarts its games), nothing is held back:
    all steps are emitted, unresolved ones with mask=False. This matches the
    original compute_loss_input.
    """

    def __init__(self, max_pending_steps: int) -> None:
        if max_pending_steps < 0:
            raise ValueError(f"max_pending_steps must be >= 0, got {max_pending_steps}")
        self.max_pending_steps = max_pending_steps
        self._tail: dict[str, np.ndarray] | None = None
        self._tail_pending: np.ndarray | None = None

    @property
    def num_pending(self) -> int:
        """Steps currently held back waiting for their game to end."""
        return 0 if self._tail_pending is None else int(self._tail_pending.sum())

    def state_dict(self) -> dict:
        """Held-back steps, for checkpointing."""
        return {"tail": self._tail, "tail_pending": self._tail_pending}

    def load_state_dict(self, state: dict) -> None:
        self._tail = state["tail"]
        self._tail_pending = state["tail_pending"]

    def process(self, data: Any, carry: bool) -> Sample:
        fields = _FIELDS + tuple(f for f in _AUX_FIELDS if getattr(data, f, None) is not None)
        steps = {f: np.asarray(getattr(data, f)) for f in fields if hasattr(data, f)}
        steps.setdefault("policy_mask", np.ones(steps["terminated"].shape, dtype=bool))
        pending = np.ones(steps["terminated"].shape, dtype=bool)
        if self._tail is not None:
            tail_len = self._tail["terminated"].shape[0]
            for f in fields:
                if f not in self._tail:  # held back before the aux target was on
                    self._tail[f] = np.zeros((tail_len, *steps[f].shape[1:]), steps[f].dtype)
            steps = {f: np.concatenate([self._tail[f], steps[f]]) for f in fields}
            pending = np.concatenate([self._tail_pending, pending])
        length = pending.shape[0]

        # resolved[t, b]: a terminal occurs at or after step t in slot b.
        terminated = steps["terminated"]
        resolved = np.flip(np.logical_or.accumulate(np.flip(terminated, 0), axis=0), 0)

        # Discounted return to the end of each game; discount is 0 at terminals.
        value_tgt = np.empty(terminated.shape, dtype=np.float32)
        v = np.zeros(terminated.shape[1], dtype=np.float32)
        for t in reversed(range(length)):
            v = steps["reward"][t] + steps["discount"][t] * v
            value_tgt[t] = v

        if carry:
            hold = pending & ~resolved
            # Steps waiting too long are emitted without a value target.
            too_old = np.arange(length)[:, None] < length - self.max_pending_steps
            hold &= ~too_old
        else:
            hold = np.zeros_like(pending)
        emit = pending & ~hold

        if hold.any():
            first = int(np.argmax(hold.any(axis=1)))
            # Copy so the tail does not keep the whole concatenated window alive.
            self._tail = {f: steps[f][first:].copy() for f in fields}
            self._tail_pending = hold[first:].copy()
        else:
            self._tail = None
            self._tail_pending = None

        return Sample(
            obs=steps["obs"][emit],
            policy_tgt=steps["action_weights"][emit],
            value_tgt=value_tgt[emit],
            mask=resolved[emit],
            policy_mask=steps["policy_mask"][emit],
            aux_win=steps["aux_win"][emit] if "aux_win" in steps else None,
            aux_valid=steps["aux_valid"][emit] if "aux_valid" in steps else None,
        )
