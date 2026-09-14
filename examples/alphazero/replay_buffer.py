# Host-memory FIFO replay buffer for AlphaZero self-play samples.
#
# Kept in its own module (like config.py) so it can be imported and tested
# without running train.py's module-level setup.

from typing import Any

import jax
import numpy as np


class ReplayBuffer:
    """FIFO ring buffer holding the most recent `capacity` samples.

    Samples are pytrees of host arrays whose leaves share a leading sample axis.
    Storage is preallocated on the first add; once full, the oldest samples are
    overwritten. The buffer lives only in memory: it is not checkpointed, so
    after a resume it refills from scratch.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._data: Any = None
        self._next = 0
        self._size = 0

    @property
    def num_samples(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        """Bytes of preallocated storage (0 before the first add)."""
        if self._data is None:
            return 0
        return sum(x.nbytes for x in jax.tree_util.tree_leaves(self._data))

    def add(self, samples: Any) -> None:
        samples = jax.tree_util.tree_map(np.asarray, samples)
        n = jax.tree_util.tree_leaves(samples)[0].shape[0]
        if self._data is None:
            self._data = jax.tree_util.tree_map(
                lambda x: np.empty((self.capacity, *x.shape[1:]), dtype=x.dtype), samples
            )
        if n > self.capacity:
            samples = jax.tree_util.tree_map(lambda x: x[n - self.capacity :], samples)
            n = self.capacity
        idxs = (self._next + np.arange(n)) % self.capacity

        def write(buf: np.ndarray, x: np.ndarray) -> None:
            buf[idxs] = x

        jax.tree_util.tree_map(write, self._data, samples)
        self._next = (self._next + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def state_dict(self) -> dict:
        """Contents for checkpointing (references, not copies)."""
        return {"capacity": self.capacity, "data": self._data, "next": self._next, "size": self._size}

    def load_state_dict(self, state: dict) -> None:
        """Restore from state_dict(). If the capacity differs, the stored
        samples are re-added oldest first, keeping the most recent ones."""
        if state["data"] is None:
            return
        if state["capacity"] == self.capacity:
            self._data, self._next, self._size = state["data"], state["next"], state["size"]
            return
        oldest = (state["next"] - state["size"]) % state["capacity"]
        order = (oldest + np.arange(state["size"])) % state["capacity"]
        self.add(jax.tree_util.tree_map(lambda x: x[order], state["data"]))

    def sample(self, rng_key: jax.Array, n: int) -> Any:
        """Draw n samples uniformly: without replacement when the buffer holds
        at least n, otherwise with replacement.

        Filled slots are always 0..num_samples-1, so with a buffer that has held
        exactly one add of n == capacity samples this is exactly a shuffle of
        that add (jax.random.permutation over its indices).
        """
        total = self._size
        if total == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if n <= total:
            idxs = np.asarray(jax.random.permutation(rng_key, total))[:n]
        else:
            idxs = np.asarray(jax.random.randint(rng_key, (n,), 0, total))
        return jax.tree_util.tree_map(lambda buf: buf[idxs], self._data)
