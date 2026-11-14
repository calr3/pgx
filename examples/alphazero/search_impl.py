"""Which mctx implementation the AlphaZero examples search with.

The examples depend on the calr3/mctx fork (see requirements.txt), whose
search loop is faster and whose tree stores the statistics of each edge once,
on the child node. The fork also carries the original upstream mctx as
`mctx._src.tests.reference`, so the two can be timed against each other on
the same machine without reinstalling anything:

    python -u examples/alphazero/train.py ... mctx_impl=reference

Both return bit-identical searches (tests/test_mctx_fork.py); only speed and
memory differ. `search_tree_bf16` is a fork-only option that does change the
results slightly, so it needs a strength test before adoption.
"""

from types import ModuleType
from typing import Any, Literal

import jax.numpy as jnp
import mctx

MctxImpl = Literal["fork", "reference"]

_NOT_FORK = (
    "The installed mctx is not the calr3/mctx fork (no Tree.child_visits). "
    "Install it with: pip install -r examples/alphazero/requirements.txt"
)


def load(impl: MctxImpl) -> ModuleType:
    """The mctx package to search with: the fork, or the upstream reference."""
    if not hasattr(mctx.Tree, "child_visits"):
        raise ImportError(_NOT_FORK)
    if impl == "fork":
        return mctx
    if impl == "reference":
        from mctx._src.tests import reference  # pylint: disable=import-outside-toplevel

        return reference
    raise ValueError(f"Unknown mctx_impl {impl!r}; expected 'fork' or 'reference'.")


def policy_kwargs(search_tree_bf16: bool) -> dict[str, Any]:
    """Extra keyword arguments for the fork's policies."""
    return {"prior_logits_dtype": jnp.bfloat16} if search_tree_bf16 else {}
