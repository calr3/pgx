# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Mctx: Monte Carlo tree search in JAX.

A copy of mctx 0.0.71 (https://github.com/google-deepmind/mctx, Apache 2.0)
with a faster search loop, used by the AlphaZero examples as
`import mctx_fast as mctx`. The API is unchanged and so are the results: the
search tree, action and action weights are bit-identical to upstream
(tests/test_mctx_fast.py). Only `_src/search.py` differs:

- The root step of each simulation runs once, before the traversal loop.
  Upstream chose the root or interior rule with `lax.cond`, which under `vmap`
  becomes a select that evaluated both rules at every depth.
- The backward pass reads the tree in its loop and applies the writes once at
  the end. Upstream updated the tree inside a vmapped `while_loop`, which
  selects over the whole carry on every iteration, i.e. copied every
  `[B, N, num_actions]` array once per level of every simulation.
- When `max_depth >= num_simulations` (the default) every simulation adds
  node `sim + 1` in every batch element, so node writes are slice updates
  rather than scatters.

When upgrading, re-copy upstream and re-apply these changes to search.py;
`git log -- examples/alphazero/mctx_fast` has the unmodified copy first.
"""

from mctx_fast._src.action_selection import gumbel_muzero_interior_action_selection
from mctx_fast._src.action_selection import gumbel_muzero_root_action_selection
from mctx_fast._src.action_selection import GumbelMuZeroExtraData
from mctx_fast._src.action_selection import muzero_action_selection
from mctx_fast._src.base import ChanceRecurrentFnOutput
from mctx_fast._src.base import DecisionRecurrentFnOutput
from mctx_fast._src.base import InteriorActionSelectionFn
from mctx_fast._src.base import LoopFn
from mctx_fast._src.base import PolicyOutput
from mctx_fast._src.base import RecurrentFn
from mctx_fast._src.base import RecurrentFnOutput
from mctx_fast._src.base import RecurrentState
from mctx_fast._src.base import RootActionSelectionFn
from mctx_fast._src.base import RootFnOutput
from mctx_fast._src.policies import gumbel_muzero_policy
from mctx_fast._src.policies import muzero_policy
from mctx_fast._src.policies import stochastic_muzero_policy
from mctx_fast._src.qtransforms import qtransform_by_min_max
from mctx_fast._src.qtransforms import qtransform_by_parent_and_siblings
from mctx_fast._src.qtransforms import qtransform_completed_by_mix_value
from mctx_fast._src.search import search
from mctx_fast._src.tree import Tree

__version__ = "0.0.71"

__all__ = (
    "ChanceRecurrentFnOutput",
    "DecisionRecurrentFnOutput",
    "GumbelMuZeroExtraData",
    "InteriorActionSelectionFn",
    "LoopFn",
    "PolicyOutput",
    "RecurrentFn",
    "RecurrentFnOutput",
    "RecurrentState",
    "RootActionSelectionFn",
    "RootFnOutput",
    "Tree",
    "gumbel_muzero_interior_action_selection",
    "gumbel_muzero_policy",
    "gumbel_muzero_root_action_selection",
    "muzero_action_selection",
    "muzero_policy",
    "qtransform_by_min_max",
    "qtransform_by_parent_and_siblings",
    "qtransform_completed_by_mix_value",
    "search",
    "stochastic_muzero_policy",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Mctx public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
