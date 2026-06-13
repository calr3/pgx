# Shared AlphaZero training configuration.
#
# Defined in its own module (rather than in train.py's __main__) so that the
# same Config class can be imported by both the trainer and run_tournament.py,
# and so checkpoints pickle/unpickle against a stable module path (config.Config).

import pgx
from pydantic import BaseModel, PrivateAttr, model_validator


class Config(BaseModel):
    env_id: pgx.EnvId = "g_hex"
    seed: int = 0
    max_num_iters: int = 400
    # path to a .ckpt file to resume training from (empty = start from scratch)
    resume_from: str = ""
    # When resuming, by default the RNG key is restored from the checkpoint so
    # self-play deterministically continues the same game stream. Set this True
    # to instead reseed the RNG from `seed`, producing fresh self-play games
    # (model/opt_state/iteration/frames are still restored as usual).
    reseed_on_resume: bool = False
    # wandb run id to resume logging into. If empty when resuming, the id stored
    # in the checkpoint (if any) is used so the original run continues.
    wandb_run_id: str = ""
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # num_heads > 0 replaces the final conv block with a multi-head
    # self-attention block using that many heads; -1 keeps an all-conv net.
    num_heads: int = -1
    # Number of self-attention blocks applied at the end (after the conv blocks),
    # used only when num_heads > 0. Must not exceed num_layers.
    num_attention_layers: int = 1
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    # Optional curriculum for MCTS search depth: a comma-separated list of
    # "<num_simulations>@<from_iteration>" entries, e.g. "8@0,16@50,32@150,64@250".
    # At each iteration the trainer uses the num_simulations of the latest entry
    # whose from_iteration <= iteration (falling back to `num_simulations` for any
    # iteration before the first entry). Empty = use the constant `num_simulations`
    # for the whole run. A cheap shallow search early (when the random-ish net
    # can't exploit deep search) and a deeper one late is more compute-efficient.
    # Changing the active value triggers one XLA recompile of the self-play step
    # (cheap, only a handful of times over a run).
    sim_schedule: str = ""
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    # Parsed `sim_schedule`, as a list of (from_iteration, num_simulations) sorted
    # ascending by from_iteration. Populated by the validator below.
    _sim_schedule: list[tuple[int, int]] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _check_attention_layers(self):
        if self.num_attention_layers > self.num_layers:
            raise ValueError(
                f"num_attention_layers ({self.num_attention_layers}) cannot exceed "
                f"num_layers ({self.num_layers})."
            )
        return self

    @model_validator(mode="after")
    def _parse_sim_schedule(self):
        entries: list[tuple[int, int]] = []
        for raw in self.sim_schedule.split(","):
            raw = raw.strip()
            if not raw:
                continue
            sims_str, sep, iter_str = raw.partition("@")
            if not sep:
                raise ValueError(
                    f"sim_schedule entry {raw!r} must be of the form "
                    "'<num_simulations>@<from_iteration>', e.g. '8@0'."
                )
            try:
                sims, from_iter = int(sims_str), int(iter_str)
            except ValueError:
                raise ValueError(
                    f"sim_schedule entry {raw!r} has non-integer parts; expected "
                    "'<num_simulations>@<from_iteration>', e.g. '16@50'."
                )
            if sims <= 0 or from_iter < 0:
                raise ValueError(
                    f"sim_schedule entry {raw!r}: num_simulations must be > 0 and "
                    "from_iteration must be >= 0."
                )
            entries.append((from_iter, sims))
        entries.sort()
        self._sim_schedule = entries
        return self

    def num_simulations_at(self, iteration: int) -> int:
        """Active MCTS simulation count for `iteration` under `sim_schedule`.

        Falls back to the constant `num_simulations` when no schedule is set (or
        for iterations before the schedule's first entry)."""
        sims = self.num_simulations
        for from_iter, s in self._sim_schedule:
            if from_iter <= iteration:
                sims = s
            else:
                break
        return sims

    class Config:
        extra = "forbid"
