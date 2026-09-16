# Shared AlphaZero training configuration.
#
# Defined in its own module (rather than in train.py's __main__) so that the
# same Config class can be imported by both the trainer and run_tournament.py,
# and so checkpoints pickle/unpickle against a stable module path (config.Config).

from typing import Literal

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
    # "resnet" is AZNet (above fields). "gessformer" is the hybrid conv-stem +
    # Chessformer-style transformer for Gess (network.GessFormer), configured by
    # the gf_* fields below, which the resnet ignores.
    architecture: Literal["resnet", "gessformer", "rayformer", "boardformer", "mlp"] = "resnet"
    # "mlp" (network.MLPNet) is for games whose observation is a flat vector of
    # counters rather than a board, e.g. pig. mlp_onehot_bins > 0 additionally
    # one-hot encodes each feature over [0, bins) (101 suits pig's 0..100
    # scores); 0 feeds the raw values only.
    mlp_width: int = 256
    mlp_layers: int = 3
    mlp_onehot_bins: int = 0
    gf_stem_channels: int = 64
    gf_stem_blocks: int = 2
    gf_embed_dim: int = 192
    gf_num_layers: int = 6
    gf_num_heads: int = 8
    gf_ffn_mult: float = 2.0
    # Geometric Attention Bias (board-conditioned); False = static bias only.
    gf_gab: bool = True
    # Rematerialize transformer blocks in the backward pass to save memory.
    gf_remat: bool = True
    # "rayformer" (network.RayFormer): full-resolution transformer for Gess
    # with attention along rows, columns and diagonals; configured by rf_*.
    # Defaults roughly match gessformer's inference cost (self-play dominates);
    # training steps are ~1.7x slower, and at training_batch_size=2048 on a
    # 16 GB GPU need train_micro_batches=2.
    rf_embed_dim: int = 96
    rf_num_layers: int = 4
    rf_num_heads: int = 4
    rf_ffn_mult: float = 2.0
    rf_stem_blocks: int = 1
    rf_remat: bool = True
    # Compute the line attention in bfloat16 (weights stay float32).
    rf_attn_bf16: bool = True
    # Tie RayFormer's position parameters across the 8 board symmetries: rows
    # and columns share an attention bias, as do diagonals and anti-diagonals,
    # each depending only on |offset|; the position embedding is shared by
    # symmetric cells. The encoder is then equivariant to rotations/reflections
    # (the conv stem is not constrained).
    rf_symmetric: bool = False
    # "boardformer" (network.BoardFormer): GessFormer's architecture for any
    # even-sized board whose action space is one action per cell (e.g.
    # epaminondas). Plain per-cell policy head.
    bf_stem_channels: int = 64
    bf_stem_blocks: int = 2
    bf_embed_dim: int = 192
    bf_num_layers: int = 6
    bf_num_heads: int = 8
    bf_ffn_mult: float = 2.0
    bf_gab: bool = True
    bf_remat: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    # Chance outcomes averaged per search edge. mctx keeps one sampled successor
    # per edge, so in a stochastic game its value stays conditioned on a single
    # draw no matter how many simulations run. Values > 1 average the value over
    # that many draws, at that many times the inference cost; only useful for
    # stochastic envs (e.g. pig), and it assumes every outcome of an action
    # leaves the same player to move.
    chance_samples: int = 1
    # How search Q values are scaled before they enter the improved policy that
    # becomes the training target.
    #   "completed_by_mix_value" - mctx's default; rescales Q to the range seen
    #     across the node's actions. With only two actions that range *is* the
    #     gap between them, so only the sign of the advantage survives.
    #   "completed_unscaled" - the same completion without that rescaling, so the
    #     size of the advantage reaches the target. Better for small action
    #     spaces (pig).
    #   "by_min_max" - rescale by a fixed [-1, 1] instead, with no completion of
    #     unvisited actions.
    qtransform: Literal["completed_by_mix_value", "completed_unscaled", "by_min_max"] = (
        "completed_by_mix_value"
    )
    # Run the self-play network in bfloat16 (weights cast per call; training and
    # evaluation stay float32). ~2x faster inference for gessformer.
    selfplay_bf16: bool = False
    # Playout cap randomization (KataGo): each self-play step uses the full
    # num_simulations search with this probability, otherwise a cheap
    # fast_num_simulations search whose policy targets are excluded from the
    # policy loss (value targets are kept). 1.0 = always full search.
    playout_cap_prob: float = 1.0
    fast_num_simulations: int = 8
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
    # Continue self-play games across iterations instead of restarting every
    # game slot from the initial position each iteration. Steps from games still
    # in progress at the end of an iteration are held back (in host memory) and
    # added to the replay buffer once the game ends and its value target is
    # known; steps held back longer than max_pending_steps are added without a
    # value target. The games and held-back steps are only checkpointed with
    # save_data_state; otherwise a resumed run starts fresh games.
    continue_games: bool = False
    max_pending_steps: int = 1024
    # Also save the replay buffer, held-back steps and in-progress self-play
    # games (to <checkpoint dir>/data_state.pkl, overwritten at every
    # checkpoint), so resume_from continues training data exactly instead of
    # starting empty with fresh games. Useful on preemptible machines; the file
    # can be several GB. On resume it is used whenever its iteration matches
    # the checkpoint's, whatever this flag's value.
    save_data_state: bool = False
    # training params
    training_batch_size: int = 4096
    # Split each device's minibatch into this many microbatches and accumulate
    # their gradients: same effective batch, less activation memory.
    train_micro_batches: int = 1
    # Train on each sample under a random one of the 8 board symmetries
    # (rotations/reflections). Gess only; see symmetry.py.
    symmetry_augmentation: bool = False
    # Replay buffer size, in self-play iterations' worth of samples
    # (replay_buffer_iters * selfplay_batch_size * max_num_steps): minibatches are
    # sampled uniformly from the most recent samples, held in host memory
    # (checkpointed only with save_data_state; otherwise it refills after a
    # resume). 1 = train only on the current iteration's samples.
    replay_buffer_iters: int = 1
    # Gradient updates per iteration. 0 = one pass over a fresh iteration's
    # samples, i.e. (selfplay_batch_size * max_num_steps) // training_batch_size.
    # Sample reuse (replay ratio) = updates * training_batch_size / samples added
    # per iteration; draws are without replacement while the buffer holds enough
    # samples for the iteration, with replacement otherwise.
    num_updates_per_iter: int = 0
    learning_rate: float = 0.001
    # With all three at their defaults the optimizer is plain Adam (compatible
    # with older checkpoints' opt_state); otherwise AdamW with a linear warmup
    # and optional global-norm gradient clipping. See network.make_optimizer.
    weight_decay: float = 0.0
    warmup_steps: int = 0
    grad_clip_norm: float = 0.0
    # Learning rate after warmup: "constant", or "cosine" decay from
    # learning_rate to learning_rate * lr_final_ratio over the whole run
    # (max_num_iters * updates_per_iter() gradient steps, warmup included).
    lr_schedule: Literal["constant", "cosine"] = "constant"
    lr_final_ratio: float = 0.1
    # eval params
    eval_interval: int = 5
    # Periodic MCTS match against a fixed opponent checkpoint (empty = off), on a
    # wall-clock cadence: at the start, then whenever mcts_eval_interval_hours
    # have passed since the previous match, and at the end of the run. Every match
    # uses the same seat-swapped random openings, so scores are comparable over
    # time. Logged as eval/mcts/*. Does not affect training or its RNG.
    mcts_eval_opponent: str = ""
    mcts_eval_interval_hours: float = 1.0
    mcts_eval_games: int = 128
    mcts_eval_batch_size: int = 128
    mcts_eval_simulations: int = 32

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
    def _check_replay(self):
        if self.replay_buffer_iters < 1:
            raise ValueError(f"replay_buffer_iters must be >= 1, got {self.replay_buffer_iters}.")
        if not 0.0 < self.playout_cap_prob <= 1.0:
            raise ValueError(f"playout_cap_prob must be in (0, 1], got {self.playout_cap_prob}.")
        if self.fast_num_simulations < 1:
            raise ValueError(f"fast_num_simulations must be >= 1, got {self.fast_num_simulations}.")
        if self.mcts_eval_opponent:
            if self.mcts_eval_interval_hours < 0:
                raise ValueError("mcts_eval_interval_hours must be >= 0.")
            if self.mcts_eval_games % self.mcts_eval_batch_size != 0:
                raise ValueError(
                    f"mcts_eval_games ({self.mcts_eval_games}) must be a multiple of "
                    f"mcts_eval_batch_size ({self.mcts_eval_batch_size})."
                )
        if self.train_micro_batches < 1:
            raise ValueError(f"train_micro_batches must be >= 1, got {self.train_micro_batches}.")
        if self.max_pending_steps < 0:
            raise ValueError(f"max_pending_steps must be >= 0, got {self.max_pending_steps}.")
        if not 0.0 <= self.lr_final_ratio <= 1.0:
            raise ValueError(f"lr_final_ratio must be in [0, 1], got {self.lr_final_ratio}.")
        if self.num_updates_per_iter < 0:
            raise ValueError(f"num_updates_per_iter must be >= 0, got {self.num_updates_per_iter}.")
        return self

    @model_validator(mode="after")
    def _check_gess_architectures(self):
        if self.symmetry_augmentation and self.env_id != "gess":
            raise ValueError(f"symmetry_augmentation requires env_id=gess, got {self.env_id!r}.")
        if self.architecture == "boardformer" and self.bf_embed_dim % self.bf_num_heads != 0:
            raise ValueError(
                f"bf_embed_dim ({self.bf_embed_dim}) must be divisible by "
                f"bf_num_heads ({self.bf_num_heads})."
            )
        if self.architecture in ("gessformer", "rayformer") and self.env_id != "gess":
            raise ValueError(
                f"architecture={self.architecture} requires env_id=gess, got {self.env_id!r}."
            )
        if self.architecture == "rayformer" and self.rf_embed_dim % self.rf_num_heads != 0:
            raise ValueError(
                f"rf_embed_dim ({self.rf_embed_dim}) must be divisible by "
                f"rf_num_heads ({self.rf_num_heads})."
            )
        if self.architecture == "gessformer":
            if self.gf_embed_dim % self.gf_num_heads != 0:
                raise ValueError(
                    f"gf_embed_dim ({self.gf_embed_dim}) must be divisible by "
                    f"gf_num_heads ({self.gf_num_heads})."
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

    def updates_per_iter(self) -> int:
        """Gradient updates per iteration once the replay buffer has filled."""
        return self.num_updates_per_iter or (
            self.selfplay_batch_size * self.max_num_steps // self.training_batch_size
        )

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
