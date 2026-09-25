# We referred to Haiku's ResNet implementation:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py

import math

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


def _to_floating(x):
    """Cast non-floating observations to float32; keep a floating compute dtype."""
    return x if jnp.issubdtype(x.dtype, jnp.floating) else x.astype(jnp.float32)


def cast_floating(tree, dtype):
    """Cast the floating-point leaves of a pytree (e.g. a (params, state) model)."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree
    )


class BlockV1(hk.Module):
    def __init__(self, num_channels, name="BlockV1"):
        super(BlockV1, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        return jax.nn.relu(x + i)


class BlockV2(hk.Module):
    def __init__(self, num_channels, name="BlockV2"):
        super(BlockV2, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


class SelfAttentionBlock(hk.Module):
    """Multi-head self-attention over the spatial grid, used as a final block.

    The (H, W) spatial dimensions are flattened into a length H*W sequence,
    self-attention is applied across that sequence, and the result is reshaped
    back to the original (H, W, C) grid and added residually.
    """

    def __init__(self, num_channels, num_heads, name="SelfAttentionBlock"):
        super(SelfAttentionBlock, self).__init__(name=name)
        self.num_channels = num_channels
        self.num_heads = num_heads

    def __call__(self, x, is_training, test_local_stats):
        i = x
        b, h, w, c = x.shape
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        # Alternative: LayerNorm (per-token over channels) instead of BatchNorm.
        # It avoids a NaN blow-up that BatchNorm causes in eval mode when its
        # running variance is seeded ~0 from a near-constant init batch (e.g. an
        # empty Go board), which divides inputs by ~sqrt(eps) and overflows the
        # attention logits. Kept disabled because it changes the parameter layout
        # and so breaks loading of existing BatchNorm-trained checkpoints.
        # x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        # Flatten the spatial grid into a sequence of length H*W.
        seq = x.reshape(b, h * w, c)
        attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=max(c // self.num_heads, 1),
            model_size=c,
            w_init=hk.initializers.VarianceScaling(1.0),
        )(seq, seq, seq)
        attn = attn.reshape(b, h, w, c)
        return attn + i


class AZNet(hk.Module):
    """AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        num_heads: int = -1,
        num_attention_layers: int = 1,
        action_cells=None,
        name="az_net",
    ):
        super().__init__(name=name)
        print(f"Creating AZ model with num_heads={num_heads}, "
              f"num_attention_layers={num_attention_layers}")
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        # num_heads > 0 replaces the final num_attention_layers convolution
        # blocks with multi-head self-attention blocks (using that many heads).
        self.num_heads = num_heads
        self.num_attention_layers = num_attention_layers
        self.resnet_cls = BlockV2 if resnet_v2 else BlockV1
        # (num_actions, 2) board cells, one per action (see `action_cells`), or
        # None for the original dense head over the flattened board.
        self.action_cells = action_cells

    def __call__(self, x, is_training, test_local_stats):
        x = _to_floating(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)

        if not self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        # The last num_attention_layers blocks become self-attention blocks
        # (when num_heads > 0); the earlier blocks stay convolutional.
        first_attention_block = self.num_blocks - self.num_attention_layers
        for i in range(self.num_blocks):
            is_attention = self.num_heads > 0 and i >= first_attention_block
            if is_attention:
                x = SelfAttentionBlock(self.num_channels, self.num_heads, name=f"block_{i}")(
                    x, is_training, test_local_stats
                )
            else:
                x = self.resnet_cls(self.num_channels, name=f"block_{i}")(
                    x, is_training, test_local_stats
                )

        if self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        # policy head
        if self.action_cells is not None:
            # One logit per board cell, read off at the cell each action names,
            # so every action's logit comes from its own place on the board.
            logits = hk.Conv2D(output_channels=self.num_channels // 2, kernel_shape=1)(x)
            logits = hk.Conv2D(output_channels=1, kernel_shape=1)(jax.nn.relu(logits))[..., 0]
            logits = logits[:, self.action_cells[:, 0], self.action_cells[:, 1]]
        else:
            logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
            logits = hk.BatchNorm(True, True, 0.9)(logits, is_training, test_local_stats)
            logits = jax.nn.relu(logits)
            logits = hk.Flatten()(logits)
            logits = hk.Linear(self.num_actions)(logits)

        # value head
        v = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        v = hk.BatchNorm(True, True, 0.9)(v, is_training, test_local_stats)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        return logits, v


# ─── GessFormer ──────────────────────────────────────────────────────────────
#
# A hybrid conv + transformer network for Gess, adapted from Chessformer
# (arXiv 2605.19091). Gess has two scales of structure: 3x3 piece footprints
# (handled by a full-resolution conv stem) and long-range sliding moves
# (handled by a transformer over 2x2-merged 10x10 tokens with Chessformer's
# Geometric Attention Bias). An upsample + skip connection restores per-cell
# features for a source->destination policy head over the 20x20 action grid.

# Gess observation: 18x18 playing area; actions address the 20x20 grid that
# also includes the border ring.
_GESS_GRID = 20
_GESS_TOKENS_SIDE = _GESS_GRID // 2
_GESS_NUM_TOKENS = _GESS_TOKENS_SIDE**2
# Chessformer GAB dimensions: d1 (per-token compression), d2 (board summary),
# d3 (per-head template code).
_GAB_D1 = 32
_GAB_D2 = 128
_GAB_D3 = 128

_TF_INIT = hk.initializers.TruncatedNormal(stddev=0.02)


def _layer_norm(name=None):
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)


class MLPNet(hk.Module):
    """A small residual MLP for flat (non-grid) observations, e.g. pig's (6,).

    Games whose observation is a handful of counters have no spatial structure
    for a conv net to exploit, but their value function is very non-linear in
    those counters (in pig, "can I reach 100 this turn?" flips behaviour). So
    when `onehot_bins` > 0 each integer feature is one-hot encoded over
    [0, onehot_bins) - clipped, so anything above the top bin lands in it - and
    concatenated with the same features scaled to roughly [0, 1]. That gives
    the trunk a near-tabular input while keeping a smooth signal for
    extrapolation.
    """

    def __init__(self, num_actions, width=256, num_layers=3, onehot_bins=0, name="MLPNet"):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.width = width
        self.num_layers = num_layers
        self.onehot_bins = onehot_bins

    def __call__(self, x, is_training=True, test_local_stats=False):
        del is_training, test_local_stats
        dtype = x.dtype
        x = x.reshape((x.shape[0], -1))
        feats = [x / max(self.onehot_bins - 1, 1) if self.onehot_bins > 0 else x]
        if self.onehot_bins > 0:
            idx = jnp.clip(jnp.round(x.astype(jnp.float32)), 0, self.onehot_bins - 1).astype(jnp.int32)
            feats.append(jax.nn.one_hot(idx, self.onehot_bins, dtype=dtype).reshape((x.shape[0], -1)))
        h = hk.Linear(self.width, w_init=_TF_INIT)(jnp.concatenate(feats, axis=-1))
        for _ in range(self.num_layers):
            r = _layer_norm()(h)
            r = jax.nn.gelu(r)
            r = hk.Linear(self.width, w_init=_TF_INIT)(r)
            r = jax.nn.gelu(r)
            r = hk.Linear(self.width, w_init=_TF_INIT)(r)
            h = h + r
        h = _layer_norm()(h)
        h = jax.nn.gelu(h)

        logits = hk.Linear(self.num_actions, w_init=_TF_INIT)(h)
        v = hk.Linear(self.width // 2, w_init=_TF_INIT)(h)
        v = jax.nn.gelu(v)
        v = hk.Linear(1, w_init=_TF_INIT)(v)
        return logits, jnp.tanh(v).reshape((-1,))


class ConvBlock(hk.Module):
    """Pre-LN residual block: [LN -> GELU -> 3x3 conv] x 2."""

    def __init__(self, num_channels, name="ConvBlock"):
        super().__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x):
        i = x
        for _ in range(2):
            x = _layer_norm()(x)
            x = jax.nn.gelu(x)
            x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


class TransformerBlock(hk.Module):
    """Pre-LN encoder block with Geometric Attention Bias (GAB).

    `gab_w`/`gab_b` are the parameters of GAB's final projection, which
    Chessformer shares across all layers; they are passed in as arrays so the
    block can be wrapped in hk.remat.
    """

    def __init__(self, num_heads, ffn_mult, use_gab, out_init, name="TransformerBlock"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        self.use_gab = use_gab
        self.out_init = out_init

    def _attention_bias(self, x, gab_w, gab_b):
        b, n, _ = x.shape
        h = self.num_heads
        bias = hk.get_parameter("static_bias", (h, n, n), init=jnp.zeros)[None]
        if self.use_gab:
            # Compress the board into a summary, then produce one template
            # code per head, which the shared projection maps to an n x n bias.
            g = hk.Linear(_GAB_D1, w_init=_TF_INIT, name="gab_compress")(x)
            g = g.reshape(b, n * _GAB_D1)
            g = hk.Linear(_GAB_D2, w_init=_TF_INIT, name="gab_summary")(g)
            g = _layer_norm("gab_summary_ln")(jax.nn.gelu(g))
            g = hk.Linear(h * _GAB_D3, w_init=_TF_INIT, name="gab_heads")(g)
            g = _layer_norm("gab_heads_ln")(jax.nn.gelu(g.reshape(b, h, _GAB_D3)))
            bias = bias + (g @ gab_w + gab_b).reshape(b, h, n, n)
        return bias

    def __call__(self, x, gab_w, gab_b):
        b, n, d = x.shape
        h = self.num_heads
        k = d // h

        # Attention
        y = _layer_norm("attn_ln")(x)
        qkv = hk.Linear(3 * d, w_init=_TF_INIT, name="qkv")(y)
        q, kk, v = jnp.split(qkv.reshape(b, n, 3, h, k), 3, axis=2)
        q, kk, v = (t.squeeze(2).transpose(0, 2, 1, 3) for t in (q, kk, v))  # (b, h, n, k)
        logits = (q @ kk.transpose(0, 1, 3, 2)) / math.sqrt(k)
        logits = logits + self._attention_bias(y, gab_w, gab_b)
        attn = jax.nn.softmax(logits, axis=-1) @ v
        attn = attn.transpose(0, 2, 1, 3).reshape(b, n, d)
        x = x + hk.Linear(d, w_init=self.out_init, name="attn_out")(attn)

        # FFN
        y = _layer_norm("ffn_ln")(x)
        y = hk.Linear(int(d * self.ffn_mult), w_init=_TF_INIT, name="ffn_in")(y)
        y = jax.nn.gelu(y)
        x = x + hk.Linear(d, w_init=self.out_init, name="ffn_out")(y)
        return x


def _gess_input(x):
    """Pad the 18x18 observation to the 20x20 action grid, add an is_border plane.

    Returns (x (b, 20, 20, 5), is_stage1 (b,), src_footprint (b, 20, 20)).
    """
    x = _to_floating(x)
    b, g = x.shape[0], _GESS_GRID
    x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    border = jnp.pad(jnp.zeros((g - 2, g - 2), x.dtype), 1, constant_values=1.0)
    x = jnp.concatenate([x, jnp.broadcast_to(border[None, :, :, None], (b, g, g, 1))], axis=-1)
    is_stage1 = x[:, :, :, 3].max(axis=(1, 2)) > 0.5
    return x, is_stage1, x[:, :, :, 2]


def _gess_policy_head(f, is_stage1, src_footprint):
    """400 logits from per-cell features f (b, 20, 20, c).

    Stage 0 (pick source): per-cell logit. Stage 1 (pick destination):
    Chessformer-style source->destination attention, with the source query
    pooled over the source footprint.
    """
    b, g, _, c = f.shape
    stage0 = hk.Linear(c, name="policy0_hidden")(f)
    stage0 = hk.Linear(1, name="policy0_out")(jax.nn.gelu(stage0)).reshape(b, g * g)

    key_size = c
    mask = src_footprint[..., None]
    pooled = (f * mask).sum(axis=(1, 2)) / jnp.maximum(mask.sum(axis=(1, 2)), 1.0)
    query = hk.Linear(key_size, name="policy1_query")(pooled)  # (b, k)
    keys = hk.Linear(key_size, name="policy1_key")(f).reshape(b, g * g, key_size)
    stage1 = jnp.einsum("bk,bnk->bn", query, keys) / math.sqrt(key_size)
    stage1 = stage1 + hk.Linear(1, name="policy1_bias")(f).reshape(b, g * g)

    return jnp.where(is_stage1[:, None], stage1, stage0)


def _gess_value_head(tok):
    """Scalar value from tokens (b, n, d): mean-pool -> LN -> MLP -> tanh."""
    v = _layer_norm("value_ln")(tok.mean(axis=1))
    v = jax.nn.gelu(hk.Linear(128, name="value_hidden")(v))
    return jnp.tanh(hk.Linear(1, w_init=_TF_INIT, name="value_out")(v)).reshape((-1,))


class GessFormer(hk.Module):
    """Hybrid conv-stem + GAB transformer network for Gess (see comment above)."""

    def __init__(
        self,
        num_actions: int,
        stem_channels: int = 64,
        stem_blocks: int = 2,
        embed_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_mult: float = 2.0,
        use_gab: bool = True,
        remat: bool = True,
        name="gess_former",
    ):
        super().__init__(name=name)
        assert num_actions == _GESS_GRID**2, "GessFormer expects the 20x20 Gess action grid"
        assert embed_dim % num_heads == 0
        self.num_actions = num_actions
        self.stem_channels = stem_channels
        self.stem_blocks = stem_blocks
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        self.use_gab = use_gab
        self.remat = remat

    def __call__(self, x, is_training=False, test_local_stats=False):
        del is_training, test_local_stats  # no BatchNorm
        x, is_stage1, src_footprint = _gess_input(x)
        b = x.shape[0]
        c, d = self.stem_channels, self.embed_dim
        t = _GESS_TOKENS_SIDE

        # Conv stem at full resolution (3x3 piece locality).
        s = hk.Conv2D(c, kernel_shape=3, name="stem_conv")(x)
        for i in range(self.stem_blocks):
            s = ConvBlock(c, name=f"stem_block_{i}")(s)

        # Patch merge to 10x10 tokens.
        tok = hk.Conv2D(d, kernel_shape=2, stride=2, padding="VALID", name="patch_merge")(s)
        tok = tok.reshape(b, _GESS_NUM_TOKENS, d)
        tok = tok + hk.get_parameter("pos_emb", (_GESS_NUM_TOKENS, d), init=_TF_INIT)

        # Transformer encoder.
        gab_w = hk.get_parameter("gab_shared_w", (_GAB_D3, _GESS_NUM_TOKENS**2), init=_TF_INIT)
        gab_b = hk.get_parameter("gab_shared_b", (_GESS_NUM_TOKENS**2,), init=jnp.zeros)
        out_init = hk.initializers.TruncatedNormal(stddev=0.02 / math.sqrt(2.0 * self.num_layers))
        for i in range(self.num_layers):
            block = TransformerBlock(
                self.num_heads, self.ffn_mult, self.use_gab, out_init, name=f"block_{i}"
            )
            block_fn = hk.remat(block) if self.remat else block
            tok = block_fn(tok, gab_w, gab_b)
        tok = _layer_norm("final_ln")(tok)

        # Decoder: upsample tokens back to the 20x20 grid with the stem skip.
        f = tok.reshape(b, t, t, d)
        f = jnp.repeat(jnp.repeat(f, 2, axis=1), 2, axis=2)
        f = hk.Linear(c, name="upsample_proj")(f) + s
        f = ConvBlock(c, name="decoder_block")(f)
        f = jax.nn.gelu(_layer_norm("decoder_ln")(f))  # (b, 20, 20, c)

        return _gess_policy_head(f, is_stage1, src_footprint), _gess_value_head(tok)

# ─── BoardFormer ─────────────────────────────────────────────────────────────
#
# GessFormer's architecture generalised to any square-grid game whose action
# space is one action per board cell (e.g. Epaminondas: a 12x14 board, 168
# actions, three actions per move). Same shape as GessFormer - conv stem, 2x2
# patch merge, pre-LN transformer with Geometric Attention Bias, upsample with a
# stem skip - but with a plain per-cell policy head, since the stage and any
# selection markers are already planes of the observation.


class BoardFormer(hk.Module):
    """Conv stem + GAB transformer with one policy logit per board cell."""

    def __init__(
        self,
        num_actions: int,
        stem_channels: int = 64,
        stem_blocks: int = 2,
        embed_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_mult: float = 2.0,
        use_gab: bool = True,
        remat: bool = True,
        action_cells=None,
        name="board_former",
    ):
        super().__init__(name=name)
        assert embed_dim % num_heads == 0
        self.num_actions = num_actions
        # (num_actions, 2) board cells, one per action, or None when every cell
        # is an action in row-major order (Epaminondas).
        self.action_cells = action_cells
        self.stem_channels = stem_channels
        self.stem_blocks = stem_blocks
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        self.use_gab = use_gab
        self.remat = remat

    def __call__(self, x, is_training=False, test_local_stats=False):
        del is_training, test_local_stats  # no BatchNorm
        x = _to_floating(x)
        b, h, w, _ = x.shape
        if h % 2 or w % 2:
            # Pad an odd board at the bottom and right to even, and mark which
            # cells are the board. On Dots and Boxes' 13x13 lattice this makes
            # each 2x2 patch a dot with the lines to its right and below and
            # the box between them, so a token is one dot's worth of board.
            board = jnp.ones((b, h, w, 1), x.dtype)
            x = jnp.concatenate([x, board], axis=-1)
            x = jnp.pad(x, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)))
            h, w = h + h % 2, w + w % 2
        if self.action_cells is None:
            assert self.num_actions == h * w, "BoardFormer expects one action per cell"
        c, d = self.stem_channels, self.embed_dim
        th, tw = h // 2, w // 2
        num_tokens = th * tw

        # Conv stem at full resolution.
        s = hk.Conv2D(c, kernel_shape=3, name="stem_conv")(x)
        for i in range(self.stem_blocks):
            s = ConvBlock(c, name=f"stem_block_{i}")(s)

        # Patch merge to (h/2, w/2) tokens.
        tok = hk.Conv2D(d, kernel_shape=2, stride=2, padding="VALID", name="patch_merge")(s)
        tok = tok.reshape(b, num_tokens, d)
        tok = tok + hk.get_parameter("pos_emb", (num_tokens, d), init=_TF_INIT)

        gab_w = hk.get_parameter("gab_shared_w", (_GAB_D3, num_tokens**2), init=_TF_INIT)
        gab_b = hk.get_parameter("gab_shared_b", (num_tokens**2,), init=jnp.zeros)
        out_init = hk.initializers.TruncatedNormal(stddev=0.02 / math.sqrt(2.0 * self.num_layers))
        for i in range(self.num_layers):
            block = TransformerBlock(
                self.num_heads, self.ffn_mult, self.use_gab, out_init, name=f"block_{i}"
            )
            tok = (hk.remat(block) if self.remat else block)(tok, gab_w, gab_b)
        tok = _layer_norm("final_ln")(tok)

        # Decoder: back to full resolution with the stem skip.
        f = tok.reshape(b, th, tw, d)
        f = jnp.repeat(jnp.repeat(f, 2, axis=1), 2, axis=2)
        f = hk.Linear(c, name="upsample_proj")(f) + s
        f = ConvBlock(c, name="decoder_block")(f)
        f = jax.nn.gelu(_layer_norm("decoder_ln")(f))

        logits = hk.Linear(c, name="policy_hidden")(f)
        logits = hk.Linear(1, name="policy_out")(jax.nn.gelu(logits))[..., 0]
        if self.action_cells is not None:
            logits = logits[:, self.action_cells[:, 0], self.action_cells[:, 1]]
        else:
            logits = logits.reshape(b, h * w)
        return logits, _gess_value_head(tok)


# ─── RayFormer ───────────────────────────────────────────────────────────────
#
# A full-resolution transformer for Gess whose attention follows piece moves:
# every cell of the 20x20 grid is a token, and it attends along its row,
# column, diagonal and anti-diagonal (the 8 compass rays). Each of the four
# line families is a separate softmax over lines of at most 20 cells, with a
# learned per-head bias for the offset along the line. A 3x3 conv stem gives
# each token its piece footprint first.


def _diagonal_lines(n, anti):
    """Gather tables for the diagonals (or anti-diagonals) of an n x n grid.

    Returns (idx, valid, inv): idx (2n-1, n) cell indices ordered along each
    line, padded with 0 where valid is False; inv (n*n,) the flat position of
    each cell in idx (every cell lies on exactly one line).
    """
    grid = np.arange(n * n).reshape(n, n)
    if anti:
        grid = grid[:, ::-1]
    idx = np.zeros((2 * n - 1, n), dtype=np.int32)
    valid = np.zeros((2 * n - 1, n), dtype=bool)
    for i, k in enumerate(range(-(n - 1), n)):
        line = np.diagonal(grid, k)
        idx[i, : len(line)] = line
        valid[i, : len(line)] = True
    inv = np.empty(n * n, dtype=np.int32)
    inv[idx[valid]] = np.flatnonzero(valid)
    return idx, valid, inv


# Ray families: rows and columns are reshapes of the grid; diagonals gather.
_RAY_FAMILIES = ("rows", "cols", "diag", "anti")
_DIAGONALS = {
    "diag": _diagonal_lines(_GESS_GRID, anti=False),
    "anti": _diagonal_lines(_GESS_GRID, anti=True),
}


def _attend_lines(q, k, v, rel_bias, valid=None, dtype=jnp.float32):
    """Attention within lines: q/k/v (b, lines, n, h, kd) -> (b, lines, n, h, kd).

    rel_bias (h, 2n-1) is indexed by the offset j - i along the line; valid
    (lines, n) masks padded key positions. The attention itself is computed in
    `dtype` and the result cast back.
    """
    out_dtype = q.dtype
    n, kd = q.shape[2], q.shape[-1]
    q, k, v = (t.transpose(0, 1, 3, 2, 4).astype(dtype) for t in (q, k, v))  # (b, lines, h, n, kd)
    logits = (q @ k.transpose(0, 1, 2, 4, 3)) / math.sqrt(kd)
    offsets = np.arange(n)[None, :] - np.arange(n)[:, None] + n - 1
    logits = logits + rel_bias[:, offsets].astype(dtype)
    if valid is not None:
        logits = jnp.where(valid[None, :, None, None, :], logits, jnp.finfo(dtype).min / 2)
    out = jax.nn.softmax(logits, axis=-1) @ v
    return out.transpose(0, 1, 3, 2, 4).astype(out_dtype)


def _symmetry_orbits(n):
    """Orbit id (0..k-1) of each cell of an n x n grid under the 8 square symmetries."""
    grid = np.arange(n * n).reshape(n, n)
    images = []
    for t in (grid, grid.T):
        for r in (t, t[::-1]):
            images += [r, r[:, ::-1]]
    canonical = np.min(np.stack(images), axis=0).reshape(-1)
    return np.unique(canonical, return_inverse=True)[1].astype(np.int32)


class LineAttention(hk.Module):
    """Multi-head attention within each line of one ray family.

    Takes per-cell projected q/k/v (b, cells, 3, h, kd) on the row-major grid and
    returns the attended values per cell (b, cells, h, kd).
    """

    def __init__(self, family, dtype=jnp.float32, name="LineAttention"):
        super().__init__(name=name)
        self.family = family
        self.dtype = dtype

    def __call__(self, qkv, rel_bias=None):
        """rel_bias (h, 2g-1): a shared offset bias; None creates this family's own."""
        b, cells, _, h, kd = qkv.shape
        g = _GESS_GRID
        if rel_bias is None:
            rel_bias = hk.get_parameter("rel_bias", (h, 2 * g - 1), init=jnp.zeros)
        if self.family in ("rows", "cols"):
            grid = qkv.reshape(b, g, g, 3, h, kd)
            if self.family == "cols":
                grid = grid.transpose(0, 2, 1, 3, 4, 5)
            out = _attend_lines(grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2], rel_bias, None, self.dtype)
            if self.family == "cols":
                out = out.transpose(0, 2, 1, 3, 4)
            return out.reshape(b, cells, h, kd)
        idx, valid, inv = _DIAGONALS[self.family]
        lines = qkv[:, idx]  # (b, 2g-1, g, 3, h, kd)
        out = _attend_lines(lines[:, :, :, 0], lines[:, :, :, 1], lines[:, :, :, 2], rel_bias, valid, self.dtype)
        return out.reshape(b, -1, h, kd)[:, inv]


class RayBlock(hk.Module):
    """Pre-LN encoder block with ray attention (see RayFormer comment above)."""

    def __init__(self, num_heads, ffn_mult, out_init, remat, attn_dtype, symmetric, name="RayBlock"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        self.out_init = out_init
        self.remat = remat
        self.attn_dtype = attn_dtype
        self.symmetric = symmetric

    def __call__(self, x):
        b, n, d = x.shape
        h = self.num_heads

        y = _layer_norm("attn_ln")(x)
        qkv = hk.Linear(3 * d, w_init=_TF_INIT, name="qkv")(y).reshape(b, n, 3, h, d // h)
        shared_bias = {}
        if self.symmetric:
            # Board symmetries map rows <-> columns and diagonals <-> anti-diagonals,
            # and reflections reverse direction along a line. So share one bias
            # between each such pair of families, depending only on |offset|.
            g = _GESS_GRID
            abs_offset = np.abs(np.arange(2 * g - 1) - (g - 1))
            orth = hk.get_parameter("rel_bias_orth", (h, g), init=jnp.zeros)[:, abs_offset]
            diag = hk.get_parameter("rel_bias_diag", (h, g), init=jnp.zeros)[:, abs_offset]
            shared_bias = {"rows": orth, "cols": orth, "diag": diag, "anti": diag}
        attn = 0.0
        for f, family in enumerate(_RAY_FAMILIES):
            line_attn = LineAttention(family, self.attn_dtype, name=f"line_attn_{f}")
            # Recompute each family separately in the backward pass, so only one
            # family's attention matrices are held at a time.
            attn = attn + (hk.remat(line_attn) if self.remat else line_attn)(
                qkv, shared_bias.get(family)
            )
        x = x + hk.Linear(d, w_init=self.out_init, name="attn_out")(attn.reshape(b, n, d))

        y = _layer_norm("ffn_ln")(x)
        y = hk.Linear(int(d * self.ffn_mult), w_init=_TF_INIT, name="ffn_in")(y)
        x = x + hk.Linear(d, w_init=self.out_init, name="ffn_out")(jax.nn.gelu(y))
        return x


class RayFormer(hk.Module):
    """Full-resolution ray-attention transformer for Gess (see comment above)."""

    def __init__(
        self,
        num_actions: int,
        embed_dim: int = 96,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_mult: float = 2.0,
        stem_blocks: int = 1,
        remat: bool = True,
        attn_bf16: bool = True,
        symmetric: bool = False,
        name="ray_former",
    ):
        super().__init__(name=name)
        assert num_actions == _GESS_GRID**2, "RayFormer expects the 20x20 Gess action grid"
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        self.stem_blocks = stem_blocks
        self.remat = remat
        self.attn_dtype = jnp.bfloat16 if attn_bf16 else jnp.float32
        # Tie position parameters across the 8 board symmetries (see RayBlock and
        # _symmetry_orbits), so the encoder is equivariant to them.
        self.symmetric = symmetric

    def __call__(self, x, is_training=False, test_local_stats=False):
        del is_training, test_local_stats  # no BatchNorm
        x, is_stage1, src_footprint = _gess_input(x)
        b, g, d = x.shape[0], _GESS_GRID, self.embed_dim

        # Conv stem at full resolution (3x3 piece locality), one token per cell.
        s = hk.Conv2D(d, kernel_shape=3, name="stem_conv")(x)
        for i in range(self.stem_blocks):
            s = ConvBlock(d, name=f"stem_block_{i}")(s)
        tok = s.reshape(b, g * g, d)
        if self.symmetric:
            orbits = _symmetry_orbits(g)
            pos_emb = hk.get_parameter("pos_emb_orbit", (orbits.max() + 1, d), init=_TF_INIT)
            tok = tok + pos_emb[orbits]
        else:
            tok = tok + hk.get_parameter("pos_emb", (g * g, d), init=_TF_INIT)

        out_init = hk.initializers.TruncatedNormal(stddev=0.02 / math.sqrt(2.0 * self.num_layers))
        for i in range(self.num_layers):
            block = RayBlock(
                self.num_heads,
                self.ffn_mult,
                out_init,
                self.remat,
                self.attn_dtype,
                self.symmetric,
                name=f"block_{i}",
            )
            tok = (hk.remat(block) if self.remat else block)(tok)
        tok = _layer_norm("final_ln")(tok)

        f = tok.reshape(b, g, g, d)
        return _gess_policy_head(f, is_stage1, src_footprint), _gess_value_head(tok)


def mlp_input_features(params, config):
    """Observation features an `mlp` checkpoint was built for, or None for
    another architecture. Read off the first layer, whose input is each feature
    raw plus, with one-hot bins, `mlp_onehot_bins` more per feature.

    Observations only ever grow at the end (pig v0 had six features, v1 seven),
    so feeding a network the leading features it was built for shows it exactly
    what it saw in training.
    """
    w = params.get("MLPNet/linear", {}).get("w") if hasattr(params, "get") else None
    if w is None:
        return None
    per_feature = 1 + config.mlp_onehot_bins if config.mlp_onehot_bins > 0 else 1
    return int(w.shape[0]) // per_feature


def action_cells(env_id: str):
    """(num_actions, 2) board cells naming where each action lives, for games
    whose actions are some of the observation's cells rather than all of them,
    or None. A network with them emits one logit per cell and reads each
    action's off its cell, instead of a dense layer over the flattened board.
    """
    if env_id == "dots_and_boxes":
        from pgx._src.games.dots_and_boxes import _LINE_CELLS

        return np.asarray(_LINE_CELLS)
    return None


def make_forward(num_actions: int, config, dtype=jnp.float32) -> hk.TransformedWithState:
    """Build the (params, state) Haiku transform for `config.architecture`.

    `dtype` is the compute dtype: inputs are cast to it and outputs back to
    float32. For bfloat16 inference, apply with cast_floating(model, dtype).
    """

    cells = action_cells(config.env_id)

    def forward_fn(x: jnp.ndarray, is_eval: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = x.astype(dtype)
        if config.architecture == "mlp":
            net = MLPNet(
                num_actions=num_actions,
                width=config.mlp_width,
                num_layers=config.mlp_layers,
                onehot_bins=config.mlp_onehot_bins,
            )
        elif config.architecture == "boardformer":
            net = BoardFormer(
                num_actions=num_actions,
                stem_channels=config.bf_stem_channels,
                stem_blocks=config.bf_stem_blocks,
                embed_dim=config.bf_embed_dim,
                num_layers=config.bf_num_layers,
                num_heads=config.bf_num_heads,
                ffn_mult=config.bf_ffn_mult,
                use_gab=config.bf_gab,
                remat=config.bf_remat,
                action_cells=cells,
            )
        elif config.architecture == "rayformer":
            net = RayFormer(
                num_actions=num_actions,
                embed_dim=config.rf_embed_dim,
                num_layers=config.rf_num_layers,
                num_heads=config.rf_num_heads,
                ffn_mult=config.rf_ffn_mult,
                stem_blocks=config.rf_stem_blocks,
                remat=config.rf_remat,
                attn_bf16=config.rf_attn_bf16,
                symmetric=config.rf_symmetric,
            )
        elif config.architecture == "gessformer":
            net = GessFormer(
                num_actions=num_actions,
                stem_channels=config.gf_stem_channels,
                stem_blocks=config.gf_stem_blocks,
                embed_dim=config.gf_embed_dim,
                num_layers=config.gf_num_layers,
                num_heads=config.gf_num_heads,
                ffn_mult=config.gf_ffn_mult,
                use_gab=config.gf_gab,
                remat=config.gf_remat,
            )
        else:
            net = AZNet(
                num_actions=num_actions,
                num_channels=config.num_channels,
                num_blocks=config.num_layers,
                resnet_v2=config.resnet_v2,
                num_heads=config.num_heads,
                num_attention_layers=config.num_attention_layers,
                action_cells=cells if config.cell_policy_head else None,
            )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out.astype(jnp.float32), value_out.astype(jnp.float32)

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def make_optimizer(config) -> optax.GradientTransformation:
    """Adam by default; AdamW with an LR schedule and clipping when configured.

    With the default weight_decay/warmup_steps/grad_clip_norm/lr_schedule this
    is exactly optax.adam(learning_rate), so older runs resume with a
    compatible opt_state.
    """
    if (
        config.weight_decay == 0.0
        and config.warmup_steps == 0
        and config.grad_clip_norm == 0.0
        and config.lr_schedule == "constant"
    ):
        return optax.adam(learning_rate=config.learning_rate)

    peak = config.learning_rate
    if config.lr_schedule == "cosine":
        # Iterations that make fewer updates while the replay buffer fills up
        # leave the decay slightly unfinished at max_num_iters.
        total_steps = max(config.max_num_iters * config.updates_per_iter(), config.warmup_steps + 1)
        end = peak * config.lr_final_ratio
        if config.warmup_steps > 0:
            lr = optax.warmup_cosine_decay_schedule(0.0, peak, config.warmup_steps, total_steps, end)
        else:
            lr = optax.cosine_decay_schedule(peak, total_steps, alpha=config.lr_final_ratio)
    elif config.warmup_steps > 0:
        lr = optax.join_schedules(
            [optax.linear_schedule(0.0, peak, config.warmup_steps), optax.constant_schedule(peak)],
            boundaries=[config.warmup_steps],
        )
    else:
        lr = peak

    def decay_mask(params):
        # Decay only weight matrices/kernels; not norms, biases or embeddings.
        return hk.data_structures.map(lambda m, n, p: n in ("w", "gab_shared_w"), params)

    tx = optax.adamw(lr, weight_decay=config.weight_decay, mask=decay_mask)
    if config.grad_clip_norm > 0.0:
        tx = optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), tx)
    return tx
