# We referred to Haiku's ResNet implementation:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py

import math

import haiku as hk
import jax
import jax.numpy as jnp
import optax


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

    def __call__(self, x, is_training, test_local_stats):
        x = x.astype(jnp.float32)
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
        x = x.astype(jnp.float32)
        b = x.shape[0]
        c, d = self.stem_channels, self.embed_dim
        g, t = _GESS_GRID, _GESS_TOKENS_SIDE

        # Input: pad the 18x18 playing area to the 20x20 action grid and add an
        # is_border plane so each cell lines up with its action index.
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        border = jnp.pad(jnp.zeros((g - 2, g - 2)), 1, constant_values=1.0)
        x = jnp.concatenate([x, jnp.broadcast_to(border[None, :, :, None], (b, g, g, 1))], axis=-1)
        is_stage1 = x[:, :, :, 3].max(axis=(1, 2)) > 0.5
        src_footprint = x[:, :, :, 2]

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

        # Policy head. Stage 0 (pick source): per-cell logit. Stage 1 (pick
        # destination): Chessformer-style source->destination attention, with
        # the source query pooled over the source footprint.
        stage0 = hk.Linear(c, name="policy0_hidden")(f)
        stage0 = hk.Linear(1, name="policy0_out")(jax.nn.gelu(stage0)).reshape(b, g * g)

        key_size = c
        mask = src_footprint[..., None]
        pooled = (f * mask).sum(axis=(1, 2)) / jnp.maximum(mask.sum(axis=(1, 2)), 1.0)
        query = hk.Linear(key_size, name="policy1_query")(pooled)  # (b, k)
        keys = hk.Linear(key_size, name="policy1_key")(f).reshape(b, g * g, key_size)
        stage1 = jnp.einsum("bk,bnk->bn", query, keys) / math.sqrt(key_size)
        stage1 = stage1 + hk.Linear(1, name="policy1_bias")(f).reshape(b, g * g)

        logits = jnp.where(is_stage1[:, None], stage1, stage0)

        # Value head: mean-pool tokens -> LN -> MLP -> tanh.
        v = _layer_norm("value_ln")(tok.mean(axis=1))
        v = jax.nn.gelu(hk.Linear(128, name="value_hidden")(v))
        v = jnp.tanh(hk.Linear(1, w_init=_TF_INIT, name="value_out")(v)).reshape((-1,))

        return logits, v


def make_forward(num_actions: int, config) -> hk.TransformedWithState:
    """Build the (params, state) Haiku transform for `config.architecture`."""

    def forward_fn(x: jnp.ndarray, is_eval: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
        if config.architecture == "gessformer":
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
            )
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def make_optimizer(config) -> optax.GradientTransformation:
    """Adam by default; AdamW + warmup + clipping when those fields are set.

    With the default weight_decay/warmup_steps/grad_clip_norm this is exactly
    optax.adam(learning_rate), so older runs resume with a compatible opt_state.
    """
    if config.weight_decay == 0.0 and config.warmup_steps == 0 and config.grad_clip_norm == 0.0:
        return optax.adam(learning_rate=config.learning_rate)

    lr = config.learning_rate
    if config.warmup_steps > 0:
        lr = optax.join_schedules(
            [optax.linear_schedule(0.0, config.learning_rate, config.warmup_steps),
             optax.constant_schedule(config.learning_rate)],
            boundaries=[config.warmup_steps],
        )

    def decay_mask(params):
        # Decay only weight matrices/kernels; not norms, biases or embeddings.
        return hk.data_structures.map(lambda m, n, p: n in ("w", "gab_shared_w"), params)

    tx = optax.adamw(lr, weight_decay=config.weight_decay, mask=decay_mask)
    if config.grad_clip_norm > 0.0:
        tx = optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), tx)
    return tx
