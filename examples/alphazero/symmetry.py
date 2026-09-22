# Board-symmetry data augmentation.
#
# Gess is invariant under the 8 symmetries of the square (the dihedral group
# D4): pieces move along all 8 compass directions, the playing area and its
# border ring are square, and nothing depends on orientation or on which colour
# is which (the observation is already from the mover's perspective). So a
# training sample stays valid when its observation planes and its policy target
# over the 20x20 action grid are transformed by the same symmetry.

import jax
import jax.numpy as jnp

from pgx._src.games.epaminondas import MIRROR_DIR_PERM, NUM_DIRS

NUM_SYMMETRIES = 8


def transform_grid(x: jnp.ndarray, sym: jnp.ndarray) -> jnp.ndarray:
    """Apply symmetry `sym` (0..7) to a square grid x of shape (n, n, ...).

    Bit 2 transposes, bit 0 flips rows and bit 1 flips columns, which together
    enumerate D4. Grids of different sizes sharing a centre (e.g. the 18x18
    playing area inside the 20x20 action grid) transform consistently.
    """
    x = jnp.where((sym & 4) > 0, jnp.swapaxes(x, 0, 1), x)
    x = jnp.where((sym & 1) > 0, jnp.flip(x, 0), x)
    return jnp.where((sym & 2) > 0, jnp.flip(x, 1), x)


def augment_gess(
    rng_key: jnp.ndarray, obs: jnp.ndarray, policy_tgt: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Transform each sample by an independent random symmetry.

    obs: (b, 18, 18, c) observations; policy_tgt: (b, 400) over the 20x20 grid.
    """
    b = obs.shape[0]
    side = int(round(policy_tgt.shape[-1] ** 0.5))
    syms = jax.random.randint(rng_key, (b,), 0, NUM_SYMMETRIES)
    obs = jax.vmap(transform_grid)(obs, syms)
    policy = jax.vmap(transform_grid)(policy_tgt.reshape(b, side, side), syms)
    return obs, policy.reshape(b, side * side)


# Epaminondas has one symmetry, not eight. The board is 12x14 rather than
# square, and the rank-by-rank tiebreak and the two home rows make the vertical
# axis meaningful, so rotations and row flips are not symmetries. The left-right
# mirror is: the board, the starting position and the eight move directions are
# all symmetric about it. Checked over 57,600 positions from random games -
# legal masks, successor boards, terminal flags and winners all agree with the
# mirror image - which is the check to redo if the rules change.
NUM_EPAMINONDAS_SYMMETRIES = 2


def augment_epaminondas(
    rng_key: jnp.ndarray, obs: jnp.ndarray, policy_tgt: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mirror a random half of the samples left to right.

    obs: (b, 12, 14, c) observations; policy_tgt: (b, 168) over the same grid.

    The trailing `NUM_DIRS` observation planes are `_travel`, one per direction,
    so mirroring the columns also changes what each of those channels means and
    they have to be permuted to match. Getting that wrong would not crash: it
    would quietly train the network on observations no position can produce.
    """
    b, h, w, c = obs.shape
    assert h * w == policy_tgt.shape[-1], "policy target must cover the observation grid"
    assert c > NUM_DIRS, "expected the travel planes to be the trailing channels"

    mirrored = jnp.flip(obs, axis=2)
    mirrored = jnp.concatenate(
        [mirrored[..., :-NUM_DIRS], mirrored[..., -NUM_DIRS:][..., MIRROR_DIR_PERM]], axis=-1
    )
    take = jax.random.bernoulli(rng_key, 0.5, (b,))
    obs = jnp.where(take[:, None, None, None], mirrored, obs)

    policy = policy_tgt.reshape(b, h, w)
    policy = jnp.where(take[:, None, None], jnp.flip(policy, axis=2), policy)
    return obs, policy.reshape(b, h * w)
