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
