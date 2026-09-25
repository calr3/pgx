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


# Dots and Boxes is invariant under all 8 symmetries of the square: the board of
# dots is square, every line and box looks alike, and nothing depends on
# orientation. The observation is the 13x13 lattice of dots, lines and boxes,
# and each symmetry of the lattice maps dots to dots, lines to lines (a
# transpose swaps horizontal and vertical ones) and boxes to boxes, so the
# planes transform as a grid. The policy target is over the 84 lines: scatter it
# onto their lattice cells, transform, and read it back off.
# `tests/test_dots_and_boxes.py` checks every symmetry against the rules -
# observations, legal masks, successors and results - which is the check to
# redo if the rules or the observation change.


def augment_dots_and_boxes(
    rng_key: jnp.ndarray, obs: jnp.ndarray, policy_tgt: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Transform each sample by an independent random symmetry.

    obs: (b, 13, 13, c) lattice observations; policy_tgt: (b, 84) over lines.
    """
    from pgx._src.games.dots_and_boxes import _LINE_CELLS

    b, g = obs.shape[0], obs.shape[1]
    rows, cols = _LINE_CELLS[:, 0], _LINE_CELLS[:, 1]
    syms = jax.random.randint(rng_key, (b,), 0, NUM_SYMMETRIES)
    obs = jax.vmap(transform_grid)(obs, syms)
    grid = jnp.zeros((b, g, g), policy_tgt.dtype).at[:, rows, cols].set(policy_tgt)
    grid = jax.vmap(transform_grid)(grid, syms)
    return obs, grid[:, rows, cols]


def dots_and_boxes_permutations():
    """(line_perm (8, 84), box_perm (8, 36)): under symmetry s, line l becomes
    line_perm[s, l] and box k becomes box_perm[s, k]. For checking the rules."""
    import numpy as np

    from pgx._src.games.dots_and_boxes import _LINE_CELLS, BOXES, GRID, LINES, SIZE

    cells = np.asarray(_LINE_CELLS)
    line_ids = np.full((GRID, GRID), -1)
    line_ids[cells[:, 0], cells[:, 1]] = np.arange(LINES)
    box_ids = np.full((GRID, GRID), -1)
    box_ids[1::2, 1::2] = np.arange(BOXES).reshape(SIZE, SIZE)
    line_perm = np.zeros((NUM_SYMMETRIES, LINES), dtype=np.int32)
    box_perm = np.zeros((NUM_SYMMETRIES, BOXES), dtype=np.int32)
    for s in range(NUM_SYMMETRIES):
        # The id that lands on each cell under s, read back as "what each id became".
        moved_lines = np.asarray(transform_grid(jnp.asarray(line_ids), jnp.int32(s)))
        moved_boxes = np.asarray(transform_grid(jnp.asarray(box_ids), jnp.int32(s)))
        at = moved_lines >= 0
        for new_cell, old in zip(np.argwhere(at), moved_lines[at]):
            line_perm[s, old] = line_ids[new_cell[0], new_cell[1]]
        for new_cell, old in zip(np.argwhere(moved_boxes >= 0), moved_boxes[moved_boxes >= 0]):
            box_perm[s, old] = box_ids[new_cell[0], new_cell[1]]
    return line_perm, box_perm
