from pgx._src.games.heckmeck import _DICE_VALS, _TILE_VALS, _WORM_VALS
from pgx.heckmeck import State as HeckmeckState

# Pips of a die face, as (column, row) offsets within a 3x3 grid. Face 0 is the
# worm, which has no pips and is drawn as a letter instead.
_PIPS = {
    1: [(1, 1)],
    2: [(0, 0), (2, 2)],
    3: [(0, 0), (1, 1), (2, 2)],
    4: [(0, 0), (2, 0), (0, 2), (2, 2)],
    5: [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
}


def _make_heckmeck_dwg(dwg, state: HeckmeckState, config):
    GRID_SIZE = config["GRID_SIZE"]
    WIDTH = config["BOARD_WIDTH"]
    HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

    dwg.add(
        dwg.rect(
            (0, 0),
            (WIDTH * GRID_SIZE, HEIGHT * GRID_SIZE),
            fill=color_set.background_color,
        )
    )
    g = dwg.g()

    grill = [bool(t) for t in state._x.grill]
    stacks = [[int(t) for t in stack] for stack in state._x.stacks]
    rolled = [int(n) for n in state._x.dice_rolled]
    taken = [int(n) for n in state._x.dice_taken]
    mover = int(state._x.color)
    winner = int(state._x.winner)

    def text(s, x, y, size=0.4, anchor="start"):
        g.add(
            dwg.text(
                text=s,
                insert=(x, y),
                fill=color_set.text_color,
                font_size=f"{GRID_SIZE * size}px",
                font_family="monospace",
                text_anchor=anchor,
            )
        )

    def die(face, x, y, side):
        g.add(
            dwg.rect(
                (x, y),
                (side, side),
                fill=color_set.background_color,
                stroke=color_set.grid_color,
                stroke_width="1.5px",
                rx="4px",
                ry="4px",
            )
        )
        if face == 0:  # the worm, worth 5 but the only face that can win a tile
            text("W", x + side / 2, y + side * 0.7, size=side / GRID_SIZE * 0.6, anchor="middle")
            return
        for col, row in _PIPS[face]:
            g.add(
                dwg.circle(
                    center=(x + (col + 0.5) * side / 3, y + (row + 0.5) * side / 3),
                    r=side / 12,
                    fill=color_set.grid_color,
                )
            )

    # The grill: the tiles still available to win, greyed out once taken.
    text("grill", 0.3 * GRID_SIZE, 0.45 * GRID_SIZE)
    tile_w = 1.3 * GRID_SIZE
    for tile in range(1, len(_TILE_VALS)):
        x, y = 0.3 * GRID_SIZE + (tile - 1) * tile_w, 0.6 * GRID_SIZE
        available = grill[tile]
        g.add(
            dwg.rect(
                (x, y),
                (tile_w * 0.9, 1.3 * GRID_SIZE),
                fill=color_set.grid_color,
                fill_opacity=0.0 if available else 0.25,
                stroke=color_set.grid_color,
                stroke_width="1.5px" if available else "0.5px",
                rx="4px",
                ry="4px",
            )
        )
        if not available:
            continue
        text(
            f"{int(_TILE_VALS[tile])}",
            x + tile_w * 0.45,
            y + 0.55 * GRID_SIZE,
            size=0.45,
            anchor="middle",
        )
        # The worms are what actually score; the number only buys the tile.
        worms = int(_WORM_VALS[tile])
        for w in range(worms):
            g.add(
                dwg.circle(
                    center=(
                        x + tile_w * 0.45 + (w - (worms - 1) / 2) * 0.22 * GRID_SIZE,
                        y + 0.95 * GRID_SIZE,
                    ),
                    r=0.07 * GRID_SIZE,
                    fill=color_set.text_color,
                )
            )

    # The dice: what was just rolled, and what has been set aside this turn.
    # A face may only be set aside once, so the kept row is the turn's history.
    side = 0.85 * GRID_SIZE
    for label, counts, row in (("rolled", rolled, 0), ("kept", taken, 1)):
        y = (2.4 + row * 1.15) * GRID_SIZE
        text(f"{label}:", 0.3 * GRID_SIZE, y + side * 0.65)
        x = 2.3 * GRID_SIZE
        for face, count in enumerate(counts):
            for _ in range(count):
                die(face, x, y, side)
                x += side * 1.3
        if row == 1:
            total = sum(_DICE_VALS[f] * n for f, n in enumerate(counts))
            worm = " + worm" if counts[0] > 0 else " (no worm yet)"
            text(f"= {total}{worm}", x + 0.2 * GRID_SIZE, y + side * 0.65)

    # Each player's stack: only the top tile is at risk on a bust, but every
    # tile's worms count at the end.
    text("stacks", 0.3 * GRID_SIZE, 5.2 * GRID_SIZE)
    for player, stack in enumerate(stacks):
        y = (5.8 + player * 0.75) * GRID_SIZE
        tiles = [t for t in stack if t != 0]
        worms = sum(int(_WORM_VALS[t]) for t in tiles)
        marker = ">" if player == mover and winner < 0 else " "
        top = f"top {int(_TILE_VALS[tiles[0]])}" if tiles else "empty"
        values = " ".join(str(int(_TILE_VALS[t])) for t in tiles[1:])
        text(
            f"{marker} player {player}: {worms} worms, {top}" + (f" over {values}" if values else ""),
            0.3 * GRID_SIZE,
            y,
        )

    if winner >= 0:
        text(f"player {winner} wins", 0.3 * GRID_SIZE, 8.4 * GRID_SIZE, size=0.5)

    return g
