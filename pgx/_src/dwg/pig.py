from pgx.pig import State as PigState

# Pips of a die face, as (column, row) offsets within a 3x3 grid.
_PIPS = {
    1: [(1, 1)],
    2: [(0, 0), (2, 2)],
    3: [(0, 0), (1, 1), (2, 2)],
    4: [(0, 0), (2, 0), (0, 2), (2, 2)],
    5: [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
    6: [(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)],
}


def _make_pig_dwg(dwg, state: PigState, config):
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

    totals = [int(t) for t in state._x.totals]
    turn_total = int(state._x.turn_total)
    last_roll = int(state._x.last_roll)
    mover = int(state._x.color)

    # Scores, with an arrow marking whose turn it is.
    for player, total in enumerate(totals):
        y = (1.2 + player * 1.2) * GRID_SIZE
        label = f"{'>' if player == mover else ' '} Player {player}: {total}"
        g.add(
            dwg.text(
                text=label,
                insert=(0.5 * GRID_SIZE, y),
                fill=color_set.text_color,
                font_size=f"{GRID_SIZE * 0.6}px",
                font_family="monospace",
            )
        )

    g.add(
        dwg.text(
            text=f"turn total: {turn_total}",
            insert=(0.5 * GRID_SIZE, (1.2 + len(totals) * 1.2) * GRID_SIZE),
            fill=color_set.text_color,
            font_size=f"{GRID_SIZE * 0.6}px",
            font_family="monospace",
        )
    )

    # The die just rolled.
    die = GRID_SIZE * 2
    ox, oy = 0.5 * GRID_SIZE, (1.2 + (len(totals) + 1) * 1.2) * GRID_SIZE
    g.add(
        dwg.rect(
            (ox, oy),
            (die, die),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width="2px",
            rx="6px",
            ry="6px",
        )
    )
    for col, row in _PIPS.get(last_roll, []):
        g.add(
            dwg.circle(
                center=(ox + (col + 0.5) * die / 3, oy + (row + 0.5) * die / 3),
                r=die / 12,
                fill=color_set.grid_color,
            )
        )
    return g
