from pgx.gess import State as GessState


def _make_gess_dwg(dwg, state: GessState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

    # Row 0 is white's home, so it is drawn at the top and black (the first
    # player) sits at the bottom, as in the usual diagrams.
    def cell_xy(row, col):
        return col * GRID_SIZE, row * GRID_SIZE

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (BOARD_WIDTH * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
            fill=color_set.background_color,
        )
    )

    board_g = dwg.g()

    # The outermost ring is not playing area: a piece centre may sit there, but
    # any stone landing on it is removed. Shade it so the two are told apart.
    for x, y, w, h in (
        (0, 0, BOARD_WIDTH, 1),
        (0, BOARD_HEIGHT - 1, BOARD_WIDTH, 1),
        (0, 1, 1, BOARD_HEIGHT - 2),
        (BOARD_WIDTH - 1, 1, 1, BOARD_HEIGHT - 2),
    ):
        board_g.add(
            dwg.rect(
                (x * GRID_SIZE, y * GRID_SIZE),
                (w * GRID_SIZE, h * GRID_SIZE),
                fill=color_set.p2_outline,
                fill_opacity=0.12,
            )
        )

    hlines = board_g.add(dwg.g(id="hlines", stroke=color_set.grid_color))
    for y in range(BOARD_HEIGHT + 1):
        hlines.add(
            dwg.line(
                start=(0, GRID_SIZE * y),
                end=(GRID_SIZE * BOARD_WIDTH, GRID_SIZE * y),
                stroke_width="0.5px",
            )
        )
    vlines = board_g.add(dwg.g(id="vlines", stroke=color_set.grid_color))
    for x in range(BOARD_WIDTH + 1):
        vlines.add(
            dwg.line(
                start=(GRID_SIZE * x, 0),
                end=(GRID_SIZE * x, GRID_SIZE * BOARD_HEIGHT),
                stroke_width="0.5px",
            )
        )

    # the playing area (the inner 18x18)
    board_g.add(
        dwg.rect(
            (GRID_SIZE, GRID_SIZE),
            ((BOARD_WIDTH - 2) * GRID_SIZE, (BOARD_HEIGHT - 2) * GRID_SIZE),
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="2px",
        )
    )

    # A move takes two steps; in stage 1 the source piece has been chosen, so
    # mark its 3x3 footprint and centre.
    if int(state._x.stage) == 1:
        src = int(state._x.source)
        src_row, src_col = src // BOARD_WIDTH, src % BOARD_WIDTH
        x, y = cell_xy(src_row - 1, src_col - 1)
        board_g.add(
            dwg.rect(
                (x, y),
                (3 * GRID_SIZE, 3 * GRID_SIZE),
                fill=color_set.grid_color,
                fill_opacity=0.15,
                stroke=color_set.grid_color,
                stroke_width="2px",
            )
        )
    # stones
    for i, stone in enumerate(state._x.board):
        if stone == 0:
            continue
        row, col = i // BOARD_WIDTH, i % BOARD_WIDTH
        x, y = cell_xy(row, col)
        color = color_set.p1_color if stone == 1 else color_set.p2_color
        outline = color_set.p1_outline if stone == 1 else color_set.p2_outline
        board_g.add(
            dwg.circle(
                center=(x + GRID_SIZE / 2, y + GRID_SIZE / 2),
                r=GRID_SIZE / 2.4,
                stroke=outline,
                fill=color,
            )
        )

    # The source centre goes on top of the stones: it is often occupied, and the
    # centre is what the stage-1 action is measured from.
    if int(state._x.stage) == 1:
        src = int(state._x.source)
        x, y = cell_xy(src // BOARD_WIDTH, src % BOARD_WIDTH)
        board_g.add(
            dwg.circle(
                center=(x + GRID_SIZE / 2, y + GRID_SIZE / 2),
                r=GRID_SIZE / 7,
                fill="none",
                stroke="crimson",
                stroke_width="2px",
            )
        )

    return board_g
