from pgx.epaminondas import State as EpaminondasState


def _make_epaminondas_dwg(dwg, state: EpaminondasState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (BOARD_WIDTH * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
            fill=color_set.background_color,
        )
    )

    board_g = dwg.g()
    board_g.add(
        dwg.rect(
            (0, 0),
            (BOARD_WIDTH * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width="2px",
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

    # The back ranks (the players' goals) are marked with a thicker line.
    for y in (0, BOARD_HEIGHT):
        board_g.add(
            dwg.line(
                start=(0, GRID_SIZE * y),
                end=(GRID_SIZE * BOARD_WIDTH, GRID_SIZE * y),
                stroke=color_set.grid_color,
                stroke_width="3px",
            )
        )

    # pieces: row 0 (white's back rank) is drawn at the bottom
    board = state._x.board
    for i, stone in enumerate(board):
        if stone == 0:
            continue
        row, col = int(i) // BOARD_WIDTH, int(i) % BOARD_WIDTH
        cx = col * GRID_SIZE + GRID_SIZE / 2
        cy = (BOARD_HEIGHT - 1 - row) * GRID_SIZE + GRID_SIZE / 2
        color = color_set.p1_color if stone == 1 else color_set.p2_color
        outline = color_set.p1_outline if stone == 1 else color_set.p2_outline
        board_g.add(
            dwg.circle(
                center=(cx, cy),
                r=GRID_SIZE / 2.4,
                stroke=outline,
                fill=color,
            )
        )

    return board_g
