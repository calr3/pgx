from pgx.g_hex import State as GHexState


def _make_g_hex_dwg(dwg, state: GHexState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (BOARD_WIDTH * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
            # stroke=svgwrite.rgb(10, 10, 16, "%"),
            fill=color_set.background_color,
        )
    )

    # board
    # grid
    board_g = dwg.g()

    vlines = board_g.add(dwg.g(id="vline", stroke=color_set.text_color))
    for x in range(1, BOARD_WIDTH):
        vlines.add(
            dwg.line(
                start=(GRID_SIZE * x, 0),
                end=(GRID_SIZE * x, GRID_SIZE * (BOARD_HEIGHT - 1)),
                stroke_width="1px",
            )
        )
    hlines = board_g.add(dwg.g(id="vline", stroke=color_set.text_color))
    for y in range(1, BOARD_HEIGHT):
        hlines.add(
            dwg.line(
                start=(0, GRID_SIZE * y),
                end=(GRID_SIZE * BOARD_WIDTH, GRID_SIZE * y),
                stroke_width="0.1px",
            )
        )

    _width = 6
    board_g.add(
        dwg.rect(
            (0, (BOARD_HEIGHT - 1) * GRID_SIZE),
            (
                BOARD_WIDTH * GRID_SIZE,
                _width,
            ),
            fill=color_set.grid_color,
            stroke=color_set.grid_color,
        )
    )
    board_g.add(
        dwg.rect(
            (-_width, 0),
            (
                _width,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill=color_set.grid_color,
            stroke=color_set.grid_color,
        )
    )

    board_g.add(
        dwg.rect(
            (GRID_SIZE * BOARD_WIDTH, 0),
            (
                _width,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill=color_set.grid_color,
            stroke=color_set.grid_color,
        )
    )

    return board_g
