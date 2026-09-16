from pgx.domineering import State as DomineeringState


def _make_domineering_dwg(dwg, state: DomineeringState, config):
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
    board_g = dwg.g()

    # The game transposes the board after every move so that the player to move
    # always places horizontally, so an odd number of moves have been played
    # whenever it is player 1's turn. Undo that, or the drawn position mirrors
    # itself between one frame and the next.
    board = state._x.board
    mover = int(state._x.color)
    if mover == 1:
        board = board.T
    size = board.shape[0]

    # Occupied squares. Which player covered a square is not recorded, and
    # cannot be recovered - a 2x2 block is two horizontal dominoes or two
    # vertical ones - so they are all drawn alike.
    for row in range(size):
        for col in range(size):
            if bool(board[row, col]):
                continue
            board_g.add(
                dwg.rect(
                    (col * GRID_SIZE, row * GRID_SIZE),
                    (GRID_SIZE, GRID_SIZE),
                    fill=color_set.p2_color,
                )
            )

    grid = board_g.add(dwg.g(id="grid", stroke=color_set.grid_color))
    for i in range(size + 1):
        grid.add(
            dwg.line(
                start=(0, GRID_SIZE * i),
                end=(GRID_SIZE * size, GRID_SIZE * i),
                stroke_width="0.5px",
            )
        )
        grid.add(
            dwg.line(
                start=(GRID_SIZE * i, 0),
                end=(GRID_SIZE * i, GRID_SIZE * size),
                stroke_width="0.5px",
            )
        )
    board_g.add(
        dwg.rect(
            (0, 0),
            (size * GRID_SIZE, size * GRID_SIZE),
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="2px",
        )
    )

    # Which way the player to move must lay their domino is the whole game, and
    # nothing on the board itself shows it.
    winner = int(state._x.winner)
    if winner >= 0:
        caption = f"player {winner} wins"
    else:
        caption = f"player {mover} to play: " + ("horizontal" if mover == 0 else "vertical")
    board_g.add(
        dwg.text(
            text=caption,
            insert=(0, (size + 0.6) * GRID_SIZE),
            fill=color_set.text_color,
            font_size=f"{GRID_SIZE * 0.45}px",
            font_family="monospace",
        )
    )

    return board_g
