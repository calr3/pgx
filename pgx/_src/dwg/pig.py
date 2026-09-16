from pgx._src.games.pig import TARGET
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
    winner = int(state._x.winner)

    def text(s, x, y, anchor="start", size=0.5):
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

    # One row per player: the banked total, and a bar of its progress to 100.
    # The mover's bar is extended by the unbanked turn total, drawn faintly,
    # which is the whole tension of the game - that part is lost on a 1.
    bar_x, bar_w = 0.5 * GRID_SIZE, (WIDTH - 1) * GRID_SIZE
    for player, total in enumerate(totals):
        top = (0.3 + player * 1.7) * GRID_SIZE
        marker = ">" if player == mover and winner < 0 else " "
        text(f"{marker} Player {player}", bar_x, top + 0.55 * GRID_SIZE)
        text(f"{total}", bar_x + bar_w, top + 0.55 * GRID_SIZE, anchor="end")

        bar_y, bar_h = top + 0.75 * GRID_SIZE, 0.45 * GRID_SIZE
        g.add(
            dwg.rect(
                (bar_x, bar_y),
                (bar_w, bar_h),
                fill="none",
                stroke=color_set.grid_color,
                stroke_width="1px",
            )
        )
        g.add(
            dwg.rect(
                (bar_x, bar_y),
                (bar_w * min(total, TARGET) / TARGET, bar_h),
                fill=color_set.grid_color,
            )
        )
        if player == mover and turn_total > 0 and winner < 0:
            pending = min(total + turn_total, TARGET) - min(total, TARGET)
            g.add(
                dwg.rect(
                    (bar_x + bar_w * min(total, TARGET) / TARGET, bar_y),
                    (bar_w * pending / TARGET, bar_h),
                    fill=color_set.grid_color,
                    fill_opacity=0.4,
                )
            )

    # The die just rolled. A turn starts with no roll, so the face is blank.
    die = 1.6 * GRID_SIZE
    ox, oy = 0.5 * GRID_SIZE, (0.3 + len(totals) * 1.7 + 0.2) * GRID_SIZE
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

    # The turn total is stale once the game is over: it belongs to the turn that
    # just ended, and the mover has already flipped to the loser.
    caption = f"player {winner} wins" if winner >= 0 else f"turn total: {turn_total}"
    text(caption, ox + die + 0.4 * GRID_SIZE, oy + 0.9 * GRID_SIZE)

    return g
