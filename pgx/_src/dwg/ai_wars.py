from pgx.ai_wars import State as AIWarsState
from pgx._src.games import ai_wars as W

BUG_COLORS = ["#1f77d0", "#d03030", "#d0b000", "#20a050", "#c040c0", "#e08000", "#40c0c0",
              "#806040", "#808080", "#000000"]


def _make_ai_wars_dwg(dwg, state: AIWarsState, config):
    G = config["GRID_SIZE"]
    color_set = config["COLOR_SET"]
    x = state._x
    grid = W.Engine.board(x)
    g = dwg.g()
    g.add(dwg.rect((0, 0), (W.COLS * G, W.ROWS * G), fill=color_set.background_color))
    for r in range(W.ROWS):
        for c in range(W.COLS):
            v = int(grid[r, c])
            if v == W.WALL:
                g.add(dwg.rect((c * G, r * G), (G, G), fill=color_set.grid_color))
            elif v == W.FLAG:
                g.add(dwg.polygon([(c * G + 2, r * G + 2), (c * G + G - 2, r * G + G / 2), (c * G + 2, r * G + G - 2)],
                                  fill="#20a050"))
            elif v == W.MINE:
                g.add(dwg.circle((c * G + G / 2, r * G + G / 2), G / 4, fill="#d03030"))
    if bool(x.node_on):
        g.add(dwg.rect((int(x.node_x) * G + 1, int(x.node_y) * G + 1), (G - 2, G - 2), fill="none",
                       stroke="#c040c0", stroke_width="2px"))
    for b in range(x.x.shape[0]):
        col = BUG_COLORS[b % len(BUG_COLORS)]
        if int(x.mis_on[b]):
            g.add(dwg.circle((int(x.mis_x[b]) * G + G / 2, int(x.mis_y[b]) * G + G / 2), G / 5, fill=col))
        if int(x.gr_on[b]):
            g.add(dwg.rect((int(x.gr_x[b]) * G + G / 3, int(x.gr_y[b]) * G + G / 3), (G / 3, G / 3), fill=col))
        if int(x.dead[b]):
            continue
        cx, cy = int(x.x[b]) * G + G / 2, int(x.y[b]) * G + G / 2
        g.add(dwg.circle((cx, cy), G / 2 - 1, fill=col, stroke=color_set.p1_outline, stroke_width="1px"))
        h = int(x.h[b])
        dx, dy = int(W.DX[h]), int(W.DY[h])
        g.add(dwg.line((cx, cy), (cx + dx * G / 2, cy + dy * G / 2), stroke=color_set.p1_outline, stroke_width="2px"))
    return g
