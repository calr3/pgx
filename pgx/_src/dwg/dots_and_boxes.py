# Copyright 2026 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pgx._src.games.dots_and_boxes import SIZE, h_line, v_line
from pgx.dots_and_boxes import State as DotsAndBoxesState


def _make_dots_and_boxes_dwg(dwg, state: DotsAndBoxesState, config):
    GRID_SIZE = config["GRID_SIZE"]
    WIDTH = config["BOARD_WIDTH"]
    HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

    dwg.add(
        dwg.rect((0, 0), (WIDTH * GRID_SIZE, HEIGHT * GRID_SIZE), fill=color_set.background_color)
    )
    g = dwg.g()

    lines = [bool(v) for v in state._x.lines]
    owner = [int(v) for v in state._x.owner]
    fills = [color_set.p1_color, color_set.p2_color]
    margin = GRID_SIZE

    def dot(r, c):
        return (margin + c * GRID_SIZE, margin + r * GRID_SIZE)

    # Boxes, filled in their owner's colour (colour 0 draws first).
    for r in range(SIZE):
        for c in range(SIZE):
            o = owner[r * SIZE + c]
            if o >= 0:
                x, y = dot(r, c)
                g.add(dwg.rect((x, y), (GRID_SIZE, GRID_SIZE), fill=fills[o], fill_opacity=0.45))

    # Lines: drawn ones solid, the rest faint so the grid is still readable.
    def line(start, end, drawn):
        g.add(
            dwg.line(
                start=start,
                end=end,
                stroke=color_set.grid_color,
                stroke_width="4px" if drawn else "1px",
                stroke_opacity=1.0 if drawn else 0.15,
                stroke_linecap="round",
            )
        )

    for r in range(SIZE + 1):
        for c in range(SIZE):
            line(dot(r, c), dot(r, c + 1), lines[h_line(r, c)])
    for r in range(SIZE):
        for c in range(SIZE + 1):
            line(dot(r, c), dot(r + 1, c), lines[v_line(r, c)])

    for r in range(SIZE + 1):
        for c in range(SIZE + 1):
            g.add(dwg.circle(center=dot(r, c), r=GRID_SIZE / 10, fill=color_set.grid_color))

    # Box counts, with a marker on the colour to move.
    mover = int(state._x.color)
    over = bool(state.terminated)
    y = margin + (SIZE + 0.9) * GRID_SIZE
    for colour in range(2):
        marker = ">" if colour == mover and not over else " "
        x = margin + colour * (SIZE / 2) * GRID_SIZE
        g.add(dwg.rect((x, y - 0.35 * GRID_SIZE), (0.4 * GRID_SIZE, 0.4 * GRID_SIZE), fill=fills[colour], fill_opacity=0.45))
        g.add(
            dwg.text(
                text=f"{marker}{owner.count(colour)}",
                insert=(x + 0.5 * GRID_SIZE, y),
                fill=color_set.text_color,
                font_size=f"{GRID_SIZE * 0.45}px",
                font_family="monospace",
            )
        )
    return g
