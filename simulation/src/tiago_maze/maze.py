"""Randomized corridor maze with a fixed number of left/right turn points.

The maze is a single self-avoiding corridor on a grid (cell = corridor width,
1.6 m — the same wall spacing as the tiago_wall_course world). The corridor
makes exactly ``n_turns`` 90-degree corners; at each corner exactly one side
is open, so the robot stops at the wall ahead and the player must pick the
correct direction. A wrong pick makes the robot face the corridor side wall
(< safe_dist away), so it stops again immediately and waits for a new
decision — the same behavior the original state machine exhibits in Gazebo.

Walls are emitted with the tiago_wall_course dimensions (0.05 m thick,
2.0 m high, grey 0.7).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .params import MazeParams

# Grid directions: 0=+x, 1=+y, 2=-x, 3=-y
DIR_VECS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass
class Wall:
    """Axis-aligned wall box (matches the SDF <box> walls of the world)."""

    cx: float
    cy: float
    sx: float  # size along x
    sy: float  # size along y
    height: float


@dataclass
class TurnPoint:
    cell: tuple[int, int]
    x: float
    y: float
    direction: str        # "LEFT" or "RIGHT" — the correct choice
    in_heading: int       # heading (0..3) when arriving
    out_heading: int      # heading (0..3) after the correct turn
    index: int            # 0-based turn number


@dataclass
class Maze:
    params: MazeParams
    cells: list[tuple[int, int]] = field(default_factory=list)
    turn_points: list[TurnPoint] = field(default_factory=list)
    walls: list[Wall] = field(default_factory=list)
    start_xy: tuple[float, float] = (0.0, 0.0)
    start_yaw: float = 0.0
    goal_xy: tuple[float, float] = (0.0, 0.0)
    seed: int = 0

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        c = self.params.cell
        return (cell[0] * c, cell[1] * c)

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [w.cx for w in self.walls]
        ys = [w.cy for w in self.walls]
        return min(xs), min(ys), max(xs), max(ys)


def _turn(heading: int, direction: str) -> int:
    # LEFT = +90 deg (counter-clockwise), RIGHT = -90 deg.
    return (heading + 1) % 4 if direction == "LEFT" else (heading - 1) % 4


def _neighbors8(cell: tuple[int, int]):
    x, y = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                yield (x + dx, y + dy)


def _carve_path(params: MazeParams, rng: random.Random) -> tuple[list[tuple[int, int]], list[int]]:
    """Backtracking search for a self-avoiding corridor with n_turns corners.

    Returns (cells, turn_cell_indices). Self-avoidance rule: a new cell may
    not coincide with or be 8-adjacent to any earlier path cell, except the
    two cells immediately preceding it. This guarantees parallel corridor
    sections never share a wall, so the laser always sees clean geometry.
    """
    n_turns = params.n_turns

    start = (0, 0)
    heading = 0

    cells = [start]
    turn_idx: list[int] = []
    # occupied includes a per-cell insertion order so we can check "last two".
    order = {start: 0}

    def can_place(cell: tuple[int, int]) -> bool:
        if cell in order:
            return False
        last = len(cells) - 1
        for nb in _neighbors8(cell):
            k = order.get(nb)
            # Allow the tail (k == last) and the cell before it (k == last-1,
            # the diagonal neighbor right after a corner); reject anything
            # older — that would make two corridor sections share a wall.
            if k is not None and k < last - 1:
                return False
        return True

    def extend(heading: int, run: int) -> int:
        """Try to append `run` cells straight ahead; returns count placed."""
        placed = 0
        for _ in range(run):
            dx, dy = DIR_VECS[heading]
            nxt = (cells[-1][0] + dx, cells[-1][1] + dy)
            if not can_place(nxt):
                break
            order[nxt] = len(cells)
            cells.append(nxt)
            placed += 1
        return placed

    def retract(count: int) -> None:
        for _ in range(count):
            order.pop(cells.pop())

    sys_limit = 200_000  # backtracking step budget (plenty; typical use ~1k)
    steps = 0

    def search(heading: int, turns_left: int) -> bool:
        nonlocal steps
        steps += 1
        if steps > sys_limit:
            return False
        if turns_left == 0:
            # Final straight run into the dead-end goal.
            gap = rng.randint(params.min_gap, params.max_gap)
            placed = extend(heading, gap)
            if placed >= params.min_gap:
                return True
            retract(placed)
            return False

        gaps = list(range(params.min_gap, params.max_gap + 1))
        rng.shuffle(gaps)
        for gap in gaps:
            placed = extend(heading, gap)
            if placed < gap:
                retract(placed)
                continue
            dirs = ["LEFT", "RIGHT"]
            rng.shuffle(dirs)
            ok = False
            for d in dirs:
                turn_idx.append(len(cells) - 1)
                if search(_turn(heading, d), turns_left - 1):
                    ok = True
                    break
                turn_idx.pop()
            if ok:
                return True
            retract(placed)
        return False

    # First run out of the start cell.
    first = extend(heading, rng.randint(params.min_gap + 1, params.max_gap + 1))
    if first < 1 or not search(heading, n_turns):
        raise RuntimeError("maze search failed (exhausted step budget)")

    return cells, turn_idx


def generate(params: MazeParams | None = None) -> Maze:
    params = params or MazeParams()
    seed = params.seed if params.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    rng = random.Random(seed)

    cells, turn_idx = _carve_path(params, rng)

    maze = Maze(params=params, cells=cells, seed=seed)

    # --- turn point metadata ---
    c = params.cell
    for i, ti in enumerate(turn_idx):
        prev_c, cur, nxt = cells[ti - 1], cells[ti], cells[ti + 1]
        in_vec = (cur[0] - prev_c[0], cur[1] - prev_c[1])
        out_vec = (nxt[0] - cur[0], nxt[1] - cur[1])
        in_h = DIR_VECS.index(in_vec)
        out_h = DIR_VECS.index(out_vec)
        direction = "LEFT" if (in_h + 1) % 4 == out_h else "RIGHT"
        maze.turn_points.append(
            TurnPoint(
                cell=cur,
                x=cur[0] * c,
                y=cur[1] * c,
                direction=direction,
                in_heading=in_h,
                out_heading=out_h,
                index=i,
            )
        )

    # --- walls: every cell edge not connecting consecutive path cells ---
    # Edge key: ((x, y), axis) = edge between cell (x,y) and its +x / +y
    # neighbor (axis 0 -> wall plane x = (x+1)*c - c/2 ... i.e. east edge).
    connected: set[tuple[tuple[int, int], int]] = set()

    def edge_key(a: tuple[int, int], b: tuple[int, int]):
        if b[0] - a[0] == 1:
            return (a, 0)
        if a[0] - b[0] == 1:
            return (b, 0)
        if b[1] - a[1] == 1:
            return (a, 1)
        return (b, 1)

    for a, b in zip(cells, cells[1:]):
        connected.add(edge_key(a, b))

    emitted: set[tuple[tuple[int, int], int]] = set()
    t = params.wall_thickness
    for cell in cells:
        x, y = cell
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            key = edge_key(cell, nb)
            if key in connected or key in emitted:
                continue
            emitted.add(key)
            (ex, ey), axis = key
            if axis == 0:  # wall between (ex,ey) and (ex+1,ey): plane x const
                maze.walls.append(
                    Wall(
                        cx=(ex + 0.5) * c,
                        cy=ey * c,
                        sx=t,
                        sy=c + t,  # overlap corners so no gaps
                        height=params.wall_height,
                    )
                )
            else:  # wall between (ex,ey) and (ex,ey+1): plane y const
                maze.walls.append(
                    Wall(
                        cx=ex * c,
                        cy=(ey + 0.5) * c,
                        sx=c + t,
                        sy=t,
                        height=params.wall_height,
                    )
                )

    # --- start / goal ---
    maze.start_xy = maze.cell_center(cells[0])
    dx, dy = cells[1][0] - cells[0][0], cells[1][1] - cells[0][1]
    maze.start_yaw = math.atan2(dy, dx)
    maze.goal_xy = maze.cell_center(cells[-1])

    return maze
