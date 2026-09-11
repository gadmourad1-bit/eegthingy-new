"""Maze generator invariants."""

import math

import pytest

from tiago_maze import maze as maze_mod
from tiago_maze.params import MazeParams


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 12345])
def test_forty_turn_points(seed):
    m = maze_mod.generate(MazeParams(n_turns=40, seed=seed))
    assert len(m.turn_points) == 40
    for i, tp in enumerate(m.turn_points):
        assert tp.index == i
        assert tp.direction in ("LEFT", "RIGHT")
        # correct exit is a real 90-degree turn
        assert (tp.in_heading + (1 if tp.direction == "LEFT" else -1)) % 4 == tp.out_heading


@pytest.mark.parametrize("seed", [3, 99, 2024])
def test_path_is_self_avoiding(seed):
    m = maze_mod.generate(MazeParams(n_turns=40, seed=seed))
    cells = m.cells
    assert len(set(cells)) == len(cells), "path revisits a cell"
    # Non-consecutive cells must never be 8-adjacent (no shared walls).
    idx = {c: i for i, c in enumerate(cells)}
    for i, (x, y) in enumerate(cells):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                j = idx.get((x + dx, y + dy))
                if j is not None:
                    assert abs(i - j) <= 2, f"cells {i} and {j} touch"


@pytest.mark.parametrize("seed", [5, 11])
def test_corridor_walls_enclose_path(seed):
    """Every path cell has walls exactly on its unconnected edges."""
    m = maze_mod.generate(MazeParams(n_turns=40, seed=seed))
    c = m.params.cell

    wall_keys = set()
    for w in m.walls:
        if w.sx < w.sy:  # x-plane wall between (ex,ey) and (ex+1,ey)
            ex = round(w.cx / c - 0.5)
            ey = round(w.cy / c)
            wall_keys.add(((ex, ey), 0))
        else:
            ex = round(w.cx / c)
            ey = round(w.cy / c - 0.5)
            wall_keys.add(((ex, ey), 1))
    assert len(wall_keys) == len(m.walls), "duplicate walls"

    def edge_key(a, b):
        if b[0] - a[0] == 1:
            return (a, 0)
        if a[0] - b[0] == 1:
            return (b, 0)
        if b[1] - a[1] == 1:
            return (a, 1)
        return (b, 1)

    connected = set()
    for a, b in zip(m.cells, m.cells[1:]):
        connected.add(edge_key(a, b))

    for cell in m.cells:
        x, y = cell
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            key = edge_key(cell, nb)
            if key in connected:
                assert key not in wall_keys, f"wall blocks the corridor at {key}"
            else:
                assert key in wall_keys, f"missing wall at {key}"


def test_seed_reproducible():
    a = maze_mod.generate(MazeParams(n_turns=40, seed=77))
    b = maze_mod.generate(MazeParams(n_turns=40, seed=77))
    assert a.cells == b.cells
    assert [(w.cx, w.cy, w.sx, w.sy) for w in a.walls] == [
        (w.cx, w.cy, w.sx, w.sy) for w in b.walls
    ]


def test_start_heading_points_down_corridor():
    m = maze_mod.generate(MazeParams(n_turns=10, seed=4))
    dx = m.cells[1][0] - m.cells[0][0]
    dy = m.cells[1][1] - m.cells[0][1]
    assert math.isclose(m.start_yaw, math.atan2(dy, dx))


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_true_labels_can_force_the_exact_maze_route(seed):
    """The replay maze must follow labels exactly: 1=left and 2=right."""
    sequence = ("LEFT", "RIGHT", "RIGHT", "LEFT", "LEFT", "RIGHT")
    m = maze_mod.generate(MazeParams(
        n_turns=len(sequence), seed=seed, turn_sequence=sequence
    ))
    assert tuple(tp.direction for tp in m.turn_points) == sequence
