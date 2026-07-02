"""Lidar raycast correctness, including exact axis-aligned rays."""

import math

from tiago_maze.lidar import Lidar
from tiago_maze.maze import Wall
from tiago_maze.params import RobotParams


def corridor_walls():
    """A +x corridor: side walls flanking y=+/-0.8, end wall at x=5.0."""
    return [
        Wall(cx=2.5, cy=0.8, sx=5.0, sy=0.05, height=2.0),   # north side
        Wall(cx=2.5, cy=-0.8, sx=5.0, sy=0.05, height=2.0),  # south side
        Wall(cx=5.0, cy=0.0, sx=0.05, sy=1.65, height=2.0),  # end wall
    ]


def test_axis_aligned_ray_ignores_flanking_walls():
    """A ray along the corridor axis (dy == 0 exactly) must hit only the
    end wall — regression test for the parallel-slab encoding bug."""
    lid = Lidar(corridor_walls(), RobotParams())
    d = lid.raycast(0.0, 0.0, 1.0, 0.0)
    assert math.isclose(d, 5.0 - 0.025, abs_tol=1e-9)

    # Backwards: open corridor, no wall -> inf.
    d_back = lid.raycast(0.0, 0.0, -1.0, 0.0)
    assert d_back == float("inf")


def test_axis_aligned_ray_inside_slab_hits():
    lid = Lidar(corridor_walls(), RobotParams())
    # Ray along +y from the center: hits the north side wall face.
    d = lid.raycast(2.5, 0.0, 0.0, 1.0)
    assert math.isclose(d, 0.8 - 0.025, abs_tol=1e-9)


def test_scan_front_distance_matches_geometry():
    lid = Lidar(corridor_walls(), RobotParams())
    scan = lid.scan(3.0, 0.0, 0.0)
    # Center ray: laser sits 0.202 ahead of the robot center.
    center_idx = min(
        range(len(scan.ranges)),
        key=lambda i: abs(scan.angle_min + i * scan.angle_increment),
    )
    expected = (5.0 - 0.025) - (3.0 + 0.202)
    assert abs(scan.ranges[center_idx] - expected) < 0.01


def test_scan_oblique_side_wall_distance():
    """Away from the end wall, the +/-15 deg cone minimum is the side wall
    seen obliquely: (0.8 - 0.025) / sin(15 deg) ~= 2.99 m."""
    lid = Lidar(
        [
            Wall(cx=0.0, cy=0.8, sx=100.0, sy=0.05, height=2.0),
            Wall(cx=0.0, cy=-0.8, sx=100.0, sy=0.05, height=2.0),
        ],
        RobotParams(),
    )
    scan = lid.scan(0.0, 0.0, 0.0)
    n = len(scan.ranges)
    # Only consider the controller's window (+/-15 deg).
    lo = math.radians(-15)
    vals = [
        r
        for i, r in enumerate(scan.ranges)
        if abs(scan.angle_min + i * scan.angle_increment) <= math.radians(15)
    ]
    expected = (0.8 - 0.025) / math.sin(math.radians(15))
    assert abs(min(vals) - expected) < 0.05
    assert n > len(vals) > 0
