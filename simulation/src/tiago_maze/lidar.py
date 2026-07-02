"""2D laser scan simulation against the maze walls.

Produces a LaserScanMsg-compatible slice around the robot's forward axis so
the ported controller can run its original front-cone windowing math
unchanged. The rays originate at the base laser frame (x=+0.202 m from the
base center, like base_laser_joint in the URDF) and are cast against the
axis-aligned wall boxes with a vectorized slab test.
"""

from __future__ import annotations

import math

import numpy as np

from .controller import LaserScanMsg
from .maze import Wall
from .params import RobotParams


class Lidar:
    def __init__(self, walls: list[Wall], params: RobotParams | None = None):
        self.params = params or RobotParams()
        p = self.params

        self.angle_min = -math.radians(p.scan_half_angle_deg)
        self.angle_increment = math.radians(p.scan_angle_increment_deg)
        n = int(round(2 * math.radians(p.scan_half_angle_deg) / self.angle_increment)) + 1
        self.n_rays = n
        self._rel_angles = self.angle_min + self.angle_increment * np.arange(n)

        self.set_walls(walls)

    def set_walls(self, walls: list[Wall]) -> None:
        # Wall AABBs as (N, 4): xmin, xmax, ymin, ymax
        if walls:
            self._aabbs = np.array(
                [
                    (
                        w.cx - w.sx / 2.0,
                        w.cx + w.sx / 2.0,
                        w.cy - w.sy / 2.0,
                        w.cy + w.sy / 2.0,
                    )
                    for w in walls
                ]
            )
        else:
            self._aabbs = np.zeros((0, 4))

    def raycast(self, ox: float, oy: float, dx: float, dy: float) -> float:
        """Distance to the nearest wall along one ray (inf if none)."""
        if len(self._aabbs) == 0:
            return float("inf")
        xmin, xmax, ymin, ymax = self._aabbs.T
        with np.errstate(divide="ignore", invalid="ignore"):
            # Parallel-ray slabs: inside -> (-inf, +inf) passes through the
            # min/max below unchanged; outside -> (+inf, +inf) forces
            # tmin=+inf, i.e. a miss. (Encoding (+inf, -inf) would be undone
            # by the min/max and turn misses into hits.)
            if abs(dx) < 1e-12:
                inside = (ox >= xmin) & (ox <= xmax)
                t1 = np.where(inside, -np.inf, np.inf)
                t2 = np.full_like(t1, np.inf)
            else:
                t1 = (xmin - ox) / dx
                t2 = (xmax - ox) / dx
            if abs(dy) < 1e-12:
                inside = (oy >= ymin) & (oy <= ymax)
                t3 = np.where(inside, -np.inf, np.inf)
                t4 = np.full_like(t3, np.inf)
            else:
                t3 = (ymin - oy) / dy
                t4 = (ymax - oy) / dy
        tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
        tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
        hit = (tmax >= tmin) & (tmax >= 0.0)
        t = np.where(tmin >= 0.0, tmin, tmax)
        t = np.where(hit, t, np.inf)
        return float(t.min())

    def scan(self, x: float, y: float, yaw: float) -> LaserScanMsg:
        p = self.params
        # Laser frame origin in world coordinates.
        ox = x + p.laser_x * math.cos(yaw)
        oy = y + p.laser_x * math.sin(yaw)

        angles = yaw + self._rel_angles
        dx = np.cos(angles)  # (R,)
        dy = np.sin(angles)

        if len(self._aabbs) == 0:
            ranges = np.full(self.n_rays, np.inf)
        else:
            xmin, xmax, ymin, ymax = self._aabbs.T  # each (W,)

            with np.errstate(divide="ignore", invalid="ignore"):
                inv_dx = 1.0 / dx
                inv_dy = 1.0 / dy

                # Slab intersections, broadcast rays (R,1) against walls (W,).
                t1 = (xmin - ox) * inv_dx[:, None]
                t2 = (xmax - ox) * inv_dx[:, None]
                t3 = (ymin - oy) * inv_dy[:, None]
                t4 = (ymax - oy) * inv_dy[:, None]

                # Rays parallel to a slab: inside -> (-inf, +inf) (no
                # constraint), outside -> (+inf, +inf) (miss survives the
                # min/max reduction below).
                para_x = np.abs(dx) < 1e-12
                if para_x.any():
                    inside = (ox >= xmin) & (ox <= xmax)  # (W,)
                    t1[para_x] = np.where(inside, -np.inf, np.inf)
                    t2[para_x] = np.inf
                para_y = np.abs(dy) < 1e-12
                if para_y.any():
                    inside = (oy >= ymin) & (oy <= ymax)
                    t3[para_y] = np.where(inside, -np.inf, np.inf)
                    t4[para_y] = np.inf

                tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
                tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))

            hit = (tmax >= tmin) & (tmax >= 0.0)
            t = np.where(tmin >= 0.0, tmin, tmax)  # if origin inside box, exit dist
            t = np.where(hit, t, np.inf)
            ranges = t.min(axis=1)

        return LaserScanMsg(
            angle_min=self.angle_min,
            angle_increment=self.angle_increment,
            range_min=p.scan_range_min,
            range_max=p.scan_range_max,
            ranges=ranges.tolist(),
        )
