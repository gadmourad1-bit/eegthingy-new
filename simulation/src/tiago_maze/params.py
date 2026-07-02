"""Simulation parameters.

The `ControlParams` defaults are a 1:1 copy of the ROS 2 parameters declared in
src/my_controller/src/safe_wall_teleop_webSocket.py; the robot constants come
from the generated TIAGo no-arm URDF (assets/tiago/tiago_no_arm.urdf); the
maze constants match the tiago_wall_course world (walls 1.6 m apart, 2.0 m
high, grey 0.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ControlParams:
    # --- safe_wall_teleop_webSocket.py parameter defaults ---
    # (forward/turn speeds are game defaults; the original node used 0.20/0.60)
    safe_dist: float = 0.8              # m, stop when front cone reads less
    forward_speed: float = 0.60         # m/s
    turn_speed: float = 0.75            # rad/s
    front_half_angle_deg: float = 15.0  # front cone half-angle
    yaw_tol_deg: float = 2.0            # turn completion tolerance
    heading_kp: float = 2.0             # forward heading-hold gain, rad/s per rad (0 = off)
    control_period: float = 0.05        # s (20 Hz control timer)
    status_period: float = 1.0          # s, periodic debug status print

    enable_ws: bool = True
    # The lab machine used ws://10.42.0.173:8765; default here to localhost,
    # same endpoint served by scripts/send_test_decisions.py.
    ws_url: str = "ws://127.0.0.1:8765"


@dataclass
class RobotParams:
    # From the generated URDF / PMB2 base.
    wheel_radius: float = 0.0985        # base_footprint -> base_link z offset
    wheel_separation: float = 0.4044    # wheel joints at y = +/-0.2022
    base_radius: float = 0.27           # PMB2 footprint radius (collision circle)
    laser_x: float = 0.202              # base_laser_joint origin xyz
    laser_z: float = -0.004             # (relative to base_link)
    base_link_z: float = 0.0985         # base_link height above ground

    # Simulated SICK TIM571 scan slice around the front (the controller only
    # ever looks at the +/-15 deg cone; we generate +/-30 deg of rays).
    scan_half_angle_deg: float = 30.0
    scan_angle_increment_deg: float = 0.33
    scan_range_min: float = 0.05
    scan_range_max: float = 25.0


@dataclass
class MazeParams:
    n_turns: int = 40                   # number of left/right decision corners
    cell: float = 1.6                   # corridor width, = wall spacing in tiago_wall_course
    wall_thickness: float = 0.05        # as in tiago_wall_course.world
    wall_height: float = 2.0            # as in tiago_wall_course.world
    min_gap: int = 1                    # min straight cells between corners
    max_gap: int = 3                    # max straight cells between corners
    seed: int | None = None


@dataclass
class GameParams:
    control: ControlParams = field(default_factory=ControlParams)
    robot: RobotParams = field(default_factory=RobotParams)
    maze: MazeParams = field(default_factory=MazeParams)
    manual_keys: bool = True            # arrow keys inject decisions locally
    view: str = "third"                 # camera: "third" | "first" | "top"
    fps: int = 60                       # frame-rate cap (physics is fixed-step)
    quiet: bool = False                 # suppress node-style console logging
    report_enabled: bool = True         # write a CSV report when the run ends
    report_dir: str = "reports"         # directory for CSV reports + NPZ traces
    record_trace: bool = True           # save an NPZ pose trace (50 ms samples)
    replay: str | None = None           # path to an NPZ trace to replay
    extra_meta: dict = field(default_factory=dict)  # user metadata -> CSV summary
    offscreen: bool = False             # render offscreen (testing)
    max_frames: int | None = None       # exit after N frames (testing)
    screenshot: str | None = None       # save a screenshot on exit (testing)

    # Fixed physics timestep (seconds). Integration + control run at this rate
    # regardless of display refresh, so behavior is identical at 60/144/240 Hz.
    phys_dt: float = 1.0 / 60.0
    trace_period: float = 0.05          # NPZ sampling period (50 ms)
