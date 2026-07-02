"""End-to-end headless episode: the robot solves a full 40-turn maze.

Runs the exact game physics (unicycle integration + lidar + ported control
loop) without Panda3D. A scripted oracle plays the decision server: whenever
the robot stops, it submits the maze's correct direction — exactly what a
websocket client would do.
"""

import math

from tiago_maze import lidar as lidar_mod
from tiago_maze import maze as maze_mod
from tiago_maze.controller import Logger, SafeWallTeleop, wrap_pi
from tiago_maze.params import ControlParams, MazeParams, RobotParams


def run_episode(seed: int, n_turns: int = 40, wrong_first: bool = False):
    maze = maze_mod.generate(MazeParams(n_turns=n_turns, seed=seed))
    lidar = lidar_mod.Lidar(maze.walls, RobotParams())
    ctrl = SafeWallTeleop(ControlParams(), logger=Logger(enabled=False))

    x, y = maze.start_xy
    yaw = maze.start_yaw
    dt = 0.05  # step at the control period

    wrong_pending = wrong_first
    completed = False

    def correct_decision() -> int:
        # Nearest turn point tells us the correct direction; at the goal
        # there is none.
        best, best_d = None, float("inf")
        for tp in maze.turn_points:
            d = math.hypot(x - tp.x, y - tp.y)
            if d < best_d:
                best, best_d = tp, d
        if best is None or best_d > maze.params.cell:
            return 0
        heading = round(yaw / (math.pi / 2)) % 4
        want = best.out_heading
        if heading == want:
            return 0  # already facing the right way (post-correction)
        if (heading + 1) % 4 == want:
            return 1  # LEFT
        if (heading - 1) % 4 == want:
            return 2  # RIGHT
        return 1  # facing backwards: two lefts will fix it

    max_steps = 200_000
    for _ in range(max_steps):
        scan = lidar.scan(x, y, yaw)
        ctrl.on_scan(scan)
        ctrl.on_odom(yaw)
        ctrl.control_loop()

        if ctrl.state == "STOPPED_WAITING_EEG":
            gd = math.hypot(x - maze.goal_xy[0], y - maze.goal_xy[1])
            if gd < 0.9:
                completed = True
                break
            want = correct_decision()
            if wrong_pending and want in (1, 2):
                ctrl.submit_decision(2 if want == 1 else 1)  # deliberately wrong
                wrong_pending = False
            elif want in (1, 2):
                ctrl.submit_decision(want)

        vx, wz = ctrl.cmd
        x += vx * math.cos(yaw) * dt
        y += vx * math.sin(yaw) * dt
        yaw = wrap_pi(yaw + wz * dt)

    return completed, x, y, yaw, maze


def test_episode_completes_40_turns():
    completed, x, y, yaw, maze = run_episode(seed=42, n_turns=40)
    assert completed, "robot did not reach the goal"
    assert math.hypot(x - maze.goal_xy[0], y - maze.goal_xy[1]) < 0.9


def test_episode_other_seed():
    completed, *_ = run_episode(seed=7, n_turns=40)
    assert completed


def test_wrong_decision_recovers():
    """A wrong turn leaves the robot facing a wall; the oracle corrects it."""
    completed, *_ = run_episode(seed=42, n_turns=10, wrong_first=True)
    assert completed


def test_small_maze():
    completed, *_ = run_episode(seed=3, n_turns=2)
    assert completed
