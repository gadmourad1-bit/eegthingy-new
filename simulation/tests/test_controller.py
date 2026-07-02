"""Fidelity tests for the ported safe-wall-teleop state machine."""

import math

from tiago_maze.controller import LaserScanMsg, Logger, SafeWallTeleop, wrap_pi
from tiago_maze.params import ControlParams


def make_ctrl():
    # Pin the speeds so the numeric assertions below are independent of the
    # game's default forward/turn speeds.
    params = ControlParams(forward_speed=0.20, turn_speed=0.60)
    ctrl = SafeWallTeleop(params, logger=Logger(enabled=False))
    return ctrl


def scan_with_front(dist: float) -> LaserScanMsg:
    """A scan whose whole front cone reads `dist`."""
    n = 181
    inc = math.radians(0.33)
    return LaserScanMsg(
        angle_min=-inc * (n // 2),
        angle_increment=inc,
        range_min=0.05,
        range_max=25.0,
        ranges=[dist] * n,
    )


def test_stops_without_scan():
    ctrl = make_ctrl()
    ctrl.control_loop()
    assert ctrl.cmd == (0.0, 0.0)
    assert ctrl.state == "FORWARD"


def test_forward_until_wall_then_wait():
    ctrl = make_ctrl()
    ctrl.on_odom(0.0)
    ctrl.on_scan(scan_with_front(5.0))
    ctrl.control_loop()
    assert ctrl.cmd == (0.20, 0.0)
    assert ctrl.state == "FORWARD"

    ctrl.on_scan(scan_with_front(0.79))
    ctrl.control_loop()
    assert ctrl.cmd == (0.0, 0.0)
    assert ctrl.state == "STOPPED_WAITING_EEG"


def test_exact_safe_dist_does_not_stop():
    # The original uses a strict `<` comparison.
    ctrl = make_ctrl()
    ctrl.on_odom(0.0)
    ctrl.on_scan(scan_with_front(0.8))
    ctrl.control_loop()
    assert ctrl.state == "FORWARD"


def test_decision_gating():
    ctrl = make_ctrl()
    ctrl.on_odom(0.0)
    ctrl.on_scan(scan_with_front(5.0))
    ctrl.control_loop()

    # Ignored while FORWARD.
    assert ctrl.submit_decision(1) == "ignored_in_FORWARD"
    # 0 is always skipped.
    assert ctrl.submit_decision(0) == "skipped"
    # invalid values ignored
    assert ctrl.submit_decision(7) == "invalid"

    ctrl.on_scan(scan_with_front(0.5))
    ctrl.control_loop()
    assert ctrl.state == "STOPPED_WAITING_EEG"

    assert ctrl.submit_decision(1) == "pending_decision"
    # A second decision while one is pending is dropped.
    assert ctrl.submit_decision(2) == "ignored_already_pending(1)"

    ctrl.control_loop()
    assert ctrl.state == "TURN_L"
    assert ctrl.target_yaw is not None
    assert abs(ctrl.target_yaw - math.pi / 2) < 1e-9


def test_left_turn_executes_90_degrees():
    ctrl = make_ctrl()
    yaw = 0.0
    ctrl.on_odom(yaw)
    ctrl.on_scan(scan_with_front(0.5))
    ctrl.control_loop()
    ctrl.submit_decision(1)
    ctrl.control_loop()
    assert ctrl.state == "TURN_L"

    # Integrate the published angular velocity like the sim does (20 Hz).
    # (The turn command itself is first published on the tick AFTER the
    # state change — same as the original node.)
    for _ in range(200):
        if ctrl.state != "TURN_L":
            break
        ctrl.on_odom(yaw)
        ctrl.on_scan(scan_with_front(5.0))
        ctrl.control_loop()
        if ctrl.state == "TURN_L":
            assert ctrl.cmd == (0.0, 0.60)
        yaw = wrap_pi(yaw + ctrl.cmd[1] * 0.05)

    assert ctrl.state == "FORWARD"
    assert abs(wrap_pi(yaw - math.pi / 2)) < math.radians(2.0)


def test_right_turn_sign():
    ctrl = make_ctrl()
    ctrl.on_odom(0.0)
    ctrl.on_scan(scan_with_front(0.5))
    ctrl.control_loop()
    ctrl.submit_decision(2)
    ctrl.control_loop()
    assert ctrl.state == "TURN_R"
    ctrl.on_scan(scan_with_front(5.0))
    ctrl.on_odom(0.0)
    ctrl.control_loop()
    assert ctrl.cmd == (0.0, -0.60)


def test_scan_front_cone_window():
    """Only the +/-15 deg window matters."""
    ctrl = make_ctrl()
    n = 181
    inc = math.radians(0.33)
    angle_min = -inc * (n // 2)
    ranges = [10.0] * n
    # Put a close obstacle outside the cone (at ~ -29.7 deg).
    ranges[0] = 0.2
    ctrl.on_scan(LaserScanMsg(angle_min, inc, 0.05, 25.0, ranges))
    assert ctrl.front_dist == 10.0
    # And inside the cone.
    center = n // 2
    ranges[center] = 0.3
    ctrl.on_scan(LaserScanMsg(angle_min, inc, 0.05, 25.0, ranges))
    assert ctrl.front_dist == 0.3


def test_scan_invalid_values_filtered():
    ctrl = make_ctrl()
    n = 181
    inc = math.radians(0.33)
    angle_min = -inc * (n // 2)
    ranges = [float("inf")] * n
    center = n // 2
    ranges[center - 1] = float("nan")
    ranges[center] = -1.0
    ranges[center + 1] = 0.01  # below range_min
    ctrl.on_scan(LaserScanMsg(angle_min, inc, 0.05, 25.0, ranges))
    assert ctrl.front_dist == float("inf")
    ctrl.on_odom(0.0)
    ctrl.control_loop()
    # inf front distance -> keeps driving (matches original behavior).
    assert ctrl.state == "FORWARD"
    assert ctrl.cmd == (0.20, 0.0)
