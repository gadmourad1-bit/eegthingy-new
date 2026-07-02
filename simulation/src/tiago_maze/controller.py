"""Port of src/my_controller/src/safe_wall_teleop_webSocket.py.

The state machine, thresholds, scan windowing, decision handling and turn
logic are kept structurally identical to the ROS 2 node, minus the rclpy
plumbing:

  - ``on_scan``     <- LaserScan callback (front-cone minimum)
  - ``on_odom``     <- Odometry callback (yaw only)
  - ``control_loop``<- 20 Hz timer callback; emits (vx, wz) via publish_cmd
  - ``submit_decision`` <- what the websocket thread does with a received
    decision value (1 = LEFT, 2 = RIGHT), including the state gating.

States: FORWARD -> STOPPED_WAITING_EEG -> TURN_L / TURN_R -> FORWARD.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .params import ControlParams


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def snap_cardinal(a: float) -> float:
    """Nearest grid heading (multiple of pi/2), wrapped to (-pi, pi]."""
    return wrap_pi(round(a / (math.pi / 2.0)) * (math.pi / 2.0))


@dataclass
class LaserScanMsg:
    """The subset of sensor_msgs/LaserScan the controller consumes."""

    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: list[float] | tuple[float, ...]


class Logger:
    """Mimics the rclpy node logger output format."""

    def __init__(self, name: str = "tiago_safe_wall_teleop", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._t0 = time.monotonic()

    def _log(self, level: str, msg: str) -> None:
        if self.enabled:
            print(f"[{level}] [t={time.monotonic() - self._t0:9.3f}] [{self.name}]: {msg}")

    def info(self, msg: str) -> None:
        self._log("INFO", msg)

    def warn(self, msg: str) -> None:
        self._log("WARN", msg)


class SafeWallTeleop:
    """The ported node. ``publish_cmd(vx, wz)`` replaces the Twist publisher."""

    def __init__(
        self,
        params: ControlParams | None = None,
        publish_cmd: Callable[[float, float], None] | None = None,
        logger: Logger | None = None,
    ):
        p = params or ControlParams()
        self.params = p

        # ---------------- Parameters ----------------
        self.safe_dist = float(p.safe_dist)
        self.forward_speed = float(p.forward_speed)
        self.turn_speed = float(p.turn_speed)
        self.front_half_angle_deg = float(p.front_half_angle_deg)
        self.yaw_tol_deg = float(p.yaw_tol_deg)
        self.heading_kp = float(getattr(p, "heading_kp", 0.0))

        # ---------------- Derived values ----------------
        self.front_half_angle = math.radians(self.front_half_angle_deg)
        self.yaw_tol = math.radians(self.yaw_tol_deg)

        # ---------------- Outputs / logging ----------------
        self._publish = publish_cmd or (lambda vx, wz: None)
        self.log = logger or Logger()
        self.cmd = (0.0, 0.0)  # last published (vx, wz)

        # ---------------- State ----------------
        self.state = "FORWARD"
        self.front_dist = float("inf")
        self.have_scan = False
        self.have_odom = False
        self.yaw = 0.0
        self.target_yaw: float | None = None

        # ---------------- EEG / websocket state ----------------
        self._lock = threading.Lock()
        # Latest decision from the server, consumed once when stopped.
        self.pending_decision: int | None = None

        # Hooks for the game layer (state change / turn complete callbacks).
        self.on_state_change: Callable[[str, str], None] | None = None
        self.on_turn_complete: Callable[[str], None] | None = None

    # ============================================================
    # ROS callbacks
    # ============================================================

    def on_scan(self, msg: LaserScanMsg) -> None:
        ang_min = msg.angle_min
        ang_inc = msg.angle_increment
        n = len(msg.ranges)

        i0 = int(max(0, math.floor(((-self.front_half_angle) - ang_min) / ang_inc)))
        i1 = int(min(n - 1, math.ceil(((+self.front_half_angle) - ang_min) / ang_inc)))

        best = float("inf")
        for i in range(i0, i1 + 1):
            r = msg.ranges[i]

            if not math.isfinite(r):
                continue
            if r <= 0.0:
                continue
            if r < msg.range_min or r > msg.range_max:
                continue

            if r < best:
                best = r

        self.front_dist = best
        self.have_scan = True

    def on_odom(self, yaw: float) -> None:
        self.yaw = yaw
        self.have_odom = True

    # ============================================================
    # WebSocket-facing logic (body of the original recv loop)
    # ============================================================

    def submit_decision(self, val: int) -> str:
        """Store a decision exactly the way the websocket loop did.

        Returns the disposition string that the original logged, e.g.
        "pending_decision" or "ignored_in_FORWARD".
        """
        if val == 0:
            self.log.info("WS recv = 0 | skipped (no consensus)")
            return "skipped"

        if val not in (1, 2):
            self.log.warn(f"Unexpected decision value {val}; ignoring")
            return "invalid"

        with self._lock:
            if self.state != "STOPPED_WAITING_EEG":
                stored = f"ignored_in_{self.state}"
            elif self.pending_decision is not None:
                stored = f"ignored_already_pending({self.pending_decision})"
            else:
                self.pending_decision = val
                stored = "pending_decision"

        self.log.info(f"WS recv = {val} | {stored}")
        return stored

    # ============================================================
    # Helpers
    # ============================================================

    def publish_cmd(self, vx: float, wz: float) -> None:
        self.cmd = (float(vx), float(wz))
        self._publish(float(vx), float(wz))

    def stop_robot(self) -> None:
        self.publish_cmd(0.0, 0.0)

    def set_state(self, new_state: str) -> None:
        changed = False
        with self._lock:
            if self.state != new_state:
                old = self.state
                self.state = new_state
                changed = True

        if changed:
            self.log.info(f"STATE -> {new_state}")
            if self.on_state_change is not None:
                self.on_state_change(old, new_state)

    def enter_stopped_waiting_EEG(self) -> None:
        with self._lock:
            old = self.state
            self.pending_decision = None
            self.state = "STOPPED_WAITING_EEG"

        self.log.info("Robot stopped at wall. Waiting for next decision from server...")
        if self.on_state_change is not None and old != "STOPPED_WAITING_EEG":
            self.on_state_change(old, "STOPPED_WAITING_EEG")

    def start_turn(self, direction: str) -> bool:
        if not self.have_odom:
            self.log.warn("No odom yet; cannot perform yaw-based turn.")
            return False

        delta = math.pi / 2.0
        base = snap_cardinal(self.yaw)  # aim from the grid heading, not the drifted one

        if direction == "LEFT":
            self.target_yaw = wrap_pi(base + delta)
            self.set_state("TURN_L")
        elif direction == "RIGHT":
            self.target_yaw = wrap_pi(base - delta)
            self.set_state("TURN_R")
        else:
            self.log.warn(f"Unknown turn direction: {direction}")
            return False

        self.log.info(f"Starting {direction} turn. target_yaw = {self.target_yaw:.3f} rad")
        return True

    def clear_for_next_cycle(self) -> None:
        with self._lock:
            self.pending_decision = None

    def status_line(self) -> str:
        with self._lock:
            state = self.state
            pending = self.pending_decision

        fd = self.front_dist if math.isfinite(self.front_dist) else float("inf")
        target_yaw_str = "None" if self.target_yaw is None else f"{self.target_yaw:.3f}"

        return (
            f"state: {state} | front: {fd:.3f} | yaw: {self.yaw:.3f} | "
            f"target: {target_yaw_str} | pending: {pending}"
        )

    def print_debug_status(self, reason: str = "periodic") -> None:
        with self._lock:
            state = self.state
            pending = self.pending_decision

        fd = self.front_dist if math.isfinite(self.front_dist) else float("inf")
        target_yaw_str = "None" if self.target_yaw is None else f"{self.target_yaw:.3f}"

        self.log.info(
            "\n"
            f"----- DEBUG STATUS ({reason}) -----\n"
            f"current state       : {state}\n"
            f"front distance      : {fd:.3f}\n"
            f"current yaw         : {self.yaw:.3f}\n"
            f"target yaw          : {target_yaw_str}\n"
            f"pending decision    : {pending}\n"
            f"-------------------------------"
        )

    # ============================================================
    # Main control loop (called at 20 Hz)
    # ============================================================

    def control_loop(self) -> None:
        if not self.have_scan:
            self.stop_robot()
            return

        if self.state == "FORWARD":
            if self.front_dist < self.safe_dist:
                self.stop_robot()
                self.enter_stopped_waiting_EEG()
            else:
                wz = self.heading_kp * wrap_pi(snap_cardinal(self.yaw) - self.yaw)
                wz = max(-self.turn_speed, min(self.turn_speed, wz))
                self.publish_cmd(self.forward_speed, wz)

        elif self.state == "STOPPED_WAITING_EEG":
            self.stop_robot()

            with self._lock:
                val = self.pending_decision
                self.pending_decision = None

            if val is None:
                return

            direction = "LEFT" if val == 1 else "RIGHT"
            ok = self.start_turn(direction)
            if not ok:
                self.stop_robot()

        elif self.state == "TURN_L":
            if not self.have_odom or self.target_yaw is None:
                self.stop_robot()
                return

            err = wrap_pi(self.target_yaw - self.yaw)

            if abs(err) < self.yaw_tol:
                self.stop_robot()
                self.log.info("LEFT turn complete.")
                self.target_yaw = None
                self.clear_for_next_cycle()
                if self.on_turn_complete is not None:
                    self.on_turn_complete("LEFT")
                self.set_state("FORWARD")
            else:
                self.publish_cmd(0.0, abs(self.turn_speed))

        elif self.state == "TURN_R":
            if not self.have_odom or self.target_yaw is None:
                self.stop_robot()
                return

            err = wrap_pi(self.target_yaw - self.yaw)

            if abs(err) < self.yaw_tol:
                self.stop_robot()
                self.log.info("RIGHT turn complete.")
                self.target_yaw = None
                self.clear_for_next_cycle()
                if self.on_turn_complete is not None:
                    self.on_turn_complete("RIGHT")
                self.set_state("FORWARD")
            else:
                self.publish_cmd(0.0, -abs(self.turn_speed))

        else:
            self.log.warn(f"Unknown state: {self.state}. Stopping robot.")
            self.stop_robot()
