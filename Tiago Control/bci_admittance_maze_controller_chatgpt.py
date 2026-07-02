"""
bci_admittance_maze_controller.py

Discrete decision-point MI-BCI maze controller with virtual muscle activation
and event-triggered admittance dynamics.

Author note:
-------------
This code implements the paper formulation discussed for maze navigation:

    EEG classifier probabilities
        -> 90% confidence-vetted left/right evidence
        -> virtual muscle activation
        -> event-triggered admittance intent state z_I
        -> discrete LEFT/RIGHT/HOLD decision at maze junction
        -> angular velocity command omega for a fixed turn maneuver

Important modeling choice:
--------------------------
For a maze, z_I is NOT directly used as the robot's continuous angular velocity.
Instead, z_I is a latent turn-intent state. When z_I crosses a commitment boundary,
the finite-state machine generates a fixed angular velocity command omega until
the robot completes a left/right turn.

This is appropriate for decision-point navigation:
    - Between junctions: robot moves forward autonomously.
    - At junctions: BCI chooses LEFT or RIGHT.
    - During turn: robot executes a clean geometric turn.

ROS convention:
---------------
In ROS geometry_msgs/Twist, angular.z > 0 usually means counter-clockwise / left turn.
Therefore this file uses:
    LEFT  -> omega = +omega_turn
    RIGHT -> omega = -omega_turn

If your simulator uses the opposite sign convention, change the signs in Config.

No heuristic reset:
-------------------
The code does NOT manually force z_I = 0 after each decision.
Instead, the event gate sigma is zero outside decision mode, so the stable
admittance dynamics naturally decays toward zero:

    M z_ddot + D z_dot + K z = sigma * f_I

When sigma = 0:
    M z_ddot + D z_dot + K z = 0

If D > 0 and K > 0, this subsystem is asymptotically stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, pi, sin
from typing import Dict, Optional


class Mode(str, Enum):
    """Robot high-level FSM modes for decision-point maze navigation."""
    FORWARD = "FORWARD"          # Robot moves forward autonomously.
    DECISION = "DECISION"        # Robot waits/slows and listens to BCI intent.
    TURN_LEFT = "TURN_LEFT"      # Robot executes a left turn.
    TURN_RIGHT = "TURN_RIGHT"    # Robot executes a right turn.
    HOLD = "HOLD"                # No confident decision within allowed time.


@dataclass
class Config:
    """
    Tunable design constants.

    These defaults match the paper-style initial design:
        decoder stride dt = 0.2 s
        confidence threshold = 0.90
        virtual muscle activation time constant = 0.6 s
        critically damped admittance with settling time about 2 s:
            M = 1, D = 4, K = 4
    """

    # ---------------------------------------------------------------------
    # Timing
    # ---------------------------------------------------------------------
    dt: float = 0.2
    # Online classifier update period in seconds.
    # In the slides/paper this is the stride of the sliding EEG window.

    # ---------------------------------------------------------------------
    # Classifier probability gate
    # ---------------------------------------------------------------------
    confidence_threshold: float = 0.90
    # A classifier output is accepted only if max(p_L, p_R) >= threshold.
    # Otherwise the BCI evidence is treated as zero.
    #
    # If your classifier already returns HOLD/STOP for probabilities below
    # 90%, you can still keep this check. It will simply preserve the same
    # behavior.

    # ---------------------------------------------------------------------
    # Virtual muscle activation
    # ---------------------------------------------------------------------
    activation_time_constant: float = 0.6
    # Larger value => slower activation, more smoothing.
    # Smaller value => faster activation, more responsiveness but more jitter.

    muscle_gain: float = 5.0
    # Converts the difference between right/left muscle activations into
    # the virtual force input f_I for the admittance system.

    # ---------------------------------------------------------------------
    # Admittance parameters
    # ---------------------------------------------------------------------
    M: float = 1.0
    # Virtual inertia.

    D: float = 4.0
    # Virtual damping.

    K: float = 4.0
    # Virtual stiffness.
    #
    # With M=1, D=4, K=4:
    #   omega_n = sqrt(K/M) = 2 rad/s
    #   zeta = D/(2*sqrt(M*K)) = 1
    #   settling time approx 4/(zeta*omega_n) = 2 s

    # ---------------------------------------------------------------------
    # Commitment boundary for maze decision
    # ---------------------------------------------------------------------
    z_on: float = 0.5
    # Turn commitment boundary.
    # If z_I >= z_on  -> RIGHT intent is committed.
    # If z_I <= -z_on -> LEFT intent is committed.
    #
    # This is NOT an idle/noise threshold.
    # Idle/no-control is already handled by the 90% classifier gate.
    # z_on only asks: has enough high-confidence evidence accumulated?

    # ---------------------------------------------------------------------
    # Turning behavior
    # ---------------------------------------------------------------------
    omega_turn: float = 0.6
    # Magnitude of angular velocity during a committed turn [rad/s].

    left_omega_sign: float = +1.0
    right_omega_sign: float = -1.0
    # ROS convention:
    #   angular.z > 0 -> left / counter-clockwise
    #   angular.z < 0 -> right / clockwise

    turn_angle: float = pi / 2.0
    # Default maze turn angle: 90 degrees.

    yaw_tolerance: float = 3.0 * pi / 180.0
    # Turn is complete when heading error is within 3 degrees.

    # ---------------------------------------------------------------------
    # Decision timeout
    # ---------------------------------------------------------------------
    max_decision_time: float = 5.0
    # If the robot is in DECISION mode for too long without z_I crossing the
    # commitment boundary, go to HOLD. The supervisor can then request another
    # EEG window, ask autonomy to choose, or keep waiting.

    # ---------------------------------------------------------------------
    # Numerical safety clamps
    # ---------------------------------------------------------------------
    max_abs_z: float = 5.0
    max_abs_zdot: float = 10.0
    # Prevent numerical blow-up if probabilities or timing are misconfigured.


def wrap_angle(angle: float) -> float:
    """
    Wrap an angle to [-pi, pi].

    This is useful for heading/yaw error calculations.
    """
    return atan2(sin(angle), cos(angle))


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


class BCIAdmittanceMazeController:
    """
    Event-triggered virtual muscle/admittance controller for discrete maze turns.

    The controller expects classifier probabilities p_left and p_right at each
    update. It also expects a boolean flag "in_decision_region" from the robot,
    for example:

        in_decision_region = (front_distance < d_th) and robot_is_aligned

    The controller returns an angular velocity omega. The forward velocity can
    be handled by the robot's existing autonomous maze/navigation controller.

    Typical usage
    -------------
    controller = BCIAdmittanceMazeController()

    while True:
        # p_left, p_right come from FBCSP/LDA or another MI classifier.
        # yaw is optional but recommended for ending 90-degree turns accurately.
        out = controller.update(
            p_left=p_left,
            p_right=p_right,
            in_decision_region=(front_distance < d_th),
            yaw=current_yaw
        )

        omega_cmd = out["omega"]
        mode = out["mode"]

        # Publish omega_cmd to ROS /cmd_vel.angular.z.
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config if config is not None else Config()

        # FSM mode.
        self.mode: Mode = Mode.FORWARD

        # Virtual muscle states.
        # a_R accumulates right-intent evidence.
        # a_L accumulates left-intent evidence.
        self.a_R: float = 0.0
        self.a_L: float = 0.0

        # Admittance state x_I = [z_I, z_dot_I].
        # z_I > 0 means right-turn intent.
        # z_I < 0 means left-turn intent.
        self.z: float = 0.0
        self.zdot: float = 0.0

        # Timers.
        self.decision_timer: float = 0.0
        self.turn_timer: float = 0.0

        # Turn bookkeeping.
        self.target_yaw: Optional[float] = None
        self.active_turn_sign: float = 0.0
        self.last_decision: str = "NONE"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        p_left: float,
        p_right: float,
        in_decision_region: bool,
        yaw: Optional[float] = None,
    ) -> Dict[str, float | str | bool]:
        """
        Update the controller by one classifier/control step.

        Parameters
        ----------
        p_left : float
            Classifier probability for left-hand / left-turn imagery.

        p_right : float
            Classifier probability for right-hand / right-turn imagery.

        in_decision_region : bool
            Event flag supplied by the robot navigation layer.
            For the maze this can be:
                front_distance < d_th
            possibly combined with an alignment check.

        yaw : Optional[float]
            Current robot yaw angle in radians.
            If provided, the controller uses yaw feedback to stop the turn.
            If not provided, it uses a timed 90-degree turn approximation.

        Returns
        -------
        dict
            A dictionary containing:
                omega : angular velocity command [rad/s]
                mode : current FSM mode
                decision : LEFT / RIGHT / HOLD / NONE
                sigma : event gate
                evidence : vetted EEG evidence u_e
                confidence : max(p_left, p_right)
                a_R, a_L : virtual muscle activations
                f_adm : actual admittance force sigma*f_I
                z, zdot : admittance states
        """

        # --------------------------------------------------------------
        # 1. Sanitize probabilities
        # --------------------------------------------------------------
        p_left = clamp(float(p_left), 0.0, 1.0)
        p_right = clamp(float(p_right), 0.0, 1.0)

        # It is not necessary that p_left + p_right = 1 if the upstream
        # classifier includes HOLD/STOP/no-control as a third output.
        # We only use the relative left/right evidence.
        confidence = max(p_left, p_right)

        # --------------------------------------------------------------
        # 2. Update high-level mode based on the maze event
        # --------------------------------------------------------------
        self._update_mode_before_dynamics(in_decision_region)

        # sigma is the event-triggering gate from the paper.
        # The admittance is actively driven by BCI only in DECISION mode
        # and only while the robot is in a decision region.
        sigma = 1.0 if (self.mode == Mode.DECISION and in_decision_region) else 0.0

        # --------------------------------------------------------------
        # 3. Convert classifier probabilities to vetted evidence
        # --------------------------------------------------------------
        # Evidence convention:
        #   u_e > 0 : right intent
        #   u_e < 0 : left intent
        #
        # Since the classifier already vets probabilities below 90%, this
        # check is aligned with the existing real-time BCI pipeline.
        if confidence >= self.cfg.confidence_threshold:
            raw_evidence = p_right - p_left
        else:
            raw_evidence = 0.0

        # The event gate prevents evidence from being accumulated when the
        # robot is not asking for a left/right decision.
        evidence = sigma * raw_evidence

        # --------------------------------------------------------------
        # 4. Virtual muscle activation dynamics
        # --------------------------------------------------------------
        self._update_virtual_muscles(evidence)

        # Virtual muscle force.
        #
        # This force is positive for right intent and negative for left intent.
        f_I = self.cfg.muscle_gain * (self.a_R - self.a_L)

        # Admittance input. sigma is included here as an additional safety
        # gate. This ensures residual muscle activation cannot drive z_I
        # outside decision mode.
        f_adm = sigma * f_I

        # --------------------------------------------------------------
        # 5. Event-triggered admittance update
        # --------------------------------------------------------------
        self._update_admittance(f_adm)

        # --------------------------------------------------------------
        # 6. FSM decision logic and omega generation
        # --------------------------------------------------------------
        omega = self._compute_omega_and_update_turn(yaw=yaw)

        # --------------------------------------------------------------
        # 7. Return debug-friendly output
        # --------------------------------------------------------------
        return {
            "omega": omega,
            "mode": self.mode.value,
            "decision": self.last_decision,
            "sigma": bool(sigma),
            "confidence": confidence,
            "evidence": evidence,
            "a_R": self.a_R,
            "a_L": self.a_L,
            "f_adm": f_adm,
            "z": self.z,
            "zdot": self.zdot,
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _update_mode_before_dynamics(self, in_decision_region: bool) -> None:
        """
        Update FSM mode before applying the admittance dynamics.

        Mode transitions:
            FORWARD -> DECISION when a maze decision region is detected.
            HOLD    -> DECISION if still in decision region and new evidence may arrive.
            DECISION remains DECISION until z crosses threshold or timeout occurs.
            TURN_* modes are handled separately after omega is computed.
        """

        dt = self.cfg.dt

        if self.mode == Mode.FORWARD:
            if in_decision_region:
                self.mode = Mode.DECISION
                self.decision_timer = 0.0
                self.last_decision = "NONE"

        elif self.mode == Mode.DECISION:
            self.decision_timer += dt

            # If the robot somehow leaves the decision region before a decision,
            # return to FORWARD. In a real maze controller, the robot is often
            # stopped/slowed at the decision point, so this may rarely happen.
            if not in_decision_region:
                self.mode = Mode.FORWARD
                self.decision_timer = 0.0
                self.last_decision = "NONE"

            elif self.decision_timer >= self.cfg.max_decision_time:
                self.mode = Mode.HOLD
                self.last_decision = "HOLD"

        elif self.mode == Mode.HOLD:
            # HOLD means no reliable decision was reached.
            # If still in the decision region, remain HOLD.
            # If the supervisor moves the robot away or clears the event,
            # return to FORWARD.
            if not in_decision_region:
                self.mode = Mode.FORWARD
                self.decision_timer = 0.0
                self.last_decision = "NONE"

        # TURN_LEFT and TURN_RIGHT are not modified here.
        # They are completed by yaw feedback or timed turn in
        # _compute_omega_and_update_turn().

    def _update_virtual_muscles(self, evidence: float) -> None:
        """
        Leaky virtual muscle activation.

        evidence > 0 excites the right virtual muscle.
        evidence < 0 excites the left virtual muscle.

        The activation update is:
            a_R[k+1] = (1-alpha) a_R[k] + alpha [evidence]_+
            a_L[k+1] = (1-alpha) a_L[k] + alpha [-evidence]_+

        where:
            alpha = dt / (T_a + dt)

        This is a stable first-order low-pass filter.
        """

        dt = self.cfg.dt
        T_a = self.cfg.activation_time_constant

        # Avoid division by zero if someone accidentally sets T_a <= 0.
        alpha = dt / (max(T_a, 1e-9) + dt)

        right_input = max(evidence, 0.0)
        left_input = max(-evidence, 0.0)

        self.a_R = (1.0 - alpha) * self.a_R + alpha * right_input
        self.a_L = (1.0 - alpha) * self.a_L + alpha * left_input

    def _update_admittance(self, force: float) -> None:
        """
        Numerical integration of the scalar admittance model:

            M z_ddot + D z_dot + K z = force

        A semi-implicit Euler step is used:
            z_dot[k+1] = z_dot[k] + dt*z_ddot[k]
            z[k+1]     = z[k]     + dt*z_dot[k+1]

        Semi-implicit Euler is simple and usually more stable than fully
        explicit Euler for damped second-order systems.
        """

        cfg = self.cfg
        dt = cfg.dt

        # Compute acceleration.
        zddot = (force - cfg.D * self.zdot - cfg.K * self.z) / cfg.M

        # Semi-implicit Euler integration.
        self.zdot += dt * zddot
        self.z += dt * self.zdot

        # Clamp states to avoid numerical issues in case of bad configuration.
        self.z = clamp(self.z, -cfg.max_abs_z, cfg.max_abs_z)
        self.zdot = clamp(self.zdot, -cfg.max_abs_zdot, cfg.max_abs_zdot)

    def _compute_omega_and_update_turn(self, yaw: Optional[float]) -> float:
        """
        Convert admittance state into an angular velocity command.

        In DECISION mode:
            z >= z_on  -> commit RIGHT turn
            z <= -z_on -> commit LEFT turn
            otherwise  -> omega = 0

        In TURN_LEFT / TURN_RIGHT mode:
            output fixed angular velocity until the turn is complete.

        In FORWARD or HOLD mode:
            omega = 0. Forward velocity is assumed to be handled elsewhere.
        """

        cfg = self.cfg

        # --------------------------------------------------------------
        # If in DECISION mode, check whether z has crossed a commitment
        # boundary. The sign convention is:
        #   z > 0  => RIGHT intent
        #   z < 0  => LEFT intent
        # --------------------------------------------------------------
        if self.mode == Mode.DECISION:
            if self.z >= cfg.z_on:
                self._start_turn(direction="RIGHT", yaw=yaw)
            elif self.z <= -cfg.z_on:
                self._start_turn(direction="LEFT", yaw=yaw)
            else:
                return 0.0

        # --------------------------------------------------------------
        # Generate omega during an active turn.
        # --------------------------------------------------------------
        if self.mode == Mode.TURN_LEFT:
            omega = cfg.left_omega_sign * cfg.omega_turn

            if self._turn_is_complete(yaw=yaw):
                self._finish_turn()

            return omega

        if self.mode == Mode.TURN_RIGHT:
            omega = cfg.right_omega_sign * cfg.omega_turn

            if self._turn_is_complete(yaw=yaw):
                self._finish_turn()

            return omega

        # FORWARD and HOLD do not generate angular velocity here.
        return 0.0

    def _start_turn(self, direction: str, yaw: Optional[float]) -> None:
        """
        Start a left or right turn.

        This method does NOT reset z or zdot.
        After entering a TURN_* mode, sigma becomes zero, so the admittance
        state passively decays under stable homogeneous dynamics.
        """

        cfg = self.cfg
        self.turn_timer = 0.0
        self.last_decision = direction

        if direction == "LEFT":
            self.mode = Mode.TURN_LEFT
            turn_sign = cfg.left_omega_sign
        elif direction == "RIGHT":
            self.mode = Mode.TURN_RIGHT
            turn_sign = cfg.right_omega_sign
        else:
            raise ValueError(f"Unknown turn direction: {direction}")

        self.active_turn_sign = turn_sign

        # If yaw is available, compute target yaw for a 90-degree turn.
        if yaw is not None:
            self.target_yaw = wrap_angle(yaw + turn_sign * cfg.turn_angle)
        else:
            self.target_yaw = None

    def _turn_is_complete(self, yaw: Optional[float]) -> bool:
        """
        Decide whether the active turn is complete.

        Preferred method:
            Use yaw feedback and stop when heading error is small.

        Fallback:
            If yaw is unavailable, stop after estimated duration:
                turn_angle / omega_turn
        """

        cfg = self.cfg
        self.turn_timer += cfg.dt

        # Yaw-feedback stopping condition.
        if yaw is not None and self.target_yaw is not None:
            heading_error = wrap_angle(self.target_yaw - yaw)
            return abs(heading_error) <= cfg.yaw_tolerance

        # Timed-turn fallback.
        # This is less accurate because wheel slip, latency, and dynamics
        # can change the true turn angle.
        estimated_turn_duration = cfg.turn_angle / max(cfg.omega_turn, 1e-9)
        return self.turn_timer >= estimated_turn_duration

    def _finish_turn(self) -> None:
        """
        Complete the turn and return to FORWARD mode.

        No manual reset of z_I is applied. The event gate sigma is zero
        outside DECISION mode, so z_I will continue to decay naturally.
        """

        self.mode = Mode.FORWARD
        self.decision_timer = 0.0
        self.turn_timer = 0.0
        self.target_yaw = None
        self.active_turn_sign = 0.0

    def get_internal_state(self) -> Dict[str, float | str]:
        """
        Return internal state for logging/debugging without updating dynamics.
        """
        return {
            "mode": self.mode.value,
            "a_R": self.a_R,
            "a_L": self.a_L,
            "z": self.z,
            "zdot": self.zdot,
            "decision_timer": self.decision_timer,
            "turn_timer": self.turn_timer,
            "last_decision": self.last_decision,
        }


# -------------------------------------------------------------------------
# Minimal example
# -------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This small example simulates a robot entering a decision region and receiving
    repeated high-confidence RIGHT probabilities from the classifier.

    In real use:
        - p_left and p_right come from your trained classifier.
        - in_decision_region comes from the maze/navigation layer.
        - yaw comes from odometry/IMU/localization.
    """

    controller = BCIAdmittanceMazeController()

    # Simulated classifier probabilities.
    # For the first 5 steps, no confident command.
    # Then the classifier gives high-confidence RIGHT evidence.
    probability_stream = (
        [(0.50, 0.50)] * 5 +
        [(0.05, 0.95)] * 20 +
        [(0.50, 0.50)] * 20
    )

    # Simulate that the robot is at a decision region for the whole example.
    in_decision_region = True

    # No yaw feedback in this simple example; timed turn fallback is used.
    yaw = None

    print("step, pL, pR, mode, omega, z, evidence, decision")

    for k, (pL, pR) in enumerate(probability_stream):
        out = controller.update(
            p_left=pL,
            p_right=pR,
            in_decision_region=in_decision_region,
            yaw=yaw,
        )

        print(
            f"{k:03d}, "
            f"{pL:.2f}, {pR:.2f}, "
            f"{out['mode']}, "
            f"{out['omega']:+.2f}, "
            f"{out['z']:+.3f}, "
            f"{out['evidence']:+.3f}, "
            f"{out['decision']}"
        )
