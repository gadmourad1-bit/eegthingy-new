"""WebSocket decision client.

Port of the ``websocket_loop`` thread in safe_wall_teleop_webSocket.py. It
connects to the decision server and uses ``smoothed.final`` (0 = no committed
choice, 1 = LEFT, 2 = RIGHT). A decision is eligible only after the robot has
stopped and the classifier's entire vote buffer contains fresh post-stop EEG.
The controller still applies its own STOPPED_WAITING_EEG/pending gate.
Reconnects after 1 second on any error, forever.

The original used the ``websocket-client`` package; this port uses
``websockets`` (BSD-3) with its synchronous client API — same wire behavior.
"""

from __future__ import annotations

import json
import threading
import time

from websockets.sync.client import connect

from .controller import SafeWallTeleop


def fresh_committed_decision(data: dict, after_timestamp: float) -> int:
    """Return a committed decision only after a complete fresh post-stop buffer.

    The classifier runs continuously while the robot is moving. Without this
    gate, a consensus left over from the previous corridor can be consumed the
    instant the robot reaches the next wall, before the subject has chosen a
    new direction.
    """
    smoothed = data.get("smoothed", {})
    buffer = data.get("buffer", [])
    required = max(1, int(smoothed.get("n", 1)))
    fresh = [
        row for row in buffer
        if float(row.get("timestamp", float("-inf"))) >= float(after_timestamp)
    ]
    if len(fresh) < required:
        return 0
    return int(smoothed.get("final", 0))


class DecisionClient:
    def __init__(self, controller: SafeWallTeleop, ws_url: str):
        self.controller = controller
        self.ws_url = ws_url
        self.status = "disconnected"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._waiting_after_server_time: float | None = None
        self._submitted_for_current_stop = False

    def start(self) -> None:
        self._thread.start()
        self.controller.log.info(f"WebSocket enabled. URL = {self.ws_url}")

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        log = self.controller.log
        while not self._stop.is_set():
            ws = None
            try:
                self.status = "connecting"
                log.info(f"Connecting WebSocket to {self.ws_url}")
                ws = connect(self.ws_url, open_timeout=5)
                self.status = "connected"
                log.info("WebSocket connected.")

                while not self._stop.is_set():
                    try:
                        msg = ws.recv(timeout=1.0)
                    except TimeoutError:
                        # No message this second; loop again so a stop
                        # request is honored promptly.
                        continue
                    if msg is None:
                        continue

                    try:
                        data = json.loads(msg)
                    except (ValueError, KeyError, TypeError) as e:
                        log.warn(f"Ignoring invalid websocket message ({e}): {msg!r}")
                        continue

                    if self.controller.state != "STOPPED_WAITING_EEG":
                        self._waiting_after_server_time = None
                        self._submitted_for_current_stop = False
                        continue
                    if self._submitted_for_current_stop:
                        continue
                    if self._waiting_after_server_time is None:
                        # Use the classifier's clock from the payload, so this
                        # also works when the simulator and decoder are on
                        # different machines with unlike local clocks.
                        self._waiting_after_server_time = float(data.get("timestamp", time.time()))
                        continue
                    try:
                        val = fresh_committed_decision(data, self._waiting_after_server_time)
                    except (ValueError, TypeError):
                        val = 0
                    if val not in (1, 2):
                        continue
                    disposition = self.controller.submit_decision(val)
                    if disposition == "pending_decision":
                        self._submitted_for_current_stop = True

            except Exception as e:
                self.status = "reconnecting"
                log.warn(f"WebSocket error: {e}. Reconnecting in 1 second...")
                time.sleep(1.0)
            finally:
                try:
                    if ws is not None:
                        ws.close()
                except Exception:
                    pass

        self.status = "stopped"
