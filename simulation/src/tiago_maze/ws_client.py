"""WebSocket decision client.

Port of the ``websocket_loop`` thread in safe_wall_teleop_webSocket.py:
connects to the decision server, parses ``{"smoothed": {"decision": v}}``
payloads (0 = skip / no consensus, 1 = LEFT, 2 = RIGHT) and hands the value
to the controller, which applies the exact same state gating as the original
(decisions are only accepted while STOPPED_WAITING_EEG and none is pending).
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


class DecisionClient:
    def __init__(self, controller: SafeWallTeleop, ws_url: str):
        self.controller = controller
        self.ws_url = ws_url
        self.status = "disconnected"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

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
                        val = int(data["smoothed"]["decision"])
                    except (ValueError, KeyError, TypeError) as e:
                        log.warn(f"Ignoring invalid websocket message ({e}): {msg!r}")
                        continue

                    self.controller.submit_decision(val)

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
