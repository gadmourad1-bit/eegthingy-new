import asyncio
import json
import threading
import websockets

from config import WS_HOST, WS_PORT


class WebSocket:
    """Broadcast-only websocket server: decisions go out to any connected client
    (e.g. the robot). Runs its asyncio loop on a background thread."""

    def __init__(self, host=WS_HOST, port=WS_PORT):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self._stop = None
        self._thread = None

    def start(self):
        ready = threading.Event()

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self._stop = self.loop.create_future()

            async def handler(ws, *_):
                self.clients.add(ws)
                try:
                    await ws.wait_closed()
                finally:
                    self.clients.discard(ws)

            async def main():
                async with websockets.serve(handler, self.host, self.port):
                    ready.set()
                    await self._stop

            try:
                self.loop.run_until_complete(main())
            except Exception as e:
                print(f"websocket server error: {e}")
                ready.set()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(timeout=2.0)
        print(f"websocket server on ws://{self.host}:{self.port}")

    def broadcast(self, payload):
        if self.loop is None or not self.clients:
            return
        msg = json.dumps(payload)

        async def send_all():
            for ws in list(self.clients):
                try:
                    await ws.send(msg)
                except Exception:
                    self.clients.discard(ws)

        asyncio.run_coroutine_threadsafe(send_all(), self.loop)

    def stop(self):
        if self.loop is not None and self._stop is not None and not self._stop.done():
            self.loop.call_soon_threadsafe(self._stop.set_result, None)
