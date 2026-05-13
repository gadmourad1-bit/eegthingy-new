import time
import threading
from collections import deque, Counter

from config import SMOOTHER_N, SMOOTHER_M, SMOOTHER_CONF_FLOOR


class Smoother:
    """M-of-N temporal smoother over per-window predictions.

    Pre-vote gate: predictions with confidence < conf_floor are recorded as
    abstain (None) and don't count toward any class. Of the N most recent
    entries, a class wins only if it has >= M votes.

    On every add(), broadcasts the full buffer state + the smoothed decision
    via the websocket's broadcast() method.
    """

    def __init__(self, ws, n=SMOOTHER_N, m=SMOOTHER_M, conf_floor=SMOOTHER_CONF_FLOOR):
        if m > n:
            raise ValueError(f"M ({m}) cannot exceed N ({n})")
        self.ws = ws
        self.n = n
        self.m = m
        self.conf_floor = conf_floor
        self.buffer = deque(maxlen=n)
        self.lock = threading.Lock()
        self.subscribers = []

    def subscribe(self, fn):
        """Register a callback that receives the broadcast payload."""
        self.subscribers.append(fn)

    def add(self, prediction, confidence, probs):
        """Record one window's result, vote, and broadcast.

        prediction: class label (int)
        confidence: max prob (0..1)
        probs: dict {class_label: prob}
        """
        vote = int(prediction) if confidence >= self.conf_floor else None

        entry = {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "probs": {str(k): float(v) for k, v in probs.items()},
            "vote": vote,
            "timestamp": time.time(),
        }

        with self.lock:
            self.buffer.append(entry)
            snapshot = list(self.buffer)

        decision, consensus, votes = self._vote(snapshot)

        payload = {
            "smoothed": {
                "decision": decision,
                "consensus": consensus,
                "votes": {str(k): v for k, v in votes.items()},
                "abstain": sum(1 for e in snapshot if e["vote"] is None),
                "n": self.n,
                "m": self.m,
                "conf_floor": self.conf_floor,
            },
            "buffer": snapshot,
            "timestamp": time.time(),
        }

        self.ws.broadcast(payload)
        for fn in self.subscribers:
            try:
                fn(payload)
            except Exception as e:
                print(f"smoother subscriber error: {e}")

        return decision, consensus

    def _vote(self, snapshot):
        votes = Counter(e["vote"] for e in snapshot if e["vote"] is not None)
        if not votes:
            return None, False, {}
        winner, count = votes.most_common(1)[0]
        if count >= self.m:
            return winner, True, dict(votes)
        return None, False, dict(votes)
