import time
import threading
from collections import deque, Counter

from config import SMOOTHER_N, SMOOTHER_M, SMOOTHER_DWELL, NORM_CONF_FLOOR


class Smoother:
    """M-of-N temporal smoother over per-window predictions.

    Pre-vote gate: predictions with confidence < conf_floor are recorded as
    abstain (None) and don't count toward any class. Of the N most recent
    entries, a class wins only if it has >= M votes.

    Dwell gate (post-consensus): a consensus class is only committed as the
    'final' decision once it has been the consensus for `dwell` consecutive
    windows. Any change of consensus class, or loss of consensus, resets the
    dwell counter to zero.

    On every add(), broadcasts the full buffer state + the smoothed decision
    via the websocket's broadcast() method.
    """

    def __init__(self, ws, n=SMOOTHER_N, m=SMOOTHER_M, conf_floor=NORM_CONF_FLOOR,
                 dwell=SMOOTHER_DWELL):
        if m > n:
            raise ValueError(f"M ({m}) cannot exceed N ({n})")
        self.ws = ws
        self.n = n
        self.m = m
        self.conf_floor = conf_floor
        self.dwell = dwell
        self.buffer = deque(maxlen=n)
        self.dwell_decision = 0
        self.dwell_count = 0
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

        if consensus and decision == self.dwell_decision:
            self.dwell_count += 1
        elif consensus:
            self.dwell_decision = decision
            self.dwell_count = 1
        else:
            self.dwell_decision = 0
            self.dwell_count = 0
        final = decision if consensus and self.dwell_count >= self.dwell else 0

        payload = {
            "smoothed": {
                "decision": decision,
                "consensus": consensus,
                "final": final,
                "dwell": self.dwell,
                "dwell_count": self.dwell_count,
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

        return decision, consensus, final

    def _vote(self, snapshot):
        votes = Counter(e["vote"] for e in snapshot if e["vote"] is not None)
        if not votes:
            return 0, False, {}
        winner, count = votes.most_common(1)[0]
        if count >= self.m:
            return winner, True, dict(votes)
        return 0, False, dict(votes)
