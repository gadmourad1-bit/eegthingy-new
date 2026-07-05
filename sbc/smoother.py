import time
import threading
from collections import deque, Counter


class Smoother:
    """M-of-N temporal vote + dwell gate over per-window predictions. Params come
    from the exported model. Every add() broadcasts the decision over the websocket."""

    def __init__(self, ws, n, m, conf_floor, dwell):
        if m > n:
            raise ValueError(f"M ({m}) cannot exceed N ({n})")
        self.ws = ws
        self.n = int(n)
        self.m = int(m)
        self.conf_floor = float(conf_floor)
        self.dwell = int(dwell)
        self.buffer = deque(maxlen=self.n)
        self.dwell_decision = 0
        self.dwell_count = 0
        self.lock = threading.Lock()

    def add(self, prediction, confidence, probs):
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

        self.ws.broadcast({
            "smoothed": {
                "decision": decision, "consensus": consensus, "final": final,
                "dwell": self.dwell, "dwell_count": self.dwell_count,
                "votes": {str(k): v for k, v in votes.items()},
                "abstain": sum(1 for e in snapshot if e["vote"] is None),
                "n": self.n, "m": self.m, "conf_floor": self.conf_floor,
            },
            "buffer": snapshot,
            "timestamp": time.time(),
        })
        return decision, consensus, final

    def _vote(self, snapshot):
        votes = Counter(e["vote"] for e in snapshot if e["vote"] is not None)
        if not votes:
            return 0, False, {}
        winner, count = votes.most_common(1)[0]
        if count >= self.m:
            return winner, True, dict(votes)
        return 0, False, dict(votes)
