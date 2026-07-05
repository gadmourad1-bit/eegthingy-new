"""Headless real-time MI decoder for an SBC — numpy + scipy + brainflow + websockets.

No mne, no scikit-learn, no matplotlib, no GUI. Loads a model.npz exported by
export_model.py (on a laptop), acquires EEG from an OpenBCI Cyton+Daisy, runs a
live EA calibration, decodes left/right motor imagery every 0.2 s with the exact
same math as EA + FB-CSP (verified bit-for-bit), smooths with M-of-N + dwell, and
broadcasts the committed decision over a websocket.

    ./lean-online                         # model.npz next to the binary, real board
    ./lean-online --model /path/model.npz
    ./lean-online --synthetic             # brainflow synthetic board (no hardware)
"""
import argparse
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import scipy.signal as sig
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from ws import WebSocket
from smoother import Smoother


def default_model_path():
    """model.npz sits next to the executable (frozen) or this script (dev), so it
    can be swapped to retrain a subject without rebuilding the binary."""
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else _HERE
    return os.path.join(base, "model.npz")


# ----------------------------- model + inference -----------------------------
class Model:
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.taps = [d["fir_taps"][b, :int(d["fir_lens"][b])] for b in range(len(d["fir_lens"]))]
        self.csp_filters = d["csp_filters"]              # (nb, nc, C)
        self.lda_coef = d["lda_coef"]                    # (nb*nc,)
        self.lda_intercept = float(d["lda_intercept"])
        self.classes = d["classes"]                      # (2,)
        self.sfreq = float(d["sfreq"])
        self.window_n = int(d["window_n"])
        self.nb = self.csp_filters.shape[0]
        self.targets = [str(x) for x in d["channels_targets"]]
        self.mapping = [str(x) for x in d["channels_mapping"]]
        self.warmup_s = float(d["warmup_s"])
        self.stride_s = float(d["stride_s"])
        self.cal_seconds = int(d["calibration_seconds"])
        sm = d["smoother"]
        self.sm_n, self.sm_m, self.sm_conf, self.sm_dwell = int(sm[0]), int(sm[1]), float(sm[2]), int(sm[3])
        self.ref_white = None  # set by calibration

    def bandbank(self, buf):
        """buf (C, >=buffer) -> filter-bank window (nb, C, window_n) via causal FIR."""
        return np.stack([sig.lfilter(self.taps[b], 1.0, buf, axis=1)[:, -self.window_n:]
                         for b in range(self.nb)], axis=0)

    @staticmethod
    def _whitener(Xb):  # Xb (n, C, T) -> R^{-1/2}  (Euclidean Alignment)
        R = np.einsum("nct,ndt->cd", Xb, Xb) / (len(Xb) * Xb.shape[2])
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 1e-12, None)
        return (V * (w ** -0.5)) @ V.T

    def set_reference(self, cal_windows):   # cal_windows (n, nb, C, T)
        self.ref_white = [self._whitener(cal_windows[:, b]) for b in range(self.nb)]

    def proba(self, fb_window):             # fb_window (nb, C, T) -> proba (2,)
        feats = []
        for b in range(self.nb):
            al = self.ref_white[b] @ fb_window[b]                 # EA whiten
            proj = self.csp_filters[b] @ al                       # CSP spatial filter
            feats.append(np.log((proj ** 2).mean(axis=1)))        # log-variance
        f = np.concatenate(feats)
        p1 = 1.0 / (1.0 + np.exp(-(f @ self.lda_coef + self.lda_intercept)))
        return np.array([1.0 - p1, p1])


# ----------------------------- OpenBCI acquisition ---------------------------
class OpenBCI:
    def __init__(self, interval, synthetic=False, serial_port=None):
        self.interval = interval
        self.synthetic = synthetic
        self.serial_port = serial_port
        self.board = None
        self.eeg = []
        self.callback = None
        self._recording = False
        self._thread = None

    def open(self):
        try:
            params = BrainFlowInputParams()
            params.timeout = 30
            if self.synthetic:
                bid = BoardIds.SYNTHETIC_BOARD.value
            else:
                if self.serial_port:
                    params.serial_port = self.serial_port
                bid = BoardIds.CYTON_DAISY_BOARD.value
            board = BoardShim(bid, params)
            board.prepare_session()
            time.sleep(1)
            board.start_stream()
            self.board = board
            self.eeg = BoardShim.get_eeg_channels(bid)
            print(f"board {'SYNTHETIC' if self.synthetic else 'CYTON_DAISY'} open · eeg channels: {self.eeg}")
        except Exception as e:
            print("failed connecting to OpenBCI:", e)
        return self

    def start(self):
        if self.board is None:
            return self
        self._recording = True
        self.board.get_board_data()  # discard buffered

        def drain():
            while self._recording:
                time.sleep(self.interval)
                if self.board is not None and self._recording:
                    chunk = self.board.get_board_data()
                    if chunk.size and self.callback is not None:
                        self.callback(chunk)

        self._thread = threading.Thread(target=drain, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2.0)
        return self

    def close(self):
        self.stop()
        if self.board is not None:
            self.board.stop_stream()
            self.board.release_session()
            self.board = None
        return self


def calibrate(bci, model, train_idx, window_n, warmup_n, seconds):
    """Stream `seconds` of EEG, slice into filter-bank windows, return them so the
    EA whitener can be computed — mirrors record_calibration()."""
    buffer_n = window_n + warmup_n
    collected, lock = [], threading.Lock()

    def collect(chunk):
        eeg_all = chunk[bci.eeg, :]
        if eeg_all.shape[1] == 0 or max(train_idx) >= eeg_all.shape[0]:
            return
        with lock:
            collected.append(eeg_all[train_idx, :].astype(np.float64) / 1e6)

    bci.callback = collect
    bci.start()
    print(f"\nCalibration: imagine the task for {seconds}s…")
    for s in range(seconds, 0, -1):
        print(f"  {s:2d}", end="\r", flush=True)
        time.sleep(1)
    bci.stop()
    bci.callback = None

    with lock:
        full = np.hstack(collected) if collected else np.zeros((len(train_idx), 0))
    if full.shape[1] < buffer_n:
        return None
    step = max(1, int(round(0.5 * model.sfreq)))
    windows = [model.bandbank(full[:, end - buffer_n:end])
               for end in range(buffer_n, full.shape[1] + 1, step)]
    return np.array(windows)


def main():
    ap = argparse.ArgumentParser(description="Lean headless SBC MI decoder")
    ap.add_argument("--model", default=None, help="path to model.npz (default: next to the binary)")
    ap.add_argument("--device", choices=["usb", "synthetic"], default=None,
                    help="EEG source: usb = OpenBCI Cyton+Daisy, synthetic = brainflow test board "
                         "(prompts if omitted on a terminal, else defaults to usb)")
    ap.add_argument("--synthetic", action="store_true", help="alias for --device synthetic")
    ap.add_argument("--serial-port", default=None)
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--cal-seconds", type=int, default=None, help="override calibration length")
    ap.add_argument("--run-seconds", type=int, default=None, help="exit after N seconds (testing)")
    ap.add_argument("--ws-host", default="0.0.0.0")
    ap.add_argument("--ws-port", type=int, default=8765)
    args = ap.parse_args()

    model_path = args.model or default_model_path()
    if not os.path.exists(model_path):
        print(f"model not found: {model_path} (export it with export_model.py and copy it next to the binary)")
        return
    model = Model(model_path)
    print(f"loaded model: {model.nb} bands, {model.window_n}-sample window, sfreq={model.sfreq:g}, "
          f"classes={[int(c) for c in model.classes]}")
    train_idx = [model.mapping.index(t) for t in model.targets]

    # device selector: --device / --synthetic, else prompt on a terminal, else usb
    device = args.device or ("synthetic" if args.synthetic else None)
    if device is None:
        if sys.stdin.isatty():
            print("\nSelect EEG device:")
            print("  1) USB       — OpenBCI Cyton+Daisy")
            print("  2) Synthetic — brainflow test board (no hardware)")
            while True:
                c = input("> ").strip().lower()
                if c in ("1", "usb"):
                    device = "usb"; break
                if c in ("2", "synthetic", "synth"):
                    device = "synthetic"; break
                print("pick 1 or 2")
        else:
            device = "usb"
    synthetic = (device == "synthetic")
    print(f"device: {device}")

    ws = WebSocket(args.ws_host, args.ws_port); ws.start()
    smoother = Smoother(ws, model.sm_n, model.sm_m, model.sm_conf, model.sm_dwell)

    bci = OpenBCI(model.stride_s, synthetic=synthetic, serial_port=args.serial_port).open()
    if bci.board is None:
        print("board not connected — aborting"); ws.stop(); return

    sfreq = float(BoardShim.get_sampling_rate(bci.board.board_id))
    if abs(sfreq - model.sfreq) > 0.5:
        print(f"warning: live sfreq={sfreq} != model sfreq={model.sfreq}; filters assume {model.sfreq}")
    window_n = model.window_n
    warmup_n = int(round(model.warmup_s * model.sfreq))
    buffer_n = window_n + warmup_n
    buffer = np.zeros((len(train_idx), 0), dtype=np.float64)
    buffer_lock = threading.Lock()

    if not args.no_calibration:
        cal_secs = args.cal_seconds if args.cal_seconds is not None else model.cal_seconds
        cal = calibrate(bci, model, train_idx, window_n, warmup_n, cal_secs)
        if cal is not None and len(cal) >= 2:
            model.set_reference(cal)
            print(f"aligned on {len(cal)} calibration windows.")
        else:
            print("calibration produced too little data — cannot align; aborting.")
            bci.close(); ws.stop(); return
    else:
        print("no calibration: EA needs a reference; aborting.")
        bci.close(); ws.stop(); return

    prev_t = [None]

    def on_chunk(chunk):
        nonlocal buffer
        eeg_all = chunk[bci.eeg, :]
        if eeg_all.shape[1] == 0 or max(train_idx) >= eeg_all.shape[0]:
            return
        eeg = eeg_all[train_idx, :].astype(np.float64, copy=False) / 1e6
        with buffer_lock:
            buffer = np.hstack([buffer, eeg])
            if buffer.shape[1] > buffer_n:
                buffer = buffer[:, -buffer_n:]
            if buffer.shape[1] < buffer_n:
                return
            buf = buffer.copy()

        proba = model.proba(model.bandbank(buf))
        idx = int(np.argmax(proba))
        pred = int(model.classes[idx]); conf = float(proba[idx])
        decision, consensus, final = smoother.add(
            prediction=pred, confidence=conf,
            probs={int(c): float(p) for c, p in zip(model.classes, proba)})

        now = time.perf_counter()
        dt = (now - prev_t[0]) * 1000 if prev_t[0] is not None else 0.0
        prev_t[0] = now
        tag = decision if consensus else "—"
        commit = f"  ✓ COMMIT {final}" if final else ""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        bd = ", ".join(f"{int(c)}={p*100:.0f}%" for c, p in zip(model.classes, proba))
        print(f"[{ts}  Δ{dt:5.0f}ms]  raw:{pred} {conf*100:4.0f}%  [{bd}]  → {tag}{commit}")

    bci.callback = on_chunk
    bci.start()
    print("\nrunning — decisions broadcasting over websocket. Ctrl-C to stop.\n")
    t0 = time.time()
    try:
        while True:
            time.sleep(1)
            if args.run_seconds and (time.time() - t0) >= args.run_seconds:
                print("run-seconds reached, stopping.")
                break
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        bci.stop(); bci.close(); ws.stop()


if __name__ == "__main__":
    main()
