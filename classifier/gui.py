import queue
import threading
import time
import tkinter as tk
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import EEG_CHANNELS

HISTORY_SECONDS = 30
TICK_MS = 16  # ~60Hz GUI tick
TOPO_HZ = 5   # mne.viz.plot_topomap is slow (~150-300ms total for 3 views)
FIG_DPI = 80

class GUI:
    """Live visualizer. Tkinter on main thread; topomap interpolation on
    a worker thread; classification on the BrainFlow drain thread.

    Three topographic views computed by projecting electrode positions:
      top  : axial   (looking down)
      left : sagittal-left
      back : coronal (from behind, subject's left on viewer's left)
    """

    def __init__(self, bci, smoother, class_labels, sfreq):
        self.bci = bci
        self.smoother = smoother
        self.class_labels = list(class_labels)
        self.sfreq = float(sfreq)
        self.queue = queue.Queue()
        smoother.subscribe(self.queue.put)

        max_points = HISTORY_SECONDS * 20
        self.t_hist = deque(maxlen=max_points)
        self.conf_hist = {c: deque(maxlen=max_points) for c in self.class_labels}
        self.t0 = None

        self.latest_probs = {c: 0.0 for c in self.class_labels}
        self.latest_decision = None
        self.latest_consensus = False

        # Channel positions from standard_1020 montage → per-view 2D projections.
        montage = mne.channels.make_standard_montage("standard_1020")
        ch_pos = montage.get_positions()["ch_pos"]
        pos3 = np.array([
            ch_pos[name] if name in ch_pos else np.zeros(3)
            for name in EEG_CHANNELS
        ])
        self.views = self._build_views(pos3)

        self.topo_queue = queue.Queue(maxsize=2)
        self.stop_evt = threading.Event()

        self._build_window()
        self._start_topo_thread()

        self.bgs = {}
        self._pending_bg_capture = True
        self._pred_dirty = False
        self._topo_dirty = False
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self.root.after(50, self._tick)

    def push_window(self, _eeg):
        """Accepted but unused — kept for compatibility with start.py."""
        pass

    # ----------------------------------------------------------------- views

    def _build_views(self, pos3):
        # Rotation R takes head coords → "view space" where +z is the view's
        # up/forward direction. After rotation, channels in the upper
        # hemisphere (z>0) are visible from that view.
        rotations = {
            "top":  np.eye(3, dtype=float),                                     # head as-is
            "left": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float),  # -x → +z
            "back": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),  # -y → +z
        }
        views = {}
        for name, R in rotations.items():
            rotated = pos3 @ R.T
            norms = np.linalg.norm(rotated, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            unit = rotated / norms
            # Azimuthal equidistant: theta = angle from +z, project to 2D disc.
            theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
            phi = np.arctan2(unit[:, 1], unit[:, 0])
            x2d = theta * np.cos(phi)
            y2d = theta * np.sin(phi)
            pos2d = np.column_stack([x2d, y2d])
            visible = unit[:, 2] > -0.05  # upper hemisphere + a sliver of equator
            views[name] = {
                "positions": pos2d[visible],
                "visible": visible,
            }
        return views

    # ----------------------------------------------------------------- build

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("EEG Live Viz")
        self.root.geometry("1400x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fig = plt.Figure(figsize=(14, 8), dpi=FIG_DPI, constrained_layout=True)
        gs = self.fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1.2, 1, 1, 1])
        self.ax_conf = self.fig.add_subplot(gs[0, :])
        self.ax_probs = self.fig.add_subplot(gs[1, 0])
        self.topo_axes = {
            "top":  self.fig.add_subplot(gs[1, 1]),
            "left": self.fig.add_subplot(gs[1, 2]),
            "back": self.fig.add_subplot(gs[1, 3]),
        }

        # Confidence time series.
        self.ax_conf.set_title("confidence over time")
        self.ax_conf.set_xlabel("seconds")
        self.ax_conf.set_ylabel("p(class)")
        self.ax_conf.set_ylim(0, 1)
        self.ax_conf.set_xlim(-HISTORY_SECONDS, 0)
        self.ax_conf.axhline(0.5, color="gray", lw=0.5, ls="--")
        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
        self.conf_lines = {}
        for i, c in enumerate(self.class_labels):
            (line,) = self.ax_conf.plot(
                [], [], color=colors[i % len(colors)], lw=1.5,
                label=f"class {c}", animated=True,
            )
            self.conf_lines[c] = line
        self.ax_conf.legend(loc="upper right")

        # Probability bars.
        self.ax_probs.set_title("current probs")
        self.ax_probs.set_ylim(0, 1)
        self.prob_bars = self.ax_probs.bar(
            [str(c) for c in self.class_labels],
            [0.0] * len(self.class_labels),
            color=[colors[i % len(colors)] for i in range(len(self.class_labels))],
        )
        for bar in self.prob_bars:
            bar.set_animated(True)

        # Three topomap views — mne.viz.plot_topomap renders into these on each refresh.
        self.topo_titles = {"top": "top view", "left": "left view", "back": "back view"}
        for name, ax in self.topo_axes.items():
            ax.set_title(self.topo_titles[name])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        self.decision_text = self.fig.text(
            0.5, 0.99, "—", ha="center", va="top",
            fontsize=24, fontweight="bold", color="gray",
            animated=True,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------ worker

    def _start_topo_thread(self):
        """Just samples EEG values at TOPO_HZ; actual plot_topomap rendering
        happens on the main thread because matplotlib isn't thread-safe."""
        def worker():
            period = 1.0 / TOPO_HZ
            while not self.stop_evt.is_set():
                t0 = time.perf_counter()
                try:
                    data = self.bci.get_data()
                    eeg = data.get("eeg")
                    if eeg is not None and len(eeg) > 0:
                        n = min(len(eeg), len(EEG_CHANNELS))
                        values = np.zeros(len(EEG_CHANNELS))
                        values[:n] = eeg[:n]
                        try:
                            self.topo_queue.put_nowait(values)
                        except queue.Full:
                            try:
                                self.topo_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.topo_queue.put_nowait(values)
                            except queue.Full:
                                pass
                except Exception as e:
                    print(f"[gui] topo worker error: {e}")
                dt = time.perf_counter() - t0
                rest = period - dt
                if rest > 0:
                    time.sleep(rest)

        threading.Thread(target=worker, daemon=True).start()

    # ----------------------------------------------------------- draw events

    def _on_draw(self, _event):
        # Topomap is part of the static background (re-rendered via plot_topomap
        # on each refresh, then captured here).
        if self._pending_bg_capture or not self.bgs:
            self.bgs["fig"] = self.canvas.copy_from_bbox(self.fig.bbox)
            self._pending_bg_capture = False
            self._pred_dirty = True

    def _on_resize(self, _event):
        self._pending_bg_capture = True
        self.canvas.draw_idle()

    # ----------------------------------------------------------- tick

    def _tick(self):
        # 1. Prediction events.
        got_pred = False
        while True:
            try:
                payload = self.queue.get_nowait()
            except queue.Empty:
                break
            self._consume_payload(payload)
            got_pred = True
        if got_pred:
            self._pred_dirty = True

        # 2. Latest EEG values for topomap. Re-render via mne.viz.plot_topomap.
        latest = None
        while True:
            try:
                latest = self.topo_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._render_topomaps(latest)
            # plot_topomap invalidates the background — force a full canvas
            # redraw and recapture on the next draw_event.
            self._pending_bg_capture = True

        if not self.bgs or self._pending_bg_capture:
            self.canvas.draw_idle()
            self.root.after(TICK_MS, self._tick)
            return

        # Prediction-driven artists (conf lines, probs, decision text) → blit.
        if self._pred_dirty:
            self._refresh_conf_lines()
            self._refresh_probs()
            self._refresh_decision()
            self.canvas.restore_region(self.bgs["fig"])
            for line in self.conf_lines.values():
                self.ax_conf.draw_artist(line)
            for bar in self.prob_bars:
                self.ax_probs.draw_artist(bar)
            self.fig.draw_artist(self.decision_text)
            self.canvas.blit(self.fig.bbox)
            self._pred_dirty = False

        self.root.after(TICK_MS, self._tick)

    def _render_topomaps(self, values):
        # Common color limits across views for comparability.
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if hi - lo < 1e-9:
                hi = lo + 1e-9
        else:
            lo, hi = 0.0, 1.0

        for name, ax in self.topo_axes.items():
            v = self.views[name]
            vals = values[v["visible"]]
            ax.clear()
            try:
                mne.viz.plot_topomap(
                    vals, v["positions"],
                    axes=ax, show=False,
                    cmap="viridis", vlim=(lo, hi),
                    sensors=True, contours=0,
                    outlines="head", extrapolate="head",
                )
            except Exception as e:
                print(f"[gui] plot_topomap error: {e}")
            ax.set_title(self.topo_titles[name])

    def _consume_payload(self, payload):
        ts = payload["timestamp"]
        if self.t0 is None:
            self.t0 = ts
        rel_t = ts - self.t0

        last = payload["buffer"][-1]
        probs = {int(k): v for k, v in last["probs"].items()}
        self.latest_probs = probs

        self.t_hist.append(rel_t)
        for c in self.class_labels:
            self.conf_hist[c].append(probs.get(c, 0.0))

        smoothed = payload["smoothed"]
        self.latest_decision = smoothed["decision"]
        self.latest_consensus = smoothed["consensus"]

    def _refresh_conf_lines(self):
        if not self.t_hist:
            return
        now = self.t_hist[-1]
        ts = np.fromiter(self.t_hist, dtype=float) - now
        for c, line in self.conf_lines.items():
            line.set_data(ts, np.fromiter(self.conf_hist[c], dtype=float))

    def _refresh_probs(self):
        for bar, c in zip(self.prob_bars, self.class_labels):
            bar.set_height(self.latest_probs.get(c, 0.0))

    def _refresh_decision(self):
        if self.latest_decision is None:
            self.decision_text.set_text("—")
            self.decision_text.set_color("gray")
        else:
            self.decision_text.set_text(f"class {self.latest_decision}")
            self.decision_text.set_color("tab:green" if self.latest_consensus else "gray")

    # ----------------------------------------------------------- shutdown

    def _on_close(self):
        self.stop_evt.set()
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
