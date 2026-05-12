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
from scipy.interpolate import griddata

from config import EEG_CHANNELS


HISTORY_SECONDS = 30
TICK_MS = 16  # ~60Hz
TOPO_HZ = 15  # topomap refresh rate (off the main thread)
TOPO_GRID = 40  # interpolation grid resolution
FIG_DPI = 80


class GUI:
    """Live visualizer. Tkinter on the main thread, blit for fast redraws.

    Threads:
      - main (tk):  drains prediction/topomap queues, blits at 60Hz
      - topo:       reads bci.get_data(), computes interp grid, pushes to queue
      - smoother:   pushes prediction events into self.queue (from drain thread)
    """

    def __init__(self, bci, smoother, class_labels):
        self.bci = bci
        self.smoother = smoother
        self.class_labels = list(class_labels)
        self.queue = queue.Queue()
        smoother.subscribe(self.queue.put)

        # Time-series history. We append a single point per smoother event
        # and replot relative to "now" each tick.
        max_points = HISTORY_SECONDS * 20  # ample headroom
        self.t_hist = deque(maxlen=max_points)
        self.conf_hist = {c: deque(maxlen=max_points) for c in self.class_labels}
        self.t0 = None

        self.latest_probs = {c: 0.0 for c in self.class_labels}
        self.latest_decision = None
        self.latest_consensus = False

        # Channel layout for the topomap.
        montage = mne.channels.make_standard_montage("standard_1020")
        info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=250.0, ch_types="eeg")
        info.set_montage(montage, on_missing="ignore")
        ch_pos = montage.get_positions()["ch_pos"]
        positions = []
        for name in EEG_CHANNELS:
            p = ch_pos.get(name)
            positions.append([p[0], p[1]] if p is not None else [0.0, 0.0])
        self.positions_2d = np.array(positions)

        # Precompute interpolation grid + head mask once.
        r = float(np.max(np.linalg.norm(self.positions_2d, axis=1))) * 1.15
        self.head_r = r
        xs = np.linspace(-r, r, TOPO_GRID)
        ys = np.linspace(-r, r, TOPO_GRID)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)
        self.head_mask = (self.grid_x ** 2 + self.grid_y ** 2) <= r ** 2
        self.topo_extent = (-r, r, -r, r)

        # Topomap thread bits.
        self.topo_queue = queue.Queue(maxsize=2)
        self.topo_stop = threading.Event()

        self._build_window()
        self._start_topo_thread()

        # Per-axes backgrounds for cheap blit. Recaptured on resize.
        self.bgs = {}  # axes -> background
        self._pending_bg_capture = True
        self._pred_dirty = False  # prediction-driven artists need a redraw
        self._topo_dirty = False  # topomap artist needs a redraw
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self.root.after(50, self._tick)

    # ------------------------------------------------------------------ build

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("EEG Live Viz")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fig = plt.Figure(figsize=(12, 8), dpi=FIG_DPI, constrained_layout=True)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1])
        self.ax_conf = self.fig.add_subplot(gs[0, :])
        self.ax_probs = self.fig.add_subplot(gs[1, 0])
        self.ax_topo = self.fig.add_subplot(gs[1, 1])

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

        # Topomap.
        self.ax_topo.set_title("EEG power (RMS)")
        self.ax_topo.set_xticks([])
        self.ax_topo.set_yticks([])
        self.ax_topo.set_aspect("equal")
        self.ax_topo.set_xlim(-self.head_r, self.head_r)
        self.ax_topo.set_ylim(-self.head_r, self.head_r)
        empty = np.full((TOPO_GRID, TOPO_GRID), np.nan)
        cmap = matplotlib.cm.get_cmap("viridis").copy()
        cmap.set_bad((1, 1, 1, 0))
        self.topo_img = self.ax_topo.imshow(
            empty, extent=self.topo_extent, origin="lower",
            cmap=cmap, animated=True, interpolation="bilinear",
        )
        # Sensor dots — static, drawn into background.
        self.ax_topo.scatter(
            self.positions_2d[:, 0], self.positions_2d[:, 1],
            s=8, c="k", zorder=3,
        )
        head = plt.Circle((0, 0), self.head_r, color="k", fill=False, lw=1)
        self.ax_topo.add_patch(head)

        # Decision label (figure-level text).
        self.decision_text = self.fig.text(
            0.5, 0.97, "—", ha="center", va="top",
            fontsize=24, fontweight="bold", color="gray",
            animated=True,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ----------------------------------------------------------- topo thread

    def _start_topo_thread(self):
        def worker():
            period = 1.0 / TOPO_HZ
            while not self.topo_stop.is_set():
                t0 = time.perf_counter()
                try:
                    data = self.bci.get_data()
                    eeg = data.get("eeg")
                    if eeg is not None and len(eeg) > 0:
                        img = self._interp(eeg)
                        # Drop oldest if queue full — we only care about latest.
                        try:
                            self.topo_queue.put_nowait(img)
                        except queue.Full:
                            try:
                                self.topo_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self.topo_queue.put_nowait(img)
                            except queue.Full:
                                pass
                except Exception as e:
                    print(f"topo worker error: {e}")
                dt = time.perf_counter() - t0
                rest = period - dt
                if rest > 0:
                    time.sleep(rest)

        self.topo_thread = threading.Thread(target=worker, daemon=True)
        self.topo_thread.start()

    def _interp(self, eeg):
        n = min(len(eeg), len(self.positions_2d))
        values = np.zeros(len(self.positions_2d))
        values[:n] = eeg[:n]
        img = griddata(
            self.positions_2d, values, (self.grid_x, self.grid_y),
            method="linear", fill_value=np.nan,
        )
        img = np.where(self.head_mask, img, np.nan)
        return img

    # ----------------------------------------------------------- draw events

    def _on_draw(self, _event):
        if self._pending_bg_capture or not self.bgs:
            self.bgs[self.ax_topo] = self.canvas.copy_from_bbox(self.ax_topo.bbox)
            self.bgs[self.fig] = self.canvas.copy_from_bbox(self.fig.bbox)
            self._pending_bg_capture = False
            # Force a full redraw of animated artists on first frame.
            self._pred_dirty = True
            self._topo_dirty = True

    def _on_resize(self, _event):
        self._pending_bg_capture = True
        self.canvas.draw_idle()

    # ----------------------------------------------------------- per-tick

    def _tick(self):
        # Drain prediction events; mark dirty only if anything arrived.
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

        # Drain latest topomap image; only the last one matters.
        latest_img = None
        while True:
            try:
                latest_img = self.topo_queue.get_nowait()
            except queue.Empty:
                break
        if latest_img is not None:
            self.topo_img.set_data(latest_img)
            finite = latest_img[np.isfinite(latest_img)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if hi - lo < 1e-9:
                    hi = lo + 1e-9
                self.topo_img.set_clim(lo, hi)
            self._topo_dirty = True

        if not self.bgs:
            self.canvas.draw_idle()
            self.root.after(TICK_MS, self._tick)
            return

        # Predictions arrive at ~DELTA_T cadence (a few Hz). When they change,
        # blit the whole figure once — fastest cumulative cost.
        if self._pred_dirty:
            self._refresh_conf_lines()
            self._refresh_probs()
            self._refresh_decision()

            self.canvas.restore_region(self.bgs[self.fig])
            for line in self.conf_lines.values():
                self.ax_conf.draw_artist(line)
            for bar in self.prob_bars:
                self.ax_probs.draw_artist(bar)
            self.ax_topo.draw_artist(self.topo_img)
            self.fig.draw_artist(self.decision_text)
            self.canvas.blit(self.fig.bbox)
            self._pred_dirty = False
            self._topo_dirty = False  # we just redrew the topo too

        elif self._topo_dirty:
            # Topomap updates more frequently (~15Hz). Just blit its axes.
            self.canvas.restore_region(self.bgs[self.ax_topo])
            self.ax_topo.draw_artist(self.topo_img)
            self.canvas.blit(self.ax_topo.bbox)
            self._topo_dirty = False

        self.root.after(TICK_MS, self._tick)

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
        # Right-justify: latest sample at x=0, older samples at negative x.
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
        self.topo_stop.set()
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
