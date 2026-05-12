import queue
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
TICK_MS = 100


class GUI:
    """Live visualizer. Owns the tkinter mainloop on the main thread.

    Receives prediction events from the smoother (thread-safe queue) and
    pulls raw EEG features from the shared OpenBCI instance via get_data().
    """

    def __init__(self, bci, smoother, class_labels):
        self.bci = bci
        self.smoother = smoother
        self.class_labels = list(class_labels)
        self.queue = queue.Queue()
        smoother.subscribe(self.queue.put)

        max_points = HISTORY_SECONDS * int(1000 / TICK_MS)
        self.t_hist = deque(maxlen=max_points)
        self.conf_hist = {c: deque(maxlen=max_points) for c in self.class_labels}
        self.t0 = None

        self.latest_probs = {c: 0.0 for c in self.class_labels}
        self.latest_decision = None
        self.latest_consensus = False

        montage = mne.channels.make_standard_montage("standard_1020")
        self.info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=250.0, ch_types="eeg")
        self.info.set_montage(montage, on_missing="ignore")

        self._build_window()
        self._tick()

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("EEG Live Viz")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1])
        self.ax_conf = self.fig.add_subplot(gs[0, :])
        self.ax_probs = self.fig.add_subplot(gs[1, 0])
        self.ax_topo = self.fig.add_subplot(gs[1, 1])

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
                [], [], color=colors[i % len(colors)], lw=1.5, label=f"class {c}"
            )
            self.conf_lines[c] = line
        self.ax_conf.legend(loc="upper right")

        self.ax_probs.set_title("current probs")
        self.ax_probs.set_ylim(0, 1)
        self.prob_bars = self.ax_probs.bar(
            [str(c) for c in self.class_labels],
            [0.0] * len(self.class_labels),
            color=[colors[i % len(colors)] for i in range(len(self.class_labels))],
        )

        self.ax_topo.set_title("EEG power (RMS)")
        self.ax_topo.set_xticks([])
        self.ax_topo.set_yticks([])

        self.decision_text = self.fig.text(
            0.5, 0.97, "—", ha="center", va="top",
            fontsize=24, fontweight="bold", color="gray",
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _tick(self):
        any_update = False
        while True:
            try:
                payload = self.queue.get_nowait()
            except queue.Empty:
                break
            self._consume_payload(payload)
            any_update = True

        if any_update:
            self._refresh_conf_line()
            self._refresh_probs()
            self._refresh_decision()

        self._refresh_topo()

        self.canvas.draw_idle()
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

    def _refresh_conf_line(self):
        if not self.t_hist:
            return
        now = self.t_hist[-1]
        ts = np.array(self.t_hist) - now
        for c, line in self.conf_lines.items():
            line.set_data(ts, list(self.conf_hist[c]))
        self.ax_conf.set_xlim(-HISTORY_SECONDS, 0)

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

    def _refresh_topo(self):
        data = self.bci.get_data()
        eeg = data.get("eeg")
        if eeg is None or len(eeg) == 0:
            return
        n = min(len(eeg), len(EEG_CHANNELS))
        values = np.zeros(len(EEG_CHANNELS))
        values[:n] = eeg[:n]
        self.ax_topo.clear()
        self.ax_topo.set_title("EEG power (RMS)")
        try:
            mne.viz.plot_topomap(
                values, self.info, axes=self.ax_topo, show=False,
                cmap="viridis", contours=0, sensors=True,
            )
        except Exception as e:
            self.ax_topo.text(
                0.5, 0.5, f"topomap error:\n{e}",
                transform=self.ax_topo.transAxes, ha="center", va="center", fontsize=8,
            )

    def _on_close(self):
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
