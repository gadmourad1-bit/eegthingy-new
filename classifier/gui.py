import math
import numpy as np
import tkinter as tk
import queue

from collections import deque
from config import FB_BANDS, GUI_HISTORY_S, GUI_REFRESH_RATE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba


NEG_COLOR = "#2a78d6"  # classes_[0] side (score < 0)
POS_COLOR = "#e34948"  # classes_[1] side (score > 0)
MID_COLOR = "#8a8780"  # inside the abstain band
TRAIL_LEN = 40


class GUI:
    def __init__(self, bci, smoother, class_labels, sfreq,
                 band_sep=None, band_abs_max=None):
        self.bci = bci
        self.smoother = smoother
        self.class_labels = list(class_labels)
        self.sfreq = float(sfreq)
        self.queue = queue.Queue()
        self.viz_queue = queue.Queue()
        self.tick_ms = math.floor((1 / GUI_REFRESH_RATE) * 1000)
        smoother.subscribe(self.queue.put)

        max_points = GUI_HISTORY_S * 20
        self.t_hist = deque(maxlen=max_points)
        self.conf_hist = {c: deque(maxlen=max_points) for c in self.class_labels}
        self.t0 = None

        self.latest_probs = {c: 0.0 for c in self.class_labels}
        self.latest_decision = None
        self.latest_consensus = False
        self.latest_final = 0
        self.latest_dwell = 0
        self.latest_dwell_count = 0
        self.transition_hist = deque(maxlen=max_points)
        self.last_decision = None
        self.decision_colors = {}
        if len(self.class_labels) > 0:
            self.decision_colors[self.class_labels[0]] = NEG_COLOR
        if len(self.class_labels) > 1:
            self.decision_colors[self.class_labels[1]] = POS_COLOR

        cf = float(getattr(smoother, "conf_floor", 0.9))
        cf = min(max(cf, 1e-3), 1 - 1e-3)
        self.gate = math.log(cf / (1.0 - cf))

        self.band_sep = None if band_sep is None else np.asarray(band_sep, float)
        self.band_abs_max = float(band_abs_max) if band_abs_max else 1.0
        self.n_bands = len(FB_BANDS)
        self.best_band = int(np.argmax(self.band_sep)) if self.band_sep is not None else None

        self.score_lim = max(6.0, self.gate * 2.5)

        self.score_hist = deque(maxlen=TRAIL_LEN)
        self.latest_band_sig = None

        self._build_window()

        self.bg = None
        self._dirty = False
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self.root.after(50, self._tick)

    def push_decision(self, score, band_signal):
        """Called from the inference thread with the live discriminant score and the
        per-band signed contributions that sum (with bias) to it."""
        self.viz_queue.put((float(score), np.asarray(band_signal, dtype=float)))

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("EEG Classifier Visualizer")
        win_w, win_h = 760, 440
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fig = Figure(figsize=(14, 8), constrained_layout=True)
        gs = self.fig.add_gridspec(
            2, 4, height_ratios=[1, 1.15], width_ratios=[0.9, 1.05, 1.05, 0.95]
        )
        self.ax_conf = self.fig.add_subplot(gs[0, :])
        self.ax_probs = self.fig.add_subplot(gs[1, 0])
        self.ax_axis = self.fig.add_subplot(gs[1, 1:3])
        self.ax_bands = self.fig.add_subplot(gs[1, 3])

        self._build_conf()
        self._build_probs()
        self._build_axis()
        self._build_bands()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_conf(self):
        self.ax_conf.set_title("confidence over time")
        self.ax_conf.set_xlabel("seconds")
        self.ax_conf.set_ylabel("p(class)")
        self.ax_conf.set_ylim(0, 1)
        self.ax_conf.set_xlim(-GUI_HISTORY_S, 0)
        self.ax_conf.axhline(0.5, color="gray", lw=0.5, ls="--")

        colors = [NEG_COLOR, POS_COLOR, "tab:green", "tab:orange"]
        self.conf_lines = {}
        for i, c in enumerate(self.class_labels):
            (line,) = self.ax_conf.plot(
                [], [], color=colors[i % len(colors)], lw=1.5,
                label=f"class {c}", animated=True,
            )
            self.conf_lines[c] = line
        self.ax_conf.legend(loc="upper right")

        self.onset_collections = {}
        for c in self.class_labels:
            lc = LineCollection(
                [], colors=self.decision_colors.get(c, "gray"), alpha=0.7,
                lw=1.2, linestyles="dashed", animated=True, zorder=1,
            )
            self.ax_conf.add_collection(lc)
            self.onset_collections[c] = lc

    def _build_probs(self):
        self.ax_probs.set_title("current probs")
        self.ax_probs.set_ylim(0, 1)
        colors = [NEG_COLOR, POS_COLOR, "tab:green", "tab:orange"]
        self.prob_bars = self.ax_probs.bar(
            [str(c) for c in self.class_labels],
            [0.0] * len(self.class_labels),
            color=[colors[i % len(colors)] for i in range(len(self.class_labels))],
        )
        for bar in self.prob_bars:
            bar.set_animated(True)

    def _build_axis(self):
        ax = self.ax_axis
        sl = self.score_lim
        neg_c = self.class_labels[0] if len(self.class_labels) > 0 else 0
        pos_c = self.class_labels[1] if len(self.class_labels) > 1 else 1

        ax.set_title("live decision axis")
        ax.set_xlabel("discriminant score  (log-odds)")
        ax.set_xlim(-sl, sl)
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        ax.axvspan(-sl, -self.gate, color=NEG_COLOR, alpha=0.07, zorder=0)
        ax.axvspan(self.gate, sl, color=POS_COLOR, alpha=0.07, zorder=0)
        ax.axvline(0, color="#444", lw=1.6, zorder=3)
        for g in (-self.gate, self.gate):
            ax.axvline(g, color="#666", lw=1.0, ls="--", zorder=3)

        ax.text(-sl * 0.96, 0.485, f"← class {neg_c}", ha="left", va="center",
                fontsize=9, color=NEG_COLOR, fontweight="bold")
        ax.text(sl * 0.96, 0.485, f"class {pos_c} →", ha="right", va="center",
                fontsize=9, color=POS_COLOR, fontweight="bold")
        ax.text(0, 0.485, "abstain", ha="center", va="center", fontsize=8, color=MID_COLOR)

        self.live_scatter = ax.scatter([], [], animated=True, zorder=6, edgecolors="none")
        self.decision_text = ax.text(
            0.5, 0.045, "—", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#bbb", animated=True, zorder=7,
        )

    def _build_bands(self):
        ax = self.ax_bands
        m = self.band_abs_max
        ax.set_title("band contribution")
        ax.set_xlim(-m * 1.05, m * 1.05)
        ax.set_ylim(self.n_bands - 0.5, -0.5)
        ax.axvline(0, color="#444", lw=1.0, zorder=3)

        ypos = np.arange(self.n_bands)
        self.band_bars = ax.barh(ypos, [0.0] * self.n_bands, height=0.66,
                                 color=MID_COLOR, zorder=2)
        for bar in self.band_bars:
            bar.set_animated(True)

        labels = [f"{int(l)}–{int(h)}" for (l, h) in FB_BANDS]
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", labelsize=8)
        if self.best_band is not None:
            ax.get_yticklabels()[self.best_band].set_fontweight("bold")

    def _on_draw(self, _event):
        self.bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._dirty = True

    def _on_resize(self, _event):
        self.bg = None
        self.canvas.draw_idle()

    def _tick(self):
        while True:
            try:
                self._consume_payload(self.queue.get_nowait())
            except queue.Empty:
                break
            self._dirty = True

        while True:
            try:
                score, band_sig = self.viz_queue.get_nowait()
            except queue.Empty:
                break
            self.score_hist.append(score)
            self.latest_band_sig = band_sig
            self._dirty = True

        if self.bg is None:
            self.canvas.draw_idle()
            self.root.after(self.tick_ms, self._tick)
            return

        if self._dirty:
            self._refresh_onsets()
            self._refresh_conf_lines()
            self._refresh_probs()
            self._refresh_axis()
            self._refresh_bands()
            self._refresh_decision()
            self.canvas.restore_region(self.bg)
            for lc in self.onset_collections.values():
                self.ax_conf.draw_artist(lc)
            for line in self.conf_lines.values():
                self.ax_conf.draw_artist(line)
            for bar in self.prob_bars:
                self.ax_probs.draw_artist(bar)
            self.ax_axis.draw_artist(self.live_scatter)
            for bar in self.band_bars:
                self.ax_bands.draw_artist(bar)
            self.ax_axis.draw_artist(self.decision_text)
            self.canvas.blit(self.fig.bbox)
            self._dirty = False

        self.root.after(self.tick_ms, self._tick)

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
        self.latest_final = smoothed.get("final", 0)
        self.latest_dwell = smoothed.get("dwell", 0)
        self.latest_dwell_count = smoothed.get("dwell_count", 0)

        new_d = self.latest_final or None
        if new_d != self.last_decision:
            if new_d is not None:
                self.transition_hist.append((rel_t, new_d))
            self.last_decision = new_d

    def _refresh_conf_lines(self):
        if not self.t_hist:
            return
        now = self.t_hist[-1]
        ts = np.fromiter(self.t_hist, dtype=float) - now
        for c, line in self.conf_lines.items():
            line.set_data(ts, np.fromiter(self.conf_hist[c], dtype=float))

    def _refresh_onsets(self):
        if not self.t_hist:
            for lc in self.onset_collections.values():
                lc.set_segments([])
            return
        now = self.t_hist[-1]
        per_class = {c: [] for c in self.class_labels}
        for rel_t, d in self.transition_hist:
            x = rel_t - now
            if x < -GUI_HISTORY_S or x > 0:
                continue
            if d in per_class:
                per_class[d].append([(x, 0.0), (x, 1.0)])
        for c, segs in per_class.items():
            self.onset_collections[c].set_segments(segs)

    def _refresh_probs(self):
        for bar, c in zip(self.prob_bars, self.class_labels):
            bar.set_height(self.latest_probs.get(c, 0.0))

    def _refresh_axis(self):
        n = len(self.score_hist)
        if n == 0:
            self.live_scatter.set_offsets(np.empty((0, 2)))
            return
        scores = np.fromiter(self.score_hist, dtype=float)
        xs = np.clip(scores, -self.score_lim * 0.985, self.score_lim * 0.985)
        frac = (n - 1 - np.arange(n)) / max(n - 1, 1)
        ys = 0.53 + frac * 0.44
        sizes = 150 * (1 - 0.78 * frac) + 12
        alphas = np.clip(1.0 - 0.82 * frac, 0.12, 1.0)
        base = np.where(scores > self.gate, POS_COLOR,
                        np.where(scores < -self.gate, NEG_COLOR, MID_COLOR))
        rgba = np.array([to_rgba(base[i], alphas[i]) for i in range(n)])
        self.live_scatter.set_offsets(np.column_stack([xs, ys]))
        self.live_scatter.set_sizes(sizes)
        self.live_scatter.set_color(rgba)

    def _refresh_bands(self):
        sig = self.latest_band_sig
        if sig is None:
            return
        dom = int(np.argmax(np.abs(sig)))
        for i, bar in enumerate(self.band_bars):
            if i >= len(sig):
                break
            w = float(sig[i])
            bar.set_width(w)
            col = POS_COLOR if w >= 0 else NEG_COLOR
            bar.set_color(to_rgba(col, 1.0 if i == dom else 0.32))

    def _refresh_decision(self):
        if self.latest_final:
            self.decision_text.set_text(f"decision: class {self.latest_final}")
            self.decision_text.set_color(self.decision_colors.get(self.latest_final, "#444"))
        elif self.latest_consensus and self.latest_decision:
            self.decision_text.set_text(
                f"class {self.latest_decision}?  {self.latest_dwell_count}/{self.latest_dwell}")
            self.decision_text.set_color("#8a8780")
        else:
            self.decision_text.set_text("decision: —")
            self.decision_text.set_color("#bbb")

    def _on_close(self):
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
