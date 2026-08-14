import math
import numpy as np
import tkinter as tk
import queue

from collections import deque
from config import FB_BANDS, GUI_HISTORY_S, GUI_REFRESH_RATE, BOUNDARY_STEP
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba


NEG_COLOR = "#2a78d6"  # classes_[0] side (score < 0)
POS_COLOR = "#e34948"  # classes_[1] side (score > 0)
MID_COLOR = "#8a8780"  # inside the abstain band
CONSENSUS_COLOR = "#ff9500"  # edge highlight for windows that held M-of-N consensus
TRAIL_LEN = 40
SAMPLE_HIST_LEN = 10


class GUI:
    def __init__(self, bci, smoother, class_labels, sfreq,
                 band_sep=None, band_abs_max=None, recenter=None, base_center=0.0):
        self.bci = bci
        self.smoother = smoother
        self.recenter = recenter
        self.base_center = float(base_center)   # plot origin: boundary at calibration
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
        self.conf_floor = cf

        self.band_sep = None if band_sep is None else np.asarray(band_sep, float)
        self.band_abs_max = float(band_abs_max) if band_abs_max else 1.0
        self.n_bands = len(FB_BANDS)
        self.best_band = int(np.argmax(self.band_sep)) if self.band_sep is not None else None

        self.score_lim = max(6.0, self.gate * 2.5)

        self.band_on = [True] * self.n_bands    # read by the inference thread
        self.manual_total = 0.0                 # cumulative arrow nudges (display only)
        self.latest_center = self.base_center

        self.score_hist = deque(maxlen=TRAIL_LEN)   # (score, center) pairs
        self.latest_band_sig = None
        self.sample_hist = deque(maxlen=SAMPLE_HIST_LEN)

        self._build_window()

        self.bg = None
        self._dirty = False
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self.root.after(50, self._tick)

    def push_decision(self, score, center, band_signal):
        """Called from the inference thread with the live (band-masked) discriminant
        score, the current boundary center, and the per-band signed contributions."""
        self.viz_queue.put((float(score), float(center), np.asarray(band_signal, dtype=float)))

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("EEG Classifier Visualizer")
        win_w, win_h = 780, 520
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fig = Figure(figsize=(14, 8), constrained_layout=True)
        gs = self.fig.add_gridspec(
            3, 4, height_ratios=[1, 1.15, 0.16], width_ratios=[0.9, 1.05, 1.05, 0.95]
        )
        self.ax_conf = self.fig.add_subplot(gs[0, :])
        self.ax_probs = self.fig.add_subplot(gs[1, 0])
        self.ax_axis = self.fig.add_subplot(gs[1, 1:3])
        self.ax_bands = self.fig.add_subplot(gs[1, 3])
        self.ax_samples = self.fig.add_subplot(gs[2, :])

        self._build_conf()
        self._build_probs()
        self._build_axis()
        self._build_bands()
        self._build_samples()

        controls = tk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)
        self._build_controls(controls)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_controls(self, bar):
        bands = tk.LabelFrame(bar, text="bands (Hz)")
        bands.pack(side=tk.LEFT, padx=(0, 10))
        self._band_vars = []
        for i, (l, h) in enumerate(FB_BANDS):
            var = tk.BooleanVar(value=True)
            tk.Checkbutton(bands, text=f"{int(l)}–{int(h)}", variable=var,
                           command=lambda i=i, var=var: self._on_band_toggle(i, var)
                           ).pack(side=tk.LEFT)
            self._band_vars.append(var)

        bound = tk.LabelFrame(bar, text="boundary")
        bound.pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(bound, text="◀", width=2,
                  command=lambda: self._nudge(-BOUNDARY_STEP)).pack(side=tk.LEFT)
        self.offset_label = tk.Label(bound, text="+0.00", width=6)
        self.offset_label.pack(side=tk.LEFT)
        tk.Button(bound, text="▶", width=2,
                  command=lambda: self._nudge(+BOUNDARY_STEP)).pack(side=tk.LEFT)
        tk.Button(bound, text="reset", command=self._reset_nudge).pack(side=tk.LEFT, padx=(4, 2))

        gate = tk.LabelFrame(bar, text="confidence gate")
        gate.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.conf_scale = tk.Scale(gate, from_=0.55, to=0.99, resolution=0.01,
                                   orient=tk.HORIZONTAL, command=self._on_conf)
        self.conf_scale.set(round(self.conf_floor, 2))
        self.conf_scale.pack(fill=tk.X, expand=True, padx=4)

    def _on_band_toggle(self, i, var):
        self.band_on[i] = bool(var.get())
        self.ax_bands.get_yticklabels()[i].set_alpha(1.0 if self.band_on[i] else 0.35)
        self.bg = None            # tick labels live in the blit background
        self.canvas.draw_idle()

    def _nudge(self, delta):
        if self.recenter is not None:
            self.recenter.nudge(delta)
        self.manual_total += delta
        self.latest_center += delta   # immediate feedback, before the next window lands
        self.offset_label.config(text=f"{self.manual_total:+.2f}")
        self._dirty = True

    def _reset_nudge(self):
        if self.manual_total:
            self._nudge(-self.manual_total)
        self.manual_total = 0.0
        self.offset_label.config(text="+0.00")

    def _on_conf(self, value):
        cf = min(max(float(value), 1e-3), 1 - 1e-3)
        self.conf_floor = cf
        self.gate = math.log(cf / (1.0 - cf))
        self.smoother.conf_floor = cf
        self._dirty = True

    def _build_conf(self):
        self.ax_conf.set_title("confidence over time")
        self.ax_conf.set_xlabel("seconds")
        self.ax_conf.set_ylabel("p(class)")
        self.ax_conf.set_ylim(0, 1)
        self.ax_conf.set_xlim(-GUI_HISTORY_S, 0)
        self.ax_conf.axhline(0.5, color="gray", lw=0.5, ls="--")
        self.conf_gate_line = self.ax_conf.axhline(self.conf_floor, color=CONSENSUS_COLOR,
                                                   lw=0.9, ls=":", animated=True)

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

        ax.set_title("confidence gate (log-odds vs calibration baseline)")
        ax.set_xlim(-sl, sl)
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        ax.axvline(0, color="#bbbbbb", lw=0.8, ls=":", zorder=1)
        ax.text(0, 0.03, "baseline", ha="center", va="bottom", fontsize=7, color="#999999")

        # boundary + gates move live (arrows, adaptive drift, conf slider) -> animated
        self.span_neg = ax.axvspan(-sl, -self.gate, color=NEG_COLOR, alpha=0.07, zorder=0)
        self.span_pos = ax.axvspan(self.gate, sl, color=POS_COLOR, alpha=0.07, zorder=0)
        self.center_line = ax.axvline(0, color="#444", lw=1.6, zorder=3)
        self.gate_lines = [ax.axvline(g, color="#666", lw=1.0, ls="--", zorder=3)
                           for g in (-self.gate, self.gate)]
        self.abstain_text = ax.text(0, 0.485, "abstain", ha="center", va="center",
                                    fontsize=8, color=MID_COLOR)
        for art in (self.span_neg, self.span_pos, self.center_line,
                    *self.gate_lines, self.abstain_text):
            art.set_animated(True)

        ax.text(0.02, 0.485, f"← class {neg_c}", ha="left", va="center",
                fontsize=9, color=NEG_COLOR, fontweight="bold", transform=ax.transAxes)
        ax.text(0.98, 0.485, f"class {pos_c} →", ha="right", va="center",
                fontsize=9, color=POS_COLOR, fontweight="bold", transform=ax.transAxes)

        self.live_scatter = ax.scatter([], [], animated=True, zorder=6, edgecolors="none")

    def _build_bands(self):
        ax = self.ax_bands
        m = self.band_abs_max
        ax.set_title("band contribution", fontsize=9)
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

    def _build_samples(self):
        ax = self.ax_samples
        n = SAMPLE_HIST_LEN
        hist_w, final_w = 0.82, 0.9
        self._final_x = n + 0.6  # committed-decision slot, past a gap at the right end
        ax.set_title("M of N voting + dwell gate", fontsize=10, pad=4)
        ax.set_xlim(-hist_w / 2, self._final_x + final_w / 2)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        xs = np.arange(n)
        self.sample_bars = ax.bar(xs, [1.0] * n, width=hist_w,
                                   color=MID_COLOR, alpha=0.0, zorder=2)
        for bar in self.sample_bars:
            bar.set_animated(True)
        self.sample_texts = [
            ax.text(x, 0.5, "", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", animated=True, zorder=3)
            for x in xs
        ]

        ax.axvline((n - 1 + self._final_x) / 2, color="#cccccc", lw=0.8, zorder=1)
        ax.text(self._final_x, 1.04, "decision", ha="center", va="bottom",
                fontsize=8, color="#555", clip_on=False, zorder=3)
        (self.final_bar,) = ax.bar([self._final_x], [1.0], width=final_w, zorder=2)
        self.final_bar.set_facecolor(to_rgba(MID_COLOR, 0.15))
        self.final_bar.set_edgecolor("#333333")
        self.final_bar.set_linewidth(1.6)
        self.final_bar.set_animated(True)
        self.final_text = ax.text(self._final_x, 0.5, "", ha="center", va="center",
                                  fontsize=10, fontweight="bold", color="white",
                                  animated=True, zorder=3)

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
                score, center, band_sig = self.viz_queue.get_nowait()
            except queue.Empty:
                break
            self.score_hist.append((score, center))
            self.latest_center = center
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
            self._refresh_samples()
            if self.bg is None:         # a refresh invalidated the background (axis rescale)
                self.canvas.draw_idle()
                self.root.after(self.tick_ms, self._tick)
                return
            self.canvas.restore_region(self.bg)
            for lc in self.onset_collections.values():
                self.ax_conf.draw_artist(lc)
            for line in self.conf_lines.values():
                self.ax_conf.draw_artist(line)
            self.ax_conf.draw_artist(self.conf_gate_line)
            for bar in self.prob_bars:
                self.ax_probs.draw_artist(bar)
            self.ax_axis.draw_artist(self.span_neg)
            self.ax_axis.draw_artist(self.span_pos)
            self.ax_axis.draw_artist(self.center_line)
            for gl in self.gate_lines:
                self.ax_axis.draw_artist(gl)
            self.ax_axis.draw_artist(self.abstain_text)
            self.ax_axis.draw_artist(self.live_scatter)
            for bar in self.band_bars:
                self.ax_bands.draw_artist(bar)
            for bar in self.sample_bars:
                self.ax_samples.draw_artist(bar)
            for text in self.sample_texts:
                self.ax_samples.draw_artist(text)
            self.ax_samples.draw_artist(self.final_bar)
            self.ax_samples.draw_artist(self.final_text)
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

        self.sample_hist.append({
            "pred": int(last["prediction"]),
            "conf": float(last["confidence"]),
            "vote": last.get("vote"),   # the vote the smoother actually counted
            "consensus": bool(self.latest_consensus),
        })

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
        g = self.gate
        raw_cx = self.latest_center - self.base_center
        need = max(6.0, abs(raw_cx) + g + 1.0)
        if need > self.score_lim:       # boundary + gate outgrew the axis: rescale (grow-only)
            self.score_lim = need
            self.ax_axis.set_xlim(-need, need)
            self.bg = None
        sl = self.score_lim
        cx = float(np.clip(raw_cx, -sl * 0.95, sl * 0.95))
        self.center_line.set_xdata([cx, cx])
        self.gate_lines[0].set_xdata([cx - g, cx - g])
        self.gate_lines[1].set_xdata([cx + g, cx + g])
        self.abstain_text.set_position((cx, 0.485))
        neg_edge, pos_edge = max(-sl, cx - g), min(sl, cx + g)
        self.span_neg.set_bounds(-sl, 0, neg_edge + sl, 1)
        self.span_pos.set_bounds(pos_edge, 0, sl - pos_edge, 1)
        self.conf_gate_line.set_ydata([self.conf_floor, self.conf_floor])

        n = len(self.score_hist)
        if n == 0:
            self.live_scatter.set_offsets(np.empty((0, 2)))
            return
        pairs = np.array(self.score_hist, dtype=float)     # columns: score, center
        xs = np.clip(pairs[:, 0] - self.base_center, -sl * 0.985, sl * 0.985)
        margins = pairs[:, 0] - pairs[:, 1]                # each dot's decision margin
        frac = (n - 1 - np.arange(n)) / max(n - 1, 1)
        ys = 0.53 + frac * 0.44
        sizes = 150 * (1 - 0.78 * frac) + 12
        alphas = np.clip(1.0 - 0.82 * frac, 0.12, 1.0)
        base = np.where(margins > g, POS_COLOR,
                        np.where(margins < -g, NEG_COLOR, MID_COLOR))
        rgba = np.array([to_rgba(base[i], alphas[i]) for i in range(n)])
        self.live_scatter.set_offsets(np.column_stack([xs, ys]))
        self.live_scatter.set_sizes(sizes)
        self.live_scatter.set_color(rgba)

    def _refresh_bands(self):
        sig = self.latest_band_sig
        if sig is None:
            return
        active = [i for i in range(min(len(sig), self.n_bands)) if self.band_on[i]]
        dom = max(active, key=lambda i: abs(float(sig[i]))) if active else None
        for i, bar in enumerate(self.band_bars):
            if i >= len(sig):
                break
            w = float(sig[i])
            bar.set_width(w)
            if not self.band_on[i]:
                bar.set_color(to_rgba(MID_COLOR, 0.25))    # excluded from the score
                continue
            col = POS_COLOR if w >= 0 else NEG_COLOR
            bar.set_color(to_rgba(col, 1.0 if i == dom else 0.32))

    def _refresh_samples(self):
        for i, (bar, text) in enumerate(zip(self.sample_bars, self.sample_texts)):
            if i >= len(self.sample_hist):
                bar.set_alpha(0.0)
                text.set_text("")
                continue

            entry = self.sample_hist[i]
            bar.set_alpha(1.0)
            vote = entry.get("vote")
            if vote is not None:
                bar.set_facecolor(self.decision_colors.get(vote, MID_COLOR))
                text.set_text(str(vote))
            else:
                bar.set_facecolor(MID_COLOR)
                text.set_text("")

        if self.latest_final:
            self.final_bar.set_facecolor(to_rgba(CONSENSUS_COLOR, 1.0))
            self.final_text.set_text(str(self.latest_final))
        else:
            self.final_bar.set_facecolor(to_rgba(MID_COLOR, 0.15))
            self.final_text.set_text("")

    def _on_close(self):
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
