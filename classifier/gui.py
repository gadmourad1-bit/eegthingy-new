import math
import matplotlib.pyplot as plt
import mne
import numpy as np
import threading
import time
import tkinter as tk
import queue

from collections import deque
from config import EEG_CHANNELS, GUI_HISTORY_S, GUI_REFRESH_RATE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, Polygon


class GUI:
    def __init__(self, bci, smoother, class_labels, sfreq):
        self.bci = bci
        self.smoother = smoother
        self.class_labels = list(class_labels)
        self.sfreq = float(sfreq)
        self.queue = queue.Queue()
        self.tick_ms = math.floor((1 / GUI_REFRESH_RATE) * 1000)
        smoother.subscribe(self.queue.put)

        max_points = GUI_HISTORY_S * 20
        self.t_hist = deque(maxlen=max_points)
        self.conf_hist = {c: deque(maxlen=max_points) for c in self.class_labels}
        self.t0 = None

        self.latest_probs = {c: 0.0 for c in self.class_labels}
        self.latest_decision = None
        self.latest_consensus = False
        self.transition_hist = deque(maxlen=max_points)
        self.last_decision = None
        self.decision_colors = {}
        if len(self.class_labels) > 0:
            self.decision_colors[self.class_labels[0]] = "tab:blue"
        if len(self.class_labels) > 1:
            self.decision_colors[self.class_labels[1]] = "tab:red"

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_pos = montage.get_positions()["ch_pos"]
        pos3 = np.array(
            [ch_pos[name] if name in ch_pos else np.zeros(3) for name in EEG_CHANNELS]
        )
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

    def _build_views(self, pos3):
        rotations = {
            "top": np.eye(3, dtype=float),
            "left": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float),
            "back": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),
        }
        views = {}
        for name, R in rotations.items():
            rotated = pos3 @ R.T
            norms = np.linalg.norm(rotated, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            unit = rotated / norms
            theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
            phi = np.arctan2(unit[:, 1], unit[:, 0])
            x2d = theta * np.cos(phi)
            y2d = theta * np.sin(phi)
            pos2d = np.column_stack([x2d, y2d])
            visible = unit[:, 2] > -0.05
            pos_vis = pos2d[visible]
            r = (
                float(np.max(np.linalg.norm(pos_vis, axis=1))) * 1.10
                if len(pos_vis)
                else np.pi / 2
            )
            views[name] = {
                "positions": pos_vis,
                "visible": visible,
                "r": r,
            }
        return views

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("EEG Live Viz")
        win_w, win_h = 700, 400
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fig = plt.Figure(figsize=(14, 8), constrained_layout=True)
        gs = self.fig.add_gridspec(
            2, 4, height_ratios=[1, 1], width_ratios=[1.2, 1, 1, 1]
        )
        self.ax_conf = self.fig.add_subplot(gs[0, :])
        self.ax_probs = self.fig.add_subplot(gs[1, 0])
        self.topo_axes = {
            "top": self.fig.add_subplot(gs[1, 1]),
            "left": self.fig.add_subplot(gs[1, 2]),
            "back": self.fig.add_subplot(gs[1, 3]),
        }

        self.ax_conf.set_title("confidence over time")
        self.ax_conf.set_xlabel("seconds")
        self.ax_conf.set_ylabel("p(class)")
        self.ax_conf.set_ylim(0, 1)
        self.ax_conf.set_xlim(-GUI_HISTORY_S, 0)
        self.ax_conf.axhline(0.5, color="gray", lw=0.5, ls="--")

        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
        self.conf_lines = {}
        for i, c in enumerate(self.class_labels):
            (line,) = self.ax_conf.plot(
                [],
                [],
                color=colors[i % len(colors)],
                lw=1.5,
                label=f"class {c}",
                animated=True,
            )
            self.conf_lines[c] = line
        self.ax_conf.legend(loc="upper right")

        from matplotlib.collections import LineCollection

        self.onset_collections = {}
        for c in self.class_labels:
            color = self.decision_colors.get(c, "gray")
            lc = LineCollection(
                [],
                colors=color,
                alpha=0.7,
                lw=1.2,
                linestyles="dashed",
                animated=True,
                zorder=1,
            )
            self.ax_conf.add_collection(lc)
            self.onset_collections[c] = lc

        self.ax_probs.set_title("current probs")
        self.ax_probs.set_ylim(0, 1)
        self.prob_bars = self.ax_probs.bar(
            [str(c) for c in self.class_labels],
            [0.0] * len(self.class_labels),
            color=[colors[i % len(colors)] for i in range(len(self.class_labels))],
        )
        for bar in self.prob_bars:
            bar.set_animated(True)

        self.topo_titles = {"top": "top view", "left": "side view", "back": "back view"}
        self.topo_imgs = {}
        for name, ax in self.topo_axes.items():
            ax.set_title(self.topo_titles[name])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_visible(False)
            placeholder = np.ones((4, 4, 4), dtype=np.uint8) * 255
            self.topo_imgs[name] = ax.imshow(placeholder, animated=True, aspect="equal")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

        self.off_figs = {}
        for name in self.topo_axes:
            fig = Figure(figsize=(3.2, 3.4), constrained_layout=True)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.off_figs[name] = (fig, canvas, ax)

        self.decision_text = self.fig.text(
            0.5,
            0.94,
            "—",
            ha="center",
            va="top",
            fontsize=24,
            fontweight="bold",
            color="gray",
            animated=True,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _start_topo_thread(self):
        def worker():
            period = 1.0 / (GUI_REFRESH_RATE)
            while not self.stop_evt.is_set():
                t0 = time.perf_counter()
                try:
                    data = self.bci.get_data()
                    eeg = data.get("eeg")
                    if eeg is not None and len(eeg) > 0:
                        n = min(len(eeg), len(EEG_CHANNELS))
                        values = np.zeros(len(EEG_CHANNELS))
                        values[:n] = eeg[:n]
                        imgs = self._render_offscreen(values)
                        if imgs is not None:
                            try:
                                self.topo_queue.put_nowait(imgs)
                            except queue.Full:
                                try:
                                    self.topo_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    self.topo_queue.put_nowait(imgs)
                                except queue.Full:
                                    pass
                except Exception as e:
                    print(f"[gui] topo worker error: {e}")
                dt = time.perf_counter() - t0
                rest = period - dt
                if rest > 0:
                    time.sleep(rest)

        threading.Thread(target=worker, daemon=True).start()

    def _render_offscreen(self, values):
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if hi - lo < 1e-9:
                hi = lo + 1e-9
        else:
            lo, hi = 0.0, 1.0

        out = {}
        for name, (_fig, canvas, ax) in self.off_figs.items():
            v = self.views[name]
            vals = values[v["visible"]]
            ax.clear()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            try:
                mne.viz.plot_topomap(
                    vals,
                    v["positions"],
                    axes=ax,
                    show=False,
                    cmap="viridis",
                    vlim=(lo, hi),
                    sensors=True,
                    contours=0,
                    outlines="head",
                    extrapolate="head",
                    sphere=(0.0, 0.0, 0.0, v["r"]),
                )
                for ln in list(ax.lines):
                    ln.remove()
            except Exception as e:
                print(f"[gui] plot_topomap error: {e}")
                continue
            marks = self.HEAD_MARKS[name]
            self._draw_head_marks(ax, v["r"], marks["nose_deg"], marks["ear_degs"])
            lim = v["r"] * 1.2
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

            canvas.draw()
            w, h = canvas.get_width_height()
            buf = (
                np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                .reshape(h, w, 4)
                .copy()
            )
            out[name] = buf
        return out

    def _on_draw(self, _event):
        if self._pending_bg_capture or not self.bgs:
            for name, ax in self.topo_axes.items():
                self.bgs[name] = self.canvas.copy_from_bbox(ax.bbox)
            self.bgs["fig"] = self.canvas.copy_from_bbox(self.fig.bbox)
            self._pending_bg_capture = False
            self._pred_dirty = True
            self._topo_dirty = True

    def _on_resize(self, _event):
        self._pending_bg_capture = True
        self.canvas.draw_idle()

    def _tick(self):
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

        latest = None
        while True:
            try:
                latest = self.topo_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            for name, img_arr in latest.items():
                topo = self.topo_imgs[name]
                cur = topo.get_array()
                if cur is None or cur.shape != img_arr.shape:
                    topo.set_extent((-1, 1, -1, 1))
                    topo.axes.set_xlim(-1, 1)
                    topo.axes.set_ylim(-1, 1)
                topo.set_data(img_arr)
            self._topo_dirty = True

        if not self.bgs or self._pending_bg_capture:
            self.canvas.draw_idle()
            self.root.after(self.tick_ms, self._tick)
            return

        if self._pred_dirty:
            self._refresh_onsets()
            self._refresh_conf_lines()
            self._refresh_probs()
            self._refresh_decision()
            self.canvas.restore_region(self.bgs["fig"])
            for lc in self.onset_collections.values():
                self.ax_conf.draw_artist(lc)
            for line in self.conf_lines.values():
                self.ax_conf.draw_artist(line)
            for bar in self.prob_bars:
                self.ax_probs.draw_artist(bar)
            for img in self.topo_imgs.values():
                img.axes.draw_artist(img)
            self.fig.draw_artist(self.decision_text)
            self.canvas.blit(self.fig.bbox)
            self._pred_dirty = False
            self._topo_dirty = False
        elif self._topo_dirty:
            for name, img in self.topo_imgs.items():
                self.canvas.restore_region(self.bgs[name])
                img.axes.draw_artist(img)
                self.canvas.blit(img.axes.bbox)
            self._topo_dirty = False

        self.root.after(self.tick_ms, self._tick)

    HEAD_MARKS = {
        "top": {"nose_deg": 90, "ear_degs": (0, 180)},
        "left": {"nose_deg": 0, "ear_degs": ()},
        "back": {"nose_deg": None, "ear_degs": (0, 180)},
    }

    def _draw_head_marks(self, ax, r, nose_deg, ear_degs):
        head = Circle((0, 0), r, color="k", fill=False, lw=1.2)
        ax.add_patch(head)
        for art in list(ax.images) + list(ax.collections):
            try:
                art.set_clip_path(head)
            except Exception:
                pass

        if nose_deg is not None:
            a = math.radians(nose_deg)
            half = math.radians(11)
            base_l = (r * math.cos(a + half), r * math.sin(a + half))
            base_r = (r * math.cos(a - half), r * math.sin(a - half))
            tip = (r * 1.10 * math.cos(a), r * 1.10 * math.sin(a))
            ax.add_patch(
                Polygon(
                    [base_l, tip, base_r], closed=True, fill=False, color="k", lw=1.2
                )
            )

        for ang in ear_degs:
            a = math.radians(ang)
            cx = r * 1.045 * math.cos(a)
            cy = r * 1.045 * math.sin(a)
            ax.add_patch(
                Ellipse(
                    (cx, cy),
                    width=0.10 * r,
                    height=0.20 * r,
                    angle=ang,
                    fill=False,
                    color="k",
                    lw=1.2,
                )
            )

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

        new_d = smoothed["decision"] if smoothed["consensus"] else None
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

    def _refresh_decision(self):
        if self.latest_decision is None:
            self.decision_text.set_text("—")
            self.decision_text.set_color("gray")
        else:
            self.decision_text.set_text(f"class {self.latest_decision}")
            self.decision_text.set_color(
                "tab:green" if self.latest_consensus else "gray"
            )

    def _on_close(self):
        self.stop_evt.set()
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
