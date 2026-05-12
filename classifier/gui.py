import queue
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import EEG_CHANNELS


HISTORY_SECONDS = 30
TICK_MS = 16  # ~60Hz
BRAIN_HZ = 2  # source-localization refresh rate (heavy)
FIG_DPI = 80


class GUI:
    """Live visualizer. Tkinter on the main thread, blit for fast redraws.

    Threads:
      - main (tk):   drains payload/brain queues, blits at ~60Hz
      - source set:  one-shot — fetches fsaverage, builds fwd + inverse
      - render:      2 Hz — applies inverse, renders stat_map, pushes image
      - smoother:    pushes prediction events into self.queue (drain thread)
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

        # Most recent filtered EEG window pushed in from start.py.
        self.window_lock = threading.Lock()
        self.latest_window = None

        # Most recent rendered brain image (RGBA numpy array).
        self.brain_lock = threading.Lock()
        self.latest_brain_img = None
        self.brain_status = "initializing source localization..."

        self.source_ready = False
        self.stop_evt = threading.Event()

        self._build_window()
        self._start_source_setup_thread()

        self.bgs = {}
        self._pending_bg_capture = True
        self._pred_dirty = False
        self._brain_dirty = False
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self.root.after(50, self._tick)

    # ------------------------------------------------------------------ ext

    def push_window(self, eeg):
        """Called from start.py's on_chunk with the latest filtered window."""
        with self.window_lock:
            self.latest_window = np.ascontiguousarray(eeg)

    # ------------------------------------------------------------------ build

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("EEG Live Viz")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fig = plt.Figure(figsize=(14, 9), dpi=FIG_DPI, constrained_layout=True)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 2])
        self.ax_conf = self.fig.add_subplot(gs[0, :])
        self.ax_probs = self.fig.add_subplot(gs[1, 0])
        self.ax_brain = self.fig.add_subplot(gs[1, 1])

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

        self.ax_probs.set_title("current probs")
        self.ax_probs.set_ylim(0, 1)
        self.prob_bars = self.ax_probs.bar(
            [str(c) for c in self.class_labels],
            [0.0] * len(self.class_labels),
            color=[colors[i % len(colors)] for i in range(len(self.class_labels))],
        )
        for bar in self.prob_bars:
            bar.set_animated(True)

        self.ax_brain.set_title("source localization (dSPM on fsaverage)")
        self.ax_brain.set_xticks([])
        self.ax_brain.set_yticks([])
        # Placeholder text shown until first render.
        self.brain_status_text = self.ax_brain.text(
            0.5, 0.5, self.brain_status,
            transform=self.ax_brain.transAxes,
            ha="center", va="center", fontsize=11, color="gray",
        )
        self.brain_img_artist = None  # created on first render

        self.decision_text = self.fig.text(
            0.5, 0.99, "—", ha="center", va="top",
            fontsize=24, fontweight="bold", color="gray",
            animated=True,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------ setup

    def _start_source_setup_thread(self):
        def setup():
            try:
                self._set_brain_status("fetching fsaverage…")
                from mne.datasets import fetch_fsaverage
                from mne.minimum_norm import make_inverse_operator

                fs_dir = fetch_fsaverage(verbose=False)
                subjects_dir = str(Path(fs_dir).parent)

                self._set_brain_status("reading BEM…")
                bem_path = Path(fs_dir) / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
                bem = mne.read_bem_solution(str(bem_path), verbose=False)

                cache_dir = Path(fs_dir) / "mne_cache"
                cache_dir.mkdir(exist_ok=True)
                vol_src_path = cache_dir / "fsaverage-vol-10mm-src.fif"

                if vol_src_path.exists():
                    self._set_brain_status("loading source space…")
                    src = mne.read_source_spaces(str(vol_src_path), verbose=False)
                else:
                    self._set_brain_status("computing volume source space (one-time, ~30s)…")
                    src = mne.setup_volume_source_space(
                        "fsaverage", pos=10.0, bem=bem,
                        subjects_dir=subjects_dir, verbose=False,
                    )
                    mne.write_source_spaces(str(vol_src_path), src, overwrite=True, verbose=False)

                montage = mne.channels.make_standard_montage("standard_1020")
                info = mne.create_info(EEG_CHANNELS, sfreq=self.sfreq, ch_types="eeg")
                info.set_montage(montage, on_missing="ignore")

                fwd_path = cache_dir / f"fsaverage-{len(EEG_CHANNELS)}ch-vol-fwd.fif"
                if fwd_path.exists():
                    self._set_brain_status("loading forward solution…")
                    fwd = mne.read_forward_solution(str(fwd_path), verbose=False)
                else:
                    self._set_brain_status("computing forward solution (one-time, ~30s)…")
                    fwd = mne.make_forward_solution(
                        info, trans="fsaverage", src=src, bem=bem,
                        eeg=True, meg=False, verbose=False,
                    )
                    mne.write_forward_solution(str(fwd_path), fwd, overwrite=True, verbose=False)

                self._set_brain_status("building inverse operator…")
                cov = mne.make_ad_hoc_cov(info, verbose=False)
                inv = make_inverse_operator(info, fwd, cov, loose=1.0, depth=0.8, verbose=False)

                self.evoked_info = info
                self.inv = inv
                self.src = src
                self.subject = "fsaverage"
                self.subjects_dir = subjects_dir
                self.source_ready = True

                self._set_brain_status("ready — waiting for data…")
                self._start_render_thread()
            except Exception as e:
                msg = f"source setup failed:\n{e}"
                print(f"[gui] {msg}")
                self._set_brain_status(msg)

        threading.Thread(target=setup, daemon=True).start()

    def _set_brain_status(self, msg):
        self.brain_status = msg
        # The status text is on the static background — flag a full redraw
        # so the new message appears on next draw.
        self._pending_bg_capture = True

    # ------------------------------------------------------------------ render

    def _start_render_thread(self):
        from mne.minimum_norm import apply_inverse

        def loop():
            period = 1.0 / BRAIN_HZ
            min_samples = int(0.25 * self.sfreq)
            while not self.stop_evt.is_set():
                t0 = time.perf_counter()
                try:
                    with self.window_lock:
                        win = None if self.latest_window is None else self.latest_window.copy()

                    if win is not None and win.shape[1] >= min_samples:
                        self._render_brain(win, apply_inverse)
                except Exception as e:
                    print(f"[gui] render error: {e}")

                dt = time.perf_counter() - t0
                rest = period - dt
                if rest > 0:
                    time.sleep(rest)

        threading.Thread(target=loop, daemon=True).start()

    def _render_brain(self, win, apply_inverse):
        evoked = mne.EvokedArray(win, self.evoked_info, tmin=0.0, verbose=False)
        evoked.set_eeg_reference("average", projection=True, verbose=False)
        evoked.apply_proj(verbose=False)

        stc = apply_inverse(
            evoked, self.inv, lambda2=1.0 / 9.0, method="dSPM", verbose=False
        )

        fig = stc.plot(
            src=self.src, subject=self.subject,
            subjects_dir=self.subjects_dir,
            mode="stat_map",
            initial_time=stc.times[-1],
            show=False, verbose=False,
        )
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)

        with self.brain_lock:
            self.latest_brain_img = img

    # ------------------------------------------------------------------ draw

    def _on_draw(self, _event):
        if self._pending_bg_capture or not self.bgs:
            self.bgs[self.fig] = self.canvas.copy_from_bbox(self.fig.bbox)
            self._pending_bg_capture = False
            self._pred_dirty = True

    def _on_resize(self, _event):
        self._pending_bg_capture = True
        self.canvas.draw_idle()

    def _tick(self):
        # Drain prediction events.
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

        # Pick up the latest brain render (replaces previous, never queues).
        new_brain = None
        with self.brain_lock:
            if self.latest_brain_img is not None:
                new_brain = self.latest_brain_img
                self.latest_brain_img = None

        # Apply new brain image — this changes static art, so triggers a full
        # canvas redraw (at most BRAIN_HZ, so the cost is fine).
        if new_brain is not None:
            self._apply_brain_image(new_brain)

        # Also redraw the canvas fully whenever bg is invalidated (status text
        # changes, brain image updates, resize).
        if self._pending_bg_capture:
            self.canvas.draw_idle()
            self.root.after(TICK_MS, self._tick)
            return

        if not self.bgs:
            self.canvas.draw_idle()
            self.root.after(TICK_MS, self._tick)
            return

        if self._pred_dirty:
            self._refresh_conf_lines()
            self._refresh_probs()
            self._refresh_decision()

            self.canvas.restore_region(self.bgs[self.fig])
            for line in self.conf_lines.values():
                self.ax_conf.draw_artist(line)
            for bar in self.prob_bars:
                self.ax_probs.draw_artist(bar)
            self.fig.draw_artist(self.decision_text)
            self.canvas.blit(self.fig.bbox)
            self._pred_dirty = False

        self.root.after(TICK_MS, self._tick)

    def _apply_brain_image(self, img):
        if self.brain_status_text is not None:
            self.brain_status_text.set_visible(False)
        h, w = img.shape[:2]
        extent = (0, w, h, 0)  # origin='upper'
        if self.brain_img_artist is None:
            self.brain_img_artist = self.ax_brain.imshow(
                img, aspect="equal", origin="upper", extent=extent,
            )
            self.ax_brain.set_anchor("C")  # center inside the axes box
        else:
            existing = self.brain_img_artist.get_array()
            if existing is not None and existing.shape == img.shape:
                self.brain_img_artist.set_data(img)
            else:
                self.brain_img_artist.remove()
                self.brain_img_artist = self.ax_brain.imshow(
                    img, aspect="equal", origin="upper", extent=extent,
                )
                self.ax_brain.set_anchor("C")
        self.ax_brain.set_xlim(0, w)
        self.ax_brain.set_ylim(h, 0)
        self._pending_bg_capture = True

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

    # ------------------------------------------------------------------ exit

    def _on_close(self):
        self.stop_evt.set()
        self.root.quit()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
