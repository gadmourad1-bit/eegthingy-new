"""Tkinter configuration and stimulus GUI for four-command SSVEP recording."""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from serial.tools import list_ports

from .acquisition import OpenBCIRecorder, PhaseLog, save_session
from .config import (
    COMMAND_LABELS,
    COMMAND_POSITIONS,
    COMMANDS,
    ConfigurationError,
    Settings,
)
from .protocol import Trial, encode_marker, make_trials


BACKGROUND = "#080a0f"
PANEL = "#141824"
TEXT = "#f4f6fb"
MUTED = "#9aa3b4"
ACCENT = "#4ee39a"
RED = "#ef3038"


def _seconds_text(seconds: float) -> str:
    minutes, remainder = divmod(max(0, int(round(seconds))), 60)
    return f"{minutes:d}:{remainder:02d}"


class SettingsWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("SSVEP data collector")
        self.root.configure(bg=BACKGROUND)
        self.root.geometry("1000x760")
        self.root.minsize(900, 700)
        self.settings = Settings.load()
        self.variables: dict[str, tk.Variable] = {}
        self.frequency_variables: dict[str, tk.StringVar] = {}
        self._build()

    def _label_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        value: object,
        *,
        width: int = 14,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
        variable = tk.StringVar(value=str(value))
        self.variables[key] = variable
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        return entry

    def _build(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22), foreground=TEXT)
        style.configure("Hint.TLabel", foreground=MUTED, font=("Segoe UI", 9))
        style.configure("TLabelframe", background=BACKGROUND, foreground=TEXT)
        style.configure("TLabelframe.Label", background=BACKGROUND, foreground=ACCENT, font=("Segoe UI Semibold", 11))
        style.configure("TCheckbutton", background=BACKGROUND, foreground=TEXT)
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 11), padding=(14, 9))

        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Four-command SSVEP recording", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Collect automatically labeled 15-channel EEG while the subject attends "
                "to FORWARD, RIGHT, BACKWARD, or LEFT flicker targets."
            ),
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(2, 16))

        columns = ttk.Frame(outer)
        columns.pack(fill="both", expand=True)
        columns.columnconfigure(0, weight=1)
        columns.columnconfigure(1, weight=1)

        session = ttk.LabelFrame(columns, text="Session", padding=14)
        session.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        session.columnconfigure(1, weight=1)
        self._label_entry(session, 0, "Subject ID", "subject", self.settings.subject)
        self._label_entry(session, 1, "Session / run", "session", self.settings.session)
        self._label_entry(
            session,
            2,
            "Repetitions per command",
            "repetitions_per_command",
            self.settings.repetitions_per_command,
        )
        self._label_entry(session, 3, "Randomization seed", "random_seed", self.settings.random_seed)

        timing = ttk.LabelFrame(columns, text="Timing", padding=14)
        timing.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))
        timing.columnconfigure(1, weight=1)
        self._label_entry(timing, 0, "Preparation (seconds)", "preparation_s", self.settings.preparation_s)
        self._label_entry(timing, 1, "Planning / cue (seconds)", "planning_s", self.settings.planning_s)
        self._label_entry(timing, 2, "Action / flicker (seconds)", "action_s", self.settings.action_s)
        self._label_entry(timing, 3, "Rest (seconds)", "rest_s", self.settings.rest_s)
        ttk.Label(
            timing,
            text="Preparation > 5 s enables the embedded how-to animation.",
            style="Hint.TLabel",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        frequency = ttk.LabelFrame(columns, text="Command frequencies", padding=14)
        frequency.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=10)
        frequency.columnconfigure(1, weight=1)
        for row, command in enumerate(COMMANDS):
            ttk.Label(frequency, text=f"{COMMAND_LABELS[command]} ({COMMAND_POSITIONS[command]})").grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=5
            )
            variable = tk.StringVar(value=str(self.settings.frequencies_hz[command]))
            self.frequency_variables[command] = variable
            ttk.Entry(frequency, textvariable=variable, width=10).grid(row=row, column=1, sticky="ew", pady=5)
            ttk.Label(frequency, text="Hz", style="Hint.TLabel").grid(row=row, column=2, sticky="w", padx=(5, 0))
        ttk.Label(
            frequency,
            text="Defaults (8, 10, 12, 15 Hz) are suitable for a typical 60 Hz monitor.",
            style="Hint.TLabel",
            wraplength=390,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

        display = ttk.LabelFrame(columns, text="Display geometry", padding=14)
        display.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=10)
        display.columnconfigure(1, weight=1)
        self._label_entry(display, 0, "Square size (cm)", "square_size_cm", self.settings.square_size_cm)
        self._label_entry(display, 1, "Horizontal offset (cm)", "horizontal_offset_cm", self.settings.horizontal_offset_cm)
        self._label_entry(display, 2, "Vertical offset (cm)", "vertical_offset_cm", self.settings.vertical_offset_cm)
        self._label_entry(display, 3, "Bright level (0-255)", "bright_level", self.settings.bright_level)
        self._label_entry(display, 4, "Dark level (0-255)", "dark_level", self.settings.dark_level)

        hardware = ttk.LabelFrame(columns, text="Hardware and output", padding=14)
        hardware.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        hardware.columnconfigure(1, weight=1)
        ttk.Label(hardware, text="Serial port").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=5)
        port_values = [port.device for port in list_ports.comports()]
        port_var = tk.StringVar(value=self.settings.serial_port)
        self.variables["serial_port"] = port_var
        ttk.Combobox(hardware, textvariable=port_var, values=port_values).grid(row=0, column=1, sticky="ew", pady=5)
        ttk.Label(hardware, text="Blank = auto-detect", style="Hint.TLabel").grid(row=0, column=2, sticky="w", padx=(8, 0))

        ttk.Label(hardware, text="Output folder").grid(row=1, column=0, sticky="w", padx=(0, 12), pady=5)
        output_var = tk.StringVar(value=self.settings.output_dir)
        self.variables["output_dir"] = output_var
        ttk.Entry(hardware, textvariable=output_var).grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Button(hardware, text="Browse…", command=self._choose_output).grid(row=1, column=2, padx=(8, 0))

        synthetic_var = tk.BooleanVar(value=self.settings.synthetic)
        fullscreen_var = tk.BooleanVar(value=self.settings.fullscreen)
        self.variables["synthetic"] = synthetic_var
        self.variables["fullscreen"] = fullscreen_var
        ttk.Checkbutton(
            hardware,
            text="Software test mode (synthetic EEG — never use as research data)",
            variable=synthetic_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 3))
        ttk.Checkbutton(hardware, text="Full-screen stimulus", variable=fullscreen_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=3
        )
        ttk.Label(
            hardware,
            text=(
                "Safety: flickering light can trigger symptoms in photosensitive people. "
                "Use the laboratory's approved screening and stop immediately if the subject feels unwell."
            ),
            style="Hint.TLabel",
            wraplength=760,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(9, 0))

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(14, 0))
        self.summary_var = tk.StringVar()
        ttk.Label(footer, textvariable=self.summary_var, style="Hint.TLabel").pack(side="left")
        ttk.Button(footer, text="Start collection", style="Accent.TButton", command=self._start).pack(side="right")
        self._update_summary()
        for variable in list(self.variables.values()) + list(self.frequency_variables.values()):
            variable.trace_add("write", lambda *_: self._update_summary())

    def _choose_output(self) -> None:
        start = self.variables["output_dir"].get() or str(Path.cwd())
        selected = filedialog.askdirectory(initialdir=start, title="Choose SSVEP recording folder")
        if selected:
            self.variables["output_dir"].set(selected)

    def _read_settings(self) -> Settings:
        try:
            settings = Settings(
                subject=str(self.variables["subject"].get()),
                session=str(self.variables["session"].get()),
                repetitions_per_command=int(str(self.variables["repetitions_per_command"].get())),
                preparation_s=float(str(self.variables["preparation_s"].get())),
                planning_s=float(str(self.variables["planning_s"].get())),
                action_s=float(str(self.variables["action_s"].get())),
                rest_s=float(str(self.variables["rest_s"].get())),
                frequencies_hz={
                    command: float(variable.get())
                    for command, variable in self.frequency_variables.items()
                },
                square_size_cm=float(str(self.variables["square_size_cm"].get())),
                horizontal_offset_cm=float(str(self.variables["horizontal_offset_cm"].get())),
                vertical_offset_cm=float(str(self.variables["vertical_offset_cm"].get())),
                bright_level=int(str(self.variables["bright_level"].get())),
                dark_level=int(str(self.variables["dark_level"].get())),
                random_seed=int(str(self.variables["random_seed"].get())),
                serial_port=str(self.variables["serial_port"].get()),
                synthetic=bool(self.variables["synthetic"].get()),
                fullscreen=bool(self.variables["fullscreen"].get()),
                output_dir=str(self.variables["output_dir"].get()),
            )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(f"A numeric setting is invalid: {error}") from error
        return settings.validate()

    def _update_summary(self) -> None:
        try:
            settings = self._read_settings()
            mode = "SOFTWARE TEST" if settings.synthetic else "OPENBCI"
            self.summary_var.set(
                f"{settings.trial_count} trials  •  about {_seconds_text(settings.estimated_duration_s)}  •  {mode}"
            )
        except ConfigurationError:
            self.summary_var.set("Complete the settings to see the estimated session time.")

    def _start(self) -> None:
        try:
            settings = self._read_settings()
            settings.save()
        except (ConfigurationError, OSError) as error:
            messagebox.showerror("Invalid SSVEP settings", str(error), parent=self.root)
            return

        if settings.synthetic and not messagebox.askokcancel(
            "Software test mode",
            "Synthetic mode checks the software only. Its EEG is not a human SSVEP recording and must not be used for training. Continue?",
            parent=self.root,
        ):
            return
        if not settings.synthetic and not messagebox.askokcancel(
            "Flicker safety check",
            "Confirm that the subject has passed your laboratory's visual-flicker safety screening, understands how to stop, and is ready to begin.",
            parent=self.root,
        ):
            return
        if max(settings.frequencies_hz.values()) > 15:
            if not messagebox.askokcancel(
                "Check monitor refresh rate",
                "A requested frequency is above 15 Hz. Confirm that the display refresh rate can render it reliably before collecting research data.",
                parent=self.root,
            ):
                return

        self.root.withdraw()
        StimulusWindow(self.root, settings, self._session_finished)

    def _session_finished(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def run(self) -> None:
        self.root.mainloop()


class StimulusWindow(tk.Toplevel):
    FRAME_INTERVAL_MS = 4

    def __init__(self, master: tk.Tk, settings: Settings, on_finished) -> None:
        super().__init__(master)
        self.settings = settings
        self.on_finished = on_finished
        self.title("SSVEP subject display")
        self.configure(bg=BACKGROUND)
        if settings.fullscreen:
            self.attributes("-fullscreen", True)
        else:
            self.geometry("1200x820")
        self.protocol("WM_DELETE_WINDOW", self._request_stop)
        self.bind("<Escape>", self._request_stop)
        self.bind("<F11>", self._toggle_fullscreen)

        self.canvas = tk.Canvas(self, bg=BACKGROUND, highlightthickness=0, cursor="none")
        self.canvas.pack(fill="both", expand=True)
        self.trials = make_trials(
            settings.repetitions_per_command,
            settings.frequencies_hz,
            settings.random_seed,
        )
        self.recorder = OpenBCIRecorder(synthetic=settings.synthetic, serial_port=settings.serial_port)
        self.phase_log: list[PhaseLog] = []
        self.current_log: PhaseLog | None = None
        self.current_trial_index = -1
        self.phase = "connecting"
        self.phase_started = 0.0
        self.session_started = 0.0
        self.finished = False
        self.after_id: str | None = None
        self.target_rectangles: dict[str, int] = {}
        self.target_states: dict[str, bool | None] = {}
        self.transition_counts: dict[str, int] = {}
        self.action_frames = 0
        self.tutorial_frame: tk.Frame | None = None
        self.tutorial_canvas: tk.Canvas | None = None
        self.connection_messages: queue.Queue[tuple[bool, str]] = queue.Queue()

        self._draw_message("Connecting to the EEG board…", "Please wait. Do not begin looking at a target yet.")
        self.update_idletasks()
        threading.Thread(target=self._connect_worker, daemon=True).start()
        self.after(50, self._poll_connection)

    @property
    def current_trial(self) -> Trial | None:
        if 0 <= self.current_trial_index < len(self.trials):
            return self.trials[self.current_trial_index]
        return None

    def _connect_worker(self) -> None:
        try:
            self.recorder.connect()
        except Exception as error:
            self.connection_messages.put((False, str(error)))
            return
        if self.finished:
            self.recorder.close()
            return
        self.connection_messages.put((True, ""))

    def _poll_connection(self) -> None:
        if self.finished:
            return
        try:
            success, message = self.connection_messages.get_nowait()
        except queue.Empty:
            self.after(50, self._poll_connection)
            return
        if success:
            self._begin_session()
        else:
            self._connection_failed(message)

    def _connection_failed(self, reason: str) -> None:
        self.finished = True
        self.recorder.close()
        messagebox.showerror("Could not start SSVEP recording", reason, parent=self)
        self.destroy()
        self.on_finished()

    def _begin_session(self) -> None:
        if self.finished:
            return
        try:
            self.recorder.begin()
        except Exception as error:
            self._connection_failed(str(error))
            return
        self.session_started = time.perf_counter()
        if self.settings.preparation_s > 0:
            self._enter_phase("preparation")
        else:
            self.current_trial_index = 0
            self._enter_phase("planning")
        self._tick()

    def _duration_for_phase(self, phase: str) -> float:
        return {
            "preparation": self.settings.preparation_s,
            "planning": self.settings.planning_s,
            "action": self.settings.action_s,
            "rest": self.settings.rest_s,
        }[phase]

    def _enter_phase(self, phase: str) -> None:
        self._close_phase_log()
        self.phase = phase
        self.phase_started = time.perf_counter()
        trial = None if phase == "preparation" else self.current_trial
        command = None if trial is None else trial.command
        number = 0 if trial is None else trial.number
        block = 0 if trial is None else trial.block
        frequency = None if trial is None else trial.frequency_hz
        duration = self._duration_for_phase(phase)
        self.current_log = PhaseLog(
            trial_number=number,
            block=block,
            phase=phase,
            command=command,
            frequency_hz=frequency,
            planned_duration_s=duration,
            onset_perf_s=self.phase_started - self.session_started,
            onset_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.phase_log.append(self.current_log)
        self.recorder.marker(encode_marker(command, phase, number))
        self._draw_phase()
        if phase == "action":
            self.target_states = {command_name: None for command_name in COMMANDS}
            self.transition_counts = {command_name: 0 for command_name in COMMANDS}
            self.action_frames = 0

    def _close_phase_log(self) -> None:
        if self.current_log is None or self.current_log.actual_duration_s is not None:
            return
        elapsed = max(0.0, time.perf_counter() - self.phase_started)
        self.current_log.actual_duration_s = elapsed
        if self.current_log.phase == "action":
            self.current_log.rendered_frames = self.action_frames
            if elapsed > 0:
                self.current_log.observed_flicker_hz = {
                    command: self.transition_counts.get(command, 0) / (2.0 * elapsed)
                    for command in COMMANDS
                }

    def _advance(self) -> None:
        if self.phase == "preparation":
            self.current_trial_index = 0
            self._enter_phase("planning")
        elif self.phase == "planning":
            self._enter_phase("action")
        elif self.phase == "action":
            self._enter_phase("rest")
        elif self.phase == "rest":
            self.current_trial_index += 1
            if self.current_trial_index >= len(self.trials):
                self._finish(aborted=False)
            else:
                self._enter_phase("planning")

    def _tick(self) -> None:
        if self.finished:
            return
        elapsed = time.perf_counter() - self.phase_started
        duration = self._duration_for_phase(self.phase)
        if elapsed >= duration:
            self._advance()
            if self.finished:
                return
            elapsed = time.perf_counter() - self.phase_started
            duration = self._duration_for_phase(self.phase)
        self._update_header(max(0.0, duration - elapsed))
        if self.phase == "action":
            self._update_flicker(elapsed)
        elif self.phase == "preparation" and self.settings.preparation_s > 5:
            self._update_tutorial(elapsed)
        self.after_id = self.after(self.FRAME_INTERVAL_MS, self._tick)

    def _canvas_size(self) -> tuple[int, int]:
        self.update_idletasks()
        return max(800, self.canvas.winfo_width()), max(600, self.canvas.winfo_height())

    def _draw_message(self, title: str, subtitle: str) -> None:
        self.canvas.delete("all")
        width, height = self._canvas_size()
        self.canvas.create_text(width / 2, height / 2 - 25, text=title, fill=TEXT, font=("Segoe UI Semibold", 26))
        self.canvas.create_text(width / 2, height / 2 + 25, text=subtitle, fill=MUTED, font=("Segoe UI", 13))

    def _draw_phase(self) -> None:
        self.canvas.delete("all")
        self.target_rectangles.clear()
        if self.tutorial_frame is not None:
            self.tutorial_frame.destroy()
            self.tutorial_frame = None
            self.tutorial_canvas = None

        width, height = self._canvas_size()
        if self.phase == "preparation":
            self.canvas.create_text(
                width / 2,
                70,
                text="PREPARATION",
                fill=TEXT,
                font=("Segoe UI Semibold", 25),
                tags="header_title",
            )
            self.canvas.create_text(
                width / 2,
                116,
                text="Keep your head still. Follow the cue with your eyes and stare at that square while it flickers.",
                fill=MUTED,
                font=("Segoe UI", 14),
                tags="header_subtitle",
            )
            if self.settings.preparation_s > 5:
                self._create_tutorial(width, height)
            else:
                self.canvas.create_text(
                    width / 2,
                    height / 2,
                    text="A direction cue will appear, then all four targets will flicker.",
                    fill=TEXT,
                    font=("Segoe UI", 18),
                )
        elif self.phase == "planning":
            trial = self.current_trial
            assert trial is not None
            self._draw_targets(highlight=trial.command, static=True)
            self.canvas.create_text(
                width / 2,
                height / 2,
                text=f"GET READY\n{COMMAND_LABELS[trial.command]}",
                fill=ACCENT,
                justify="center",
                font=("Segoe UI Semibold", 24),
            )
        elif self.phase == "action":
            trial = self.current_trial
            assert trial is not None
            self._draw_targets(highlight=trial.command, static=False)
            self.canvas.create_text(
                width / 2,
                height / 2,
                text=f"LOOK {COMMAND_LABELS[trial.command]}",
                fill=ACCENT,
                font=("Segoe UI Semibold", 19),
            )
            self.canvas.create_text(width / 2, height / 2 + 35, text="+", fill=TEXT, font=("Consolas", 22))
        elif self.phase == "rest":
            self.canvas.create_text(width / 2, height / 2 - 20, text="+", fill=TEXT, font=("Consolas", 36))
            self.canvas.create_text(width / 2, height / 2 + 45, text="REST", fill=MUTED, font=("Segoe UI", 14))
        self.canvas.create_text(
            width / 2,
            height - 28,
            text="Keep the head and jaw relaxed  •  Esc = stop and save  •  F11 = toggle full screen",
            fill=MUTED,
            font=("Segoe UI", 10),
        )

    def _positions(self) -> tuple[dict[str, tuple[float, float]], float]:
        width, height = self._canvas_size()
        px_per_cm = float(self.winfo_fpixels("1i")) / 2.54
        size = max(35.0, self.settings.square_size_cm * px_per_cm)
        x_offset = self.settings.horizontal_offset_cm * px_per_cm
        y_offset = self.settings.vertical_offset_cm * px_per_cm
        # Preserve the requested physical layout when it fits, otherwise keep all
        # targets safely on a smaller screen and record the requested geometry in JSON.
        x_offset = min(x_offset, width / 2 - size / 2 - 35)
        y_offset = min(y_offset, height / 2 - size / 2 - 85)
        center_x, center_y = width / 2, height / 2
        return {
            "forward": (center_x, center_y - y_offset),
            "right": (center_x + x_offset, center_y),
            "backward": (center_x, center_y + y_offset),
            "left": (center_x - x_offset, center_y),
        }, size

    def _draw_targets(self, *, highlight: str, static: bool) -> None:
        positions, size = self._positions()
        half = size / 2
        for command in COMMANDS:
            x, y = positions[command]
            outline = ACCENT if command == highlight else "#555d6d"
            line_width = 6 if command == highlight else 2
            fill = RED if static and command == highlight else self._gray(self.settings.dark_level)
            rectangle = self.canvas.create_rectangle(
                x - half,
                y - half,
                x + half,
                y + half,
                fill=fill,
                outline=outline,
                width=line_width,
            )
            self.target_rectangles[command] = rectangle
            label_y = y + half + 18
            self.canvas.create_text(
                x,
                label_y,
                text=f"{COMMAND_LABELS[command]}  {self.settings.frequencies_hz[command]:g} Hz",
                fill=TEXT if command == highlight else MUTED,
                font=("Segoe UI Semibold" if command == highlight else "Segoe UI", 11),
            )

    @staticmethod
    def _gray(level: int) -> str:
        level = max(0, min(255, int(level)))
        return f"#{level:02x}{level:02x}{level:02x}"

    def _update_flicker(self, elapsed: float) -> None:
        self.action_frames += 1
        for command, rectangle in self.target_rectangles.items():
            frequency = self.settings.frequencies_hz[command]
            bright = int(elapsed * frequency * 2.0) % 2 == 0
            previous = self.target_states.get(command)
            if previous is not None and previous != bright:
                self.transition_counts[command] += 1
            self.target_states[command] = bright
            level = self.settings.bright_level if bright else self.settings.dark_level
            self.canvas.itemconfigure(rectangle, fill=self._gray(level))

    def _create_tutorial(self, width: int, height: int) -> None:
        self.tutorial_frame = tk.Frame(self.canvas, bg=PANEL, highlightbackground="#374055", highlightthickness=2)
        self.tutorial_canvas = tk.Canvas(
            self.tutorial_frame,
            width=540,
            height=310,
            bg=PANEL,
            highlightthickness=0,
        )
        self.tutorial_canvas.pack(padx=12, pady=12)
        self.canvas.create_window(width / 2, height / 2 + 35, window=self.tutorial_frame)

    def _update_tutorial(self, elapsed: float) -> None:
        canvas = self.tutorial_canvas
        if canvas is None:
            return
        canvas.delete("all")
        width, height = 540, 310
        centers = {
            "forward": (width / 2, 55),
            "right": (width - 75, height / 2),
            "backward": (width / 2, height - 62),
            "left": (75, height / 2),
        }
        command = COMMANDS[int(elapsed / 1.4) % len(COMMANDS)]
        for name, (x, y) in centers.items():
            color = "#f5f5f5" if name == command else "#474e5d"
            canvas.create_rectangle(x - 23, y - 23, x + 23, y + 23, fill=color, outline="")
            canvas.create_text(x, y + 36, text=COMMAND_LABELS[name], fill=MUTED, font=("Segoe UI", 8))
        eye_x, eye_y = width / 2, height / 2
        target_x, target_y = centers[command]
        progress = 0.5 - 0.5 * math.cos(min(1.0, (elapsed % 1.4) / 0.45) * math.pi)
        gaze_x = eye_x + (target_x - eye_x) * progress * 0.72
        gaze_y = eye_y + (target_y - eye_y) * progress * 0.72
        canvas.create_line(eye_x, eye_y, target_x, target_y, fill=ACCENT, width=2, arrow=tk.LAST)
        canvas.create_oval(gaze_x - 10, gaze_y - 7, gaze_x + 10, gaze_y + 7, fill=ACCENT, outline="")
        canvas.create_text(
            width / 2,
            height - 14,
            text=f"When cued {COMMAND_LABELS[command]}, keep your gaze on that square during flicker.",
            fill=TEXT,
            font=("Segoe UI Semibold", 10),
        )

    def _update_header(self, remaining: float) -> None:
        width, _ = self._canvas_size()
        trial = self.current_trial
        if self.phase == "preparation":
            title = f"PREPARATION  {remaining:0.1f}s"
            progress = "Session starts next"
        else:
            title = f"{self.phase.upper()}  {remaining:0.1f}s"
            number = 0 if trial is None else trial.number
            block = 0 if trial is None else trial.block
            progress = f"Trial {number}/{len(self.trials)}  •  Block {block}/{self.settings.repetitions_per_command}"
        if self.canvas.find_withtag("live_header"):
            self.canvas.itemconfigure("live_header", text=title)
            self.canvas.itemconfigure("live_progress", text=progress)
        else:
            self.canvas.create_text(
                30,
                26,
                anchor="w",
                text=title,
                fill=TEXT,
                font=("Segoe UI Semibold", 17),
                tags="live_header",
            )
            self.canvas.create_text(
                width - 30,
                26,
                anchor="e",
                text=progress,
                fill=MUTED,
                font=("Segoe UI", 11),
                tags="live_progress",
            )

    def _toggle_fullscreen(self, _event=None) -> str:
        enabled = bool(self.attributes("-fullscreen"))
        self.attributes("-fullscreen", not enabled)
        return "break"

    def _request_stop(self, _event=None) -> str:
        if self.finished:
            return "break"
        if self.phase == "connecting":
            if messagebox.askyesno(
                "Cancel connection?",
                "Cancel the EEG-board connection and return to SSVEP settings?",
                parent=self,
            ):
                self.finished = True
                self.destroy()
                self.on_finished()
            return "break"
        if messagebox.askyesno(
            "Stop SSVEP session?",
            "Stop now and save the partial recording with aborted=true?",
            parent=self,
        ):
            self._finish(aborted=True)
        return "break"

    def _finish(self, *, aborted: bool) -> None:
        if self.finished:
            return
        self.finished = True
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except tk.TclError:
                pass
        self._close_phase_log()
        self._draw_message("Saving recording…", "EEG, markers, event table, and metadata are being written.")
        self.update()
        try:
            captured = self.recorder.finish()
        except Exception as error:
            self.recorder.close()
            messagebox.showerror("Could not finish recording", str(error), parent=self)
            self.destroy()
            self.on_finished()
            return
        self.recorder.close()
        try:
            paths = save_session(captured, self.settings, self.phase_log, aborted=aborted)
        except Exception as error:
            messagebox.showerror("Could not save SSVEP recording", str(error), parent=self)
        else:
            state = "Partial recording saved" if aborted else "SSVEP session complete"
            messagebox.showinfo(
                state,
                "Saved automatically labeled files:\n\n"
                f"EEG: {paths['fif']}\n\n"
                f"Events: {paths['events']}\n\n"
                f"Metadata: {paths['metadata']}",
                parent=self,
            )
        self.destroy()
        self.on_finished()


def run() -> None:
    SettingsWindow().run()
