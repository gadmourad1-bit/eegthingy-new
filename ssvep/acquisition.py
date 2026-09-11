"""Independent OpenBCI Cyton+Daisy acquisition and SSVEP file writer."""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from serial.tools import list_ports

from .config import EEG_CHANNEL_NAMES, USABLE_EEG_INDICES, Settings
from .protocol import decode_marker, marker_description


@dataclass
class PhaseLog:
    trial_number: int
    block: int
    phase: str
    command: str | None
    frequency_hz: float | None
    planned_duration_s: float
    onset_perf_s: float
    onset_utc: str
    actual_duration_s: float | None = None
    rendered_frames: int | None = None
    observed_flicker_hz: dict[str, float] | None = None


@dataclass
class CapturedData:
    eeg_uv: np.ndarray
    acceleration: np.ndarray
    markers: np.ndarray
    timestamps: np.ndarray
    sample_rate: float
    board_name: str


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned or "unknown"


def find_serial_port() -> str:
    """Choose the most likely OpenBCI/FTDI serial port without importing MI code."""
    ports = list(list_ports.comports())
    likely = [
        port.device
        for port in ports
        if "ftdi" in (port.manufacturer or "").lower()
        or "usb serial" in (port.description or "").lower()
        or "usbserial" in port.device.lower()
        or (port.vid, port.pid) == (0x0403, 0x6015)
    ]
    if likely:
        return likely[0]
    if len(ports) == 1:
        return ports[0].device
    return ""


class OpenBCIRecorder:
    """Own one BrainFlow session and collect data directly from its ring buffer."""

    def __init__(self, *, synthetic: bool = False, serial_port: str = ""):
        self.synthetic = synthetic
        self.serial_port = serial_port.strip()
        self.board: BoardShim | None = None
        self.board_id: int | None = None
        self.eeg_rows: list[int] = []
        self.acc_rows: list[int] = []
        self.marker_row = -1
        self.timestamp_row = -1
        self.recording = False

    def connect(self) -> "OpenBCIRecorder":
        if self.board is not None:
            return self
        params = BrainFlowInputParams()
        params.timeout = 30
        if self.synthetic:
            board_id = BoardIds.SYNTHETIC_BOARD.value
        else:
            port = self.serial_port or find_serial_port()
            if not port:
                raise RuntimeError(
                    "No OpenBCI serial port was found. Plug in the Cyton dongle, "
                    "or enter its COM port in the SSVEP settings."
                )
            params.serial_port = port
            self.serial_port = port
            board_id = BoardIds.CYTON_DAISY_BOARD.value

        BoardShim.disable_board_logger()
        board = BoardShim(board_id, params)
        try:
            board.prepare_session()
            board.start_stream()
        except Exception:
            try:
                board.release_session()
            except Exception:
                pass
            raise

        all_eeg_rows = list(BoardShim.get_eeg_channels(board_id))
        if len(all_eeg_rows) < 16:
            board.stop_stream()
            board.release_session()
            raise RuntimeError(
                f"The selected board exposes {len(all_eeg_rows)} EEG channels; "
                "this experiment requires the 16-input Cyton+Daisy setup."
            )

        self.board = board
        self.board_id = board_id
        self.eeg_rows = [all_eeg_rows[index] for index in USABLE_EEG_INDICES]
        self.acc_rows = list(BoardShim.get_accel_channels(board_id))[:3]
        self.marker_row = BoardShim.get_marker_channel(board_id)
        self.timestamp_row = BoardShim.get_timestamp_channel(board_id)
        return self

    @property
    def sample_rate(self) -> float:
        if self.board_id is None:
            raise RuntimeError("The board is not connected.")
        return float(BoardShim.get_sampling_rate(self.board_id))

    @property
    def board_name(self) -> str:
        if self.board_id is None:
            return "not-connected"
        return "BrainFlow synthetic" if self.synthetic else "OpenBCI Cyton+Daisy"

    def begin(self) -> None:
        if self.board is None:
            raise RuntimeError("The board is not connected.")
        self.board.get_board_data()  # discard connection/setup samples
        self.recording = True

    def marker(self, value: int) -> None:
        if self.board is None or not self.recording:
            raise RuntimeError("Cannot insert a marker before recording starts.")
        self.board.insert_marker(float(value))

    def finish(self) -> CapturedData:
        if self.board is None:
            raise RuntimeError("The board is not connected.")
        raw = self.board.get_board_data()
        self.recording = False
        if raw.size == 0:
            return CapturedData(
                eeg_uv=np.empty((15, 0)),
                acceleration=np.empty((len(self.acc_rows), 0)),
                markers=np.empty(0),
                timestamps=np.empty(0),
                sample_rate=self.sample_rate,
                board_name=self.board_name,
            )
        return CapturedData(
            eeg_uv=raw[self.eeg_rows, :],
            acceleration=raw[self.acc_rows, :] if self.acc_rows else np.empty((0, raw.shape[1])),
            markers=raw[self.marker_row, :],
            timestamps=raw[self.timestamp_row, :],
            sample_rate=self.sample_rate,
            board_name=self.board_name,
        )

    def close(self) -> None:
        board, self.board = self.board, None
        self.recording = False
        if board is None:
            return
        try:
            board.stop_stream()
        finally:
            board.release_session()


def save_session(
    captured: CapturedData,
    settings: Settings,
    phase_log: list[PhaseLog],
    *,
    aborted: bool,
) -> dict[str, Path]:
    """Save EEG, annotations, an event table, and complete reproducibility metadata."""
    if captured.eeg_uv.shape[0] != len(EEG_CHANNEL_NAMES):
        raise ValueError(
            f"Expected {len(EEG_CHANNEL_NAMES)} EEG channels, got {captured.eeg_uv.shape[0]}."
        )
    if captured.eeg_uv.shape[1] == 0:
        raise ValueError("No EEG samples were recorded, so no file was written.")

    output_dir = Path(settings.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = (
        f"sub-{_safe_component(settings.subject)}_"
        f"ses-{_safe_component(settings.session)}_{stamp}_ssvep"
    )
    fif_path = output_dir / f"{stem}_raw.fif"
    csv_path = output_dir / f"{stem}_events.csv"
    json_path = output_dir / f"{stem}_metadata.json"

    n_samples = captured.eeg_uv.shape[1]
    acc = captured.acceleration
    if acc.shape[1] != n_samples:
        acc = np.empty((0, n_samples))
    marker = np.asarray(captured.markers, dtype=float).reshape(-1)
    if marker.size != n_samples:
        marker = np.zeros(n_samples, dtype=float)

    # BrainFlow reports EEG in microvolts; MNE stores EEG in volts.
    data_rows = [captured.eeg_uv / 1e6]
    channel_names = list(EEG_CHANNEL_NAMES)
    channel_types = ["eeg"] * len(EEG_CHANNEL_NAMES)
    if acc.shape[0]:
        data_rows.append(acc)
        acc_names = ["ACCX", "ACCY", "ACCZ"][: acc.shape[0]]
        channel_names.extend(acc_names)
        channel_types.extend(["misc"] * len(acc_names))
    data_rows.append(marker[np.newaxis, :])
    channel_names.append("STI 014")
    channel_types.append("stim")

    info = mne.create_info(channel_names, captured.sample_rate, channel_types)
    raw = mne.io.RawArray(np.vstack(data_rows), info, verbose="ERROR")
    timestamps = np.asarray(captured.timestamps, dtype=float).reshape(-1)
    if timestamps.size and np.isfinite(timestamps[0]) and timestamps[0] > 0:
        raw.set_meas_date(datetime.fromtimestamp(float(timestamps[0]), tz=timezone.utc))
    raw.set_montage("standard_1020", on_missing="warn")

    marker_indices = np.flatnonzero(marker != 0)
    if marker_indices.size:
        onsets = marker_indices / captured.sample_rate
        ends = np.append(marker_indices[1:], n_samples)
        durations = (ends - marker_indices) / captured.sample_rate
        descriptions = [
            marker_description(marker[index], settings.frequencies_hz)
            for index in marker_indices
        ]
        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                orig_time=raw.info["meas_date"],
            )
        )

    raw.info["description"] = json.dumps(
        {
            "experiment": "four-command SSVEP data collection",
            "attended_label_definition": (
                "During action, all four squares flicker; command/frequency is the "
                "square the subject was instructed to look at."
            ),
            "settings": asdict(settings),
            "aborted": aborted,
        }
    )
    raw.save(fif_path, overwrite=False, verbose="ERROR")

    fieldnames = [field.name for field in PhaseLog.__dataclass_fields__.values()]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in phase_log:
            row = asdict(item)
            if row["observed_flicker_hz"] is not None:
                row["observed_flicker_hz"] = json.dumps(row["observed_flicker_hz"], sort_keys=True)
            writer.writerow(row)

    nonzero_markers: list[dict[str, Any]] = []
    for index in marker_indices:
        nonzero_markers.append(
            {
                "sample": int(index),
                "onset_s": float(index / captured.sample_rate),
                "value": int(round(marker[index])),
                "description": marker_description(marker[index], settings.frequencies_hz),
                **decode_marker(marker[index]),
            }
        )
    metadata = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "four-command SSVEP data collection",
        "subject": settings.subject,
        "session": settings.session,
        "board": captured.board_name,
        "sample_rate_hz": captured.sample_rate,
        "eeg_channel_names": list(EEG_CHANNEL_NAMES),
        "unused_physical_channel": "Cyton channel 8",
        "sample_count": n_samples,
        "duration_s": n_samples / captured.sample_rate,
        "aborted": aborted,
        "settings": asdict(settings),
        "markers": nonzero_markers,
        "phase_log": [asdict(item) for item in phase_log],
        "files": {"eeg": fif_path.name, "events": csv_path.name},
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"fif": fif_path, "events": csv_path, "metadata": json_path}
