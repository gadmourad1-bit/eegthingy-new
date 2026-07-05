import sys
import time
import ctypes
import threading
import serial
import numpy as np

try:
    import requests
except ImportError:
    requests = None
try:
    import pyglet
except ImportError:
    pyglet = None
try:
    from sdk.vernier.bindings import GoIOSDK, SKIP_CMD_ID_START_MEASUREMENTS, SKIP_CMD_ID_STOP_MEASUREMENTS
except Exception:
    GoIOSDK = None
    SKIP_CMD_ID_START_MEASUREMENTS = SKIP_CMD_ID_STOP_MEASUREMENTS = None

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from serial import Serial
from serial.tools import list_ports

# OpenBCI Cyton Daisy Board
class OpenBCI:
    def __init__(self, interval=1.0, synthetic=False):
        self.interval = interval
        self.synthetic = synthetic
        self.board = None
        self.filter = None
        self.worker = 0
        self.callback = None

        self.last_sample = {
            'eeg': np.zeros(16),
            'acc': np.zeros(3),
            'marker': 0,
            'timestamp': -1
        }
        
        # channel indices into BrainFlow's data array
        self.eeg = []
        self.acc = []
        self.marker = -1
        self.timestamp = -1
 
        # recording state
        self._recording = False
        self._chunks = []
        self._drain_thread = None

    def find_serial_port(self):
        openbci_port = ''
        for port in list_ports.comports():
            try:
                s = Serial(port=port.device, baudrate=115200, timeout=10)
                s.write(b'v')
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    while '$$$' not in line:
                        line += s.read().decode('utf-8', errors='replace')
                    if "OpenBCI" in line:
                        openbci_port = port.device
                        s.close()
                        break
                s.close()
            except (OSError, serial.SerialException):
                pass

        return openbci_port

    def open(self):
        try:
            params = BrainFlowInputParams()
            params.timeout = 30
            if self.synthetic:
                board_id = BoardIds.SYNTHETIC_BOARD.value
            else:
                params.serial_port = self.find_serial_port()
                board_id = BoardIds.CYTON_DAISY_BOARD.value
            temp_board = BoardShim(board_id, params)
            temp_board.prepare_session()
            time.sleep(1)
            temp_board.start_stream()
            self.board = temp_board
 
            bid = self.board.board_id
            self.eeg = BoardShim.get_eeg_channels(bid)
            self.acc = BoardShim.get_accel_channels(bid)
            self.marker = BoardShim.get_marker_channel(bid)
            self.timestamp = BoardShim.get_timestamp_channel(bid)

            print(f"eeg channels: {self.eeg}")
            print(f"acc channels: {self.acc}")
            print(f"marker channel: {self.marker}")
            print(f"timestamp channel: {self.timestamp}")
        except Exception as error:
            print('failed connecting to OpenBCI port, retrying: ', error)
        
        return self

    def close(self):
        if self._recording:
            self.stop()
        if self.board is not None:
            self.board.stop_stream()
            self.board.release_session()
        return self

    def start(self):
        """Discard any buffered data and begin accumulating from this moment.
        If already recording, the current recording is dropped and restarts."""
        if self.board is None:
            print("OpenBCI.start: board not connected yet")
            return self

        if self._recording:
            print("OpenBCI.start: already recording, restarting")
            self._recording = False
            if self._drain_thread is not None:
                self._drain_thread.join(timeout=self.interval + 2.0)

        self.worker += 1
        worker_id = self.worker

        self.board.get_board_data()
        self._chunks = []
        self._recording = True

        def drain_worker():
            while self._recording and self.worker == worker_id:
                time.sleep(self.interval)
                if self.board is not None and self._recording and self.worker == worker_id:
                    chunk = self.board.get_board_data()
                    if chunk.size:
                        self._chunks.append(chunk)
                        if self.callback is not None:
                            self.callback(chunk)

            if self.board is not None and self.worker == worker_id:
                chunk = self.board.get_board_data()
                if chunk.size:
                    self._chunks.append(chunk)
                    if self.callback is not None:
                        self.callback(chunk)

        self._drain_thread = threading.Thread(target=drain_worker, daemon=True)
        self._drain_thread.start()
        return self

    def stop(self):
        """Stop recording. Returns dict with eeg (16, n), acc (3, n), marker (n,), timestamp (n,)."""
        empty = {'eeg': np.empty((16, 0)), 'acc': np.empty((3, 0)), 'markers': np.empty(0), 'timestamp': np.empty(0)}
        if not self._recording:
            return empty
 
        self._recording = False
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=self.interval + 2.0)

        self.last_sample['marker'] = 0

        if not self._chunks:
            return empty
 
        full = np.concatenate(self._chunks, axis=1)
        eeg = full[self.eeg, :]
        acc = full[self.acc, :]

        if self.filter is not None:
            eeg = self.filter.apply(eeg, 'eeg')

        return {
            'eeg':       eeg,
            'acc':       acc,
            'markers':    full[self.marker, :],
            'timestamp': full[self.timestamp, :],
        }

    def insert_marker(self, value: float):
        if self.board is not None and value > 0:
            self.board.insert_marker(float(value))

    def add_callback(self, callback):
        self.callback = callback
 
    def get_data(self):
        if self.board is not None:
            window = self.board.get_current_board_data(256)

            if window.shape[1] == 0:
                return self.last_sample

            latest = window[:, -1]
            eeg_window = window[self.eeg, :]

            if self.filter is not None:
                eeg_window = self.filter.apply(eeg_window, 'eeg')

            # RMS over the window, after demeaning so the DC offset doesn't dominate
            demeaned = eeg_window - eeg_window.mean(axis=1, keepdims=True)
            eeg = np.sqrt(np.mean(demeaned ** 2, axis=1))

            markers = window[self.marker, :]
            nonzero = np.flatnonzero(markers)
            marker = markers[nonzero[-1]] if nonzero.size else self.last_sample['marker']

            self.last_sample = {
                'eeg':       eeg,
                'acc':       latest[self.acc],
                'marker':    marker,
                'timestamp': latest[self.timestamp]
            }

        return self.last_sample
 
    def set_filter(self, _filter):
        self.filter = _filter
        return self

# Vernier hand dynamometer
class Vernier:
    def __init__(self):
        self.sensor = 0
        self.filter = None
        self.last_sample = 0

        self.lib = GoIOSDK().get_library()
        self.lib.GoIO_Init()

    def open(self):
        try:
            time.sleep(0.5)
            found, name, vendor, product = GoIOSDK().get_device()
            if not found:
                raise Exception("failed connecting to Vernier device, retrying.")

            self.sensor = self.lib.GoIO_Sensor_Open(name.encode(), vendor, product, 0)
            if self.sensor == 0:
                raise Exception("failed to open Vernier sensor, retrying.")

            char_id = ctypes.c_ubyte()
            self.lib.GoIO_Sensor_DDSMem_GetSensorNumber(self.sensor, ctypes.byref(char_id), 0, 0)
            self.lib.GoIO_Sensor_SendCmdAndGetResponse(self.sensor, SKIP_CMD_ID_START_MEASUREMENTS, None, 0,
                                                               None, None, 1000)
        except Exception as error:
            print("error while connecting to Vernier device, retrying:", error)

        return self

    def close(self):
        if self.sensor != 0:
            self.lib.GoIO_Sensor_SendCmdAndGetResponse(self.sensor, SKIP_CMD_ID_STOP_MEASUREMENTS, None, 0, None, None,
                                                       500)
            time.sleep(0.5)
            self.lib.GoIO_Sensor_Close(self.sensor)
        return self

    def get_data(self):
        if self.sensor != 0 and self.lib.GoIO_Sensor_GetNumMeasurementsAvailable(self.sensor) > 0:
            latest = self.lib.GoIO_Sensor_GetLatestRawMeasurement(self.sensor)
            volts = self.lib.GoIO_Sensor_ConvertToVoltage(self.sensor, latest)
            sample = self.lib.GoIO_Sensor_CalibrateData(self.sensor, volts)

            if self.filter is not None:
                sample = self.filter.apply(sample, 'hd')

            self.last_sample = sample

        return self.last_sample

    def set_filter(self, _filter):
        self.filter = _filter
        return self


class Noraxon:
    def __init__(self):
        self.session = None
        self.address = None
        self.port = None
        self.filter = None
        self.last_sample = np.zeros(4)

    def open(self, address, port):
        self.address = address
        self.port = port

        def worker():
            while self.session is None:
                try:
                    temp_session = requests.Session()
                    headers = temp_session.get(f"http://{address}:{port}/headers")
                    
                    print(headers.json()["headers"])

                    if len(headers.json()["headers"]) != 4:
                        raise Exception(f"expected 4 emg sensors, got {len(headers.json()['headers'])}")
                    else:
                        enabled = temp_session.get(f"http://{address}:{port}/enable/all")

                        if enabled.status_code != 200:
                            raise Exception("failed to enable all channels.")

                    self.session = temp_session
                    break
                except Exception as error:
                    print(f"failed connecting to Noraxon stream, retrying: {error}")

                time.sleep(0.5)

        threading.Thread(target=worker, daemon=True).start()
        return self

    def close(self):
        if self.session is not None:
            disable = self.session.get(f"http://{self.address}:{self.port}/disable/all")
            if disable.status_code != 200:
                raise Exception("failed to disable all channels.")
            self.session.close()
        return self

    def get_data(self):
        if self.session is not None:
            data = self.session.get(f"http://{self.address}:{self.port}/samples")

            if data.status_code == 200:
                channels = data.json()["channels"]

                sample = np.zeros(4)
                
                for _, channel in enumerate(channels):
                    index = channel["index"]
                    latest = channel["samples"][-1]
                    sample[index] = latest

                if self.filter is not None:
                    sample = self.filter.apply(sample, 'emg')

                self.last_sample = sample

        return self.last_sample

    def set_filter(self, _filter):
        self.filter = _filter
        return self

if sys.platform == 'darwin':
    from pyglet.libs.darwin.cocoapy import cf, kCFRunLoopDefaultMode
    cf.CFRunLoopRunInMode.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_bool]
    cf.CFRunLoopRunInMode.restype = ctypes.c_int32
    _CF_RUN_LOOP_MODE = kCFRunLoopDefaultMode

    def _pyglet_loop_start():
        pass

    def _pump_runloop():
        while cf.CFRunLoopRunInMode(_CF_RUN_LOOP_MODE, 0.0, True) == 4:
            pass  # 4 = kCFRunLoopRunHandledSource — drain pending sources
else:
    _pyglet_loop_started = False

    def _pyglet_loop_start():
        global _pyglet_loop_started
        if not _pyglet_loop_started:
            pyglet.app.platform_event_loop.start()
            _pyglet_loop_started = True

    def _pump_runloop():
        try:
            pyglet.app.platform_event_loop.step(timeout=0)
            pyglet.app.platform_event_loop.dispatch_posted_events()
        except Exception:
            pass

class Joystick:
    def __init__(self):
        self.joystick = None
        self.filter = None
        self.last_sample = np.zeros(2, dtype=np.float64)

    def open(self):
        _pyglet_loop_start()
        sticks = pyglet.input.get_joysticks()
        if not sticks:
            print("no joystick found")
            return
        
        self.joystick = sticks[0]
        self.joystick.open()
        print(f"connected: {self.joystick.device.name}")

        @self.joystick.event
        def on_joyaxis_motion(joystick, axis, value):
            if axis == 'x':
                self.last_sample[0] = value
            elif axis == 'y':
                self.last_sample[1] = -value
            if self.filter is not None:
                self.last_sample = self.filter.apply(self.last_sample, 'joystick')
        
        return self

    def tick(self):
        if self.joystick is None:
            return
        _pump_runloop()

    def close(self):
        if self.joystick is not None:
            self.joystick.close()
            self.joystick = None
        return self

    def get_data(self):
        return self.last_sample.copy()

    def set_filter(self, f):
        self.filter = f
        return self