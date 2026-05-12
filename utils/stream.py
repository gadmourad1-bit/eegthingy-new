import time
import threading
import numpy as np
import mne

from brainflow.board_shim import BoardShim
from utils.globals import openbci, vernier, noraxon, joystick

LABELS = {1: 'left_hand', 2: 'right_hand', 3: 'left_foot', 4: 'right_foot'}
PHASES = {1: 'prep', 2: 'plan', 3: 'task', 4: 'rest'}

EEG_NAMES = [
    'Cz',  'Pz',  'C3', 'C4', 'T5', 'T6', 'Fz', 'DEAD',  # Cyton ch 1-8  (ch8 unplugged)
    'F7',  'F8',  'F3', 'F4', 'T3', 'T4', 'P3', 'P4',    # Daisy  ch 9-16
]

def decode_marker(value):
    v = int(value)
    label = LABELS.get(v // 1_000_000, '?')
    phase = PHASES.get((v // 100_000) % 10, '?')
    trial = v % 100_000
    return f'{label}/{phase}/t{trial}'


class Stream:
    def __init__(self):
        self.brainflow_data = {
            'eeg':       np.empty((16, 0)), 
            'acc':       np.empty((3, 0)),
            'markers':   np.empty(0),
            'timestamp': np.empty(0)
        }

        self.misc_data = []
        self.running = False
        self._worker = None

    def start(self):
        """This method creates a stream and collects all data and features at the board's sample rate."""
        self.running = True

        def worker():
            perf_counter_ns = time.perf_counter_ns
            np_hstack = np.hstack

            while self.running and openbci.board is None:
                time.sleep(0.1)
            if not self.running:
                return

            sample_rate = BoardShim.get_sampling_rate(openbci.board.board_id)
            target = (1 / sample_rate) * (10 ** 9)

            openbci.start()

            start = perf_counter_ns()
            count = 0
            
            while self.running:
                now = perf_counter_ns()
                next_sample = start + (target * count)

                if now > next_sample:
                    self.misc_data.append(np_hstack([noraxon.get_data(), joystick.get_data(), vernier.get_data()]))
                    count += 1

                time.sleep(0.00001)

            elapsed = (time.perf_counter_ns() - start) / 1e9
            self.brainflow_data = openbci.stop()
            print(f"Stream stopped: misc={count/elapsed:.2f}Hz, "
                  f"eeg={self.brainflow_data['eeg'].shape[1]} samples")

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def stop(self):
        """This method stops the stream."""
        self.running = False

    def event(self, value: float):
        openbci.insert_marker(value)

    def save(self, filename, params=None):
        if self.running:
            self.stop()

        if self._worker is not None:
            self._worker.join(timeout=5.0)

        if openbci.board is None:
            print("Stream.save: board never connected, skipping")
            return

        sample_rate = BoardShim.get_sampling_rate(openbci.board.board_id)
        eeg = self.brainflow_data['eeg']
        acc = self.brainflow_data['acc']
        ts  = self.brainflow_data['timestamp']
        markers = self.brainflow_data.get('markers', np.empty(0))
        misc = np.array(self.misc_data)
 
        if eeg.size == 0 or misc.size == 0:
            print("Stream.save: no data, skipping")
            return

        n = min(eeg.shape[1], misc.shape[0])

        eeg = eeg[:, :n]
        acc = acc[:, :n]
        ts  = ts[:n]
        markers = markers[:n] if markers.size else np.zeros(n)
        misc = misc[:n].T
 
        eeg_v = eeg / 1e6
        data = np.vstack([eeg_v, acc, misc])
 
        ch_names = (EEG_NAMES
                    + ['ACCX', 'ACCY', 'ACCZ']
                    + ['EMG1', 'EMG2', 'EMG3', 'EMG4']
                    + ['JOYX', 'JOYY']
                    + ['FORCE'])
        
        ch_types = (['eeg'] * 16
                    + ['misc'] * 3
                    + ['emg'] * 4
                    + ['misc'] * 2
                    + ['misc'])
 
        info = mne.create_info(ch_names=ch_names, sfreq=sample_rate, ch_types=ch_types)
        raw  = mne.io.RawArray(data, info)
        raw.set_meas_date(float(ts[0]))
        
        # we have a disabled electrode at ch8
        raw.set_channel_types({'DEAD': 'misc'})

        raw.set_montage('standard_1020', on_missing='warn')
 
        idx = np.where(markers != 0)[0]

        if idx.size:
            onsets = idx / sample_rate
            next_starts = np.append(idx[1:], n)
            durations = (next_starts - idx) / sample_rate
            descs = [decode_marker(markers[i]) for i in idx]

            raw.set_annotations(mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descs,
                orig_time=raw.info['meas_date'],
            ))
 
        if params is not None:
            raw.info['description'] = params
 
        raw.save(filename, overwrite=True)
        print(f"Stream saved {raw.n_times} samples, "
              f"{len(raw.annotations)} annotations -> {filename}")
 
        self.brainflow_data = {'eeg': np.empty((16, 0)), 'acc': np.empty((3, 0)),
                               'timestamp': np.empty(0), 'markers': np.empty(0)}
        self.misc_data = []