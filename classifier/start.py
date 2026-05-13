
import glob
import mne
import numpy as np
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brainflow.board_shim import BoardShim
from config import DATA_DIR, DELTA_T, EEG_CHANNELS, EPOCH_TMIN, EPOCH_TMAX, FILTER_KWARGS, TARGET_MAPPINGS
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from utils.devices import OpenBCI
from ws import WebSocket
from smoother import Smoother
from gui import GUI

def bandpass(data, sfreq):
    return mne.filter.filter_data(data, sfreq=sfreq, verbose=False, **FILTER_KWARGS)

def build_event_id(raw, target_id_dict):
    """Expand prefix-based target mappings to the exact annotation descriptions
    present in this raw file. Each annotation is assigned the code of the first
    target whose key matches it (exact or hierarchical prefix with '/')."""
    event_id = {}
    for desc in set(raw.annotations.description):
        d = str(desc).strip().lower()
        for prefix, code in target_id_dict.items():
            p = prefix.lower()
            if d == p or d.startswith(p + '/'):
                event_id[str(desc)] = code
                break
    return event_id


def process_data(file_name, target_id_dict):
    print(f"\n--- Loading: {file_name} ---")
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.pick(EEG_CHANNELS)
    raw.set_montage('standard_1020', on_missing='ignore')
    raw.annotations.description = np.array([str(d).strip().lower() for d in raw.annotations.description])

    raw.filter(**FILTER_KWARGS)

    event_id = build_event_id(raw, target_id_dict)
    if not event_id:
        raise ValueError(
            f"No annotations in {file_name} matched any prefix in {target_id_dict}. "
            f"Annotations present: {sorted(set(raw.annotations.description))[:5]}…"
        )

    events, event_id_used = mne.events_from_annotations(raw, event_id=event_id)
    epochs = mne.Epochs(raw, events, event_id=event_id_used,
                        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=None,
                        preload=True, proj=False, on_missing='warn')

    print(f"Successfully created {len(epochs)} epochs for classes: {epochs.event_id}")
    return epochs

def discover_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, '*_mi_raw.fif')))

def build_pipeline():
    return Pipeline([
        ('CSP', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ('Classifier', LogisticRegression()),
    ])

def parse_indices(raw, n):
    out = []
    for tok in raw.replace(' ', '').split(','):
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"'{tok}' is not a number")
        i = int(tok)
        if i < 1 or i > n:
            raise ValueError(f"{i} is out of range (1..{n})")
        if i - 1 not in out:
            out.append(i - 1)
    return out

def select_files(prompt, files):
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            print("pick at least one")
            continue
        try:
            idxs = parse_indices(raw, len(files))
        except ValueError as e:
            print(f"invalid: {e}")
            continue
        if not idxs:
            print("pick at least one")
            continue
        return [files[i] for i in idxs]

def run_offline():
    files = discover_files()
    if not files:
        print(f"no .fif files found in {DATA_DIR}")
        return

    print("\nAvailable files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {os.path.basename(f)}")
    print()

    train_files = select_files("training files (e.g. 1,2,3): ", files)
    if train_files is None:
        return
    test_files = select_files("testing files  (e.g. 1,2,3): ", files)
    if test_files is None:
        return

    overlap = set(train_files) & set(test_files)
    if overlap:
        print(f"warning: file(s) used in both train and test: {[os.path.basename(f) for f in overlap]}")

    epochs_train = mne.concatenate_epochs(
        [process_data(f, TARGET_MAPPINGS) for f in train_files]
    )

    epochs_test = mne.concatenate_epochs(
        [process_data(f, TARGET_MAPPINGS) for f in test_files]
    ) if len(test_files) > 1 else process_data(test_files[0], TARGET_MAPPINGS)

    X_train = epochs_train.get_data()
    y_train = epochs_train.events[:, -1]
    X_test = epochs_test.get_data()
    y_test = epochs_test.events[:, -1]

    clf = build_pipeline()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Y_TEST:", y_test)
    print("Y_PRED:", y_pred)
    print("\n" + "=" * 35)
    print(f"Motor Imagery Test Accuracy: {acc * 100:.2f}%")
    print("=" * 35)

def run_online():
    files = discover_files()
    if not files:
        print(f"no .fif files found in {DATA_DIR}")
        return

    print("\nAvailable files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {os.path.basename(f)}")
    print()

    train_files = select_files("training files (e.g. 1,2,3): ", files)
    if train_files is None:
        return

    print(f"\nTraining on {len(train_files)} file(s):")
    for f in train_files:
        print(f"  - {os.path.basename(f)}")

    epochs_all = mne.concatenate_epochs([
        process_data(f, TARGET_MAPPINGS) for f in train_files
    ]) if len(train_files) > 1 else process_data(train_files[0], TARGET_MAPPINGS)
    X_train = epochs_all.get_data()
    y_train = epochs_all.events[:, -1]
    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]

    clf = build_pipeline()
    clf.fit(X_train, y_train)
    print(f"\nTrained on {len(X_train)} epochs across {n_channels} channels x {n_times} samples.")

    # The OpenBCI chunk delivers the same channel layout as the raw .fif
    # (before process_data picks EEG_CHANNELS). Look up the indices of the
    # training channels in that full ordering and use them to slice the chunk.
    raw_full = mne.io.read_raw_fif(train_files[0], preload=False)
    full_names = raw_full.ch_names
    train_sfreq = float(raw_full.info['sfreq'])
    try:
        train_idx = [full_names.index(name) for name in EEG_CHANNELS]
    except ValueError as e:
        print(f"channel mismatch between training data and EEG_CHANNELS: {e}")
        return

    ws = WebSocket()
    ws.start()
    smoother = Smoother(ws)

    bci = OpenBCI(interval=DELTA_T).open()
    if bci.board is None:
        print("OpenBCI failed to open.")
        ws.stop()
        return

    sfreq = float(BoardShim.get_sampling_rate(bci.board.board_id))
    if abs(sfreq - train_sfreq) > 0.5:
        print(f"warning: live sfreq={sfreq} differs from training sfreq={train_sfreq}; "
              "predictions may degrade")

    window_n = int(round(DELTA_T * sfreq))
    # Past context for the minimum-phase FIR (~filter length ≈ 413 samples at
    # 250Hz with our transition bandwidths). 2s is comfortably above that.
    context_n = int(round(2.0 * sfreq))
    buffer_n = window_n + context_n
    buffer = np.zeros((len(train_idx), 0), dtype=np.float64)
    buffer_lock = threading.Lock()

    print(f"window: {DELTA_T}s ({window_n} samples), buffer: {buffer_n} samples")

    # Create the GUI before BCI starts streaming so push_window has a target.
    gui = GUI(bci, smoother, clf.classes_, sfreq)

    def on_chunk(chunk):
        nonlocal buffer
        eeg_all = chunk[bci.eeg, :]
        if eeg_all.shape[1] == 0:
            return
        if max(train_idx) >= eeg_all.shape[0]:
            print(f"(got {eeg_all.shape[0]} channels, need index {max(train_idx)})")
            return

        eeg = eeg_all[train_idx, :].astype(np.float64, copy=False) / 1e6

        with buffer_lock:
            buffer = np.hstack([buffer, eeg])
            if buffer.shape[1] > buffer_n:
                buffer = buffer[:, -buffer_n:]
            if buffer.shape[1] < buffer_n:
                return  # priming the filter context
            buf = buffer.copy()

        # Filter the full buffer (matches raw.filter on a continuous signal),
        # then take only the last DELTA_T worth of samples for prediction.
        filtered = bandpass(buf, sfreq)
        window = filtered[:, -window_n:][np.newaxis, :, :]
        gui.push_window(filtered[:, -window_n:])

        probs = clf.predict_proba(window)[0]
        idx = int(np.argmax(probs))
        pred = clf.classes_[idx]
        conf = probs[idx]
        breakdown = ", ".join(f"{c}={p*100:.1f}%" for c, p in zip(clf.classes_, probs))

        decision, consensus = smoother.add(
            prediction=pred,
            confidence=conf,
            probs=dict(zip(clf.classes_, probs)),
        )
        tag = decision if consensus else "—"
        print(f"raw: {pred}  conf: {conf*100:.1f}%  [{breakdown}]  → smoothed: {tag}")

    bci.callback = on_chunk
    bci.start()

    print("\nLive classification — close the GUI window or Ctrl+C to stop.\n")
    try:
        gui.mainloop()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        bci.stop()
        bci.close()
        ws.stop()

def menu():
    print("=" * 35)
    print(" EEG Motor Imagery Classifier")
    print("=" * 35)
    print(" 1) Offline  (pick training/testing files)")
    print(" 2) Online   (train on all files, classify live)")
    print(" q) Quit")
    print("=" * 35)
    return input("> ").strip().lower()

def main():
    while True:
        try:
            choice = menu()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice in ('1', 'offline'):
            run_offline()
            return
        if choice in ('2', 'online'):
            run_online()
            return
        if choice in ('q', 'quit', 'exit'):
            return
        print("invalid choice\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
