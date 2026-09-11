import os
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Import your config
from config import DATA_DIR, EEG_CHANNELS_TARGETS, TARGET_MAPPINGS

def bandpower(data, sfreq, band):
    """Compute average band power for each epoch."""
    b, a = butter(4, [band[0]/(sfreq/2), band[1]/(sfreq/2)], btype='band')
    filtered = filtfilt(b, a, data, axis=-1)
    return np.mean(filtered**2, axis=-1)  # (n_epochs, n_chans)

def load_epochs(file_path, tmin=0.5, tmax=2.5):
    raw = mne.io.read_raw_fif(file_path, preload=True)
    raw.pick(EEG_CHANNELS_TARGETS)
    raw.set_montage('standard_1020', on_missing='ignore')
    raw.annotations.description = np.array([str(d).strip().lower() for d in raw.annotations.description])

    # Build event_id
    event_id = {}
    for desc in set(raw.annotations.description):
        d = str(desc).strip().lower()
        for prefix, code in TARGET_MAPPINGS.items():
            p = prefix.lower()
            if d == p or d.startswith(p + '/'):
                event_id[str(desc)] = code
                break

    events, event_id_used = mne.events_from_annotations(raw, event_id=event_id)
    epochs = mne.Epochs(raw, events, event_id=event_id_used,
                        tmin=tmin, tmax=tmax, baseline=None, preload=True)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y, raw.info['sfreq']

def main():
    files = glob.glob(os.path.join(DATA_DIR, '*_mi_raw.fif'))
    if not files:
        print("No files found")
        return

    # Pick a file with balanced classes
    file = files[0]  # adjust as needed
    print(f"Analyzing {file}")
    X, y, sfreq = load_epochs(file)

    # Separate classes
    left = X[y == 1]
    right = X[y == 2]
    print(f"Left epochs: {len(left)}, Right epochs: {len(right)}")

    # Compute bandpower
    mu_left = bandpower(left, sfreq, (8, 12))
    mu_right = bandpower(right, sfreq, (8, 12))
    beta_left = bandpower(left, sfreq, (13, 30))
    beta_right = bandpower(right, sfreq, (13, 30))

    # Plot average bandpower per channel
    ch_names = EEG_CHANNELS_TARGETS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(np.arange(len(ch_names)), mu_left.mean(axis=0), alpha=0.7, label='Left')
    axes[0].bar(np.arange(len(ch_names)), mu_right.mean(axis=0), alpha=0.7, label='Right')
    axes[0].set_xticks(np.arange(len(ch_names)))
    axes[0].set_xticklabels(ch_names, rotation=45)
    axes[0].set_title('Mu band (8-12 Hz)')
    axes[0].legend()

    axes[1].bar(np.arange(len(ch_names)), beta_left.mean(axis=0), alpha=0.7, label='Left')
    axes[1].bar(np.arange(len(ch_names)), beta_right.mean(axis=0), alpha=0.7, label='Right')
    axes[1].set_xticks(np.arange(len(ch_names)))
    axes[1].set_xticklabels(ch_names, rotation=45)
    axes[1].set_title('Beta band (13-30 Hz)')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # Compute r² map (time-frequency)
    times = np.linspace(0.5, 2.5, X.shape[-1])  # adjust if tmin/tmax differ
    from mne.decoding import compute_patterns
    from sklearn.feature_selection import r_regression

    # For each channel and time point, compute r²
    r2_map = np.zeros((X.shape[1], X.shape[-1]))
    for ch in range(X.shape[1]):
        for t in range(X.shape[-1]):
            r2_map[ch, t] = r_regression(
                X[:, ch, t].reshape(-1, 1), y
            )[0] ** 2  # r²

    plt.figure(figsize=(10, 6))
    plt.imshow(r2_map, aspect='auto', extent=[times[0], times[-1], 0, X.shape[1]-1],
               origin='lower', cmap='RdBu_r', vmin=0, vmax=0.1)
    plt.colorbar(label='r²')
    plt.yticks(np.arange(X.shape[1]), ch_names)
    plt.xlabel('Time (s)')
    plt.title('r² map (left vs right)')
    plt.tight_layout()
    plt.show()

    # Print max r²
    print(f"Maximum r²: {r2_map.max():.4f} at channel {ch_names[np.unravel_index(r2_map.argmax(), r2_map.shape)[0]]}, time {times[np.unravel_index(r2_map.argmax(), r2_map.shape)[1]]:.2f}s")

if __name__ == "__main__":
    main()