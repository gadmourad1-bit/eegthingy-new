# Standalone SBC decoder (EA + FB-CSP, lean)

A single Linux/arm64 executable that runs the online MI decoder on an SBC (Orange
Pi Zero 2W, DietPi, …) **with no Python, uv, mne, scikit-learn, or matplotlib**.
The online inference is re-expressed in pure numpy/scipy and verified **bit-for-bit
identical** to `EAFilterBankCSP` (max proba diff ≈ 2e-16). Baked-in deps: numpy,
scipy, brainflow, websockets.

```
sbc/
  export_model.py   # (Mac) train EA+FB-CSP on data/*.fif  ->  model.npz  (~12 KB)
  lean_online.py    # the SBC runtime: numpy/scipy/brainflow/websockets only
  ws.py smoother.py # self-contained websocket + M-of-N/dwell smoother
  Dockerfile        # builds brainflow (aarch64) + PyInstaller -> one binary
  build.sh          # one command on the Mac to produce the binary
```

## Two inputs, as designed
- **Training data** → folded into `model.npz` on the Mac at export time (so the
  binary carries no training stack). Retrain a subject = re-export + recopy the
  12 KB file; the binary never changes.
- **Live calibration** → 45 s on the SBC computes the Euclidean-Alignment
  reference from your own signal (the online mechanic), exactly like `set_reference`.

## Workflow

### 1. Export the model (on the Mac)
```bash
PYTHONPATH=.:classifier python sbc/export_model.py                 # all data/*.fif
PYTHONPATH=.:classifier python sbc/export_model.py data/exp4_subject1_*.fif   # subset
# -> sbc/model.npz
```

### 2. Build the Linux/arm64 binary (on the Mac, needs Docker/OrbStack/colima)
```bash
bash sbc/build.sh
# -> sbc/lean-online-linux-arm64   (native arm64 build; brainflow compiled from source)
```
The `linux/arm64` container runs natively on Apple Silicon (no emulation). First
build is slow (brainflow compiles); later builds are cached.

### 3. Deploy to the SBC
```bash
scp sbc/lean-online-linux-arm64 sbc/model.npz root@<pi-ip>:/root/eeg/
# on the SBC:
chmod +x lean-online-linux-arm64
./lean-online-linux-arm64                 # model.npz must sit beside the binary
```
On a terminal it prompts for the EEG device (**1 = USB Cyton+Daisy, 2 = Synthetic**);
pass `--device usb` / `--device synthetic` to skip the prompt (a service with no
terminal defaults to `usb`). Decisions broadcast on `ws://<pi-ip>:8765`; the robot
acts on `smoothed.final` (1 = left, 2 = right). Other options: `--cal-seconds N`,
`--ws-port N`, `--serial-port /dev/ttyUSB0`.

## The OpenBCI dongle on the SBC
The decoder talks to the Cyton via the FTDI USB dongle — the kernel's `ftdi_sio`
handles it (no driver install). Add yourself to `dialout`, and set the FTDI
latency timer to 1 ms for smooth streaming:
```bash
usermod -aG dialout $USER      # re-login after
echo 'ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"' \
  > /etc/udev/rules.d/99-openbci-ftdi.rules && udevadm control --reload && udevadm trigger
```

## Retrain a new subject
Re-run step 1 with that subject's `.fif` files, copy the new `model.npz` to the
SBC. **No rebuild** — the binary is subject-agnostic; the model + live calibration
carry the subject specifics.
