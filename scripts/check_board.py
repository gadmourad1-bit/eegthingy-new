"""OpenBCI Cyton+Daisy hardware diagnostic.

Walks the whole chain and prints a verdict per stage, so a failure points at the
exact component: FTDI dongle -> RF link -> Cyton board -> Daisy module -> ADS1299
front-end -> per-channel signal quality.

Run with the same Python environment the live pipeline uses (needs brainflow,
pyserial, numpy). The board should be ON (switch to PC) with the dongle plugged in.

    python scripts/check_board.py            # full check (~30 s)
    python scripts/check_board.py --scan     # just enumerate serial ports
    python scripts/check_board.py --synthetic  # software-path sanity check only
"""

import argparse
import sys
import time

import numpy as np
import serial as pyserial
from serial import Serial
from serial.tools import list_ports


def stage(title):
    print(f"\n=== {title} " + "=" * max(0, 58 - len(title)))


def scan_ports():
    """Stage 1: does the FTDI dongle even enumerate?"""
    stage("1. Serial ports (is the dongle enumerating?)")
    ports = list(list_ports.comports())
    if not ports:
        print("  ✗ NO serial ports found at all.")
        print("    -> Dongle not enumerating: try another USB port/cable, check that the")
        print("       dongle's blue LED is on. If it never appears, the DONGLE (not the")
        print("       board) is the failed part. FTDI driver issues also look like this.")
        return []
    for p in ports:
        marker = "  <- likely OpenBCI (FTDI)" if ("usbserial" in p.device.lower()
                 or (p.manufacturer or "").lower().startswith("ftdi")) else ""
        print(f"  {p.device}   {p.manufacturer or '?'} {p.description or ''}{marker}")
    return ports


def probe_reset_string(ports):
    """Stage 2: the definitive board self-report. Sending 'v' makes the Cyton print
    its reset string, which names every chip it can talk to."""
    stage("2. Board reset string ('v' probe over the RF link)")
    for p in ports:
        try:
            s = Serial(port=p.device, baudrate=115200, timeout=3)
            time.sleep(0.5)
            s.reset_input_buffer()
            s.write(b"v")
            time.sleep(2.5)
            raw = s.read(s.in_waiting or 1).decode("utf-8", errors="replace")
            s.close()
        except (OSError, pyserial.SerialException) as e:
            print(f"  {p.device}: cannot open ({e})")
            continue
        if not raw.strip():
            continue
        print(f"  {p.device} replied:")
        for line in raw.strip().splitlines():
            print(f"    | {line}")
        text = raw.lower()
        ok = True
        if "openbci" not in text:
            print("  ✗ No OpenBCI banner — this port is some other device.")
            continue
        if "failure" in text or "not attached" in text or "timeout" in text:
            print("  ✗ Dongle is fine but the RF LINK to the board failed.")
            print("    -> Board off/dead battery, out of pairing, or RFduino radio fault.")
            ok = False
        if "ads1299" not in text:
            print("  ✗ Cyton's ADS1299 (the EEG front-end chip) did NOT identify itself —")
            print("    that is real board damage on the Cyton.")
            ok = False
        if "daisy" not in text:
            print("  ✗ NO Daisy in the reset string — Cyton alive but the DAISY MODULE is")
            print("    not detected. Reseat the Daisy on its header pins (most common fix),")
            print("    check for bent pins. If it never returns, the Daisy is the failure.")
            ok = False
        if ok:
            print("  ✓ Dongle, RF link, Cyton ADS1299, and Daisy all identify correctly.")
        return p.device, ok
    print("  ✗ No port produced an OpenBCI reply. If stage 1 showed an FTDI port, the")
    print("    RF link/board side is down (battery first, then pairing, then board).")
    return None, False


def stream_check(synthetic=False, seconds=15):
    """Stage 3: stream and inspect what the amplifier actually measures."""
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

    stage(f"3. Signal check ({'SYNTHETIC board' if synthetic else 'real board'}, {seconds}s stream)")
    params = BrainFlowInputParams()
    params.timeout = 30
    if synthetic:
        board_id = BoardIds.SYNTHETIC_BOARD.value
    else:
        board_id = BoardIds.CYTON_DAISY_BOARD.value
        ports = [p.device for p in list_ports.comports()
                 if "usbserial" in p.device.lower() or (p.manufacturer or "").lower().startswith("ftdi")]
        if not ports:
            print("  ✗ no FTDI port to stream from"); return False
        params.serial_port = ports[0]
    BoardShim.disable_board_logger()
    board = BoardShim(board_id, params)
    try:
        board.prepare_session()
        board.start_stream()
    except Exception as e:
        print(f"  ✗ BrainFlow could not start a stream: {e}")
        print("    -> If stage 2 passed, this is a driver/library issue, not hardware.")
        return False
    print(f"  streaming {seconds}s ...")
    time.sleep(seconds)
    data = board.get_board_data()
    board.stop_stream(); board.release_session()

    sfreq = BoardShim.get_sampling_rate(board_id)
    eeg_ch = BoardShim.get_eeg_channels(board_id)
    n = data.shape[1]
    print(f"  received {n} samples  (expected ~{int(seconds * sfreq)} at {sfreq} Hz)")
    if n < seconds * sfreq * 0.5:
        print("  ✗ Less than half the expected samples — severe packet loss.")
        print("    -> Weak batteries are the #1 cause; also RF interference/distance.")

    # packet-loss estimate from the sample counter (channel 0 on Cyton)
    counter = data[0]
    if not synthetic and len(counter) > 10:
        d = np.diff(counter)
        d = d[(d != 0)]
        wraps = (d < 0).sum()
        skips = int(np.sum(d[d > 1] - 1)) if (d > 1).any() else 0
        print(f"  packet counter: {skips} skipped packets, {wraps} wraps "
              f"({'OK' if skips < n * 0.02 else '✗ HIGH LOSS — batteries/RF'})")

    print(f"\n  {'ch':>4s} {'mean µV':>12s} {'std µV':>10s}  verdict")
    bad = []
    for i, ch in enumerate(eeg_ch, 1):
        x = data[ch]
        mean, std = float(np.mean(x)), float(np.std(x))
        if std < 0.5:
            v = "✗ FLAT (dead/unplugged input or dead ADC channel)"
        elif abs(mean) > 150_000 or std > 60_000:
            v = "✗ RAILED (saturated: bad reference/ground or damaged input)"
        elif std > 15_000:
            v = "~ very noisy (electrode/wiring, or nothing connected)"
        else:
            v = "ok"
        if v.startswith("✗"):
            bad.append(i)
        print(f"  {i:>4d} {mean:>12.1f} {std:>10.1f}  {v}")

    cyton_bad = [b for b in bad if b <= 8]
    daisy_bad = [b for b in bad if b > 8]
    print()
    if not synthetic:
        if daisy_bad and len(daisy_bad) == 8 and not cyton_bad:
            print("  ✗ ALL Daisy channels (9-16) bad while Cyton (1-8) fine:")
            print("    -> Daisy module/header connection problem. Reseat the Daisy.")
        elif bad:
            print(f"  ✗ bad channels: {bad} — if the SAME channels are bad with pins")
            print("    shorted to SRB/ground (see below), those inputs are damaged;")
            print("    if they recover, it's electrodes/wiring, not the board.")
        else:
            print("  ✓ all 16 channels alive with plausible noise floors.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seconds", type=int, default=15)
    args = ap.parse_args()

    if args.synthetic:
        stream_check(synthetic=True, seconds=min(args.seconds, 5)); return
    ports = scan_ports()
    if args.scan or not ports:
        return
    port, link_ok = probe_reset_string(ports)
    if link_ok:
        stream_check(synthetic=False, seconds=args.seconds)
    stage("Manual checks if anything failed above")
    print("""  1. BATTERIES first — fresh/charged. Brownout = packet loss, random resets,
     'worked yesterday, dead today'. This is the most common failure by far.
  2. Board power switch on PC (not OFF/BLE); dongle switch (older ones) on GPIO6.
  3. Blue LED on board when powered? On dongle when plugged?
  4. Reseat the DAISY on its snap-in headers (pins work loose; #1 Daisy fault).
  5. Distance/interference: board within ~1 m of dongle, away from chargers.
  6. Independent cross-check: OpenBCI GUI -> if it streams clean signals there,
     the hardware is FINE and the problem is software/config on our side.
  7. Electrode sanity: short an input pin to SRB with a wire; that channel should
     go near-flat. A channel that stays railed/flat regardless = damaged input.""")


if __name__ == "__main__":
    main()
