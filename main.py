import os
import ctypes
import shlex
import subprocess
import sys

from platform import system


def run_collector():
    # Get joystick data in headless mode
    os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    if system() == "Windows":
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # this is to fix an issue with matplotlib scaling on windows

    from gui.window import Window
    Window().start()


def run_classifier():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier", "run.py")
    subprocess.run([sys.executable, script])


def run_ssvep_collector():
    """Launch the independent four-command SSVEP recording package."""
    root = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, "-m", "ssvep"], cwd=root)


def run_mirepnet_script(name, *args):
    root = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(root, name)
    subprocess.run([sys.executable, script, *args], cwd=root)


def run_mirepnet_workbench():
    print("\nMIRepNet foundation-model workbench")
    print(" 1) Offline subject test (zero-shot / unlabeled / calibrated + Excel)")
    print(" 2) Online live decoding (GUI)")
    print(" 3) Online live decoding (headless)")
    print(" 4) Fine-tune selected subjects or individual files")
    print(" 5) Test recorded subject EEG at each corner of a fixed maze")
    print(" 6) Live OpenBCI headset control on a fixed maze")
    print(" b) Back")
    while True:
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("1", "offline", "test"):
            run_mirepnet_script(os.path.join("scripts", "test_mirepnet_patients.py"))
            return
        if choice in ("2", "online", "gui"):
            run_mirepnet_script(os.path.join("classifier", "run.py"), "--mirepnet-online")
            return
        if choice in ("3", "headless", "online-headless"):
            run_mirepnet_script(os.path.join("classifier", "run.py"), "--mirepnet-headless")
            return
        if choice in ("4", "fine-tune", "finetune", "train"):
            run_mirepnet_script(os.path.join("scripts", "finetune_mirepnet_selected.py"))
            return
        if choice in ("5", "maze", "replay", "maze-test"):
            run_mirepnet_script(os.path.join("scripts", "mirepnet_maze_test.py"))
            return
        if choice in ("6", "live-maze", "headset-maze", "openbci-maze"):
            run_mirepnet_script(os.path.join("classifier", "run.py"), "--mirepnet-live-maze")
            return
        if choice in ("b", "back", "q", "quit"):
            return
        print("enter 1-6 or b")


def run_simulation():
    sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation")
    src_dir = os.path.join(sim_dir, "src")
    try:
        subject = input("subject id> ").strip()
        test = input("test id> ").strip()
        
        print("simulation args (blank prints the sim's help):")
        print("  --ws-url URL               decision server (default ws://127.0.0.1:8765)")
        print("  --no-ws                    disable the websocket client")
        print("  --no-manual-keys           disable arrow-key decisions")
        print("  --view {third,first}   camera (default third)")
        
        raw = input("args> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    id_args = []
    if subject:
        id_args += ["--subject", subject]
    if test:
        id_args += ["--test", test]

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_dir, env.get("PYTHONPATH")) if p)
    subprocess.run([sys.executable, "-m", "tiago_maze", *shlex.split(raw), *id_args],
                   cwd=sim_dir, env=env)


def menu():
    print("=" * 35)
    print(" 1) Data Collector")
    print(" 2) Classifier")
    print(" 3) Robot Simulation")
    print(" 4) MIRepNet Workbench (offline / online / fine-tune / maze)")
    print(" 5) SSVEP Data Collector (forward / right / backward / left)")
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
        if choice in ("1", "collector", "data"):
            run_collector()
            return
        if choice in ("2", "classifier"):
            run_classifier()
            return
        if choice in ("3", "simulation", "sim"):
            run_simulation()
            return
        if choice in ("4", "mirepnet", "foundation"):
            run_mirepnet_workbench()
            continue
        if choice in ("5", "ssvep"):
            run_ssvep_collector()
            return
        if choice in ("q", "quit", "exit"):
            return
        print("invalid choice\n")


if __name__ == "__main__":
    main()
