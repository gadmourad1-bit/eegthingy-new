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
        if choice in ("q", "quit", "exit"):
            return
        print("invalid choice\n")


if __name__ == "__main__":
    main()
