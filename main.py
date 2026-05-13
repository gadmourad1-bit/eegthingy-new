import os
import ctypes
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


def menu():
    print("=" * 35)
    print(" 1) Data Collector")
    print(" 2) Classifier")
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
        if choice in ("q", "quit", "exit"):
            return
        print("invalid choice\n")


if __name__ == "__main__":
    main()
