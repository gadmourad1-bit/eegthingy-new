import os
import ctypes

from platform import system
from gui.window import Window

# Get joystick data in headless mode
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"

if system() == "Windows":
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # this is to fix an issue with matplotlib scaling on windows

Window().start()