from tkinter import *
from tkinter.ttk import *

from gui.eeg import EEGTab
from gui.hd import HDTab
from gui.emg import EMGTab
from gui.joystick import JoystickTab
from gui.channels import ChannelsTab
from gui.exp4.root import EXP4Tab
from utils.globals import openbci, vernier, noraxon, joystick


class Window:
    def __init__(self):
        self.master = Tk()

        self.master.title("Data collection UI")
        self.master.configure(background="white")
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.master.resizable(False, False)

        width, height = 800, 600
        self.master.update_idletasks()
        x = (self.master.winfo_screenwidth() - width) // 2
        y = (self.master.winfo_screenheight() - height) // 2
        self.master.geometry(f"{width}x{height}+{x}+{y}")

        self.tabcontrol = Notebook(self.master)
        self.tabcontrol.pack(expand=1, fill='both')

        self.tabs = [EEGTab(self.tabcontrol), HDTab(self.tabcontrol), EMGTab(self.tabcontrol),
                     JoystickTab(self.tabcontrol), ChannelsTab(self.tabcontrol), EXP4Tab(self.tabcontrol)]

        for tab in self.tabs:
            self.tabcontrol.add(tab, text=tab.name)

    def _on_closing(self):
        """Stop all devices and close the application."""
        try:
            self.master.quit()
            self.master.destroy()

            openbci.close()
            vernier.close()
            noraxon.close()
            joystick.close()
        except:
            exit(1)

    def _pump_pyglet(self):
        """Pump pyglet's event loop on the tkinter main thread. Pyglet's
        joystick HID dispatch needs a running event loop on the thread that
        opened the device; running pyglet.app.run() would block tkinter, so
        we step it cooperatively from after()."""
        joystick.tick()
        self.master.after(8, self._pump_pyglet)

    def start(self):
        """Run the main gui event loop."""
        self.master.lift()
        self.master.attributes("-topmost", True)
        self.master.after(100, lambda: self.master.attributes("-topmost", False))
        self.master.focus_force()
        self._pump_pyglet()
        self.master.mainloop()
