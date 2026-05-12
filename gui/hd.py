import numpy as np

from tkinter import *
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.globals import vernier

class HDTab(Frame):
    def __init__(self, notebook):
        super().__init__(notebook)

        self.name = "HD"
        self.anim = None
        self.values = np.array(np.zeros(100))

        self._create_widgets()

        self.bind("<Map>", self._start_animation)
        self.bind("<Unmap>", self._stop_animation)

    def _start_animation(self, event):
        """Start the animation when the tab becomes visible."""
        if self.anim is None:
            self.anim = animation.FuncAnimation(self.figure, self._update_canvas, blit=True, cache_frame_data=False,
                                                interval=1000 / 60)
        else:
            self.anim.event_source.start()

    def _stop_animation(self, event):
        """Stop the animation when the tab is hidden."""
        if self.anim is not None:
            self.anim.event_source.stop()

    def _update_canvas(self, frame):
        """Update function for FuncAnimation to dynamically update the canvas."""
        self.values = np.roll(self.values, -1)
        self.values[-1] = vernier.get_data()
        self.line.set_ydata(self.values)
        return self.line,

    def _create_widgets(self):
        """Create the widgets for the tab."""
        self.figure = Figure(constrained_layout=True)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.line = self.ax.plot([], [])[0]

        self.ax.set_title('HD data')
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('Force (N)')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-10, 500)

        self.line.set_data(np.arange(100), self.values)

        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack()
