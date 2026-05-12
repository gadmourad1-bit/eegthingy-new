import numpy as np

from tkinter import *
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.globals import noraxon

class EMGTab(Frame):
    def __init__(self, notebook):
        super().__init__(notebook, background="white")
        self.name = "EMG"
        self.anim = None
        self.values = np.zeros((4, 100))

        self._create_widgets()

        self.bind("<Map>", self._start_animation)
        self.bind("<Unmap>", self._stop_animation)

    def _start_animation(self, event):
        """Start the animation when the tab becomes visible."""
        if self.anim is None:
            self.anim = animation.FuncAnimation(self.figure, self._update_canvas, blit=True,
                                                cache_frame_data=False, interval=1000/60)
        else:
            self.anim.event_source.start()

    def _stop_animation(self, event):
        """Stop the animation when the tab is hidden."""
        if self.anim is not None:
            self.anim.event_source.stop()

    def _update_canvas(self, frame):
        """Update the canvas with the new data."""
        data = noraxon.get_data()

        for index, values in enumerate(self.values):
            new_values = np.roll(values, -1)
            new_values[-1] = data[index]
            self.values[index] = new_values

        for i, line in enumerate(self.lines):
            line.set_ydata(self.values[i])
        return self.lines

    def _create_widgets(self):
        """Create the widgets for the tab."""
        self.figure = Figure(constrained_layout=True)
        self.ax = self.figure.add_subplot(1, 1, 1)
        
        self.lines = []

        for i in range(4):
            line, = self.ax.plot(np.arange(100), np.zeros(100), label=f'EMG channel {i + 1}')
            self.lines.append(line)

        self.ax.set_title('EMG data')
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('Activity (µV)')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-50, 100000)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack()
