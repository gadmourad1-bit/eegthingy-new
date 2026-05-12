import numpy as np

from tkinter import *
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.globals import openbci

class EEGTab(Frame):
    def __init__(self, notebook):
        super().__init__(notebook)

        self.name = "EEG"
        self.anim = None
        self.values = np.zeros(16)

        self._create_widgets()

        self.bind("<Map>", self._start_animation)
        self.bind("<Unmap>", self._stop_animation)

    def _start_animation(self, event):
        """Start the animation when the tab becomes visible."""
        if self.anim is None:
            self.anim = animation.FuncAnimation(self.figure, self._update_canvas, blit=True, cache_frame_data=False, interval=1000/60)
        else:
            self.anim.event_source.start()

    def _stop_animation(self, event):
        """Stop the animation when the tab is hidden."""
        if self.anim is not None:
            self.anim.event_source.stop()

    def _update_canvas(self, frame):
        """Update function for FuncAnimation to dynamically update the canvas."""
        sample = openbci.get_data()

        for bar, value in zip(self.bars, sample["eeg"]):
            if bar.get_height() != value:
                bar.set_height(value)

        acc = sample["acc"]
        self.meta.set_text(
            f'M={sample["marker"]:.2f}\n'
            f'T={sample["timestamp"]:.3f}s\n'
            f'X={acc[0]:+.2f}g\n'
            f'Y={acc[1]:+.2f}g\n'
            f'Z={acc[2]:+.2f}g'
        )

        return [*self.bars, self.meta]
        
    def _bci_callback(self, sample):
        """This method is called every time a new sample is ready from the OpenBCI device, it sets the latest sample."""
        self.values = sample
        print(sample)

    def _create_widgets(self):
        """Create the widgets for the tab."""
        self.figure = Figure(constrained_layout=True)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.bars = self.ax.bar(np.arange(16) + 1, self.values, edgecolor='black')

        self.ax.set_title('EEG View')
        self.ax.set_xlabel('Channel')
        self.ax.set_ylabel('Activity (µVrms)')
        self.ax.set_ylim(0, 200)
        self.ax.set_xlim(0.5, 16.5)
        self.ax.set_xticks(np.arange(16) + 1)

        self.meta = self.ax.text(
            0.02, 0.98, 'M=-1\nT=-1\nX=0.00\nY=0.00\nZ=0.00',
            transform=self.ax.transAxes,
            ha='left', va='top',
            fontsize=9, family='monospace',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=4),
        )
        
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack()