from tkinter import *
from matplotlib import animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.globals import joystick

class JoystickTab(Frame):
    def __init__(self, notebook):
        super().__init__(notebook)

        self.name = "Joystick"
        self.anim = None
        self.x = 0
        self.y = 0
        
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
        sample = joystick.get_data()
        self.scatter.set_offsets((sample[0], sample[1]))
        return [self.scatter]

    def _joystick_callback(self, data):
        """This method is called every time a new sample is ready from the Joystick device, it sets the latest sample."""
        self.x = data[0]
        self.y = data[1]

    def _create_widgets(self):
        """Create the widgets for the tab."""
        self.figure = Figure(constrained_layout=True)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.scatter = self.ax.scatter(0, 0, c='blue', alpha=0.6, edgecolors='black')

        self.ax.set_title('Joystick visualizer')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xticks([-1, 0, 1])
        self.ax.set_yticks([-1, 0, 1])
        self.ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        self.ax.axvline(0, color='gray', linestyle='--', linewidth=1)

        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack()
