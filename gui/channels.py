from tkinter import *
from tkinter.ttk import *
from utils.filter import ChannelFilter
from utils.globals import openbci, vernier, noraxon, joystick


class ChannelsTab(Frame):
    def __init__(self, master):
        super().__init__()
        self.master = master
        self.filter = ChannelFilter().load()
        self.name = "Channels"

        openbci.set_filter(self.filter)
        vernier.set_filter(self.filter)
        noraxon.set_filter(self.filter)
        joystick.set_filter(self.filter)

        self._create_widgets()

    def _close(self):
        """Clean up and close the window."""
        self.destroy()
        self.master.channels_window = None

    def _save(self):
        """Save the filter settings."""
        for i in range(16):
            self.filter.eeg_enabled[i] = self.eeg_enabled_array[i].get()
            self.filter.eeg_low[i] = float(self.eeg_low_array[i].get())
            self.filter.eeg_high[i] = float(self.eeg_high_array[i].get())
            self.filter.eeg_offset[i] = float(self.eeg_offset_array[i].get())

        for i in range(4):
            self.filter.emg_enabled[i] = self.emg_enabled_array[i].get()
            self.filter.emg_low[i] = float(self.emg_low_array[i].get())
            self.filter.emg_high[i] = float(self.emg_high_array[i].get())
            self.filter.emg_offset[i] = float(self.emg_offset_array[i].get())

        self.filter.hd_enabled = self.hd_enabled.get()
        self.filter.hd_low = float(self.hd_low.get())
        self.filter.hd_high = float(self.hd_high.get())
        self.filter.hd_offset = float(self.hd_offset.get())

        self.filter.joystick_x_enabled = self.js_x_enabled.get()
        self.filter.joystick_x_low = float(self.js_x_low.get())
        self.filter.joystick_x_high = float(self.js_x_high.get())
        self.filter.joystick_x_offset = float(self.js_x_offset.get())

        self.filter.joystick_y_enabled = self.js_y_enabled.get()
        self.filter.joystick_y_low = float(self.js_y_low.get())
        self.filter.joystick_y_high = float(self.js_y_high.get())
        self.filter.joystick_y_offset = float(self.js_y_offset.get())

        openbci.set_filter(self.filter)
        vernier.set_filter(self.filter)
        noraxon.set_filter(self.filter)
        joystick.set_filter(self.filter)

        self.filter.save()

    def _create_widgets(self):
        """Create the widgets for the check channels window."""
        eeg_frame = LabelFrame(self, text="EEG Parameters")
        eeg_frame.grid(row=0, column=0, sticky="s")

        eeg_number_label = Label(eeg_frame, text="Channel #")
        eeg_number_label.grid(row=1, column=0)

        eeg_enabled_label = Label(eeg_frame, text="Enabled")
        eeg_enabled_label.grid(row=1, column=1)

        eeg_low_label = Label(eeg_frame, text="Low")
        eeg_low_label.grid(row=1, column=2)

        eeg_high_label = Label(eeg_frame, text="High")
        eeg_high_label.grid(row=1, column=3)

        eeg_offset_label = Label(eeg_frame, text="Offset")
        eeg_offset_label.grid(row=1, column=4)

        self.eeg_enabled_array = [None] * 16
        self.eeg_low_array = [None] * 16
        self.eeg_high_array = [None] * 16
        self.eeg_offset_array = [None] * 16

        for i in range(16):
            eeg_number_i_label = Label(eeg_frame, text=i + 1)
            eeg_number_i_label.grid(row=i + 2, column=0)

            self.eeg_enabled_array[i] = BooleanVar()
            self.eeg_enabled_array[i].set(self.filter.eeg_enabled[i])
            eeg_enabled_checkbutton = Checkbutton(eeg_frame, variable=self.eeg_enabled_array[i])
            eeg_enabled_checkbutton.grid(row=i + 2, column=1)

            self.eeg_low_array[i] = StringVar()
            self.eeg_low_array[i].set(self.filter.eeg_low[i])
            eeg_low_entry = Entry(eeg_frame, width=5, textvariable=self.eeg_low_array[i])
            eeg_low_entry.grid(row=i + 2, column=2)

            self.eeg_high_array[i] = StringVar()
            self.eeg_high_array[i].set(self.filter.eeg_high[i])
            eeg_high_entry = Entry(eeg_frame, width=5, textvariable=self.eeg_high_array[i])
            eeg_high_entry.grid(row=i + 2, column=3)

            self.eeg_offset_array[i] = StringVar()
            self.eeg_offset_array[i].set(self.filter.eeg_offset[i])
            eeg_offset_entry = Entry(eeg_frame, width=5, textvariable=self.eeg_offset_array[i])
            eeg_offset_entry.grid(row=i + 2, column=4)

        right_frame = Frame(self)
        right_frame.grid(row=0, column=1, sticky="nsew")

        hd_frame = LabelFrame(right_frame, text="HD Parameters")
        hd_frame.grid(row=0, column=0, sticky="nsew")

        hd_number_label = Label(hd_frame, text="Channel #")
        hd_number_label.grid(row=1, column=0)

        hd_enabled_label = Label(hd_frame, text="Enabled")
        hd_enabled_label.grid(row=1, column=1)

        hd_low_label = Label(hd_frame, text="Low")
        hd_low_label.grid(row=1, column=2)

        hd_high_label = Label(hd_frame, text="High")
        hd_high_label.grid(row=1, column=3)

        hd_offset_label = Label(hd_frame, text="Offset")
        hd_offset_label.grid(row=1, column=4)

        hd_number_i_label = Label(hd_frame, text=1)
        hd_number_i_label.grid(row=2, column=0)

        self.hd_enabled = BooleanVar()
        self.hd_enabled.set(self.filter.hd_enabled)
        hd_enabled_checkbutton = Checkbutton(hd_frame, variable=self.hd_enabled)
        hd_enabled_checkbutton.grid(row=2, column=1)

        self.hd_low = StringVar()
        self.hd_low.set(self.filter.hd_low)
        hd_low_entry = Entry(hd_frame, width=5, textvariable=self.hd_low)
        hd_low_entry.grid(row=2, column=2)

        self.hd_high = StringVar()
        self.hd_high.set(self.filter.hd_high)
        hd_high_entry = Entry(hd_frame, width=5, textvariable=self.hd_high)
        hd_high_entry.grid(row=2, column=3)

        self.hd_offset = StringVar()
        self.hd_offset.set(self.filter.hd_offset)
        hd_offset_entry = Entry(hd_frame, width=5, textvariable=self.hd_offset)
        hd_offset_entry.grid(row=2, column=4)

        emg_frame = LabelFrame(right_frame, text="EMG Parameters")
        emg_frame.grid(row=1, column=0, sticky="nsew")

        emg_number_label = Label(emg_frame, text="Channel #")
        emg_number_label.grid(row=1, column=0)

        emg_enabled_label = Label(emg_frame, text="Enabled")
        emg_enabled_label.grid(row=1, column=1)

        emg_low_label = Label(emg_frame, text="Low")
        emg_low_label.grid(row=1, column=2)

        emg_high_label = Label(emg_frame, text="High")
        emg_high_label.grid(row=1, column=3)

        emg_offset_label = Label(emg_frame, text="Offset")
        emg_offset_label.grid(row=1, column=4)

        self.emg_enabled_array = [None] * 4
        self.emg_low_array = [None] * 4
        self.emg_high_array = [None] * 4
        self.emg_offset_array = [None] * 4

        for i in range(4):
            emg_number_i_label = Label(emg_frame, text=i + 1)
            emg_number_i_label.grid(row=i + 2, column=0)

            self.emg_enabled_array[i] = BooleanVar()
            self.emg_enabled_array[i].set(self.filter.emg_enabled[i])
            emg_enabled_checkbutton = Checkbutton(emg_frame, variable=self.emg_enabled_array[i])
            emg_enabled_checkbutton.grid(row=i + 2, column=1)

            self.emg_low_array[i] = StringVar()
            self.emg_low_array[i].set(self.filter.emg_low[i])
            emg_low_entry = Entry(emg_frame, width=5, textvariable=self.emg_low_array[i])
            emg_low_entry.grid(row=i + 2, column=2)

            self.emg_high_array[i] = StringVar()
            self.emg_high_array[i].set(self.filter.emg_high[i])
            emg_high_entry = Entry(emg_frame, width=5, textvariable=self.emg_high_array[i])
            emg_high_entry.grid(row=i + 2, column=3)

            self.emg_offset_array[i] = StringVar()
            self.emg_offset_array[i].set(self.filter.emg_offset[i])
            emg_offset_entry = Entry(emg_frame, width=5, textvariable=self.emg_offset_array[i])
            emg_offset_entry.grid(row=i + 2, column=4)

        js_frame = LabelFrame(right_frame, text="Joystick Parameters")
        js_frame.grid(row=2, column=0, sticky="nsew")

        js_number_label = Label(js_frame, text="Channel #")
        js_number_label.grid(row=1, column=0)

        js_enabled_label = Label(js_frame, text="Enabled")
        js_enabled_label.grid(row=1, column=1)

        js_low_label = Label(js_frame, text="Low")
        js_low_label.grid(row=1, column=2)

        js_high_label = Label(js_frame, text="High")
        js_high_label.grid(row=1, column=3)

        js_offset_label = Label(js_frame, text="Offset")
        js_offset_label.grid(row=1, column=4)

        js_number_x_label = Label(js_frame, text=1)
        js_number_x_label.grid(row=2, column=0)

        self.js_x_enabled = BooleanVar()
        self.js_x_enabled.set(self.filter.joystick_x_enabled)
        js_x_enabled_checkbutton = Checkbutton(js_frame, variable=self.js_x_enabled)
        js_x_enabled_checkbutton.grid(row=2, column=1)

        self.js_x_low = StringVar()
        self.js_x_low.set(self.filter.joystick_x_low)
        js_x_low_entry = Entry(js_frame, width=5, textvariable=self.js_x_low)
        js_x_low_entry.grid(row=2, column=2)

        self.js_x_high = StringVar()
        self.js_x_high.set(self.filter.joystick_x_high)
        js_x_high_entry = Entry(js_frame, width=5, textvariable=self.js_x_high)
        js_x_high_entry.grid(row=2, column=3)

        self.js_x_offset = StringVar()
        self.js_x_offset.set(self.filter.joystick_x_offset)
        js_x_offset_entry = Entry(js_frame, width=5, textvariable=self.js_x_offset)
        js_x_offset_entry.grid(row=2, column=4)

        js_number_y_label = Label(js_frame, text=2)
        js_number_y_label.grid(row=3, column=0)

        self.js_y_enabled = BooleanVar()
        self.js_y_enabled.set(self.filter.joystick_y_enabled)
        js_y_enabled_checkbutton = Checkbutton(js_frame, variable=self.js_y_enabled)
        js_y_enabled_checkbutton.grid(row=3, column=1)

        self.js_y_low = StringVar()
        self.js_y_low.set(self.filter.joystick_y_low)
        js_y_low_entry = Entry(js_frame, width=5, textvariable=self.js_y_low)
        js_y_low_entry.grid(row=3, column=2)

        self.js_y_high = StringVar()
        self.js_y_high.set(self.filter.joystick_y_high)
        js_y_high_entry = Entry(js_frame, width=5, textvariable=self.js_y_high)
        js_y_high_entry.grid(row=3, column=3)

        self.js_y_offset = StringVar()
        self.js_y_offset.set(self.filter.joystick_y_offset)
        js_y_offset_entry = Entry(js_frame, width=5, textvariable=self.js_y_offset)
        js_y_offset_entry.grid(row=3, column=4)

        common_frame = LabelFrame(right_frame, text="Common Settings")
        common_frame.grid(row=3, column=0, sticky="nsew")

        common_apply_filters_button = Button(common_frame, text="Apply all filters", command=self._save)
        common_apply_filters_button.grid(sticky="nesw")
