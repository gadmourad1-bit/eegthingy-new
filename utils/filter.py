import json
import numpy as np


class ChannelFilter:
    def __init__(self):
        self.eeg_enabled = [True] * 16
        self.eeg_low = [-999999.0] * 16
        self.eeg_high = [999999.0] * 16
        self.eeg_offset = [0.0] * 16

        self.hd_enabled = True
        self.hd_low = -15.0
        self.hd_high = 1000.0
        self.hd_offset = 0.0

        self.emg_enabled = [True] * 4
        self.emg_low = [-999999.0] * 4
        self.emg_high = [999999.0] * 4
        self.emg_offset = [0.0] * 4

        self.joystick_x_enabled = True
        self.joystick_x_low = -1.0
        self.joystick_x_high = 1.0
        self.joystick_x_offset = 0.0

        self.joystick_y_enabled = True
        self.joystick_y_low = -1.0
        self.joystick_y_high = 1.0
        self.joystick_y_offset = 0.0

    def load(self):
        """Load the filter configuration from a json file."""
        try:
            with open('./filter.json') as json_file:
                self.__dict__ = json.load(json_file)
        except:
            pass
        return self

    def save(self):
        """Save the filter configuration to a json file."""
        with open('./filter.json', 'w+') as outfile:
            json.dump(self.__dict__, outfile, indent=4)

    def apply(self, data, type='none'):
        """Apply the filter to the data."""
        if type == 'eeg':
            offset  = np.asarray(self.eeg_offset)
            low     = np.asarray(self.eeg_low)
            high    = np.asarray(self.eeg_high)
            enabled = np.asarray(self.eeg_enabled)

            if data.ndim == 2:
                offset, low, high, enabled = (a[:, None] for a in (offset, low, high, enabled))

            data = data + offset
            data = np.clip(data, low, high)
            data = np.where(enabled, data, 0.0)
        elif type == 'emg':
            offset  = np.asarray(self.emg_offset)
            low     = np.asarray(self.emg_low)
            high    = np.asarray(self.emg_high)
            enabled = np.asarray(self.emg_enabled)

            if data.ndim == 2:
                offset, low, high, enabled = (a[:, None] for a in (offset, low, high, enabled))

            data = data + offset
            data = np.clip(data, low, high)
            data = np.where(enabled, data, 0.0)
        elif type == 'hd':
            data = (data + self.hd_offset)
            data = np.clip(data, self.hd_low, self.hd_high)
            data = np.where(self.hd_enabled, data, 0.0)
        elif type == 'joystick':
            data[0] = (data[0] + self.joystick_x_offset)
            data[1] = (data[1] + self.joystick_y_offset)
            data[0] = np.clip(data[0], self.joystick_x_low, self.joystick_x_high)
            data[1] = np.clip(data[1], self.joystick_y_low, self.joystick_y_high)
            data[0] = np.where(self.joystick_x_enabled, data[0], 0.0)
            data[1] = np.where(self.joystick_y_enabled, data[1], 0.0)

        return data
