import json


class Config:
    def __init__(self):
        self.id = 1
        self.run = 1
        self.cues = 40
        self.time_prep = 10
        self.time_plan = 2
        self.time_task = 2
        self.time_rest = 2
        self.time_buffer = 0.1
        self.run_type = "Training"
        self.load()

    def load(self):
        """Load the subject configuration from a json file."""
        try:
            with open('./exp4_config.json', 'r') as json_file:
                self.__dict__ = json.load(json_file)
        except:
            pass
        return self

    def save(self):
        """Save the subject configuration to a json file."""
        with open('./exp4_config.json', 'w+') as outfile:
            json.dump(self.__dict__, outfile, indent=4)
        return self
