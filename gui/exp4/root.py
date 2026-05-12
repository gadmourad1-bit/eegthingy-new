from tkinter import *
from gui.exp4.config import Config
from gui.exp4.subject.root import SubjectInterface


class EXP4Tab(Frame):
    def __init__(self, notebook):
        super().__init__(notebook)
        self.name = "EXP4"

        self.subject_window = None
        self.subject = Config().load()
        self._create_widgets()

    def _create_widgets(self):
        """Create the widgets for the exp1 tab."""
        left_frame = LabelFrame(self, text="Session Parameters", background="white", foreground="black", padx=10, pady=10)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True)

        subject_label = Label(left_frame, text="Subject", background="white", foreground="black", font=("Arial", 12))
        subject_label.grid(row=1, column=0, pady=5, padx=5)

        run_label = Label(left_frame, text="Run", background="white", foreground="black", font=("Arial", 12))
        run_label.grid(row=1, column=1, pady=5, padx=5)

        num_cues_label = Label(left_frame, text="Cues", background="white", foreground="black", font=("Arial", 12))
        num_cues_label.grid(row=1, column=2, pady=5, padx=5)

        self.subject_entry = Entry(left_frame, width=10, relief=SOLID)
        self.subject_entry.insert(0, str(self.subject.id))
        self.subject_entry.grid(row=2, column=0, pady=5, padx=5)

        self.run_entry = Entry(left_frame, width=10, relief=SOLID)
        self.run_entry.insert(0, str(self.subject.run))
        self.run_entry.grid(row=2, column=1, pady=5, padx=5)

        self.num_cues_entry = Entry(left_frame, width=10, relief=SOLID)
        self.num_cues_entry.insert(0, str(self.subject.cues))
        self.num_cues_entry.grid(row=2, column=2, pady=5, padx=5)

        border = Frame(left_frame, width=500, height=1, background="white")
        border.grid(row=3, column=0, columnspan=4, pady=10)

        label_prep = Label(left_frame, text="Preparation (s)", background="white", foreground="black", font=("Arial", 12))
        label_prep.grid(row=4, column=0, pady=5, padx=5)

        self.entry_prep = Entry(left_frame, width=10, relief=SOLID)
        self.entry_prep.insert(0, self.subject.time_prep)
        self.entry_prep.grid(row=4, column=1, pady=5, padx=5)

        label_plan = Label(left_frame, text="Planning (s)", background="white", foreground="black", font=("Arial", 12))
        label_plan.grid(row=5, column=0, pady=5, padx=5)

        self.entry_plan = Entry(left_frame, width=10, relief=SOLID)
        self.entry_plan.insert(0, self.subject.time_plan)
        self.entry_plan.grid(row=5, column=1, pady=5, padx=5)

        label_task = Label(left_frame, text="Task (s)", background="white", foreground="black", font=("Arial", 12))
        label_task.grid(row=6, column=0, pady=5, padx=5)

        self.entry_task = Entry(left_frame, width=10, relief=SOLID)
        self.entry_task.insert(0, self.subject.time_task)
        self.entry_task.grid(row=6, column=1, pady=5, padx=5)

        label_rest = Label(left_frame, text="Rest (s)", background="white", foreground="black", font=("Arial", 12))
        label_rest.grid(row=7, column=0, pady=5, padx=5)

        self.entry_rest = Entry(left_frame, width=10, relief=SOLID)
        self.entry_rest.insert(0, self.subject.time_rest)
        self.entry_rest.grid(row=7, column=1, pady=5, padx=5)

        frame_run_type = LabelFrame(left_frame, text="Run type", background="white", foreground="black", padx=10, pady=10)
        frame_run_type.grid(row=4, column=2, rowspan=3, padx=5)

        self.run_type = StringVar()
        self.run_type.set(self.subject.run_type)

        radio_training = Radiobutton(frame_run_type, text="Training", variable=self.run_type, value="Training", background="white", foreground="black", selectcolor="white")
        radio_training.grid(row=0, column=0, sticky="nesw")

        radio_demo = Radiobutton(frame_run_type, text="Demo", variable=self.run_type, value="Demo",
                                    background="white", foreground="black", selectcolor="white")
        radio_demo.grid(row=1, column=0, sticky="nesw")

        radio_testing = Radiobutton(frame_run_type, text="Testing", variable=self.run_type, value="Testing", background="white", foreground="black", selectcolor="white")
        radio_testing.grid(row=2, column=0, sticky="nesw")

        right_frame = LabelFrame(self, text="Actions", background="white", foreground="black", padx=10, pady=10)
        right_frame.pack(side=LEFT, fill=BOTH, expand=True)

        button_start = Button(right_frame, text="Start", command=self._show_subject_window)
        button_start.grid(row=1, column=0, sticky="nesw", padx=5, pady=5)

    def _save(self):
        """Save the current subject parameters."""
        self.subject.id = int(self.subject_entry.get())
        self.subject.run = int(self.run_entry.get())
        self.subject.cues = int(self.num_cues_entry.get())
        self.subject.time_prep = int(self.entry_prep.get())
        self.subject.time_plan = int(self.entry_plan.get())
        self.subject.time_task = int(self.entry_task.get())
        self.subject.time_rest = int(self.entry_rest.get())
        self.subject.run_type = self.run_type.get()
        self.subject.save()

    def _show_subject_window(self):
        """Open the subject window."""
        self._save()
        if self.subject_window is None:
            self.subject_window = SubjectInterface(self)
        else:
            self.subject_window.focus()