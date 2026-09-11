# MIRepNet maze test — very simple explanation

## What MIRepNet does

The EEG headset records brain signals while a person imagines moving left or
right. MIRepNet looks at one short piece of that EEG and makes one guess:

- `1` means **LEFT**
- `2` means **RIGHT**

MIRepNet was already taught general EEG patterns using a much larger dataset.
Fine-tuning is the extra training that teaches it the patterns of this project,
this headset, and these patients.

## What the maze proves

This project already has four standard 40-corner mazes. They are fixed and do
not change:

- Standard maze 1 = original seed 11
- Standard maze 2 = original seed 12
- Standard maze 3 = original seed 13
- Standard maze 4 = original seed 14

For every tested EEG epoch, the file contains the instruction recorded during
the experiment. That instruction is called the **true label**.

The maze test uses the two pieces separately:

1. You select one of the four original fixed mazes.
2. When the robot reaches a corner, the program takes the next unused recorded
   EEG epoch whose instruction is the direction that this maze requires.
3. MIRepNet calculates a new prediction at that moment. Predictions are not
   calculated first and replayed later.
4. The **new MIRepNet prediction drives the robot**.
5. A correct prediction sends the robot through the open corridor.
6. A wrong prediction turns the robot toward a wall. The mistake is counted and
   shown on screen. The robot then corrects itself automatically so the replay
   can continue; this correction does not consume another EEG epoch.

The final score is simply:

`correct MIRepNet guesses / EEG epochs shown in the maze`

The program also prints the accuracy on **all eligible test epochs**. The maze
score uses exactly 40 route-matched EEG epochs because every standard maze has
40 decision corners.

## The four test choices

- **Zero-shot:** directly tests the saved model. It does not adjust anything
  using the selected patient first. This is the strict unseen-patient test.
- **Unlabeled EA:** uses the selected patient's EEG values to make their signal
  scale and shape look more like the training data. It never reads the answers
  while making that adjustment.
- **Calibrated:** uses the first recordings and their answers to learn this
  patient, then tests only on later recordings.
The balanced-block method is intentionally not available in this maze test. It
must inspect all scores in a completed batch before assigning decisions, which
would contradict the requirement to make one new prediction at each corner.

## How to run it

Run `python main.py`, then choose:

1. `4) MIRepNet Workbench`
2. `5) Patient maze replay`
3. Pick the saved weights, one patient, a test method, and standard maze 1--4.

The maze opens automatically. Its CSV report is saved in `simulation/reports`,
and the exact EEG/prediction replay plan is saved in `results/mirepnet`.

## Live headset maze

The workbench has a separate option `6) Live OpenBCI headset control on a fixed
maze`. Unlike option 5, it does not replay saved EEG. It connects to the
Cyton+Daisy, completes the chosen live calibration, and opens standard maze
1--4. When the robot stops at a wall, imagine the left hand for LEFT or the
right hand for RIGHT. The maze ignores predictions made while the robot was
driving and waits for a full fresh post-stop MIRepNet vote before turning.

Closing the maze saves the raw live EEG, decision CSV/Excel workbook, maze CSV
report, and pose trace.
