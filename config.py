DATA_DIR = './data'
DELTA_T=0.6
EEG_CHANNELS_MAPPING = [
    'Cz', 'Pz', 'C3', 'C4', 'T5', 'T6', 'Fz', 'DEAD',  # Cyton ch 1-8  (ch8 unplugged)
    'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'P3', 'P4',    # Daisy  ch 9-16
]
EEG_CHANNELS_TARGETS = [
    'Cz', 'Pz', 'C3', 'C4', 'T5', 'T6', 'Fz',       # ch8 is marked as dead
    'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'P3', 'P4'
]
EPOCH_TMIN = 0.5
EPOCH_TMAX = 2.5
# mu band (8-13 Hz): the sensorimotor rhythm where motor-imagery ERD lives.
# Narrowing from the old 4-42 Hz dropped theta/EMG noise out of the CSP
# covariance estimates and was the single biggest accuracy lever.
FILTER_KWARGS = dict(
    l_freq=8.0, h_freq=13.0,
    l_trans_bandwidth=2.0, h_trans_bandwidth=3.0,
    method='fir', phase='minimum', fir_design='firwin',
)
# Peak-to-peak rejection: drops epochs with movement/drowsiness artifacts.
# Free on clean recordings (keeps all epochs), balances noisier sessions.
EPOCH_REJECT = dict(eeg=100e-6)
GUI_HISTORY_S = 30
GUI_REFRESH_RATE = 60
LABELS = {1: "left_hand", 2: "right_hand", 3: "left_foot", 4: "right_foot"}
PHASES = {1: "prep", 2: "plan", 3: "task", 4: "rest"}
SMOOTHER_N = 3
SMOOTHER_M = 2
SMOOTHER_CONF_FLOOR = 0.85
TARGET_MAPPINGS = {
    'left_hand/task':  1,
    'right_hand/task': 2,
}
WS_HOST = '0.0.0.0'
WS_PORT = 8765