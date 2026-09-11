DATA_DIR = './data'
STRIDE_S = 0.2
FILTER_WARMUP_S = 2.0
CALIBRATION_SECONDS = 45          # classical decoders: EA/Riemann alignment reference + boundary seed
CALIBRATION_SECONDS_DEEP = 15     # deep decoders: boundary seed only (no alignment reference to fit)
DEEP_EPOCHS = 200                 # training passes for the deep decoders (early stopping usually ends sooner)
DEEP_SEED = 7
DEEP_DEVICE = 'auto'              # 'auto' picks CUDA, then Apple Metal (MPS), then CPU; or force one
MIREPNET_EPOCHS = 20              # foundation-model fine-tuning passes (early stopping usually ends sooner)
MIREPNET_SEED = 7
MIREPNET_DEVICE = 'auto'
MIREPNET_WINDOW_MODE = 'task-repeat'  # audited Exp4 task 0--2 s, repeated to MIRepNet's required 4 s
LOCAL_EXP4_VALID_RUNS = {
    1: (1, 2, 3, 4), 3: (1, 2, 3, 4), 4: (1, 2, 3, 4),
    5: (1, 2, 3, 4), 6: (1, 2, 3, 4), 7: (1, 2, 3, 4),
    8: (1, 2, 3, 4), 10: (5, 6, 7, 8),
}
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
FB_BANDS = [(8.0, 12.0), (11.0, 15.0), (14.0, 20.0), (20.0, 30.0)]
FB_TRANS = dict(l_trans_bandwidth=2.0, h_trans_bandwidth=2.5)
CSP_COMPONENTS = 4
EPOCH_REJECT = dict(eeg=100e-6)
GUI_HISTORY_S = 30
GUI_REFRESH_RATE = 60
LABELS = {1: "left_hand", 2: "right_hand", 3: "left_foot", 4: "right_foot"}
PHASES = {1: "prep", 2: "plan", 3: "task", 4: "rest"}
SMOOTHER_N = 5
SMOOTHER_M = 4
SMOOTHER_DWELL = 3
CONF_FLOOR = 0.85    # commit gate on the recentered confidence; also acts as the rest dead-zone
RECENTER_ADAPTIVE = True   # False = boundary frozen at its calibration seed (no feedback loop); True = EMA-track it live
RECENTER_ALPHA = 0.01       # EMA rate for the adaptive decision boundary (applied on rest-like windows only)
RECENTER_CLAMP = 2.0        # max drift of the neutral from its calibration seed, in log-odds
RECENTER_REST_CONF = 0.65   # windows below this recentered confidence are 'rest' and update the neutral
TARGET_MAPPINGS = {
    'left_hand/task':  1,
    'right_hand/task': 2,
}
WS_HOST = '0.0.0.0'
WS_PORT = 8765
