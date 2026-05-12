DATA_DIR = './data'
DELTA_T=0.25
EEG_CHANNELS = ['Cz', 'Pz', 'C3', 'C4', 'T5', 'T6', 'Fz', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'P3', 'P4']
EPOCH_TMIN = 0.0
EPOCH_TMAX = 2.0
FILTER_KWARGS = dict(
    l_freq=4.0, h_freq=42.0,
    l_trans_bandwidth=2.0, h_trans_bandwidth=3.0,
    method='fir', phase='minimum', fir_design='firwin',
)
TARGET_MAPPINGS = {'21': 1, '22': 2}
WS_HOST = '0.0.0.0'
WS_PORT = 8765

# Temporal smoother — M-of-N voting on the last N predictions.
# Pre-vote: any prediction with confidence < SMOOTHER_CONF_FLOOR is dropped
# (treated as abstain). Of those that vote, a class wins if it has >= M votes.
SMOOTHER_N = 3
SMOOTHER_M = 2
SMOOTHER_CONF_FLOOR = 0.60