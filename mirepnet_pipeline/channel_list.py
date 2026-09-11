# Vendored from staraink/MIRepNet (utils/channel_list.py), MIT licensed.
# Trimmed to just what's needed for inference: the 45-channel template
# MIRepNet's PatchEmbedding was trained on, and the 2D scalp coordinates used
# to interpolate missing channels for headsets with fewer/different electrodes.

MIREPNET_TEMPLATE_CHANNELS = [
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
]
assert len(MIREPNET_TEMPLATE_CHANNELS) == 45

LEGACY_CHANNEL_ALIASES = {
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
}


def normalize_channel_name(name):
    """'T3' -> 'T7', 'Cz' -> 'CZ', etc. Anything already-modern passes through."""
    name = name.strip().upper()
    return LEGACY_CHANNEL_ALIASES.get(name, name)


CHANNEL_POSITIONS = {
    'FP1': (-0.3, 0.9), 'FPZ': (0, 1.0), 'FP2': (0.3, 0.9),
    'AF7': (-0.3, 0.8), 'AF3': (-0.2, 0.8), 'AFZ': (0, 0.8), 'AF4': (0.2, 0.8), 'AF8': (0.3, 0.8),
    'F9': (-0.6, 0.7), 'F7': (-0.5, 0.7), 'F5': (-0.4, 0.7), 'F3': (-0.3, 0.7),
    'F1': (-0.15, 0.7), 'FZ': (0, 0.7), 'F2': (0.15, 0.7),
    'F4': (0.3, 0.7), 'F6': (0.4, 0.7), 'F8': (0.5, 0.7), 'F10': (0.6, 0.7),
    'FT9': (-0.7, 0.6), 'FT7': (-0.6, 0.6), 'FC5': (-0.5, 0.6), 'FC3': (-0.4, 0.6),
    'FC1': (-0.2, 0.6), 'FCZ': (0, 0.6), 'FC2': (0.2, 0.6),
    'FC4': (0.4, 0.6), 'FC6': (0.5, 0.6), 'FT8': (0.6, 0.6), 'FT10': (0.6, 0.7),
    'FTT9': (-1.1, 0.55), 'T7': (-1.0, 0.5), 'TPP7': (-0.9, 0.45), 'C5': (-0.7, 0.5), 'C3': (-0.4, 0.5),
    'C1': (-0.2, 0.5), 'CZ': (0, 0.5), 'C2': (0.2, 0.5),
    'C4': (0.4, 0.5), 'C6': (0.7, 0.5), 'TPP8': (0.9, 0.45), 'T8': (1.0, 0.5), 'FTT10': (1.1, 0.55),
    'TP9': (-0.8, 0.4), 'TPP9': (-0.7, 0.4), 'TP7': (-0.6, 0.4), 'CP5': (-0.5, 0.4), 'CP3': (-0.4, 0.4),
    'CP1': (-0.2, 0.4), 'CPZ': (0, 0.4), 'CP2': (0.2, 0.4),
    'CP4': (0.4, 0.4), 'CP6': (0.5, 0.4), 'TP8': (0.6, 0.4), 'TPP10': (0.7, 0.4), 'TP10': (0.8, 0.4),
    'P9': (-0.6, 0.3), 'P7': (-0.5, 0.3), 'P5': (-0.4, 0.3), 'P3': (-0.3, 0.3),
    'P1': (-0.15, 0.3), 'PZ': (0, 0.3), 'P2': (0.15, 0.3),
    'P4': (0.3, 0.3), 'P6': (0.4, 0.3), 'P8': (0.5, 0.3), 'P10': (0.6, 0.3),
    'PO9': (-0.5, 0.2), 'PO7': (-0.4, 0.2), 'PO5': (-0.3, 0.2), 'PO3': (-0.2, 0.2), 'POZ': (0, 0.2),
    'PO4': (0.2, 0.2), 'PO6': (0.3, 0.2), 'PO8': (0.4, 0.2), 'PO10': (0.5, 0.2),
    'O1': (-0.2, 0.1), 'OZ': (0, 0.1), 'O2': (0.2, 0.1),
}