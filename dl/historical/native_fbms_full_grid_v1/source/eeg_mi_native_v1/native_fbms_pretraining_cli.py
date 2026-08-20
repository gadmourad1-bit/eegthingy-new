"""Locked write-once CardinalFBMS native-montage pretraining entry point.

This command has no model, dataset, subject, montage, cache-building, or
confirmation switch.  It reuses the audited native-pretraining publisher while
pinning a distinct CardinalFBMS identity and artifact schema.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from . import native_pretraining_cli as _core
from .native_pretraining import NativePretrainConfig


ARTIFACT_SCHEMA = "eeg-mi-native-cardinal-fbms-pretraining-development-v1"
CHECKPOINT_SCHEMA = "eeg-mi-native-cardinal-fbms-pretraining-checkpoint-v1"
MODEL_KEY = "cardinal_fbms"
CHECKPOINT_FILENAME = _core.CHECKPOINT_FILENAME
PROVENANCE_FILENAME = _core.PROVENANCE_FILENAME
FROZEN_CONFIG = NativePretrainConfig()

ENTRY_POINT = _core.NativePretrainingEntryPoint(
    model_key=MODEL_KEY,
    artifact_schema=ARTIFACT_SCHEMA,
    checkpoint_schema=CHECKPOINT_SCHEMA,
    description=(
        "Pretrain canonical CardinalFBMS on the locked v2 native development "
        "corpus and atomically write a new artifact directory."
    ),
    extra_source_files=(("eeg_mi/native_fbms_pretraining_cli.py", Path(__file__)),),
)


def build_parser() -> argparse.ArgumentParser:
    """Expose paths and execution device, never scientific hyperparameters."""

    parser = argparse.ArgumentParser(description=ENTRY_POINT.description)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=_core._device, default=FROZEN_CONFIG.device)
    return parser


def run(argv: Sequence[str] | None = None) -> Path:
    arguments = build_parser().parse_args(argv)
    return _core.execute_locked_native_pretraining(
        ENTRY_POINT,
        cache_root=arguments.cache_root,
        output=arguments.output,
        config=replace(FROZEN_CONFIG, device=arguments.device),
    )


def main(argv: Sequence[str] | None = None) -> int:
    output = run(argv)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
