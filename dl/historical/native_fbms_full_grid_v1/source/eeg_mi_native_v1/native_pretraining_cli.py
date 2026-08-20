"""Atomic development-only launcher for locked native-montage pretraining.

This entry point is intentionally narrow. It can only read the frozen v2
development corpus through :func:`load_primary_native_pretraining_corpus`, and
it always constructs the canonical 21-channel CardinalFBC model. It neither
imports nor invokes a cache builder and exposes no confirmation-stage,
dataset, subject, montage-profile, or model-selection switches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import make_model
from .config import (
    CANONICAL_21_CHANNELS,
    CHANNEL_SCALING,
    PREPROCESSING,
)
from .models import CANONICAL_21_POSITIONS, parameter_count
from .native_pretraining import (
    PRIMARY_NATIVE_DEVELOPMENT_SOURCES,
    LoadedNativePretrainingCorpus,
    NativePretrainConfig,
    NativePretrainingResult,
    fit_native_montage_pretraining,
    load_primary_native_pretraining_corpus,
    state_dict_sha256,
)


ARTIFACT_SCHEMA = "eeg-mi-native-pretraining-development-v1"
CHECKPOINT_SCHEMA = "eeg-mi-native-pretraining-checkpoint-v1"
MODEL_KEY = "cardinal_fbc"
CHECKPOINT_FILENAME = "checkpoint.pt"
PROVENANCE_FILENAME = "provenance.json"


@dataclass(frozen=True)
class NativePretrainingEntryPoint:
    """Locked model/schema identity for one non-search pretraining command."""

    model_key: str
    artifact_schema: str
    checkpoint_schema: str
    description: str
    extra_source_files: tuple[tuple[str, Path], ...] = ()


LEGACY_ENTRY_POINT = NativePretrainingEntryPoint(
    model_key=MODEL_KEY,
    artifact_schema=ARTIFACT_SCHEMA,
    checkpoint_schema=CHECKPOINT_SCHEMA,
    description=(
        "Pretrain canonical CardinalFBC on the locked v2 native development "
        "corpus and atomically write a new artifact directory."
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_exists(path: Path) -> bool:
    """Return true for any directory entry, including a broken symlink."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _OutputClaim:
    """Coordinate output publication and reject cooperating CLI races."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.lock = output.parent / f".{output.name}.native-pretraining.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> "_OutputClaim":
        if _path_exists(self.output):
            raise FileExistsError(f"refusing to overwrite output {self.output}")
        if not self.output.parent.is_dir():
            raise FileNotFoundError(
                f"output parent must already exist: {self.output.parent}"
            )
        try:
            self._descriptor = os.open(
                self.lock,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"another native-pretraining publisher has claimed {self.output}"
            ) from exc
        try:
            os.write(
                self._descriptor,
                f"pid={os.getpid()} output={self.output}\n".encode("utf-8"),
            )
            os.fsync(self._descriptor)
            if _path_exists(self.output):
                raise FileExistsError(
                    f"output appeared while acquiring publication claim: {self.output}"
                )
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *unused: object) -> None:
        del unused
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        try:
            self.lock.unlink()
        except FileNotFoundError:
            pass


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"provenance value is not JSON serializable: {type(value)!r}")


def _source_paths(
    extra_source_files: Sequence[tuple[str, Path]] = (),
) -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    project = package.parent
    paths = {
        "eeg_mi/baselines.py": package / "baselines.py",
        "eeg_mi/config.py": package / "config.py",
        "eeg_mi/data.py": package / "data.py",
        "eeg_mi/models.py": package / "models.py",
        "eeg_mi/native_pretraining.py": package / "native_pretraining.py",
        "eeg_mi/native_pretraining_cli.py": Path(__file__).resolve(),
        "eeg_mi/training.py": package / "training.py",
        "eeg_mi/requirements-cu128.txt": package / "requirements-cu128.txt",
    }
    root_requirements = project / "requirements.txt"
    if root_requirements.is_file():
        paths["requirements.txt"] = root_requirements
    for name, path in extra_source_files:
        if name in paths:
            raise ValueError(f"duplicate source-file provenance key {name}")
        paths[str(name)] = Path(path).resolve()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required source files are missing: {missing}")
    return paths


def _source_file_hashes(
    extra_source_files: Sequence[tuple[str, Path]] = (),
) -> dict[str, str]:
    return {
        name: _sha256_file(path)
        for name, path in sorted(_source_paths(extra_source_files).items())
    }


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _environment_record(device: str) -> dict[str, object]:
    cuda_devices: list[dict[str, object]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(
                        torch.cuda.get_device_capability(index)
                    ),
                }
            )
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "requested_device": device,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "cuda_devices": cuda_devices,
        "mps_available": bool(
            mps_backend is not None and mps_backend.is_available()
        ),
        "packages": {
            name: _distribution_version(name)
            for name in (
                "braindecode",
                "mne",
                "moabb",
                "numpy",
                "scipy",
                "torch",
            )
        },
    }


def _model_factory(model_key: str = MODEL_KEY) -> torch.nn.Module:
    return make_model(
        model_key,
        n_channels=21,
        n_outputs=2,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=torch.tensor(
            CANONICAL_21_POSITIONS,
            dtype=torch.float32,
        ),
    )


def _model_record(
    result: NativePretrainingResult,
    *,
    model_key: str = MODEL_KEY,
) -> dict[str, object]:
    model = result.model
    return {
        "identity": model_key,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "construction": {
            "n_channels": 21,
            "n_outputs": 2,
            "n_times": 320,
            "sfreq_hz": 128.0,
            "channel_names": list(CANONICAL_21_CHANNELS),
            "channel_positions": [list(row) for row in CANONICAL_21_POSITIONS],
        },
        "config": _jsonable(getattr(model, "config", {})),
        "trainable_parameter_count": parameter_count(model),
        "uses_positions": bool(getattr(model, "uses_positions", False)),
    }


def _corpus_identity(corpus: LoadedNativePretrainingCorpus) -> dict[str, object]:
    identities: list[dict[str, object]] = []
    for source in corpus.sources:
        identities.append(
            {
                "dataset": source["dataset"],
                "subject": source["subject"],
                "cache_file_sha256": source["cache_file_sha256"],
                "cache_identity_sha256": source["cache_identity_sha256"],
                "cache_array_sha256": source["cache_array_sha256"],
                "authorized_rows_sha256": source["authorized_rows_sha256"],
            }
        )
    return {
        "sha256": corpus.sha256,
        "cache_schema": corpus.cache_schema,
        "montage_profile": corpus.montage_profile,
        "source_record_count": len(corpus.sources),
        "source_identities": identities,
    }


def _checkpoint_payload(
    result: NativePretrainingResult,
    corpus: LoadedNativePretrainingCorpus,
    config: NativePretrainConfig,
    model_record: Mapping[str, object],
    *,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in result.checkpoint_state.items()
    }
    state_hash = state_dict_sha256(state)
    if state_hash != result.checkpoint_sha256:
        raise RuntimeError(
            "CPU checkpoint state does not match the pretraining result hash"
        )
    if any(value.device.type != "cpu" for value in state.values()):
        raise RuntimeError("checkpoint serialization attempted to retain device tensors")
    payload: dict[str, object] = {
        "schema": checkpoint_schema,
        "mode": "development",
        "confirmation_access": False,
        "model": dict(model_record),
        "model_state_dict": state,
        "training_config": asdict(config),
        "partition": asdict(result.partition),
        "selection_scalers": result.selection_scalers,
        "refit_scalers": result.refit_scalers,
        "state_hashes": {
            "initial": result.initial_state_sha256,
            "selection": result.selection_state_sha256,
            "checkpoint": result.checkpoint_sha256,
        },
        "corpus": _corpus_identity(corpus),
        "best_epoch_zero_based": result.best_epoch,
        "selected_epoch_count": result.best_epoch + 1,
    }
    if source_hashes is not None:
        payload["source_code"] = {
            "hash_algorithm": "sha256",
            "files": dict(source_hashes),
        }
    return payload


def _provenance_payload(
    *,
    result: NativePretrainingResult,
    corpus: LoadedNativePretrainingCorpus,
    config: NativePretrainConfig,
    model_record: Mapping[str, object],
    source_hashes: Mapping[str, str],
    checkpoint_sha256: str,
    checkpoint_size_bytes: int,
    started_utc: str,
    elapsed_seconds: float,
    artifact_schema: str = ARTIFACT_SCHEMA,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
) -> dict[str, object]:
    authorized_rows = sum(len(subject.y) for subject in corpus.subjects)
    train_subjects = len(result.partition.train)
    validation_subjects = len(result.partition.validation)
    return {
        "schema": artifact_schema,
        "mode": "development",
        "confirmation_access": False,
        "created_utc": _utc_now(),
        "started_utc": started_utc,
        "elapsed_seconds": float(elapsed_seconds),
        "model": dict(model_record),
        "corpus": {
            **_corpus_identity(corpus),
            "authorized_trial_count": authorized_rows,
            "sources": list(corpus.sources),
        },
        "training_config": asdict(config),
        "partition": asdict(result.partition),
        "scaling": {
            "selection": result.selection_scalers,
            "refit": result.refit_scalers,
            "validation_statistics_used_for_selection_scaling": False,
            "test_statistics_used": False,
        },
        "selection": {
            "best_epoch_zero_based": result.best_epoch,
            "selected_epoch_count": result.best_epoch + 1,
            "best_validation_equal_dataset_ce": (
                result.best_validation_equal_dataset_ce
            ),
            "steps_per_epoch": result.selection_steps_per_epoch,
            "history": result.selection_history,
        },
        "refit": {
            "epoch_count": len(result.refit_history),
            "steps_per_epoch": result.refit_steps_per_epoch,
            "history": result.refit_history,
        },
        "state_hashes": {
            "initial": result.initial_state_sha256,
            "selection": result.selection_state_sha256,
            "checkpoint": result.checkpoint_sha256,
        },
        "counts": {
            "source_subjects": len(corpus.subjects),
            "partition_train_subjects": train_subjects,
            "partition_validation_subjects": validation_subjects,
            "authorized_trials": authorized_rows,
            "dataset_subjects": result.dataset_subject_counts,
            "selection_steps_total": (
                len(result.selection_history)
                * result.selection_steps_per_epoch
            ),
            "refit_steps_total": (
                len(result.refit_history) * result.refit_steps_per_epoch
            ),
        },
        "checkpoint": {
            "schema": checkpoint_schema,
            "filename": CHECKPOINT_FILENAME,
            "file_sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size_bytes,
            "state_sha256": result.checkpoint_sha256,
            "all_tensors_cpu": True,
        },
        "source_code": {
            "hash_algorithm": "sha256",
            "files": dict(source_hashes),
        },
        "configuration": {
            "preprocessing": PREPROCESSING,
            "channel_scaling": CHANNEL_SCALING,
            "locked_native_development_sources": {
                dataset: list(subjects)
                for dataset, subjects in PRIMARY_NATIVE_DEVELOPMENT_SOURCES.items()
            },
            "cache_loading_only": True,
            "cache_building_available": False,
        },
        "environment": _environment_record(config.device),
    }


def _write_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("xb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _publish_artifacts(
    output: Path,
    *,
    result: NativePretrainingResult,
    corpus: LoadedNativePretrainingCorpus,
    config: NativePretrainConfig,
    model_record: Mapping[str, object],
    source_hashes: Mapping[str, str],
    started_utc: str,
    elapsed_seconds: float,
    artifact_schema: str = ARTIFACT_SCHEMA,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
) -> None:
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    checkpoint_path = staging / CHECKPOINT_FILENAME
    provenance_path = staging / PROVENANCE_FILENAME
    try:
        checkpoint = _checkpoint_payload(
            result,
            corpus,
            config,
            model_record,
            checkpoint_schema=checkpoint_schema,
            source_hashes=source_hashes,
        )
        _write_checkpoint(checkpoint_path, checkpoint)
        checkpoint_file_hash = _sha256_file(checkpoint_path)
        provenance = _provenance_payload(
            result=result,
            corpus=corpus,
            config=config,
            model_record=model_record,
            source_hashes=source_hashes,
            checkpoint_sha256=checkpoint_file_hash,
            checkpoint_size_bytes=checkpoint_path.stat().st_size,
            started_utc=started_utc,
            elapsed_seconds=elapsed_seconds,
            artifact_schema=artifact_schema,
            checkpoint_schema=checkpoint_schema,
        )
        _write_json(provenance_path, provenance)
        _fsync_directory(staging)
        if _path_exists(output):
            raise FileExistsError(
                f"refusing output that appeared before publication: {output}"
            )
        # The exclusive sibling claim prevents another instance of this CLI
        # from racing this no-overwrite check. Directory rename publishes both
        # complete, fsynced artifacts as one namespace operation.
        os.rename(staging, output)
        _fsync_directory(output.parent)
    except BaseException:
        # Cleanup is deliberately scoped to the two files in our unique
        # staging directory. Never recurse through a caller-owned path.
        for path in (provenance_path, checkpoint_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            staging.rmdir()
        except FileNotFoundError:
            pass
        raise


def _bounded_int(
    label: str,
    minimum: int,
    maximum: int,
    *,
    even: bool = False,
) -> Any:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"{label} must be in [{minimum}, {maximum}]"
            )
        if even and number % 2:
            raise argparse.ArgumentTypeError(f"{label} must be even")
        return number

    return parse


def _bounded_float(label: str, minimum: float, maximum: float) -> Any:
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be numeric") from exc
        if not np.isfinite(number) or not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"{label} must be finite and in [{minimum}, {maximum}]"
            )
        return number

    return parse


def _device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"cpu", "mps", "cuda"}:
        return normalized
    if normalized.startswith("cuda:"):
        suffix = normalized.removeprefix("cuda:")
        if suffix.isdigit() and 0 <= int(suffix) <= 31:
            return normalized
    raise argparse.ArgumentTypeError(
        "device must be cpu, mps, cuda, or cuda:N with N in [0, 31]"
    )


def _build_parser(description: str) -> argparse.ArgumentParser:
    defaults = NativePretrainConfig()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--macro-epochs",
        type=_bounded_int("macro epochs", 1, 5000),
        default=defaults.macro_epochs,
    )
    parser.add_argument(
        "--batch-size-per-dataset",
        type=_bounded_int("batch size per dataset", 2, 4096, even=True),
        default=defaults.batch_size_per_dataset,
    )
    parser.add_argument(
        "--steps-per-macro-epoch",
        type=_bounded_int("steps per macro epoch", 1, 1_000_000),
        default=defaults.steps_per_macro_epoch,
    )
    parser.add_argument(
        "--learning-rate",
        type=_bounded_float("learning rate", 1e-8, 1.0),
        default=defaults.learning_rate,
    )
    parser.add_argument(
        "--weight-decay",
        type=_bounded_float("weight decay", 0.0, 1.0),
        default=defaults.weight_decay,
    )
    parser.add_argument(
        "--patience",
        type=_bounded_int("patience", 1, 5000),
        default=defaults.patience,
    )
    parser.add_argument(
        "--min-delta",
        type=_bounded_float("minimum delta", 0.0, 10.0),
        default=defaults.min_delta,
    )
    parser.add_argument(
        "--label-smoothing",
        type=_bounded_float("label smoothing", 0.0, 0.5),
        default=defaults.label_smoothing,
    )
    parser.add_argument(
        "--segment-probability",
        type=_bounded_float("segment probability", 0.0, 1.0),
        default=defaults.segment_probability,
    )
    parser.add_argument(
        "--segment-count",
        type=_bounded_int("segment count", 1, 64),
        default=defaults.segment_count,
    )
    parser.add_argument(
        "--time-shift-samples",
        type=_bounded_int("time shift samples", 0, 320),
        default=defaults.time_shift_samples,
    )
    parser.add_argument(
        "--noise-std",
        type=_bounded_float("noise standard deviation", 0.0, 1.0),
        default=defaults.noise_std,
    )
    parser.add_argument(
        "--gradient-clip",
        type=_bounded_float("gradient clip", 1e-8, 10_000.0),
        default=defaults.gradient_clip,
    )
    parser.add_argument(
        "--validation-batch-size",
        type=_bounded_int("validation batch size", 1, 65_536),
        default=defaults.validation_batch_size,
    )
    parser.add_argument(
        "--validation-fraction",
        type=_bounded_float("validation fraction", 1e-6, 0.5),
        default=defaults.validation_fraction,
    )
    parser.add_argument(
        "--seed",
        type=_bounded_int("seed", 0, 2**32 - 1),
        default=defaults.seed,
    )
    parser.add_argument("--device", type=_device, default=defaults.device)
    return parser


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(LEGACY_ENTRY_POINT.description)


def _config_from_args(args: argparse.Namespace) -> NativePretrainConfig:
    return NativePretrainConfig(
        macro_epochs=args.macro_epochs,
        batch_size_per_dataset=args.batch_size_per_dataset,
        steps_per_macro_epoch=args.steps_per_macro_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        label_smoothing=args.label_smoothing,
        segment_probability=args.segment_probability,
        segment_count=args.segment_count,
        time_shift_samples=args.time_shift_samples,
        noise_std=args.noise_std,
        gradient_clip=args.gradient_clip,
        validation_batch_size=args.validation_batch_size,
        validation_fraction=args.validation_fraction,
        # The partition salt remains the dataclass's frozen pilot default and
        # deliberately is not exposed as a CLI model-selection knob.
        seed=args.seed,
        device=args.device,
    )


def _validate_entry_point(spec: NativePretrainingEntryPoint) -> None:
    if spec.model_key not in {"cardinal_fbc", "cardinal_fbms"}:
        raise ValueError("unsupported locked native-pretraining model identity")
    for label, value in (
        ("artifact schema", spec.artifact_schema),
        ("checkpoint schema", spec.checkpoint_schema),
        ("description", spec.description),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonempty string")
    names = [name for name, _ in spec.extra_source_files]
    if len(names) != len(set(names)):
        raise ValueError("extra source-file provenance keys must be unique")


def run_locked_native_pretraining(
    spec: NativePretrainingEntryPoint,
    argv: Sequence[str] | None = None,
) -> Path:
    """Execute one fixed model family without exposing a model-selection CLI."""

    _validate_entry_point(spec)
    args = _build_parser(spec.description).parse_args(argv)
    return execute_locked_native_pretraining(
        spec,
        cache_root=args.cache_root,
        output=args.output,
        config=_config_from_args(args),
    )


def execute_locked_native_pretraining(
    spec: NativePretrainingEntryPoint,
    *,
    cache_root: str | Path,
    output: str | Path,
    config: NativePretrainConfig,
) -> Path:
    """Execute an already parsed fixed-family pretraining specification."""

    _validate_entry_point(spec)
    output = Path(output).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()

    # Hold the claim across source loading and training: a pre-existing output
    # or a second invocation is rejected before any cache archive is opened.
    with _OutputClaim(output):
        started_utc = _utc_now()
        started = time.monotonic()
        source_hashes = _source_file_hashes(spec.extra_source_files)
        corpus = load_primary_native_pretraining_corpus(
            cache_root,
            montage_profile="native",
        )
        result = fit_native_montage_pretraining(
            lambda: _model_factory(spec.model_key),
            corpus.subjects,
            config=config,
        )
        if _source_file_hashes(spec.extra_source_files) != source_hashes:
            raise RuntimeError("source files changed during native pretraining")
        model_record = _model_record(result, model_key=spec.model_key)
        _publish_artifacts(
            output,
            result=result,
            corpus=corpus,
            config=config,
            model_record=model_record,
            source_hashes=source_hashes,
            started_utc=started_utc,
            elapsed_seconds=time.monotonic() - started,
            artifact_schema=spec.artifact_schema,
            checkpoint_schema=spec.checkpoint_schema,
        )
    return output


def run(argv: Sequence[str] | None = None) -> Path:
    return run_locked_native_pretraining(LEGACY_ENTRY_POINT, argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = run(argv)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
