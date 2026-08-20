"""Resumable, audited all-architecture tournament on the local Exp4 cohort.

The primary comparison is deliberately limited to distinct architectures that
can be trained by :mod:`benchmark.runner` under one common recipe.  Training
aliases that change only width, dropout, reflection augmentation, or another
hyperparameter are ablations, not additional architecture families.

This cohort is development-only.  A numerical winner produced here is not an
independent confirmation result because every fourth recording has previously
been inspected during project development.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import runner as benchmark
from .analysis import _load_artifacts, analyze
from .baselines import make_model
from .config import CHANNEL_SCALING, dataset_spec, preprocessing_for_dataset
from .data import (
    apply_channel_scaler,
    fit_channel_scaler,
    load_subject_cache,
    reflection_index,
    split_indices,
)
from .models import CardinalFieldConfig, ScopeConfig, parameter_count
from .training import (
    TrainConfig,
    _forward,
    classification_metrics,
    configure_determinism,
)


PLAN_SCHEMA = "eeg-mi-local-model-tournament-plan-v1"
RESULT_SCHEMA = "eeg-mi-local-model-tournament-result-v1"
DATASET = "local_exp4"
SUBJECTS: tuple[int, ...] = (1, 3, 4, 5, 6, 7, 8, 10)
FOLDS: tuple[int, ...] = (0,)
SEEDS: tuple[int, ...] = benchmark.FORMAL_DEVELOPMENT_SEEDS

# Every distinct architecture accepted by the common raw-trial factory.  The
# ordering is frozen into plan.json and is also the deterministic shard order.
ARCHITECTURES: tuple[str, ...] = (
    "scope",
    "free_scope",
    "cardinal",
    "free_cardinal",
    "cardinal_dynamics",
    "cardinal_dynamics_compact",
    "cardinal_dynamics_extended",
    "cardinal_dynamics_sinc",
    "cardinal_dynamics_sinc_residual",
    "cardinal_dynamics_sinc_extended",
    "eegnet",
    "shallow",
    "deep4",
    "eegconformer",
    "eegconformer_compact",
    "atcnet",
    "atcnet_aggressive_pool",
    "fbcnet",
    "cardinal_fbc",
    "cardinal_fbc_extended",
    "cardinal_fbc_corr",
    "cardinal_fbc_corr_extended",
    "cardinal_fbc_compactdyn",
    "cardinal_fbc_compactdyn_extended",
    "cardinal_fbc_compactdyn_scale010",
    "cardinal_fbc_compactdyn_scale010_extended",
    "cardinal_fbc_compactdyn_scale025",
    "cardinal_fbc_compactdyn_scale025_extended",
    "cardinal_fbc_micro",
    "cardinal_fbc_micro_extended",
    "cardinal_fbc_physical",
    "cardinal_fbc_physical_extended",
    "eegtcnet",
    "fbmsnet",
    "cardinal_fbms",
    "cardinal_fbms_extended",
    "cardinal_mix",
    "cardinal_mix_drop",
    "ctnet",
    "ctnet_compact",
    "eegsym",
    "eegsym_wide",
    "tcformer",
)

BENCHMARK_SOURCE_NAMES: tuple[str, ...] = (
    "runner.py",
    "analysis.py",
    "models.py",
    "training.py",
    "data.py",
    "config.py",
    "baselines.py",
    "requirements-cu128.txt",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def strict_load(path: Path) -> dict[str, Any]:
    """Load research JSON while rejecting duplicate keys and NaN/Infinity."""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (*BENCHMARK_SOURCE_NAMES, Path(__file__).name)
    return {name: _sha256(root / name) for name in names}


def _cache_manifest(cache_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for subject in SUBJECTS:
        identity = load_subject_cache(DATASET, subject, cache_root=cache_root)[
            "identity"
        ]
        result[str(subject)] = str(identity["array_sha256"])
    return result


def _planned_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        epochs=int(args.epochs),
        patience=int(args.patience),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )


def _plan(cache_root: Path, train_config: TrainConfig) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "created_for": "resumable_all_architecture_local_development_comparison",
        "evidence_scope": "development_only_not_confirmation",
        "confirmation_evidence": False,
        "dataset": DATASET,
        "subjects": list(SUBJECTS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "architectures": list(ARCHITECTURES),
        "n_records": len(ARCHITECTURES) * len(SUBJECTS) * len(FOLDS) * len(SEEDS),
        "primary_metric": (
            "mean_subject_balanced_accuracy_after_averaging_seeds_within_subject"
        ),
        "train_config": asdict(train_config),
        "preprocessing": preprocessing_for_dataset(DATASET),
        "channel_scaling": CHANNEL_SCALING,
        "source_sha256": _source_manifest(),
        "cache_array_sha256": _cache_manifest(cache_root),
        "environment": benchmark._environment(),
    }


def ensure_plan(
    results_root: Path,
    *,
    cache_root: Path,
    train_config: TrainConfig,
) -> dict[str, Any]:
    """Create plan.json once, or require an exact match on resume."""

    expected = _plan(cache_root, train_config)
    path = results_root / "plan.json"
    if path.exists():
        observed = strict_load(path)
        # Creation time is intentionally absent, making the contract stable.
        if observed != expected:
            raise ValueError(f"{path} differs from the requested frozen plan")
        return observed
    try:
        _write_json_exclusive(path, expected)
    except FileExistsError:
        observed = strict_load(path)
        if observed != expected:
            raise ValueError(f"{path} differs from the requested frozen plan")
        return observed
    return expected


def _metric_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _metric_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _metric_values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12))
    return left == right


def _relative_artifact_sources(artifact: dict[str, Any]) -> dict[str, str]:
    source = artifact.get("source_sha256")
    if not isinstance(source, dict):
        raise ValueError("artifact has no source manifest")
    result: dict[str, str] = {}
    for raw_name, raw_digest in source.items():
        name = Path(str(raw_name)).name
        if name in result:
            raise ValueError(f"artifact source basename is duplicated: {name}")
        result[name] = str(raw_digest)
    return result


def validate_artifact(
    path: Path,
    *,
    model: str,
    plan: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    """Strictly validate one completed per-model benchmark artifact."""

    artifact = strict_load(path)
    _load_artifacts([path])
    exact_top_level = {
        "schema": benchmark.DEVELOPMENT_ARTIFACT_SCHEMA,
        "mode": "development",
        "artifact_mode": "formal",
        "confirmation_evidence": False,
        "dataset": DATASET,
        "subjects": list(SUBJECTS),
        "models": [model],
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "preprocessing": plan["preprocessing"],
        "channel_scaling": plan["channel_scaling"],
        "train_config": plan["train_config"],
        "environment": plan["environment"],
    }
    for key, expected in exact_top_level.items():
        if artifact.get(key) != expected:
            raise ValueError(f"{path} has an invalid {key!r} contract")
    expected_sources = {
        name: plan["source_sha256"][name] for name in BENCHMARK_SOURCE_NAMES
    }
    if _relative_artifact_sources(artifact) != expected_sources:
        raise ValueError(f"{path} source manifest differs from plan.json")

    records = artifact.get("records")
    expected_keys = set(product(SUBJECTS, FOLDS, SEEDS))
    if not isinstance(records, list) or len(records) != len(expected_keys):
        raise ValueError(f"{path} must contain exactly {len(expected_keys)} records")
    seen: set[tuple[int, int, int]] = set()
    caches: dict[int, dict[str, Any]] = {}
    expected_summaries: list[dict[str, Any]] = []
    for record in records:
        subject = int(record["subject"])
        fold = int(record["fold"])
        seed = int(record["seed"])
        key = (subject, fold, seed)
        if key not in expected_keys or key in seen:
            raise ValueError(f"{path} has an invalid/duplicate record key {key}")
        seen.add(key)
        data = caches.setdefault(
            subject,
            load_subject_cache(DATASET, subject, cache_root=cache_root),
        )
        train_rows, validation_rows, test_rows = split_indices(
            DATASET,
            data["y"],
            data["sessions"],
            data["runs"],
            fold=fold,
            subject=subject,
        )
        source_rows = np.sort(np.concatenate((train_rows, validation_rows)))
        split = record["split"]
        expected_split = {
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "refit_source_count": len(source_rows),
            "test_count": len(test_rows),
            "train_runs": sorted(set(data["runs"][train_rows].tolist())),
            "validation_runs": sorted(set(data["runs"][validation_rows].tolist())),
            "test_runs": sorted(set(data["runs"][test_rows].tolist())),
            "train_rows_sha256": hashlib.sha256(train_rows.tobytes()).hexdigest(),
            "validation_rows_sha256": hashlib.sha256(
                validation_rows.tobytes()
            ).hexdigest(),
            "test_rows_sha256": hashlib.sha256(test_rows.tobytes()).hexdigest(),
        }
        if split != expected_split:
            raise ValueError(f"{path} record {key} has an invalid split")
        if record.get("cache_identity") != data["identity"]:
            raise ValueError(f"{path} record {key} has an invalid cache identity")
        if plan["cache_array_sha256"][str(subject)] != data["identity"][
            "array_sha256"
        ]:
            raise ValueError(f"{path} record {key} cache differs from plan.json")

        channel_names = tuple(str(value) for value in data["channel_names"].tolist())
        selection_mean, selection_std = fit_channel_scaler(
            data["x"][train_rows], channel_names
        )
        refit_mean, refit_std = fit_channel_scaler(
            data["x"][source_rows], channel_names
        )
        expected_scalers = {
            "selection_train": {
                "mean": selection_mean.reshape(-1).tolist(),
                "std": selection_std.reshape(-1).tolist(),
            },
            "refit_source": {
                "mean": refit_mean.reshape(-1).tolist(),
                "std": refit_std.reshape(-1).tolist(),
            },
        }
        if not _metric_values_equal(record.get("scaler"), expected_scalers):
            raise ValueError(f"{path} record {key} has invalid scaler values")

        test = record["test"]
        expected_test_arrays = {
            "rows": test_rows.tolist(),
            "labels": data["y"][test_rows].tolist(),
            "sessions": data["sessions"][test_rows].tolist(),
            "runs": data["runs"][test_rows].tolist(),
        }
        for field, expected in expected_test_arrays.items():
            if test.get(field) != expected:
                raise ValueError(f"{path} record {key} has invalid test {field}")
        probabilities = np.asarray(test["probabilities"], dtype=np.float64)
        if (
            probabilities.shape != (len(test_rows), 2)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError(f"{path} record {key} has invalid probabilities")
        recomputed = classification_metrics(data["y"][test_rows], probabilities)
        if not _metric_values_equal(test.get("metrics"), recomputed):
            raise ValueError(f"{path} record {key} has stale test metrics")

        fit = record.get("fit", {})
        best_epoch = int(fit.get("best_epoch", -1))
        if not 0 <= best_epoch < int(plan["train_config"]["epochs"]):
            raise ValueError(f"{path} record {key} has invalid best epoch")
        if int(fit.get("refit_epochs_run", -1)) != best_epoch + 1:
            raise ValueError(f"{path} record {key} has invalid refit duration")
        for state_name in ("selection_state_sha256", "refit_state_sha256"):
            state_hash = fit.get(state_name)
            if not isinstance(state_hash, str) or len(state_hash) != 64:
                raise ValueError(f"{path} record {key} has invalid {state_name}")
        if int(fit.get("parameter_count", 0)) <= 0:
            raise ValueError(f"{path} record {key} has invalid parameter count")
        architecture = fit.get("architecture", {})
        if architecture.get("requested_name") != model:
            raise ValueError(f"{path} record {key} has invalid architecture identity")
        expected_effective_config = dict(plan["train_config"])
        expected_effective_config["seed"] = seed
        if fit.get("train_config") != expected_effective_config:
            raise ValueError(f"{path} record {key} has invalid train configuration")
        expected_summaries.append(record)
    if seen != expected_keys:
        raise ValueError(f"{path} is missing record identities")
    recomputed_summary = benchmark._summary(expected_summaries)
    if not _metric_values_equal(artifact.get("summary"), recomputed_summary):
        raise ValueError(f"{path} has a stale or invalid stored summary")
    return artifact


def _selected_models(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ARCHITECTURES
    values = tuple(token.strip() for token in value.split(",") if token.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("--models must be a nonempty unique comma-separated list")
    unknown = set(values) - set(ARCHITECTURES)
    if unknown:
        raise ValueError(f"unknown architectures: {sorted(unknown)}")
    return values


def _attempt_log(log_root: Path, model: str) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 10_000):
        path = log_root / f"{model}.attempt-{attempt:04d}.log"
        if not path.exists():
            return path
    raise RuntimeError(f"too many attempts for {model}")


def _process_identity(pid: int) -> dict[str, Any]:
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    stat_path = Path(f"/proc/{pid}/stat")
    boot_id = boot_path.read_text(encoding="utf-8").strip() if boot_path.exists() else None
    start_ticks = None
    if stat_path.exists():
        fields = stat_path.read_text(encoding="utf-8").split()
        if len(fields) > 21:
            start_ticks = fields[21]
    return {"pid": pid, "boot_id": boot_id, "start_ticks": start_ticks}


def _lock_is_live(lock: dict[str, Any]) -> bool:
    try:
        pid = int(lock["pid"])
        os.kill(pid, 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    current = _process_identity(pid)
    for key in ("boot_id", "start_ticks"):
        if lock.get(key) is not None and current.get(key) != lock.get(key):
            return False
    return True


def _acquire_lock(path: Path, *, recover_stale: bool) -> None:
    if path.exists():
        lock = strict_load(path)
        if _lock_is_live(lock):
            raise RuntimeError(f"active model lock exists: {path}")
        if not recover_stale:
            raise RuntimeError(f"stale model lock exists; rerun with --recover-stale: {path}")
        recovered = path.with_name(f"{path.name}.stale-{datetime.now().strftime('%Y%m%dT%H%M%S')}")
        path.rename(recovered)
    identity = _process_identity(os.getpid())
    _write_json_exclusive(
        path,
        {
            **identity,
            "host": socket.gethostname(),
            "created_at": _utc_now(),
        },
    )


def _append_journal(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _utc_now(), **event}, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _preflight(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_root).resolve()
    models = _selected_models(args.models)
    data = load_subject_cache(DATASET, SUBJECTS[0], cache_root=cache_root)
    train_rows, _, _ = split_indices(
        DATASET,
        data["y"],
        data["sessions"],
        data["runs"],
        subject=SUBJECTS[0],
    )
    channel_names = tuple(str(value) for value in data["channel_names"].tolist())
    mean, std = fit_channel_scaler(data["x"][train_rows], channel_names)
    values = apply_channel_scaler(data["x"][train_rows[:2]], mean, std)
    device = torch.device(args.device)
    results: list[dict[str, Any]] = []
    failures = 0
    for requested in models:
        print(f"preflight {requested}", flush=True)
        try:
            model_name, variant = benchmark._model_definition(requested)
            scope_config = (
                benchmark._scope_config(variant)
                if model_name in {"scope", "free_scope"}
                else ScopeConfig()
            )
            cardinal_config = (
                benchmark._cardinal_config(variant)
                if model_name in {"cardinal", "free_cardinal"}
                else CardinalFieldConfig()
            )
            configure_determinism(7)
            model = make_model(
                model_name,
                n_channels=values.shape[1],
                n_outputs=2,
                n_times=values.shape[2],
                sfreq=float(preprocessing_for_dataset(DATASET)["sfreq_hz"]),
                scope_config=scope_config,
                cardinal_config=cardinal_config,
                channel_names=channel_names,
                channel_positions=torch.as_tensor(
                    data["positions"], dtype=torch.float32
                ),
            ).to(device)
            x = torch.as_tensor(values, dtype=torch.float32, device=device)
            positions = torch.as_tensor(
                data["positions"], dtype=torch.float32, device=device
            )
            logits = _forward(model, x, positions)
            if logits.shape != (2, 2) or not bool(torch.all(torch.isfinite(logits))):
                raise RuntimeError(f"invalid logits {tuple(logits.shape)}")
            loss = torch.nn.functional.cross_entropy(
                logits, torch.as_tensor((0, 1), dtype=torch.long, device=device)
            )
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(bool(torch.all(torch.isfinite(g))) for g in gradients):
                raise RuntimeError("model has missing or non-finite trainable gradients")
            results.append(
                {
                    "model": requested,
                    "status": "pass",
                    "parameter_count": parameter_count(model),
                    "logit_shape": list(logits.shape),
                }
            )
        except Exception as error:  # continue to report every incompatible model
            failures += 1
            results.append(
                {
                    "model": requested,
                    "status": "fail",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        finally:
            if "model" in locals():
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    payload = {
        "schema": "eeg-mi-local-model-tournament-preflight-v1",
        "created_at": _utc_now(),
        "dataset": DATASET,
        "device": str(device),
        "models": list(models),
        "failures": failures,
        "results": results,
    }
    _write_json_exclusive(Path(args.output).resolve(), payload)
    print(f"preflight: {len(models) - failures}/{len(models)} passed", flush=True)
    return 1 if failures else 0


def _run(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_root).resolve()
    results_root = Path(args.results_root).resolve()
    train_config = _planned_train_config(args)
    plan = ensure_plan(
        results_root,
        cache_root=cache_root,
        train_config=train_config,
    )
    models = _selected_models(args.models)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    assigned = tuple(
        model
        for index, model in enumerate(models)
        if index % args.shard_count == args.shard_index
    )
    artifact_root = results_root / "artifacts"
    lock_root = results_root / "locks"
    log_root = results_root / "logs"
    journal = results_root / f"journal-shard-{args.shard_index:02d}.jsonl"
    artifact_root.mkdir(parents=True, exist_ok=True)
    lock_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    completed = 0
    skipped = 0
    for model in assigned:
        output = artifact_root / f"{model}.json"
        if output.exists():
            try:
                validate_artifact(
                    output,
                    model=model,
                    plan=plan,
                    cache_root=cache_root,
                )
            except Exception as error:
                failures.append(model)
                _append_journal(
                    journal,
                    {
                        "event": "invalid_existing_artifact",
                        "model": model,
                        "error": str(error),
                    },
                )
            else:
                skipped += 1
                print(f"skip validated {model}", flush=True)
            continue
        lock_path = lock_root / f"{model}.lock.json"
        _acquire_lock(lock_path, recover_stale=bool(args.recover_stale))
        try:
            partial = output.with_suffix(output.suffix + ".partial")
            if partial.exists():
                recovered = partial.with_name(
                    f"{partial.name}.stale-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
                )
                partial.rename(recovered)
                _append_journal(
                    journal,
                    {
                        "event": "recovered_stale_partial",
                        "model": model,
                        "path": str(recovered),
                    },
                )
            log_path = _attempt_log(log_root, model)
            command = [
                sys.executable,
                "-m",
                "benchmark.runner",
                "--dataset",
                DATASET,
                "--subjects",
                ",".join(str(value) for value in SUBJECTS),
                "--models",
                model,
                "--seeds",
                ",".join(str(value) for value in SEEDS),
                "--folds",
                "0",
                "--artifact-mode",
                "formal",
                "--cache-root",
                str(cache_root),
                "--output",
                str(output),
                "--epochs",
                str(train_config.epochs),
                "--patience",
                str(train_config.patience),
                "--batch-size",
                str(train_config.batch_size),
                "--device",
                train_config.device,
            ]
            print(f"run {model}; log={log_path}", flush=True)
            _append_journal(
                journal,
                {"event": "started", "model": model, "command": command},
            )
            with log_path.open("x", encoding="utf-8") as log_handle:
                process = subprocess.run(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log_handle.flush()
                os.fsync(log_handle.fileno())
            if process.returncode != 0:
                failures.append(model)
                _append_journal(
                    journal,
                    {
                        "event": "subprocess_failed",
                        "model": model,
                        "returncode": process.returncode,
                        "log": str(log_path),
                    },
                )
                continue
            try:
                validate_artifact(
                    output,
                    model=model,
                    plan=plan,
                    cache_root=cache_root,
                )
            except Exception as error:
                failures.append(model)
                _append_journal(
                    journal,
                    {
                        "event": "validation_failed",
                        "model": model,
                        "error": str(error),
                        "log": str(log_path),
                    },
                )
                continue
            completed += 1
            _append_journal(
                journal,
                {
                    "event": "completed",
                    "model": model,
                    "artifact": str(output),
                    "artifact_sha256": _sha256(output),
                },
            )
        finally:
            if lock_path.exists():
                lock_path.unlink()
    print(
        f"shard {args.shard_index}: completed={completed} skipped={skipped} "
        f"failed={len(failures)}",
        flush=True,
    )
    if failures:
        print("failed models: " + ",".join(failures), file=sys.stderr, flush=True)
        return 1
    return 0


def _sign_flip_p_value(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    values: list[float] = []
    for bits in product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(float(np.mean(differences * np.asarray(bits)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def _subject_seed_matrix(artifact: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {str(subject): {} for subject in SUBJECTS}
    for record in artifact["records"]:
        result[str(record["subject"])][str(record["seed"])] = float(
            record["test"]["metrics"]["balanced_accuracy"]
        )
    return result


def _status(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    plan = strict_load(results_root / "plan.json")
    valid: dict[str, dict[str, Any]] = {}
    invalid: dict[str, str] = {}
    for model in ARCHITECTURES:
        path = results_root / "artifacts" / f"{model}.json"
        if not path.exists():
            continue
        try:
            valid[model] = validate_artifact(
                path, model=model, plan=plan, cache_root=cache_root
            )
        except Exception as error:
            invalid[model] = str(error)
    ranking = sorted(
        (
            (model, float(artifact["summary"][model]["balanced_accuracy_mean"]))
            for model, artifact in valid.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    active_locks = sorted(
        path.name.removesuffix(".lock.json")
        for path in (results_root / "locks").glob("*.lock.json")
    )
    print(
        json.dumps(
            {
                "complete": len(valid),
                "total": len(ARCHITECTURES),
                "remaining": len(ARCHITECTURES) - len(valid),
                "active_locks": active_locks,
                "invalid": invalid,
                "provisional_ranking": [
                    {"model": model, "balanced_accuracy": score}
                    for model, score in ranking
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if invalid else 0


def _finalize(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    plan = strict_load(results_root / "plan.json")
    artifacts: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    for model in ARCHITECTURES:
        path = results_root / "artifacts" / f"{model}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing completed artifact: {path}")
        artifacts[model] = validate_artifact(
            path, model=model, plan=plan, cache_root=cache_root
        )
        paths.append(path)
    analysis = analyze(
        paths,
        reference=str(args.reference),
        bootstrap_repetitions=int(args.bootstrap_repetitions),
        random_seed=int(args.random_seed),
    )
    dataset_results = analysis["datasets"][DATASET]
    ranking_names = sorted(
        ARCHITECTURES,
        key=lambda model: (-float(dataset_results[model]["mean"]), model),
    )
    top_mean = float(dataset_results[ranking_names[0]]["mean"])
    co_winners = [
        model
        for model in ranking_names
        if abs(float(dataset_results[model]["mean"]) - top_mean) <= 1e-12
    ]
    runner_up = next(model for model in ranking_names if model not in co_winners)
    winner = co_winners[0]
    winner_subjects = dataset_results[winner]["subject_balanced_accuracy"]
    runner_subjects = dataset_results[runner_up]["subject_balanced_accuracy"]
    differences = np.asarray(
        [
            float(winner_subjects[str(subject)])
            - float(runner_subjects[str(subject)])
            for subject in SUBJECTS
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(int(args.random_seed))
    indices = generator.integers(
        0,
        len(differences),
        size=(int(args.bootstrap_repetitions), len(differences)),
    )
    paired_bootstrap = differences[indices].mean(axis=1)
    rows: list[dict[str, Any]] = []
    for rank, model in enumerate(ranking_names, start=1):
        artifact = artifacts[model]
        parameter_counts = {
            int(record["fit"]["parameter_count"]) for record in artifact["records"]
        }
        if len(parameter_counts) != 1:
            raise ValueError(f"{model} parameter count changes across local records")
        fit_seconds = sum(
            float(record["fit"]["selection_fit_seconds"])
            + float(record["fit"]["refit_fit_seconds"])
            for record in artifact["records"]
        )
        values = [
            float(dataset_results[model]["subject_balanced_accuracy"][str(subject)])
            for subject in SUBJECTS
        ]
        rows.append(
            {
                "rank": rank,
                "model": model,
                "balanced_accuracy_mean": float(dataset_results[model]["mean"]),
                "balanced_accuracy_standard_deviation": float(
                    dataset_results[model]["standard_deviation"]
                ),
                "bootstrap_95_ci": dataset_results[model]["bootstrap_95_ci"],
                "minimum_subject_balanced_accuracy": min(values),
                "median_subject_balanced_accuracy": float(np.median(values)),
                "subject_balanced_accuracy": dataset_results[model][
                    "subject_balanced_accuracy"
                ],
                "subject_seed_balanced_accuracy": _subject_seed_matrix(artifact),
                "parameter_count": parameter_counts.pop(),
                "total_fit_seconds": fit_seconds,
                "artifact": str(paths[ARCHITECTURES.index(model)]),
                "artifact_sha256": _sha256(paths[ARCHITECTURES.index(model)]),
            }
        )
    result = {
        "schema": RESULT_SCHEMA,
        "created_at": _utc_now(),
        "evidence_scope": "development_only_not_confirmation",
        "confirmation_evidence": False,
        "selection_warning": (
            "Winner-versus-runner-up inference is descriptive and post-selection; "
            "recording 4 was previously opened during project development."
        ),
        "primary_metric": plan["primary_metric"],
        "winner": winner,
        "co_winners": co_winners,
        "runner_up": runner_up,
        "winner_balanced_accuracy": top_mean,
        "winner_vs_runner_up": {
            "mean_delta": float(differences.mean()),
            "median_delta": float(np.median(differences)),
            "wins_ties_losses": [
                int(np.sum(differences > 0.0)),
                int(np.sum(differences == 0.0)),
                int(np.sum(differences < 0.0)),
            ],
            "paired_subject_bootstrap_95_ci": [
                float(value)
                for value in np.quantile(paired_bootstrap, (0.025, 0.975))
            ],
            "exact_sign_flip_p_descriptive": _sign_flip_p_value(differences),
        },
        "ranking": rows,
        "reference_analysis": analysis,
        "plan_sha256": _sha256(results_root / "plan.json"),
    }
    output_json = Path(args.output_json).resolve()
    output_markdown = Path(args.output_markdown).resolve()
    _write_json_exclusive(output_json, result)
    lines = [
        "# Local Exp4 all-architecture tournament",
        "",
        "> Development-only result. Recording 4 has previously been inspected; this is not independent confirmation evidence.",
        "",
        f"**Numerical winner:** `{winner}` — {100.0 * top_mean:.3f}% mean balanced accuracy.",
        "",
        f"Runner-up: `{runner_up}`. Mean paired margin: {100.0 * differences.mean():+.3f} percentage points; "
        f"95% paired subject-bootstrap interval [{100.0 * np.quantile(paired_bootstrap, 0.025):+.3f}, "
        f"{100.0 * np.quantile(paired_bootstrap, 0.975):+.3f}] points.",
        "",
        "Winner-versus-runner-up statistics are post-selection and descriptive.",
        "",
        "| Rank | Model | Mean BA | Subject SD | 95% bootstrap CI | Parameters |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        low, high = row["bootstrap_95_ci"]
        lines.append(
            f"| {row['rank']} | `{row['model']}` | "
            f"{100.0 * row['balanced_accuracy_mean']:.3f}% | "
            f"{100.0 * row['balanced_accuracy_standard_deviation']:.3f}% | "
            f"[{100.0 * low:.3f}%, {100.0 * high:.3f}%] | "
            f"{row['parameter_count']:,} |"
        )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    with output_markdown.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"winner={winner} balanced_accuracy={top_mean:.6f}")
    return 0


def _add_common_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--device", default=TrainConfig.device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--cache-root", required=True)
    preflight.add_argument("--models")
    preflight.add_argument("--device", default=TrainConfig.device)
    preflight.add_argument("--output", required=True)
    preflight.set_defaults(handler=_preflight)

    run = commands.add_parser("run")
    run.add_argument("--cache-root", required=True)
    run.add_argument("--results-root", required=True)
    run.add_argument("--models")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--recover-stale", action="store_true")
    _add_common_training_arguments(run)
    run.set_defaults(handler=_run)

    status = commands.add_parser("status")
    status.add_argument("--cache-root", required=True)
    status.add_argument("--results-root", required=True)
    status.set_defaults(handler=_status)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--cache-root", required=True)
    finalize.add_argument("--results-root", required=True)
    finalize.add_argument("--reference", default="fbmsnet")
    finalize.add_argument("--bootstrap-repetitions", type=int, default=100_000)
    finalize.add_argument("--random-seed", type=int, default=20260721)
    finalize.add_argument("--output-json", required=True)
    finalize.add_argument("--output-markdown", required=True)
    finalize.set_defaults(handler=_finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
