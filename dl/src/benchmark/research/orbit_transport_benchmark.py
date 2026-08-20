"""Locked development and one-shot confirmation runner for OrbitTransportNet.

Development is restricted to local recordings 1--3, Cho2017 subjects 1--26,
and BNCI2014-001 subjects 1--4.  Local recording 4 is never passed to the
loader.  BNCI subjects 5--9 remain behind an explicit token, frozen
configuration/source/development-artifact manifest, exact output pin, permanent
study-wide access receipt, exclusive output lock, and ordered-cohort contract.

Candidate test metrics are descriptive only.  Candidate and checkpoint
selection occur exclusively on the protected validation split inside
``OrbitTransportClassifier.fit``.  Every record retains exact data/source
provenance, all candidate logits and metrics, expert/orientation diagnostics,
invariant routing diagnostics, parameter counts, and signed-action errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .cameo_benchmark import _git_state, _stratified_val
from .config import DEFAULT_DATA_CONFIG, PROJECT_ROOT, SUBJECT_RUNS, VALID_SUBJECTS
from .data import SessionKey, load_sessions
from .external_bnci2014 import (
    BNCI_CHANNELS,
    CONFIRMATION_SUBJECTS,
    DEVELOPMENT_SUBJECTS,
    load_bnci2014_subject,
    protocol_metadata,
)
from .external_cho2017 import load_cho_subject
from .orbit_transport_net import (
    OrbitTransportClassifier,
    OrbitTransportConfig,
    OrbitTransportNet,
    OrbitTransportOutput,
    _configure_torch_determinism,
)
from .parity_fuse_benchmark import (
    _OutputLock,
    _array_sha256,
    _atomic_json,
    _bnci_provenance,
    _cho_provenance,
    _completed_keys,
    _data_manifest,
    _exclusive_json_create,
    _expected_record_keys,
    _file_sha256,
    _json_sha256,
    _local_provenance,
    _metrics,
    _parse_ints,
    _probabilities_from_logit,
    _sanitized_command,
    _split_manifest,
    _update_completion,
    _utc_now,
    _environment,
)

SCHEMA_VERSION = 2
FROZEN_MANIFEST_SCHEMA_VERSION = 2
CONFIRMATION_RECEIPT_SCHEMA_VERSION = 1
ARCHITECTURE = "OrbitTransportNet"
MODES = ("local-dev", "cho-dev", "bnci-dev", "bnci-confirm")
CONFIRMATION_TOKEN = "CONFIRM-BNCI2014-001-S5-9-ORBIT-ONCE"
CONFIRMATION_STUDY_ID = "eegthingy-orbit-bnci2014-001-s5-9-final-v1"
CONFIRMATION_RECEIPT_PATH = (
    PROJECT_ROOT
    / "output"
    / "research"
    / "parity"
    / ".confirmation-receipts"
    / f"{CONFIRMATION_STUDY_ID}.json"
)
PROTOCOLS = {
    "local-dev": "local_rec1-fit_rec2-select_rec3-development_rec4-not-loaded",
    "cho-dev": "cho2017_s1-26_inner-select_outer-5fold-development",
    "bnci-dev": "bnci2014-001_s1-4_T-runs0-4-fit_T-run5-select_E-development",
    "bnci-confirm": "bnci2014-001_s5-9_T-runs0-4-fit_T-run5-select_E-confirmation",
}

_SOURCE_FILES = (
    "src/benchmark/research/__init__.py",
    "src/benchmark/research/orbit_transport_benchmark.py",
    "src/benchmark/research/orbit_transport_net.py",
    "src/benchmark/research/cameo_net.py",
    "src/benchmark/research/cameo_benchmark.py",
    # Audited atomic/provenance primitives imported above are source-pinned too.
    "src/benchmark/research/parity_fuse_benchmark.py",
    # Importing the benchmark executes this first-party dependency as well.
    "src/benchmark/research/parity_fuse_net.py",
    "src/benchmark/shared/augment.py",
    "src/benchmark/research/config.py",
    "src/benchmark/research/data.py",
    "src/benchmark/research/external_cho2017.py",
    "src/benchmark/research/external_bnci2014.py",
    "src/benchmark/shared/spd.py",
)


def _source_manifest() -> dict[str, str]:
    return {name: _file_sha256(PROJECT_ROOT / name) for name in _SOURCE_FILES}


def _json_config(config: OrbitTransportConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _run_environment(config: OrbitTransportConfig) -> dict[str, Any]:
    """Environment fingerprint used both for provenance and safe resume."""

    environment = dict(_environment())
    warn_only = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: None
    )()
    environment["determinism"] = {
        "requested": bool(config.deterministic),
        "algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": warn_only,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    return environment


def _validate_common_contract(
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("subjects must be non-empty and unique")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if folds < 2:
        raise ValueError("folds must be at least two")
    if mode == "local-dev":
        allowed = set(VALID_SUBJECTS)
    elif mode == "cho-dev":
        allowed = set(range(1, 27))
    elif mode == "bnci-dev":
        allowed = set(DEVELOPMENT_SUBJECTS)
    else:
        allowed = set(CONFIRMATION_SUBJECTS)
    if not set(subjects).issubset(allowed):
        raise ValueError(
            f"{mode} subjects must stay inside the locked partition "
            f"{min(allowed)}--{max(allowed)}"
        )


def _experiment_contract(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: OrbitTransportConfig,
    frozen_manifest_sha256: str | None,
) -> dict[str, Any]:
    return {
        "architecture_name": ARCHITECTURE,
        "mode": mode,
        "protocol": PROTOCOLS[mode],
        "subjects": list(subjects),
        "seeds": list(seeds),
        "folds": int(folds) if mode == "cho-dev" else None,
        "config": _json_config(config),
        "frozen_manifest_sha256": frozen_manifest_sha256,
    }


def _new_payload(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: OrbitTransportConfig,
    frozen_manifest: Mapping[str, Any] | None,
    frozen_manifest_path: Path | None,
    frozen_manifest_sha256: str | None,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    source = _source_manifest()
    contract = _experiment_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        folds=folds,
        config=config,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    expected = _expected_record_keys(subjects, seeds)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": None,
        "architecture": {
            "name": ARCHITECTURE,
            "implementation": "benchmark.research.orbit_transport_net.OrbitTransportNet",
            "trainable_parameter_contract_lt": 50_000,
            "exact_z2_label_anti_equivariance": True,
        },
        **contract,
        "contract_sha256": _json_sha256(contract),
        "confirmation_access": mode == "bnci-confirm",
        "confirmation": (
            {
                "token_verified": True,
                "frozen_manifest_path": str(frozen_manifest_path.resolve()),
                "frozen_manifest_sha256": frozen_manifest_sha256,
                "development_artifact": frozen_manifest["development_artifact"],
                "confirmation_contract": frozen_manifest["confirmation_contract"],
            }
            if frozen_manifest is not None and frozen_manifest_path is not None
            else None
        ),
        "candidate_test_metrics_are_descriptive_only": True,
        "candidate_selection_source": "validation_only",
        "command": _sanitized_command(),
        "repository": _git_state(),
        "source_sha256": source,
        "environment": dict(environment),
        "records": [],
        "summary": {},
        "completion": {
            "status": "running",
            "expected_record_keys": expected,
            "actual_record_keys": [],
            "expected_records": len(expected),
            "actual_records": 0,
            "complete": False,
        },
    }


def _load_or_initialize(
    output: Path,
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: OrbitTransportConfig,
    resume: bool,
    frozen_manifest: Mapping[str, Any] | None,
    frozen_manifest_path: Path | None,
    frozen_manifest_sha256: str | None,
) -> dict[str, Any]:
    current_environment = _run_environment(config)
    expected_contract = _experiment_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        folds=folds,
        config=config,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing result: {output}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        observed = {key: payload.get(key) for key in expected_contract}
        if observed != expected_contract:
            raise ValueError(
                "cannot resume an output with a different experiment contract"
            )
        if payload.get("contract_sha256") != _json_sha256(expected_contract):
            raise ValueError("cannot resume an output with a corrupt contract digest")
        if payload.get("source_sha256") != _source_manifest():
            raise ValueError("cannot resume after benchmark source files have changed")
        if payload.get("environment") != current_environment:
            raise ValueError("cannot resume in a different execution environment")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("cannot resume an unsupported result schema")
        if mode == "bnci-confirm":
            if frozen_manifest is None or frozen_manifest_path is None:
                raise ValueError("confirmation resume lacks its frozen manifest")
            expected_confirmation = {
                "token_verified": True,
                "frozen_manifest_path": str(frozen_manifest_path.resolve()),
                "frozen_manifest_sha256": frozen_manifest_sha256,
                "development_artifact": frozen_manifest["development_artifact"],
                "confirmation_contract": frozen_manifest["confirmation_contract"],
            }
            if (
                payload.get("confirmation_access") is not True
                or payload.get("confirmation") != expected_confirmation
            ):
                raise ValueError("confirmation resume metadata is corrupt")
        _update_completion(payload, status="running")
        return payload
    if resume:
        raise FileNotFoundError(f"--resume requires an existing result: {output}")
    return _new_payload(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        folds=folds,
        config=config,
        frozen_manifest=frozen_manifest,
        frozen_manifest_path=frozen_manifest_path,
        frozen_manifest_sha256=frozen_manifest_sha256,
        environment=current_environment,
    )


def _tensor_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "abs_mean": float(np.mean(np.abs(array))),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _gate_summary(weights: np.ndarray) -> dict[str, Any]:
    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError("fusion weights must have shape (N, 3)")
    clipped = np.clip(weights, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    result: dict[str, Any] = {
        "shape": list(weights.shape),
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
    }
    for index, name in enumerate(("energy", "dynamics", "anchor")):
        result[name] = _tensor_stats(weights[:, index])
    return result


def _model_output(
    classifier: OrbitTransportClassifier,
    raw: np.ndarray,
    covariances: np.ndarray,
    *,
    swapped: bool = False,
) -> OrbitTransportOutput:
    views = classifier._prepare_views(raw, covariances)
    if swapped:
        views = (views[1], views[0], views[3], views[2])
    tensors = [torch.from_numpy(value).to(classifier.device_) for value in views]
    classifier.model_.eval()
    with torch.no_grad():
        return classifier.model_(*tensors)


def _prediction_trace(
    labels: np.ndarray,
    probabilities: np.ndarray,
    candidate_probabilities: Mapping[str, np.ndarray],
    output: OrbitTransportOutput,
    selected_candidate: str,
    provenance: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    labels = np.asarray(labels)
    if len(provenance) != len(labels):
        raise ValueError("provenance length does not match predictions")
    candidate_logits = {
        name: value.detach().cpu().numpy()
        for name, value in output.candidate_logits().items()
    }
    gate = output.fusion_weights.detach().cpu().numpy()
    energy_views = output.energy_view_logits.detach().cpu().numpy()
    dynamics_views = output.dynamics_view_logits.detach().cpu().numpy()
    energy_alpha = output.energy_alpha.detach().cpu().numpy()
    dynamics_alpha = output.dynamics_alpha.detach().cpu().numpy()
    energy_q = output.energy_orientation.detach().cpu().numpy()
    dynamics_q = output.dynamics_orientation.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for index in range(len(labels)):
        rows.append(
            {
                "window_index": int(index),
                "provenance": dict(provenance[index]),
                "label": int(labels[index]),
                "selected_candidate": selected_candidate,
                "probability_left": float(probabilities[index, 0]),
                "probability_right": float(probabilities[index, 1]),
                "candidate_probability_right": {
                    name: float(value[index, 1])
                    for name, value in candidate_probabilities.items()
                },
                "candidate_logits": {
                    name: float(value[index])
                    for name, value in candidate_logits.items()
                },
                "fusion_weights": {
                    "energy": float(gate[index, 0]),
                    "dynamics": float(gate[index, 1]),
                    "anchor": float(gate[index, 2]),
                },
                "orientation": {
                    "energy_alpha": float(energy_alpha[index]),
                    "dynamics_alpha": float(dynamics_alpha[index]),
                    "energy_score": float(energy_q[index]),
                    "dynamics_score": float(dynamics_q[index]),
                },
                "view_logits": {
                    "energy": energy_views[index].astype(float).tolist(),
                    "dynamics": dynamics_views[index].astype(float).tolist(),
                },
            }
        )
    return rows


def _fit_score(
    *,
    raw_train: np.ndarray,
    cov_train: np.ndarray,
    y_train: np.ndarray,
    raw_validation: np.ndarray,
    cov_validation: np.ndarray,
    y_validation: np.ndarray,
    raw_test: np.ndarray,
    cov_test: np.ndarray,
    y_test: np.ndarray,
    channels: Sequence[str],
    config: OrbitTransportConfig,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    dict[str, np.ndarray],
    OrbitTransportOutput,
]:
    classifier = OrbitTransportClassifier(config).fit(
        raw_train,
        cov_train,
        y_train,
        raw_validation,
        cov_validation,
        y_validation,
        channels=channels,
    )
    output = _model_output(classifier, raw_test, cov_test)
    reflected = _model_output(classifier, raw_test, cov_test, swapped=True)
    logits = {
        name: value.detach().cpu().numpy()
        for name, value in output.candidate_logits().items()
    }
    reflected_logits = {
        name: value.detach().cpu().numpy()
        for name, value in reflected.candidate_logits().items()
    }
    candidate_probabilities = {
        name: _probabilities_from_logit(value) for name, value in logits.items()
    }
    selected = classifier.selected_candidate_
    probability = candidate_probabilities[selected]
    gate = output.fusion_weights.detach().cpu().numpy()
    reflected_gate = reflected.fusion_weights.detach().cpu().numpy()
    energy_alpha = output.energy_alpha.detach().cpu().numpy()
    reflected_energy_alpha = reflected.energy_alpha.detach().cpu().numpy()
    dynamics_alpha = output.dynamics_alpha.detach().cpu().numpy()
    reflected_dynamics_alpha = reflected.dynamics_alpha.detach().cpu().numpy()
    energy_views = output.energy_view_logits.detach().cpu().numpy()
    reflected_energy_views = reflected.energy_view_logits.detach().cpu().numpy()
    dynamics_views = output.dynamics_view_logits.detach().cpu().numpy()
    reflected_dynamics_views = reflected.dynamics_view_logits.detach().cpu().numpy()
    per_candidate_errors = {
        name: float(np.max(np.abs(value + reflected_logits[name])))
        for name, value in logits.items()
    }
    candidate_metrics = {
        name: _metrics(y_test, value) for name, value in candidate_probabilities.items()
    }
    detail = {
        "selected_candidate": selected,
        "selection_source": "validation_only",
        "selection_trace": classifier.selection_trace_,
        "best_epoch": int(classifier.best_epoch_),
        "epochs_run": int(classifier.epochs_run_),
        "best_validation_loss": float(classifier.best_validation_loss_),
        "best_validation_balanced_accuracy": float(
            classifier.best_validation_balanced_accuracy_
        ),
        "parameter_count": int(classifier.param_count_),
        "trainable_parameter_count": int(classifier.trainable_param_count_),
        "neural_parameter_count": int(classifier.param_count_),
        "neural_trainable_parameter_count": int(classifier.trainable_param_count_),
        "convex_anchor_learned_parameter_count": int(
            classifier.convex_anchor_learned_param_count_
        ),
        "total_learned_parameter_count": int(classifier.total_learned_param_count_),
        "train_seconds": float(classifier.train_seconds_),
        "effective_seed": int(config.seed),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "n_test": int(len(y_test)),
        "candidate_test_metrics_descriptive_only": candidate_metrics,
        "equivariance": {
            "max_abs_selected_logit_sum": per_candidate_errors[selected],
            "max_abs_candidate_logit_sum": per_candidate_errors,
            "max_abs_gate_difference": float(np.max(np.abs(gate - reflected_gate))),
            "max_abs_energy_alpha_complement_error": float(
                np.max(np.abs(energy_alpha + reflected_energy_alpha - 1.0))
            ),
            "max_abs_dynamics_alpha_complement_error": float(
                np.max(np.abs(dynamics_alpha + reflected_dynamics_alpha - 1.0))
            ),
            "max_abs_energy_view_swap_error": float(
                np.max(np.abs(energy_views - reflected_energy_views[:, ::-1]))
            ),
            "max_abs_dynamics_view_swap_error": float(
                np.max(np.abs(dynamics_views - reflected_dynamics_views[:, ::-1]))
            ),
        },
        "experts": {
            "energy_transported": _tensor_stats(logits["energy"]),
            "dynamics_transported": _tensor_stats(logits["dynamics"]),
            "anchor_projected": _tensor_stats(logits["anchor"]),
            "energy_view_first": _tensor_stats(energy_views[:, 0]),
            "energy_view_reflected": _tensor_stats(energy_views[:, 1]),
            "dynamics_view_first": _tensor_stats(dynamics_views[:, 0]),
            "dynamics_view_reflected": _tensor_stats(dynamics_views[:, 1]),
        },
        "orientation": {
            "energy_alpha": _tensor_stats(energy_alpha),
            "dynamics_alpha": _tensor_stats(dynamics_alpha),
            "energy_score": _tensor_stats(
                output.energy_orientation.detach().cpu().numpy()
            ),
            "dynamics_score": _tensor_stats(
                output.dynamics_orientation.detach().cpu().numpy()
            ),
        },
        "routing": _gate_summary(gate),
        "source_only_preprocessing": True,
        "unlabeled_test_calibration": False,
    }
    return detail, probability, candidate_probabilities, output


def _record(
    *,
    subject: int,
    seed: int,
    metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, Mapping[str, float]],
    fits: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    data_sha256: Mapping[str, Any],
    protocol: Mapping[str, Any] | str,
) -> dict[str, Any]:
    anchor_metrics = candidate_metrics["anchor"]
    return {
        "subject": int(subject),
        "seed": int(seed),
        "protocol": protocol,
        "data_sha256": data_sha256,
        "metrics": dict(metrics),
        "anchor_metrics": dict(anchor_metrics),
        "candidate_test_metrics_descriptive_only": {
            name: dict(value) for name, value in candidate_metrics.items()
        },
        "delta_balanced_accuracy_vs_anchor": float(
            metrics["balanced_accuracy"] - anchor_metrics["balanced_accuracy"]
        ),
        "fits": [dict(fit) for fit in fits],
        "predictions": [dict(row) for row in predictions],
    }


def _summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = list(payload["records"])
    by_subject: dict[int, list[float]] = {}
    anchor_by_subject: dict[int, list[float]] = {}
    candidate_by_subject: dict[str, dict[int, list[float]]] = {}
    fits: list[Mapping[str, Any]] = []
    for row in records:
        subject = int(row["subject"])
        by_subject.setdefault(subject, []).append(
            float(row["metrics"]["balanced_accuracy"])
        )
        anchor_by_subject.setdefault(subject, []).append(
            float(row["anchor_metrics"]["balanced_accuracy"])
        )
        for name, metrics in row["candidate_test_metrics_descriptive_only"].items():
            candidate_by_subject.setdefault(name, {}).setdefault(subject, []).append(
                float(metrics["balanced_accuracy"])
            )
        fits.extend(row["fits"])
    values = np.asarray(
        [np.mean(by_subject[key]) for key in sorted(by_subject)], dtype=np.float64
    )
    anchors = np.asarray(
        [np.mean(anchor_by_subject[key]) for key in sorted(anchor_by_subject)],
        dtype=np.float64,
    )
    selected = [str(fit["selected_candidate"]) for fit in fits]
    equivariance = [
        float(fit["equivariance"]["max_abs_selected_logit_sum"]) for fit in fits
    ]
    gate_errors = [
        float(fit["equivariance"]["max_abs_gate_difference"]) for fit in fits
    ]
    return {
        "balanced_accuracy_mean": float(values.mean()) if len(values) else None,
        "balanced_accuracy_std": (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ),
        "anchor_balanced_accuracy_mean": (
            float(anchors.mean()) if len(anchors) else None
        ),
        "mean_delta_vs_anchor": (
            float((values - anchors).mean()) if len(values) else None
        ),
        "candidate_balanced_accuracy_means_descriptive_only": {
            name: float(np.mean([np.mean(rows[key]) for key in sorted(rows)]))
            for name, rows in sorted(candidate_by_subject.items())
        },
        "selected_candidate_counts": {
            name: selected.count(name) for name in sorted(set(selected))
        },
        "n_participants": int(len(values)),
        "n_records": int(len(records)),
        "n_fits": int(len(fits)),
        "parameter_counts": sorted({int(fit["parameter_count"]) for fit in fits}),
        "trainable_parameter_counts": sorted(
            {int(fit["trainable_parameter_count"]) for fit in fits}
        ),
        "neural_parameter_counts": sorted(
            {int(fit["neural_parameter_count"]) for fit in fits}
        ),
        "convex_anchor_learned_parameter_counts": sorted(
            {int(fit["convex_anchor_learned_parameter_count"]) for fit in fits}
        ),
        "total_learned_parameter_counts": sorted(
            {int(fit["total_learned_parameter_count"]) for fit in fits}
        ),
        "best_epoch_counts": {
            str(epoch): sum(int(fit["best_epoch"]) == epoch for fit in fits)
            for epoch in sorted({int(fit["best_epoch"]) for fit in fits})
        },
        "max_selected_equivariance_error": (
            max(equivariance) if equivariance else None
        ),
        "max_gate_invariance_error": max(gate_errors) if gate_errors else None,
        "mean_routing_weights": {
            name: (
                float(np.mean([fit["routing"][name]["mean"] for fit in fits]))
                if fits
                else None
            )
            for name in ("energy", "dynamics", "anchor")
        },
    }


def _save_payload(output: Path, payload: dict[str, Any], *, status: str) -> None:
    payload["updated_at"] = _utc_now()
    payload["summary"] = _summarize(payload)
    _update_completion(payload, status=status)
    if status == "complete" and not payload["completion"]["complete"]:
        raise RuntimeError("refusing to mark an incomplete result complete")
    _atomic_json(output, payload)


def _run_local(
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: OrbitTransportConfig,
    output: Path,
    payload: dict[str, Any],
) -> None:
    completed = _completed_keys(payload)
    if all((int(s), int(seed)) in completed for s in subjects for seed in seeds):
        return
    # Physical isolation: recording 4 is never named or loaded.
    keys = [
        SessionKey(subject, run)
        for subject in subjects
        for run in SUBJECT_RUNS[subject][:3]
    ]
    data = load_sessions(keys, DEFAULT_DATA_CONFIG)
    task = data.labels >= 0
    for subject in subjects:
        runs = list(SUBJECT_RUNS[subject][:3])

        def rows(run: int) -> np.ndarray:
            return np.flatnonzero(
                (data.subject_ids == subject) & (data.run_ids == run) & task
            )

        train, validation, test = rows(runs[0]), rows(runs[1]), rows(runs[2])
        metadata = {
            "subject_ids": data.subject_ids,
            "session_ids": data.session_ids,
            "run_ids": data.run_ids,
            "trial_ids": data.trial_ids,
        }
        manifest = _data_manifest(
            {
                "train": _split_manifest(
                    raw=data.broadband_epochs,
                    covariances=data.covariances,
                    labels=data.labels,
                    indices=train,
                    metadata=metadata,
                ),
                "validation": _split_manifest(
                    raw=data.broadband_epochs,
                    covariances=data.covariances,
                    labels=data.labels,
                    indices=validation,
                    metadata=metadata,
                ),
                "test": _split_manifest(
                    raw=data.broadband_epochs,
                    covariances=data.covariances,
                    labels=data.labels,
                    indices=test,
                    metadata=metadata,
                ),
            }
        )
        for seed in seeds:
            if (int(subject), int(seed)) in completed:
                continue
            detail, probability, candidates, model_output = _fit_score(
                raw_train=data.broadband_epochs[train],
                cov_train=data.covariances[train],
                y_train=data.labels[train],
                raw_validation=data.broadband_epochs[validation],
                cov_validation=data.covariances[validation],
                y_validation=data.labels[validation],
                raw_test=data.broadband_epochs[test],
                cov_test=data.covariances[test],
                y_test=data.labels[test],
                channels=DEFAULT_DATA_CONFIG.channels,
                config=replace(config, seed=int(seed)),
            )
            truth = data.labels[test]
            candidate_metrics = {
                name: _metrics(truth, value) for name, value in candidates.items()
            }
            row = _record(
                subject=subject,
                seed=seed,
                metrics=_metrics(truth, probability),
                candidate_metrics=candidate_metrics,
                fits=[detail],
                predictions=_prediction_trace(
                    truth,
                    probability,
                    candidates,
                    model_output,
                    detail["selected_candidate"],
                    _local_provenance(data, test),
                ),
                data_sha256=manifest,
                protocol=PROTOCOLS["local-dev"],
            )
            payload["records"].append(row)
            _save_payload(output, payload, status="running")
            print(
                f"{ARCHITECTURE} local-dev S{subject} seed={seed}: "
                f"bacc={100.0 * row['metrics']['balanced_accuracy']:.2f}% "
                f"candidate={detail['selected_candidate']}",
                flush=True,
            )


def _run_cho(
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: OrbitTransportConfig,
    output: Path,
    payload: dict[str, Any],
) -> None:
    completed = _completed_keys(payload)
    for subject in subjects:
        pending = [seed for seed in seeds if (int(subject), int(seed)) not in completed]
        if not pending:
            continue
        data = load_cho_subject(subject)
        raw = np.asarray(data["broadband"])
        cov = np.asarray(data["covariances"])
        labels = np.asarray(data["labels"])
        manifest = _data_manifest(
            {
                "all_trials_before_fixed_outer_split": {
                    key: np.asarray(value) for key, value in data.items()
                }
            }
        )
        splits = list(
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=0).split(
                np.zeros(len(labels)), labels
            )
        )
        for seed in pending:
            all_truth: list[np.ndarray] = []
            all_probability: list[np.ndarray] = []
            all_candidates: dict[str, list[np.ndarray]] = {
                name: [] for name in OrbitTransportNet.CANDIDATE_NAMES
            }
            fits: list[dict[str, Any]] = []
            predictions: list[dict[str, Any]] = []
            trace_offset = 0
            for fold, (outer_train, test) in enumerate(splits):
                train_rel, validation_rel = _stratified_val(
                    labels[outer_train], int(seed) + fold
                )
                train = outer_train[train_rel]
                validation = outer_train[validation_rel]
                detail, probability, candidates, model_output = _fit_score(
                    raw_train=raw[train],
                    cov_train=cov[train],
                    y_train=labels[train],
                    raw_validation=raw[validation],
                    cov_validation=cov[validation],
                    y_validation=labels[validation],
                    raw_test=raw[test],
                    cov_test=cov[test],
                    y_test=labels[test],
                    channels=DEFAULT_DATA_CONFIG.channels,
                    config=replace(config, seed=int(seed) + fold),
                )
                detail.update(
                    {
                        "fold": int(fold),
                        "outer_train_indices_sha256": _array_sha256(outer_train),
                        "train_indices_sha256": _array_sha256(train),
                        "validation_indices_sha256": _array_sha256(validation),
                        "test_indices_sha256": _array_sha256(test),
                    }
                )
                fold_trace = _prediction_trace(
                    labels[test],
                    probability,
                    candidates,
                    model_output,
                    detail["selected_candidate"],
                    _cho_provenance(test, fold),
                )
                for trace in fold_trace:
                    trace["window_index"] += trace_offset
                trace_offset += len(fold_trace)
                predictions.extend(fold_trace)
                fits.append(detail)
                all_truth.append(labels[test])
                all_probability.append(probability)
                for name, value in candidates.items():
                    all_candidates[name].append(value)
            truth = np.concatenate(all_truth)
            probability = np.concatenate(all_probability)
            candidates = {
                name: np.concatenate(parts) for name, parts in all_candidates.items()
            }
            candidate_metrics = {
                name: _metrics(truth, value) for name, value in candidates.items()
            }
            row = _record(
                subject=subject,
                seed=seed,
                metrics=_metrics(truth, probability),
                candidate_metrics=candidate_metrics,
                fits=fits,
                predictions=predictions,
                data_sha256=manifest,
                protocol={
                    "protocol_id": PROTOCOLS["cho-dev"],
                    "folds": int(folds),
                    "outer_split_random_state": 0,
                    "inner_validation_fraction": 0.2,
                },
            )
            payload["records"].append(row)
            _save_payload(output, payload, status="running")
            print(
                f"{ARCHITECTURE} cho-dev S{subject} seed={seed}: "
                f"bacc={100.0 * row['metrics']['balanced_accuracy']:.2f}%",
                flush=True,
            )


def _run_bnci(
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: OrbitTransportConfig,
    output: Path,
    payload: dict[str, Any],
) -> None:
    completed = _completed_keys(payload)
    for subject in subjects:
        pending = [seed for seed in seeds if (int(subject), int(seed)) not in completed]
        if not pending:
            continue
        data = load_bnci2014_subject(subject)
        manifest = _data_manifest(data)
        train = data["train"]
        validation = data["validation"]
        test = data["test"]
        for seed in pending:
            detail, probability, candidates, model_output = _fit_score(
                raw_train=train["broadband"],
                cov_train=train["covariances"],
                y_train=train["labels"],
                raw_validation=validation["broadband"],
                cov_validation=validation["covariances"],
                y_validation=validation["labels"],
                raw_test=test["broadband"],
                cov_test=test["covariances"],
                y_test=test["labels"],
                channels=BNCI_CHANNELS,
                config=replace(config, seed=int(seed)),
            )
            detail.update(
                {
                    "train_run_ids": sorted(set(train["run_ids"].astype(str).tolist())),
                    "validation_run_ids": sorted(
                        set(validation["run_ids"].astype(str).tolist())
                    ),
                    "test_run_ids": sorted(set(test["run_ids"].astype(str).tolist())),
                }
            )
            truth = test["labels"]
            candidate_metrics = {
                name: _metrics(truth, value) for name, value in candidates.items()
            }
            row = _record(
                subject=subject,
                seed=seed,
                metrics=_metrics(truth, probability),
                candidate_metrics=candidate_metrics,
                fits=[detail],
                predictions=_prediction_trace(
                    truth,
                    probability,
                    candidates,
                    model_output,
                    detail["selected_candidate"],
                    _bnci_provenance(test["run_ids"]),
                ),
                data_sha256=manifest,
                protocol=protocol_metadata(subject),
            )
            payload["records"].append(row)
            _save_payload(output, payload, status="running")
            print(
                f"{ARCHITECTURE} {mode} S{subject} seed={seed}: "
                f"bacc={100.0 * row['metrics']['balanced_accuracy']:.2f}% "
                f"candidate={detail['selected_candidate']}",
                flush=True,
            )


def _resolve_development_artifact(manifest_path: Path, reference: str) -> Path:
    path = Path(reference).expanduser()
    return path if path.is_absolute() else manifest_path.parent / path


def _same_metrics(
    observed: Mapping[str, Any], expected: Mapping[str, float], *, name: str
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{name} metric keys are invalid")
    for key, value in expected.items():
        actual = float(observed[key])
        if not math.isfinite(actual) or not math.isclose(
            actual, float(value), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{name} metric {key} does not recompute")


def _validate_array_manifest(
    value: Any, *, shape: Sequence[int], dtype: str, name: str
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"development {name} array manifest is missing")
    digest = value.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"development {name} array digest is invalid")
    if value.get("shape") != list(shape) or value.get("dtype") != dtype:
        raise ValueError(f"development {name} array shape/dtype is invalid")


def _validate_development_record(
    record: Mapping[str, Any], *, subject: int, config: OrbitTransportConfig
) -> None:
    if int(record.get("subject", -1)) != subject or int(record.get("seed", -1)) != 7:
        raise ValueError("development record identity is invalid")
    if record.get("protocol") != protocol_metadata(subject):
        raise ValueError(f"development S{subject} protocol is invalid")
    manifests = record.get("data_sha256")
    if not isinstance(manifests, dict) or set(manifests) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(f"development S{subject} data manifest is invalid")
    expected_counts = {"train": 120, "validation": 24, "test": 144}
    for split, count in expected_counts.items():
        arrays = manifests.get(split)
        if not isinstance(arrays, dict) or set(arrays) != {
            "broadband",
            "covariances",
            "labels",
            "run_ids",
        }:
            raise ValueError(f"development S{subject} {split} manifest is invalid")
        _validate_array_manifest(
            arrays["broadband"],
            shape=(count, len(BNCI_CHANNELS), 251),
            dtype="<f4",
            name=f"S{subject}/{split}/broadband",
        )
        _validate_array_manifest(
            arrays["covariances"],
            shape=(count, 4, len(BNCI_CHANNELS), len(BNCI_CHANNELS)),
            dtype="<f4",
            name=f"S{subject}/{split}/covariances",
        )
        _validate_array_manifest(
            arrays["labels"],
            shape=(count,),
            dtype="<i8",
            name=f"S{subject}/{split}/labels",
        )
        _validate_array_manifest(
            arrays["run_ids"],
            shape=(count,),
            dtype="<U1",
            name=f"S{subject}/{split}/run_ids",
        )

    fits = record.get("fits")
    if not isinstance(fits, list) or len(fits) != 1 or not isinstance(fits[0], dict):
        raise ValueError(f"development S{subject} must contain exactly one fit")
    fit = fits[0]
    if (
        fit.get("selected_candidate") not in config.candidate_names
        or fit.get("selection_source") != "validation_only"
        or fit.get("effective_seed") != 7
        or fit.get("n_train") != 120
        or fit.get("n_validation") != 24
        or fit.get("n_test") != 144
        or fit.get("train_run_ids") != ["0", "1", "2", "3", "4"]
        or fit.get("validation_run_ids") != ["5"]
        or fit.get("test_run_ids") != ["0", "1", "2", "3", "4", "5"]
        or fit.get("source_only_preprocessing") is not True
        or fit.get("unlabeled_test_calibration") is not False
    ):
        raise ValueError(f"development S{subject} fit contract is invalid")
    selection_trace = fit.get("selection_trace")
    if (
        not isinstance(selection_trace, list)
        or len(selection_trace) != len(config.candidate_names)
        or [row.get("candidate") for row in selection_trace]
        != list(config.candidate_names)
    ):
        raise ValueError(f"development S{subject} selection trace is invalid")
    for row in selection_trace:
        if (
            not isinstance(row.get("best_epoch"), int)
            or not math.isfinite(float(row.get("validation_loss", float("nan"))))
            or not 0.0
            <= float(row.get("validation_balanced_accuracy", float("nan")))
            <= 1.0
        ):
            raise ValueError(f"development S{subject} selection trace is invalid")
    preference = {name: -index for index, name in enumerate(config.candidate_names)}
    winner = max(
        selection_trace,
        key=lambda row: (
            float(row["validation_balanced_accuracy"]),
            -float(row["validation_loss"]),
            preference[str(row["candidate"])],
        ),
    )
    if (
        winner["candidate"] != fit["selected_candidate"]
        or winner["best_epoch"] != fit.get("best_epoch")
        or winner["validation_loss"] != fit.get("best_validation_loss")
        or winner["validation_balanced_accuracy"]
        != fit.get("best_validation_balanced_accuracy")
    ):
        raise ValueError(f"development S{subject} selected checkpoint is invalid")
    for key in (
        "neural_parameter_count",
        "neural_trainable_parameter_count",
        "convex_anchor_learned_parameter_count",
        "total_learned_parameter_count",
    ):
        if not isinstance(fit.get(key), int) or int(fit[key]) <= 0:
            raise ValueError(f"development S{subject} {key} is invalid")
    if (
        fit.get("parameter_count") != fit["neural_parameter_count"]
        or fit.get("trainable_parameter_count")
        != fit["neural_trainable_parameter_count"]
        or fit["total_learned_parameter_count"]
        != fit["neural_parameter_count"] + fit["convex_anchor_learned_parameter_count"]
    ):
        raise ValueError(f"development S{subject} parameter accounting is invalid")
    equivariance = fit.get("equivariance")
    if not isinstance(equivariance, dict):
        raise ValueError(f"development S{subject} equivariance audit is missing")
    for value in equivariance.values():
        if isinstance(value, dict):
            numeric = value.values()
        else:
            numeric = (value,)
        numeric_values = [float(item) for item in numeric]
        if any(not math.isfinite(item) or abs(item) > 1e-3 for item in numeric_values):
            raise ValueError(f"development S{subject} equivariance audit is invalid")

    predictions = record.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 144:
        raise ValueError(f"development S{subject} prediction count is invalid")
    labels: list[int] = []
    selected_probability: list[list[float]] = []
    candidate_probability: dict[str, list[list[float]]] = {
        name: [] for name in OrbitTransportNet.CANDIDATE_NAMES
    }
    run_counts = {str(run): 0 for run in range(6)}
    run_label_counts = {str(run): [0, 0] for run in range(6)}
    selected = str(fit["selected_candidate"])
    for index, row in enumerate(predictions):
        if not isinstance(row, dict) or row.get("window_index") != index:
            raise ValueError(f"development S{subject} prediction order is invalid")
        label = row.get("label")
        if label not in (0, 1) or row.get("selected_candidate") != selected:
            raise ValueError(
                f"development S{subject} prediction label/route is invalid"
            )
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("dataset") != "BNCI2014-001"
            or provenance.get("session") != "1test"
            or provenance.get("source_trial_index") != index
        ):
            raise ValueError(f"development S{subject} prediction provenance is invalid")
        run_id = str(provenance.get("run_id"))
        ordinal = provenance.get("trial_index_within_run")
        if run_id not in run_counts or ordinal != run_counts[run_id]:
            raise ValueError(f"development S{subject} test-run ordering is invalid")
        run_counts[run_id] += 1
        run_label_counts[run_id][int(label)] += 1
        left, right = float(row["probability_left"]), float(row["probability_right"])
        if (
            not math.isfinite(left)
            or not math.isfinite(right)
            or not math.isclose(left + right, 1.0, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError(f"development S{subject} probability is invalid")
        candidate_right = row.get("candidate_probability_right")
        candidate_logits = row.get("candidate_logits")
        expected_names = set(OrbitTransportNet.CANDIDATE_NAMES)
        if (
            not isinstance(candidate_right, dict)
            or set(candidate_right) != expected_names
            or not isinstance(candidate_logits, dict)
            or set(candidate_logits) != expected_names
        ):
            raise ValueError(f"development S{subject} candidate trace is invalid")
        if not math.isclose(
            right, float(candidate_right[selected]), rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError(f"development S{subject} selected probability is invalid")
        labels.append(int(label))
        selected_probability.append([left, right])
        for name in OrbitTransportNet.CANDIDATE_NAMES:
            probability_right = float(candidate_right[name])
            logit = float(candidate_logits[name])
            if not math.isfinite(probability_right) or not math.isfinite(logit):
                raise ValueError(f"development S{subject} candidate value is invalid")
            expected_probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
            if not math.isclose(
                probability_right,
                expected_probability,
                rel_tol=0.0,
                abs_tol=2e-6,
            ):
                raise ValueError(
                    f"development S{subject} candidate logit/probability mismatch"
                )
            candidate_probability[name].append(
                [1.0 - probability_right, probability_right]
            )
    if (
        run_counts != {str(run): 24 for run in range(6)}
        or run_label_counts != {str(run): [12, 12] for run in range(6)}
        or set(labels) != {0, 1}
    ):
        raise ValueError(f"development S{subject} test cohort is invalid")

    y = np.asarray(labels, dtype=np.int64)
    selected_metrics = _metrics(y, np.asarray(selected_probability, dtype=np.float64))
    _same_metrics(record.get("metrics", {}), selected_metrics, name=f"S{subject}")
    stored_candidates = record.get("candidate_test_metrics_descriptive_only")
    if not isinstance(stored_candidates, dict) or set(stored_candidates) != set(
        OrbitTransportNet.CANDIDATE_NAMES
    ):
        raise ValueError(f"development S{subject} candidate metrics are invalid")
    recomputed_candidates: dict[str, dict[str, float]] = {}
    for name, values in candidate_probability.items():
        recomputed_candidates[name] = _metrics(y, np.asarray(values, dtype=np.float64))
        _same_metrics(
            stored_candidates[name],
            recomputed_candidates[name],
            name=f"S{subject}/{name}",
        )
    fit_candidates = fit.get("candidate_test_metrics_descriptive_only")
    if not isinstance(fit_candidates, dict) or set(fit_candidates) != set(
        OrbitTransportNet.CANDIDATE_NAMES
    ):
        raise ValueError(f"development S{subject} fit candidate metrics are invalid")
    for name, expected_metrics in recomputed_candidates.items():
        _same_metrics(
            fit_candidates[name],
            expected_metrics,
            name=f"S{subject}/fit/{name}",
        )
    _same_metrics(
        record.get("anchor_metrics", {}),
        recomputed_candidates["anchor"],
        name=f"S{subject}/anchor",
    )
    expected_delta = (
        selected_metrics["balanced_accuracy"]
        - recomputed_candidates["anchor"]["balanced_accuracy"]
    )
    if not math.isclose(
        float(record.get("delta_balanced_accuracy_vs_anchor", float("nan"))),
        expected_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"development S{subject} anchor delta is invalid")


def _validate_development_payload(
    payload: Mapping[str, Any],
    *,
    config: OrbitTransportConfig,
    current_source: Mapping[str, str],
) -> None:
    expected_contract = _experiment_contract(
        mode="bnci-dev",
        subjects=DEVELOPMENT_SUBJECTS,
        seeds=(7,),
        folds=5,
        config=config,
        frozen_manifest_sha256=None,
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("development artifact schema is unsupported")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("name") != ARCHITECTURE:
        raise ValueError("development artifact names a different architecture")
    observed_contract = {key: payload.get(key) for key in expected_contract}
    if observed_contract != expected_contract:
        raise ValueError("development artifact does not match the frozen contract")
    if payload.get("contract_sha256") != _json_sha256(expected_contract):
        raise ValueError("development artifact contract digest is invalid")
    if payload.get("source_sha256") != dict(current_source):
        raise ValueError("development artifact source digest is invalid")
    if (
        payload.get("confirmation_access") is not False
        or payload.get("candidate_selection_source") != "validation_only"
        or payload.get("candidate_test_metrics_are_descriptive_only") is not True
    ):
        raise ValueError("development artifact selection/access contract is invalid")

    expected_keys = _expected_record_keys(DEVELOPMENT_SUBJECTS, (7,))
    completion = payload.get("completion")
    if (
        not isinstance(completion, dict)
        or completion.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("expected_record_keys") != expected_keys
        or completion.get("actual_record_keys") != expected_keys
        or completion.get("expected_records") != len(expected_keys)
        or completion.get("actual_records") != len(expected_keys)
    ):
        raise ValueError("development artifact completion contract is invalid")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(expected_keys):
        raise ValueError("development artifact has the wrong record count")
    identities = [
        (
            (int(record.get("subject", -1)), int(record.get("seed", -1)))
            if isinstance(record, dict)
            else (-1, -1)
        )
        for record in records
    ]
    expected_identities = [(subject, 7) for subject in DEVELOPMENT_SUBJECTS]
    if identities != expected_identities or len(set(identities)) != len(identities):
        raise ValueError("development artifact record identities are invalid")
    for record, subject in zip(records, DEVELOPMENT_SUBJECTS, strict=True):
        _validate_development_record(record, subject=subject, config=config)
    if payload.get("summary") != _summarize(payload):
        raise ValueError("development artifact summary does not recompute")


def _confirmation_manifest_contract(output: Path) -> dict[str, Any]:
    return {
        "study_id": CONFIRMATION_STUDY_ID,
        "dataset": "BNCI2014-001",
        "subjects": list(CONFIRMATION_SUBJECTS),
        "seeds": [7],
        "output_path": str(output.expanduser().resolve()),
        "receipt_path": str(Path(CONFIRMATION_RECEIPT_PATH).expanduser().resolve()),
    }


def _read_and_validate_frozen_manifest(
    path: Path | None, config: OrbitTransportConfig
) -> tuple[dict[str, Any], str]:
    if path is None:
        raise ValueError("bnci-confirm requires --frozen-manifest")
    if not path.is_file():
        raise FileNotFoundError(f"frozen manifest does not exist: {path}")
    # Hash exactly the bytes parsed below; do not re-read the path later.
    manifest_bytes = path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != FROZEN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported frozen-manifest schema")
    if manifest.get("architecture") != ARCHITECTURE:
        raise ValueError("frozen manifest names a different architecture")
    if manifest.get("config") != _json_config(config):
        raise ValueError("confirmation config differs from the frozen manifest")
    current_source = _source_manifest()
    if manifest.get("source_sha256") != current_source:
        raise ValueError("confirmation source differs from the frozen manifest")
    confirmation_contract = manifest.get("confirmation_contract")
    if not isinstance(confirmation_contract, dict):
        raise ValueError("frozen manifest lacks a confirmation contract")
    expected_receipt = str(Path(CONFIRMATION_RECEIPT_PATH).expanduser().resolve())
    if (
        confirmation_contract.get("study_id") != CONFIRMATION_STUDY_ID
        or confirmation_contract.get("dataset") != "BNCI2014-001"
        or confirmation_contract.get("subjects") != list(CONFIRMATION_SUBJECTS)
        or confirmation_contract.get("seeds") != [7]
        or confirmation_contract.get("receipt_path") != expected_receipt
        or not isinstance(confirmation_contract.get("output_path"), str)
        or not Path(confirmation_contract["output_path"]).is_absolute()
    ):
        raise ValueError("frozen manifest confirmation contract is invalid")
    development = manifest.get("development_artifact")
    if not isinstance(development, dict) or not isinstance(
        development.get("path"), str
    ):
        raise ValueError("frozen manifest lacks a development_artifact object")
    artifact_path = _resolve_development_artifact(path, development["path"])
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"frozen development artifact is missing: {artifact_path}"
        )
    development_bytes = artifact_path.read_bytes()
    development_sha256 = hashlib.sha256(development_bytes).hexdigest()
    if development.get("sha256") != development_sha256:
        raise ValueError("frozen development artifact hash no longer matches")
    payload = json.loads(development_bytes)
    _validate_development_payload(payload, config=config, current_source=current_source)
    if development.get("summary_sha256") != _json_sha256(payload.get("summary", {})):
        raise ValueError("development summary hash no longer matches the manifest")
    expected_development = {
        "mode": "bnci-dev",
        "subjects": list(DEVELOPMENT_SUBJECTS),
        "seeds": [7],
        "expected_records": len(DEVELOPMENT_SUBJECTS),
    }
    if any(
        development.get(key) != value for key, value in expected_development.items()
    ):
        raise ValueError("frozen development metadata is invalid")
    return manifest, manifest_sha256


def _validate_frozen_manifest(
    path: Path | None, config: OrbitTransportConfig
) -> dict[str, Any]:
    """Compatibility wrapper used by tests and manifest-audit callers."""

    return _read_and_validate_frozen_manifest(path, config)[0]


def write_frozen_manifest(
    development_artifact: Path,
    output: Path,
    *,
    config: OrbitTransportConfig,
    confirmation_output: Path | None = None,
) -> dict[str, Any]:
    """Freeze the exact completed Orbit S1--4/seed-7 BNCI development run."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")
    if not development_artifact.is_file():
        raise FileNotFoundError(development_artifact)
    current_source = _source_manifest()
    development_bytes = development_artifact.read_bytes()
    development_sha256 = hashlib.sha256(development_bytes).hexdigest()
    payload = json.loads(development_bytes)
    _validate_development_payload(payload, config=config, current_source=current_source)
    if confirmation_output is None:
        confirmation_output = output.with_name(
            output.stem + ".confirmation-result.json"
        )
    confirmation_contract = _confirmation_manifest_contract(confirmation_output)
    if Path(confirmation_contract["output_path"]) in {
        output.expanduser().resolve(),
        development_artifact.expanduser().resolve(),
        Path(confirmation_contract["receipt_path"]),
    }:
        raise ValueError("confirmation output must be distinct from frozen artifacts")
    manifest = {
        "schema_version": FROZEN_MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "architecture": ARCHITECTURE,
        "config": _json_config(config),
        "source_sha256": current_source,
        "confirmation_contract": confirmation_contract,
        "development_artifact": {
            "path": str(development_artifact.resolve()),
            "sha256": development_sha256,
            "summary_sha256": _json_sha256(payload.get("summary", {})),
            "mode": "bnci-dev",
            "subjects": list(DEVELOPMENT_SUBJECTS),
            "seeds": [7],
            "expected_records": len(DEVELOPMENT_SUBJECTS),
        },
    }
    _exclusive_json_create(output, manifest)
    return manifest


def _validate_confirmation_contract(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: OrbitTransportConfig,
    confirmation_token: str | None,
    frozen_manifest: Path | None,
) -> tuple[dict[str, Any], str] | None:
    if mode != "bnci-confirm":
        if confirmation_token is not None or frozen_manifest is not None:
            raise ValueError(
                "confirmation token/manifest may only be used with bnci-confirm"
            )
        return None
    if list(subjects) != list(CONFIRMATION_SUBJECTS):
        raise ValueError("bnci-confirm requires the exact ordered subjects 5--9")
    if list(seeds) != [7] or config.seed != 7:
        raise ValueError("bnci-confirm requires the sole frozen seed 7")
    if not config.deterministic:
        raise ValueError("bnci-confirm requires strict deterministic execution")
    if confirmation_token != CONFIRMATION_TOKEN:
        raise ValueError("bnci-confirm requires the exact explicit confirmation token")
    # No confirmation loader can be invoked until all frozen contracts validate.
    return _read_and_validate_frozen_manifest(frozen_manifest, config)


def _receipt_payload(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    config: OrbitTransportConfig,
) -> dict[str, Any]:
    contract = manifest["confirmation_contract"]
    immutable = {
        "schema_version": CONFIRMATION_RECEIPT_SCHEMA_VERSION,
        "study_id": CONFIRMATION_STUDY_ID,
        "manifest_sha256": manifest_sha256,
        "development_artifact_sha256": manifest["development_artifact"]["sha256"],
        "source_sha256": manifest["source_sha256"],
        "config": _json_config(config),
        "subjects": list(CONFIRMATION_SUBJECTS),
        "seeds": [7],
        "output_path": contract["output_path"],
        "receipt_path": contract["receipt_path"],
    }
    return {
        **immutable,
        "contract_sha256": _json_sha256(immutable),
        "state": "claimed",
        "created_at": _utc_now(),
        "updated_at": None,
    }


def _claim_confirmation_receipt(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    config: OrbitTransportConfig,
    output: Path,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    contract = manifest["confirmation_contract"]
    pinned_output = Path(contract["output_path"])
    if output.expanduser().resolve() != pinned_output:
        raise ValueError("confirmation output differs from the frozen exact path")
    if resume and not output.is_file():
        raise FileNotFoundError(
            "confirmation resume requires the pinned result artifact"
        )
    if not resume and output.exists():
        raise FileExistsError("refusing to replace the pinned confirmation result")
    receipt_path = Path(contract["receipt_path"])
    expected = _receipt_payload(
        manifest=manifest, manifest_sha256=manifest_sha256, config=config
    )
    immutable_keys = tuple(expected.keys() - {"state", "created_at", "updated_at"})
    if receipt_path.exists():
        if not resume:
            raise FileExistsError(
                "permanent study confirmation receipt already exists; refusing re-access"
            )
        observed = json.loads(receipt_path.read_text(encoding="utf-8"))
        allowed_receipt_keys = set(expected) | {"last_error", "output_sha256"}
        if not isinstance(observed, dict) or not set(observed).issubset(
            allowed_receipt_keys
        ):
            raise ValueError("confirmation receipt schema is invalid")
        if any(observed.get(key) != expected.get(key) for key in immutable_keys):
            raise ValueError(
                "confirmation receipt does not match the exact frozen contract"
            )
        state = observed.get("state")
        if state == "complete":
            if observed.get("output_sha256") != _file_sha256(output):
                raise ValueError(
                    "completed confirmation output no longer matches its receipt"
                )
            raise FileExistsError(
                "confirmation is already complete; refusing any reevaluation"
            )
        if state not in {"claimed", "running", "interrupted"}:
            raise ValueError("confirmation receipt has an invalid state")
        return receipt_path, observed
    if resume:
        raise FileNotFoundError(
            "confirmation resume requires the permanent study receipt"
        )
    _exclusive_json_create(receipt_path, expected)
    return receipt_path, expected


def _update_confirmation_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    state: str,
    output: Path,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("permanent confirmation receipt disappeared")
    updated = dict(receipt)
    updated["state"] = state
    updated["updated_at"] = _utc_now()
    if error is not None:
        updated["last_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    elif state == "complete":
        updated.pop("last_error", None)
        updated["output_sha256"] = _file_sha256(output)
    _atomic_json(path, updated)
    return updated


def run(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: OrbitTransportConfig,
    output: Path,
    folds: int = 5,
    resume: bool = False,
    frozen_manifest: Path | None = None,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    subjects = [int(value) for value in subjects]
    seeds = [int(value) for value in seeds]
    output = Path(output)
    # This precedes environment capture and any possible CUDA initialization.
    _configure_torch_determinism(config.deterministic)
    _validate_common_contract(mode, subjects, seeds, folds)
    validated_manifest = _validate_confirmation_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        confirmation_token=confirmation_token,
        frozen_manifest=frozen_manifest,
    )
    manifest: Mapping[str, Any] | None = None
    manifest_sha256: str | None = None
    receipt_path: Path | None = None
    receipt: dict[str, Any] | None = None
    if validated_manifest is not None:
        manifest, manifest_sha256 = validated_manifest
        receipt_path, receipt = _claim_confirmation_receipt(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            config=config,
            output=output,
            resume=resume,
        )
    with _OutputLock(output):
        payload = _load_or_initialize(
            output,
            mode=mode,
            subjects=subjects,
            seeds=seeds,
            folds=folds,
            config=config,
            resume=resume,
            frozen_manifest=manifest,
            frozen_manifest_path=frozen_manifest,
            frozen_manifest_sha256=manifest_sha256,
        )
        if not output.exists():
            _exclusive_json_create(output, payload)
        else:
            _save_payload(output, payload, status="running")
        if receipt_path is not None and receipt is not None:
            receipt = _update_confirmation_receipt(
                receipt_path, receipt, state="running", output=output
            )
        try:
            if mode == "local-dev":
                _run_local(subjects, seeds, config, output, payload)
            elif mode == "cho-dev":
                _run_cho(subjects, seeds, folds, config, output, payload)
            else:
                _run_bnci(mode, subjects, seeds, config, output, payload)
        except BaseException as error:
            payload["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "at": _utc_now(),
            }
            _save_payload(output, payload, status="interrupted")
            if receipt_path is not None and receipt is not None:
                _update_confirmation_receipt(
                    receipt_path,
                    receipt,
                    state="interrupted",
                    output=output,
                    error=error,
                )
            raise
        payload.pop("failure", None)
        _save_payload(output, payload, status="complete")
        if receipt_path is not None and receipt is not None:
            _update_confirmation_receipt(
                receipt_path, receipt, state="complete", output=output
            )
        return payload


def _build_config(args: argparse.Namespace) -> OrbitTransportConfig:
    return OrbitTransportConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        temporal_filters=args.temporal_filters,
        temporal_kernel=args.temporal_kernel,
        dynamics_channels=args.dynamics_channels,
        dynamics_kernel=args.dynamics_kernel,
        pool_kernel=args.pool_kernel,
        pool_stride=args.pool_stride,
        normalization=args.normalization,
        orientation_rank=args.orientation_rank,
        orientation_hidden=args.orientation_hidden,
        gate_hidden=args.gate_hidden,
        transport_auxiliary_weight=args.transport_auxiliary_weight,
        view_auxiliary_weight=args.view_auxiliary_weight,
        orientation_penalty=args.orientation_penalty,
        gradient_clip=args.gradient_clip,
        seed=7,
        device=args.device,
        deterministic=not args.allow_nondeterministic,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--subjects")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--confirmation-token")
    parser.add_argument("--write-frozen-manifest", type=Path)
    parser.add_argument(
        "--confirmation-output",
        type=Path,
        help="exact future BNCI confirmation result path pinned into a new manifest",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalization", choices=("group", "batch"), default="group")
    for name, field_type in (
        ("epochs", int),
        ("batch_size", int),
        ("learning_rate", float),
        ("weight_decay", float),
        ("patience", int),
        ("min_delta", float),
        ("temporal_filters", int),
        ("temporal_kernel", int),
        ("dynamics_channels", int),
        ("dynamics_kernel", int),
        ("pool_kernel", int),
        ("pool_stride", int),
        ("orientation_rank", int),
        ("orientation_hidden", int),
        ("gate_hidden", int),
        ("transport_auxiliary_weight", float),
        ("view_auxiliary_weight", float),
        ("orientation_penalty", float),
        ("gradient_clip", float),
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=field_type,
            default=getattr(OrbitTransportConfig, name),
        )
    parser.add_argument("--allow-nondeterministic", action="store_true")
    args = parser.parse_args(argv)
    defaults: dict[str, Sequence[int]] = {
        "local-dev": VALID_SUBJECTS,
        "cho-dev": tuple(range(1, 27)),
        "bnci-dev": DEVELOPMENT_SUBJECTS,
        "bnci-confirm": CONFIRMATION_SUBJECTS,
    }
    subjects = (
        list(defaults[args.mode])
        if args.subjects is None
        else _parse_ints(args.subjects)
    )
    seeds = _parse_ints(args.seeds)
    config = _build_config(args)
    if args.write_frozen_manifest is not None:
        if args.mode != "bnci-dev":
            raise ValueError("--write-frozen-manifest is only valid with bnci-dev")
        if subjects != list(DEVELOPMENT_SUBJECTS) or seeds != [7]:
            raise ValueError(
                "a frozen manifest requires exact BNCI development subjects 1--4 "
                "and sole seed 7"
            )
        if args.write_frozen_manifest.resolve() == args.output.resolve():
            raise ValueError(
                "result and frozen-manifest outputs must be different files"
            )
        if args.write_frozen_manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen manifest: {args.write_frozen_manifest}"
            )
    elif args.confirmation_output is not None:
        raise ValueError("--confirmation-output requires --write-frozen-manifest")
    payload = run(
        mode=args.mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        output=args.output,
        folds=args.folds,
        resume=args.resume,
        frozen_manifest=args.frozen_manifest,
        confirmation_token=args.confirmation_token,
    )
    if args.write_frozen_manifest is not None:
        write_frozen_manifest(
            args.output,
            args.write_frozen_manifest,
            config=config,
            confirmation_output=args.confirmation_output,
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE",
    "CONFIRMATION_RECEIPT_PATH",
    "CONFIRMATION_STUDY_ID",
    "CONFIRMATION_TOKEN",
    "MODES",
    "run",
    "write_frozen_manifest",
]
