"""Development and new-dataset confirmation runner for HemiParityNet.

The already-opened CAMEO confirmation partitions are intentionally unavailable
here.  Architecture work uses ``local-dev`` and ``cho-dev`` only.  Final evidence
must come from the locked BNCI2014-001 split: subjects 1--4 for development and
subjects 5--9 for one-shot confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .cameo_benchmark import (
    _atomic_json,
    _environment,
    _git_state,
    _prediction_trace,
    _record_metrics,
    _stratified_val,
)
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
from .parity_net import HemiParityClassifier, ParityConfig


MODES = ("local-dev", "cho-dev", "bnci-dev", "bnci-confirm")
PROTOCOLS = {
    "local-dev": "local_rec1-fit_rec2-select_rec3-development",
    "cho-dev": "cho2017_s1-26_inner-select_outer-5fold-development",
    "bnci-dev": "bnci2014-001_s1-4_T-runs0-4-fit_T-run5-select_E-development",
    "bnci-confirm": "bnci2014-001_s5-9_T-runs0-4-fit_T-run5-select_E-confirmation",
}


def _source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "deepnet/parity_net.py",
        "deepnet/cameo_net.py",
        "deepnet/parity_benchmark.py",
        "deepnet/external_bnci2014.py",
    ):
        path = PROJECT_ROOT / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _json_config(config: ParityConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _new_payload(
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: ParityConfig,
    *,
    folds: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "architecture": "HemiParityNet",
        "mode": mode,
        "protocol": PROTOCOLS[mode],
        "subjects": list(subjects),
        "seeds": list(seeds),
        "folds": folds,
        "config": _json_config(config),
        "command": sys.argv,
        "repository": _git_state(),
        "source_sha256": _source_manifest(),
        "environment": _environment(),
        "records": [],
        "summary": {},
    }


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    by_subject: dict[int, list[float]] = {}
    errors: list[float] = []
    gates: list[float] = []
    trainable: list[int] = []
    for row in payload["records"]:
        by_subject.setdefault(int(row["subject"]), []).append(
            float(row["metrics"]["balanced_accuracy"])
        )
        for fit in row["fits"]:
            errors.append(float(fit["max_equivariance_error"]))
            gates.append(float(fit["mean_raw_gate"]))
            trainable.append(int(fit["trainable_parameter_count"]))
    values = np.asarray([np.mean(rows) for rows in by_subject.values()], dtype=np.float64)
    return {
        "balanced_accuracy_mean": float(values.mean()) if len(values) else None,
        "balanced_accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "n_participants": len(values),
        "n_records": len(payload["records"]),
        "max_equivariance_error": max(errors) if errors else None,
        "mean_raw_gate": float(np.mean(gates)) if gates else None,
        "trainable_parameter_count": int(np.median(trainable)) if trainable else None,
    }


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
    config: ParityConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    classifier = HemiParityClassifier(config).fit(
        raw_train,
        cov_train,
        y_train,
        raw_validation,
        cov_validation,
        y_validation,
        channels=channels,
    )
    probability = classifier.predict_proba(raw_test, cov_test)
    output = classifier._predict_output(raw_test, cov_test)
    detail = {
        "best_epoch": classifier.best_epoch_,
        "epochs_run": classifier.epochs_run_,
        "best_validation_loss": classifier.best_validation_loss_,
        "best_validation_balanced_accuracy": classifier.best_validation_balanced_accuracy_,
        "parameter_count": classifier.param_count_,
        "trainable_parameter_count": classifier.trainable_param_count_,
        "train_seconds": classifier.train_seconds_,
        "max_equivariance_error": classifier.max_equivariance_error(raw_test, cov_test),
        "mean_raw_gate": float(output.fusion_weights[:, 0].mean().cpu()),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "n_test": int(len(y_test)),
    }
    return detail, probability


def _save_record(output: Path, payload: dict[str, Any], record: dict[str, Any]) -> None:
    payload["records"].append(record)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload["summary"] = _summarize(payload)
    _atomic_json(output, payload)


def run_local_dev(
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: ParityConfig,
    output: Path,
) -> dict[str, Any]:
    if not subjects or not set(subjects).issubset(set(VALID_SUBJECTS)):
        raise ValueError("local development subjects must belong to the fixed local cohort")
    # Physical isolation: do not load recording 4 into the development process.
    keys = [
        SessionKey(subject, run)
        for subject in subjects
        for run in SUBJECT_RUNS[subject][:3]
    ]
    data = load_sessions(keys, DEFAULT_DATA_CONFIG)
    payload = _new_payload("local-dev", subjects, seeds, config)
    for subject in subjects:
        runs = list(SUBJECT_RUNS[subject][:3])
        task = data.labels >= 0

        def rows(run: int) -> np.ndarray:
            return np.flatnonzero(
                (data.subject_ids == subject) & (data.run_ids == run) & task
            )

        train, validation, test = rows(runs[0]), rows(runs[1]), rows(runs[2])
        for seed in seeds:
            detail, probability = _fit_score(
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
                config=replace(config, seed=seed),
            )
            truth = data.labels[test]
            metrics = _record_metrics(truth, probability)
            _save_record(
                output,
                payload,
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "metrics": metrics,
                    "fits": [detail],
                    "predictions": _prediction_trace(truth, probability),
                },
            )
            print(
                f"HemiParity local-dev S{subject} seed={seed}: "
                f"bacc={100*metrics['balanced_accuracy']:.2f}%",
                flush=True,
            )
    return payload


def run_cho_dev(
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: ParityConfig,
    output: Path,
) -> dict[str, Any]:
    if not subjects or not set(subjects).issubset(set(range(1, 27))):
        raise ValueError("Cho development is locked to subjects 1--26")
    payload = _new_payload("cho-dev", subjects, seeds, config, folds=folds)
    for subject in subjects:
        data = load_cho_subject(subject)
        raw, cov, labels = data["broadband"], data["covariances"], data["labels"]
        splits = list(
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=0).split(
                np.zeros(len(labels)), labels
            )
        )
        for seed in seeds:
            all_truth: list[np.ndarray] = []
            all_probability: list[np.ndarray] = []
            fits: list[dict[str, Any]] = []
            for fold, (outer_train, test) in enumerate(splits):
                train_rel, validation_rel = _stratified_val(
                    labels[outer_train], seed + fold
                )
                train, validation = outer_train[train_rel], outer_train[validation_rel]
                detail, probability = _fit_score(
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
                    config=replace(config, seed=seed + fold),
                )
                detail["fold"] = int(fold)
                fits.append(detail)
                all_truth.append(labels[test])
                all_probability.append(probability)
            truth = np.concatenate(all_truth)
            probability = np.concatenate(all_probability)
            metrics = _record_metrics(truth, probability)
            _save_record(
                output,
                payload,
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "metrics": metrics,
                    "fits": fits,
                    "predictions": _prediction_trace(truth, probability),
                },
            )
            print(
                f"HemiParity cho-dev S{subject} seed={seed}: "
                f"bacc={100*metrics['balanced_accuracy']:.2f}%",
                flush=True,
            )
    return payload


def run_bnci(
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: ParityConfig,
    output: Path,
) -> dict[str, Any]:
    expected = set(DEVELOPMENT_SUBJECTS if mode == "bnci-dev" else CONFIRMATION_SUBJECTS)
    if not subjects or not set(subjects).issubset(expected):
        raise ValueError(
            f"{mode} subjects must stay inside {min(expected)}--{max(expected)}"
        )
    payload = _new_payload(mode, subjects, seeds, config)
    payload["dataset_protocol"] = protocol_metadata(subjects[0])
    for subject in subjects:
        data = load_bnci2014_subject(subject)
        for seed in seeds:
            detail, probability = _fit_score(
                raw_train=data["train"]["broadband"],
                cov_train=data["train"]["covariances"],
                y_train=data["train"]["labels"],
                raw_validation=data["validation"]["broadband"],
                cov_validation=data["validation"]["covariances"],
                y_validation=data["validation"]["labels"],
                raw_test=data["test"]["broadband"],
                cov_test=data["test"]["covariances"],
                y_test=data["test"]["labels"],
                channels=BNCI_CHANNELS,
                config=replace(config, seed=seed),
            )
            truth = data["test"]["labels"]
            metrics = _record_metrics(truth, probability)
            _save_record(
                output,
                payload,
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "metrics": metrics,
                    "fits": [detail],
                    "predictions": _prediction_trace(truth, probability),
                },
            )
            print(
                f"HemiParity {mode} S{subject} seed={seed}: "
                f"bacc={100*metrics['balanced_accuracy']:.2f}%",
                flush=True,
            )
    return payload


def _parse_ints(value: str) -> list[int]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            low, high = token.split("-", 1)
            result.extend(range(int(low), int(high) + 1))
        else:
            result.append(int(token))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=ParityConfig.epochs)
    parser.add_argument("--patience", type=int, default=ParityConfig.patience)
    parser.add_argument("--batch-size", type=int, default=ParityConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=ParityConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ParityConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=ParityConfig.dropout)
    parser.add_argument("--temporal-filters", type=int, default=ParityConfig.temporal_filters)
    parser.add_argument("--dynamics-channels", type=int, default=ParityConfig.dynamics_channels)
    parser.add_argument("--raw-rank", type=int, default=ParityConfig.raw_rank)
    parser.add_argument("--tangent-rank", type=int, default=ParityConfig.tangent_rank)
    parser.add_argument("--auxiliary-weight", type=float, default=ParityConfig.auxiliary_weight)
    parser.add_argument("--gate-balance-weight", type=float, default=ParityConfig.gate_balance_weight)
    args = parser.parse_args(argv)

    defaults = {
        "local-dev": list(VALID_SUBJECTS),
        "cho-dev": list(range(1, 27)),
        "bnci-dev": list(DEVELOPMENT_SUBJECTS),
        "bnci-confirm": list(CONFIRMATION_SUBJECTS),
    }
    subjects = defaults[args.mode] if args.subjects is None else _parse_ints(args.subjects)
    seeds = _parse_ints(args.seeds)
    config = ParityConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        temporal_filters=args.temporal_filters,
        dynamics_channels=args.dynamics_channels,
        raw_rank=args.raw_rank,
        tangent_rank=args.tangent_rank,
        auxiliary_weight=args.auxiliary_weight,
        gate_balance_weight=args.gate_balance_weight,
        device=args.device,
    )
    if args.mode == "local-dev":
        payload = run_local_dev(subjects, seeds, config, args.output)
    elif args.mode == "cho-dev":
        payload = run_cho_dev(subjects, seeds, args.folds, config, args.output)
    else:
        payload = run_bnci(args.mode, subjects, seeds, config, args.output)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

