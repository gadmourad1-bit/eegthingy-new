"""Leakage-controlled development and confirmation benchmarks for CAMEO-Net.

Development and confirmation are deliberately separate:

* ``local-dev`` trains on recording 1, selects on recording 2, and scores
  recording 3.  Recording 4 is not loaded into any fit/selection operation.
* ``local-confirm`` uses recordings 1--2 / 3 / 4 and must only be run after the
  architecture configuration has been frozen.
* ``external-dev`` is intended for Cho2017 subjects 1--26.
* ``external-confirm`` is intended for subjects 27--52.  Each subject uses an
  outer five-fold split and a train-only inner validation split.

Every output retains probabilities, labels, validation routing traces, runtime,
and the exact configuration so reported metrics can be reconstructed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .cameo_net import CAMEOClassifier, CAMEOConfig
from .config import (
    DEFAULT_DATA_CONFIG,
    PROJECT_ROOT,
    SUBJECT_RUNS,
    VALID_SUBJECTS,
)
from .data import SessionData, SessionKey, load_sessions
from .external_cho2017 import load_cho_subject


SCHEMA_VERSION = 1
PROTOCOLS = {
    "local-dev": "local_train-rec1_select-rec2_development-rec3",
    "local-confirm": "local_train-rec1-2_select-rec3_confirm-rec4",
    "external-dev": "cho2017_subjects1-26_inner-select_outer-5fold-development",
    "external-confirm": "cho2017_subjects27-52_inner-select_outer-5fold-confirmation",
}


def _stratified_val(
    labels: np.ndarray, seed: int, fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/validation positions without importing the baseline stack."""

    generator = np.random.RandomState(seed)
    validation = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        generator.shuffle(positions)
        validation[positions[: max(1, int(round(fraction * len(positions))))]] = True
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip()

    commit = run("git", "rev-parse", "HEAD") or None
    status_text = run("git", "status", "--porcelain") if commit is not None else ""
    return {
        "available": commit is not None,
        "commit": commit,
        # Do not silently describe a source snapshot without .git metadata as
        # clean.  The result still records its command and a separate source
        # manifest can pin exact file hashes.
        "dirty": bool(status_text) if commit is not None else None,
        "status": status_text.splitlines() if commit is not None else None,
    }


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _json_config(config: CAMEOConfig) -> dict[str, Any]:
    """Normalize tuple-valued dataclass fields exactly as JSON will store them."""

    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}-",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _record_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, probabilities.argmax(axis=1))
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])),
    }


def _prediction_trace(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "label": int(label),
            "probability_left": float(probability[0]),
            "probability_right": float(probability[1]),
        }
        for label, probability in zip(labels, probabilities, strict=True)
    ]


def _local_rows(
    data: SessionData, subject: int, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs = sorted(np.unique(data.run_ids[data.subject_ids == subject]).tolist())
    expected = list(SUBJECT_RUNS[subject])
    if runs != expected:
        raise ValueError(f"subject {subject} has runs {runs}, expected {expected}")
    task = data.labels >= 0

    def selected(chosen: Sequence[int]) -> np.ndarray:
        return np.flatnonzero(
            (data.subject_ids == subject) & np.isin(data.run_ids, chosen) & task
        )

    if mode == "local-dev":
        return selected([runs[0]]), selected([runs[1]]), selected([runs[2]])
    if mode == "local-confirm":
        return selected(runs[:2]), selected([runs[2]]), selected([runs[3]])
    raise ValueError(mode)


def _fit_and_score(
    *,
    raw: np.ndarray,
    covariances: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    evaluation: np.ndarray,
    config: CAMEOConfig,
) -> tuple[CAMEOClassifier, np.ndarray, dict[str, Any]]:
    classifier = CAMEOClassifier(config).fit(
        raw[train],
        covariances[train],
        labels[train],
        raw[validation],
        covariances[validation],
        labels[validation],
    )
    probabilities = classifier.predict_proba(raw[evaluation], covariances[evaluation])
    detail = {
        "selected_mixture": classifier.selected_mixture_,
        "selected_rho": classifier.selected_rho_,
        "best_epoch": classifier.best_epoch_,
        "epochs_run": classifier.epochs_run_,
        "best_validation_loss": classifier.best_validation_loss_,
        "parameter_count": classifier.param_count_,
        "train_seconds": classifier.train_seconds_,
        "selection_trace": classifier.selection_trace_,
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_evaluation": int(len(evaluation)),
    }
    return classifier, probabilities, detail


def _new_payload(mode: str, subjects: Sequence[int], seeds: Sequence[int], config: CAMEOConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "mode": mode,
        "protocol": PROTOCOLS[mode],
        "subjects": list(subjects),
        "seeds": list(seeds),
        "config": _json_config(config),
        "command": sys.argv,
        "repository": _git_state(),
        "environment": _environment(),
        "records": [],
        "summary": {},
    }


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    by_subject: dict[int, list[float]] = {}
    routes: dict[str, int] = {}
    rhos: dict[str, int] = {}
    for row in payload["records"]:
        by_subject.setdefault(int(row["subject"]), []).append(
            float(row["metrics"]["balanced_accuracy"])
        )
        for detail in row["fits"]:
            name = str(detail["selected_mixture"])
            routes[name] = routes.get(name, 0) + 1
            rho = str(detail["selected_rho"])
            rhos[rho] = rhos.get(rho, 0) + 1
    values = np.asarray([np.mean(rows) for rows in by_subject.values()], dtype=np.float64)
    return {
        "balanced_accuracy_mean": float(values.mean()) if len(values) else None,
        "balanced_accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "n_participants": len(values),
        "n_records": len(payload["records"]),
        "route_counts": routes,
        "rho_counts": rhos,
    }


def _load_or_initialize(
    output: Path,
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: CAMEOConfig,
    resume: bool,
) -> dict[str, Any]:
    if resume and output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        expected = (mode, list(subjects), list(seeds), _json_config(config))
        observed = (
            payload.get("mode"),
            payload.get("subjects"),
            payload.get("seeds"),
            payload.get("config"),
        )
        if observed != expected:
            raise ValueError("cannot resume an output with a different experiment contract")
        return payload
    return _new_payload(mode, subjects, seeds, config)


def run_local(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: CAMEOConfig,
    output: Path,
    resume: bool,
) -> dict[str, Any]:
    keys = [SessionKey(subject, run) for subject in subjects for run in SUBJECT_RUNS[subject]]
    data = load_sessions(keys, replace(DEFAULT_DATA_CONFIG, include_rest=False))
    payload = _load_or_initialize(
        output,
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        resume=resume,
    )
    completed = {(int(row["subject"]), int(row["seed"])) for row in payload["records"]}
    for subject in subjects:
        train, validation, evaluation = _local_rows(data, subject, mode)
        for seed in seeds:
            if (subject, seed) in completed:
                continue
            run_config = replace(config, seed=seed)
            classifier, probabilities, detail = _fit_and_score(
                raw=data.broadband_epochs,
                covariances=data.covariances,
                labels=data.labels,
                train=train,
                validation=validation,
                evaluation=evaluation,
                config=run_config,
            )
            labels = data.labels[evaluation]
            metrics = _record_metrics(labels, probabilities)
            payload["records"].append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "metrics": metrics,
                    "fits": [detail],
                    "predictions": _prediction_trace(labels, probabilities),
                }
            )
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            payload["summary"] = _summarize(payload)
            _atomic_json(output, payload)
            print(
                f"CAMEO {mode} S{subject} seed={seed}: "
                f"bacc={100*metrics['balanced_accuracy']:.2f}% "
                f"route={classifier.selected_mixture_} rho={classifier.selected_rho_} "
                f"({classifier.train_seconds_:.1f}s)",
                flush=True,
            )
    return payload


def run_external(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: CAMEOConfig,
    output: Path,
    resume: bool,
) -> dict[str, Any]:
    allowed = set(range(1, 27) if mode == "external-dev" else range(27, 53))
    if not subjects or not set(subjects).issubset(allowed):
        raise ValueError(
            f"{mode} subjects must stay inside the locked partition "
            f"{min(allowed)}--{max(allowed)}"
        )
    payload = _load_or_initialize(
        output,
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        resume=resume,
    )
    if "folds" in payload and int(payload["folds"]) != int(folds):
        raise ValueError("cannot resume with a different external fold count")
    payload["folds"] = int(folds)
    completed = {(int(row["subject"]), int(row["seed"])) for row in payload["records"]}
    for subject in subjects:
        data = load_cho_subject(subject)
        raw = data["broadband"]
        covariances = data["covariances"]
        labels = data["labels"]
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        outer_splits = list(splitter.split(np.zeros(len(labels)), labels))
        for seed in seeds:
            if (subject, seed) in completed:
                continue
            run_config = replace(config, seed=seed)
            all_labels: list[np.ndarray] = []
            all_probabilities: list[np.ndarray] = []
            fit_details: list[dict[str, Any]] = []
            fold_records: list[dict[str, Any]] = []
            for fold_index, (outer_train, outer_test) in enumerate(outer_splits):
                inner_train_rel, validation_rel = _stratified_val(
                    labels[outer_train], seed + fold_index
                )
                train = outer_train[inner_train_rel]
                validation = outer_train[validation_rel]
                classifier, probabilities, detail = _fit_and_score(
                    raw=raw,
                    covariances=covariances,
                    labels=labels,
                    train=train,
                    validation=validation,
                    evaluation=outer_test,
                    config=replace(run_config, seed=seed + fold_index),
                )
                truth = labels[outer_test]
                fold_metrics = _record_metrics(truth, probabilities)
                detail["fold"] = int(fold_index)
                fit_details.append(detail)
                fold_records.append(
                    {
                        "fold": int(fold_index),
                        "metrics": fold_metrics,
                        "predictions": _prediction_trace(truth, probabilities),
                    }
                )
                all_labels.append(truth)
                all_probabilities.append(probabilities)
            truth = np.concatenate(all_labels)
            probabilities = np.concatenate(all_probabilities)
            metrics = _record_metrics(truth, probabilities)
            payload["records"].append(
                {
                    "subject": int(subject),
                    "seed": int(seed),
                    "metrics": metrics,
                    "fits": fit_details,
                    "fold_records": fold_records,
                    "predictions": _prediction_trace(truth, probabilities),
                }
            )
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            payload["summary"] = _summarize(payload)
            _atomic_json(output, payload)
            print(
                f"CAMEO {mode} S{subject} seed={seed}: "
                f"bacc={100*metrics['balanced_accuracy']:.2f}%",
                flush=True,
            )
    return payload


def _parse_ints(value: str) -> list[int]:
    values: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            low, high = token.split("-", 1)
            values.extend(range(int(low), int(high) + 1))
        else:
            values.append(int(token))
    return values


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(token) for token in value.split(",") if token.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=CAMEOConfig.epochs)
    parser.add_argument("--patience", type=int, default=CAMEOConfig.patience)
    parser.add_argument("--batch-size", type=int, default=CAMEOConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=CAMEOConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=CAMEOConfig.weight_decay)
    parser.add_argument("--dropout", type=float, default=CAMEOConfig.dropout)
    parser.add_argument("--temporal-filters", type=int, default=CAMEOConfig.temporal_filters)
    parser.add_argument("--dynamics-channels", type=int, default=CAMEOConfig.dynamics_channels)
    parser.add_argument("--mirror-penalty", type=float, default=CAMEOConfig.mirror_penalty)
    parser.add_argument("--rhos", default=",".join(str(x) for x in CAMEOConfig.rho_grid))
    parser.add_argument("--mixtures", default=",".join(CAMEOConfig.mixture_names))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    if args.subjects is None:
        if args.mode.startswith("local"):
            subjects = list(VALID_SUBJECTS)
        elif args.mode == "external-dev":
            subjects = list(range(1, 27))
        else:
            subjects = list(range(27, 53))
    else:
        subjects = _parse_ints(args.subjects)
    seeds = _parse_ints(args.seeds)
    config = CAMEOConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        temporal_filters=args.temporal_filters,
        dynamics_channels=args.dynamics_channels,
        mirror_penalty=args.mirror_penalty,
        rho_grid=_parse_floats(args.rhos),
        mixture_names=tuple(token for token in args.mixtures.split(",") if token),
        device=args.device,
    )
    if args.mode.startswith("local"):
        payload = run_local(
            mode=args.mode,
            subjects=subjects,
            seeds=seeds,
            config=config,
            output=args.output,
            resume=not args.no_resume,
        )
    else:
        payload = run_external(
            mode=args.mode,
            subjects=subjects,
            seeds=seeds,
            folds=args.folds,
            config=config,
            output=args.output,
            resume=not args.no_resume,
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_external", "run_local"]
