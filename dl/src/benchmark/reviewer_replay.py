"""Bounded, post-publication replay of one common-grid result.

The formal :mod:`benchmark.full_grid` runner deliberately has
no subset switch:
its publication claim is the complete 96,320-job Cartesian product.  This
module is a separate verification protocol.  It selects one already-published
job, subject, dataset cell, or model row; preserves every seed/fold needed by
that selected estimand; and compares the replay with the sealed v6 tables.

Trial rows and probabilities are never persisted.  They live only long enough
to compute an exact prediction-archive digest, a confusion matrix, accuracy,
and balanced accuracy.  Each immutable job receipt is therefore useful for
resume and aggregation without becoming a trial-level prediction release.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import tomllib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# The sanitized v6 replay authority records ``eeg_mi`` as its logical source
# namespace.  Its frozen source copy is loaded under a private package name so
# relative imports remain functional without shadowing the active package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_V6_SOURCE_ROOT = (
    PROJECT_ROOT / "historical" / "common_grid_v6" / "source"
)
HISTORICAL_PRIVATE_PACKAGE = "_eeg_mi_historical_common_grid_v6"
_historical_package_root = HISTORICAL_V6_SOURCE_ROOT / "eeg_mi_v6"
_historical_spec = importlib.util.spec_from_file_location(
    HISTORICAL_PRIVATE_PACKAGE,
    _historical_package_root / "__init__.py",
    submodule_search_locations=[str(_historical_package_root)],
)
if _historical_spec is None or _historical_spec.loader is None:
    raise RuntimeError("could not construct the private historical v6 adapter")
_historical_package = importlib.util.module_from_spec(_historical_spec)
sys.modules[HISTORICAL_PRIVATE_PACKAGE] = _historical_package
try:
    _historical_spec.loader.exec_module(_historical_package)
    full_grid = importlib.import_module(
        f"{HISTORICAL_PRIVATE_PACKAGE}.full_grid"
    )
    project_gpu_leases = importlib.import_module(
        f"{HISTORICAL_PRIVATE_PACKAGE}.project_gpu_leases"
    )
except BaseException:
    for _module_name in tuple(sys.modules):
        if _module_name == HISTORICAL_PRIVATE_PACKAGE or _module_name.startswith(
            f"{HISTORICAL_PRIVATE_PACKAGE}."
        ):
            sys.modules.pop(_module_name, None)
    raise
if (
    Path(full_grid.__file__).resolve().parent
    != HISTORICAL_V6_SOURCE_ROOT / "eeg_mi_v6"
):
    raise RuntimeError("historical v6 module resolved outside its isolated source root")


LEGACY_REPLAY_SCHEMA = "eeg-mi-reviewer-replay-plan-v1"
REPLAY_SCHEMA = "eeg-mi-reviewer-replay-plan-v2"
LEGACY_RECORD_SCHEMA = "eeg-mi-reviewer-replay-record-v1"
RECORD_SCHEMA = "eeg-mi-reviewer-replay-record-v2"
LEGACY_ENVIRONMENT_SCHEMA = "eeg-mi-reviewer-replay-environment-v1"
ENVIRONMENT_SCHEMA = "eeg-mi-reviewer-replay-environment-v2"
LEGACY_COMPARISON_SCHEMA = "eeg-mi-reviewer-replay-comparison-v1"
COMPARISON_SCHEMA = "eeg-mi-reviewer-replay-comparison-v2"
LEGACY_STATUS_SCHEMA = "eeg-mi-reviewer-replay-status-v1"
STATUS_SCHEMA = "eeg-mi-reviewer-replay-status-v2"
ESTIMATE_SCHEMA = "eeg-mi-reviewer-replay-estimate-v2"
LEGACY_TRACK_SCOPE = "reviewer_replay_v1"
TRACK_SCOPE = "reviewer_replay_v2"
SCOPES = ("job", "subject", "dataset", "model")
STRICT_ABSOLUTE_TOLERANCE = 1e-12
CPU_THREADS = 4
REFERENCE_ROOT = PROJECT_ROOT / "results" / "common_grid_v6"
ANALYSIS_ROOT = REFERENCE_ROOT / "analysis"
MANIFEST_NAME = "replay_plan.json"
MANIFEST_DIGEST_NAME = "replay_plan.sha256"
ENVIRONMENT_NAME = "environment.json"
ENVIRONMENT_DIGEST_NAME = "environment.sha256"
COMPARISON_NAME = "comparison.json"
COMPARISON_DIGEST_NAME = "comparison.sha256"
RECORD_FILES = frozenset({"record.json", "record.sha256"})
THREAD_VARIABLES = full_grid.THREAD_ENVIRONMENT_VARIABLES
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
class ReviewerReplayError(RuntimeError):
    """A replay input, identity, record, or comparison failed closed."""


def _artifact_schema(replay_plan: Mapping[str, Any], kind: str) -> str:
    legacy = replay_plan.get("schema") == LEGACY_REPLAY_SCHEMA
    schemas = {
        "record": (LEGACY_RECORD_SCHEMA, RECORD_SCHEMA),
        "environment": (LEGACY_ENVIRONMENT_SCHEMA, ENVIRONMENT_SCHEMA),
        "comparison": (LEGACY_COMPARISON_SCHEMA, COMPARISON_SCHEMA),
        "status": (LEGACY_STATUS_SCHEMA, STATUS_SCHEMA),
    }
    try:
        selected = schemas[kind]
    except KeyError as error:
        raise ReviewerReplayError(f"unknown replay artifact kind: {kind}") from error
    return selected[0] if legacy else selected[1]


@contextmanager
def _historical_tcformer_environment() -> Iterable[None]:
    """Map the active TCFormer root only while consulting frozen v6 code."""

    active = os.environ.get("EEG_MI_TCFORMER_ROOT")
    if active is None:
        active = str(PROJECT_ROOT / "third_party" / "TCFormer")
    previous = os.environ.get("EEG_MI_TCFORMER_ROOT")
    os.environ["EEG_MI_TCFORMER_ROOT"] = active
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("EEG_MI_TCFORMER_ROOT", None)
        else:
            os.environ["EEG_MI_TCFORMER_ROOT"] = previous


@dataclass(frozen=True)
class Selection:
    scope: str
    model: str
    dataset: str | None = None
    subject: int | None = None
    fold: int | None = None
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "model": self.model,
            "dataset": self.dataset,
            "subject": self.subject,
            "fold": self.fold,
            "seed": self.seed,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_regular(path))


def _read_regular(path: Path) -> bytes:
    """Read a stable unique regular file without imposing checkout modes."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    before_path = absolute.lstat()
    if (
        stat.S_ISLNK(before_path.st_mode)
        or not stat.S_ISREG(before_path.st_mode)
        or before_path.st_nlink != 1
    ):
        raise ReviewerReplayError(f"not a unique regular file: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(before_path, name) for name in stable):
            raise ReviewerReplayError(f"file changed while opening: {absolute}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        after_path = absolute.lstat()
        if any(
            getattr(before, name) != getattr(observed, name)
            for observed in (after, after_path)
            for name in stable
        ):
            raise ReviewerReplayError(f"file changed during read: {absolute}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ReviewerReplayError(f"incomplete read: {absolute}")
        return payload
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewerReplayError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReviewerReplayError(f"non-finite JSON value {value!r}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_regular(path).decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewerReplayError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReviewerReplayError(f"JSON root is not an object: {path}")
    return value


def load_reference_plan() -> dict[str, Any]:
    """Load the distributed plan without requiring lab-only 0444 modes."""

    plan_path = REFERENCE_ROOT / "plan.json"
    sidecar_path = REFERENCE_ROOT / "plan.sha256"
    plan = _load_json(plan_path)
    sidecar = _read_regular(sidecar_path).decode("ascii").strip()
    digest = full_grid.plan_sha256(plan)
    if (
        _read_regular(plan_path) != _canonical_bytes(plan) + b"\n"
        or plan.get("plan_sha256") != digest
        or sidecar != digest
    ):
        raise ReviewerReplayError("sealed reference plan identity is invalid")
    try:
        full_grid.validate_plan_semantics(plan, require_runtime_identity=False)
    except Exception as error:
        raise ReviewerReplayError("sealed reference plan semantics are invalid") from error
    return plan


def _verify_results_bundle() -> None:
    path = PROJECT_ROOT / "results" / "verify_bundle.py"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-S", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.SubprocessError as error:
        raise ReviewerReplayError("distributed results bundle failed verification") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReviewerReplayError(
            f"distributed results bundle failed verification: {detail}"
        )


def _historical_source_identity() -> dict[str, str]:
    """Hash the exact sanitized v6 replay closure under its logical names."""

    paths = {
        f"eeg_mi/{name}": _historical_package_root / name
        for name in full_grid.SOURCE_FILES
    }
    paths.update(
        {
            "pyproject.toml": HISTORICAL_V6_SOURCE_ROOT / "pyproject.toml",
            "uv.lock": HISTORICAL_V6_SOURCE_ROOT / "uv.lock",
        }
    )
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def source_compatibility(
    plan: Mapping[str, Any], *, legacy_v1_snapshot: bool = False
) -> dict[str, Any]:
    """Validate the exact sanitized historical source authority."""

    if legacy_v1_snapshot:
        raise ReviewerReplayError("v1 replay snapshots are retired")

    planned = plan.get("source_identity")
    if not isinstance(planned, Mapping):
        raise ReviewerReplayError("reference plan lacks source identity")
    current = _historical_source_identity()
    rows: list[dict[str, str]] = []
    if set(current) != set(planned):
        raise ReviewerReplayError("current and reference source closures differ")
    for path in sorted(planned):
        expected = str(planned[path])
        observed = str(current[path])
        if observed != expected:
            raise ReviewerReplayError(f"unapproved source difference: {path}")
        rows.append(
            {
                "path": str(path),
                "reference_sha256": expected,
                "current_sha256": observed,
                "status": "exact",
            }
        )
    with _historical_tcformer_environment():
        current_tcformer = full_grid._current_tcformer_source_identity()
    planned_tcformer = plan.get("tcformer_source_identity")
    if not isinstance(planned_tcformer, Mapping):
        raise ReviewerReplayError("reference TCFormer identity is invalid")
    for key in (
        "source_mode",
        "repository",
        "commit",
        "license",
        "manifest_sha256",
        "file_identity",
    ):
        if current_tcformer.get(key) != planned_tcformer.get(key):
            raise ReviewerReplayError(f"TCFormer identity differs at {key}")
    analysis_path = (
        HISTORICAL_V6_SOURCE_ROOT / "eeg_mi_v6" / "full_grid_analysis.py"
    )
    planned_analysis = plan.get("analysis_contract", {}).get("source_identity", {})
    expected_analysis = planned_analysis.get("eeg_mi/full_grid_analysis.py")
    if _sha256_file(analysis_path) != expected_analysis:
        raise ReviewerReplayError("analysis source differs from the sealed plan")
    return {
        "formal_sources": rows,
        "tcformer_source_root_is_location_only": True,
        "tcformer_identity": {
            key: current_tcformer[key]
            for key in current_tcformer
            if key != "source_root"
        },
        "analysis_source_sha256": expected_analysis,
    }


def validate_selection(plan: Mapping[str, Any], selection: Selection) -> None:
    if selection.scope not in SCOPES:
        raise ReviewerReplayError(f"unknown scope: {selection.scope}")
    if selection.model not in plan["architectures"]:
        raise ReviewerReplayError(f"unknown common-grid model: {selection.model}")
    requires = {
        "job": ("dataset", "subject", "fold", "seed"),
        "subject": ("dataset", "subject"),
        "dataset": ("dataset",),
        "model": (),
    }[selection.scope]
    values = {
        "dataset": selection.dataset,
        "subject": selection.subject,
        "fold": selection.fold,
        "seed": selection.seed,
    }
    missing = [name for name in requires if values[name] is None]
    forbidden = [name for name in values if name not in requires and values[name] is not None]
    if missing or forbidden:
        raise ReviewerReplayError(
            f"scope {selection.scope} selector mismatch; missing={missing}, forbidden={forbidden}"
        )
    if selection.dataset is not None:
        if selection.dataset not in plan["dataset_order"]:
            raise ReviewerReplayError(f"unknown dataset: {selection.dataset}")
        contract = plan["datasets"][selection.dataset]
        if selection.subject is not None and selection.subject not in contract["subjects"]:
            raise ReviewerReplayError("subject is outside the selected dataset")
        if selection.fold is not None and selection.fold not in contract["folds"]:
            raise ReviewerReplayError("fold is outside the selected dataset")
    if selection.seed is not None and selection.seed not in plan["seeds"]:
        raise ReviewerReplayError("seed is outside the frozen five-seed schedule")


def select_jobs(plan: Mapping[str, Any], selection: Selection) -> tuple[full_grid.Job, ...]:
    validate_selection(plan, selection)
    jobs: list[full_grid.Job] = []
    for job in full_grid.iter_jobs(plan):
        if job.model != selection.model:
            continue
        if selection.dataset is not None and job.dataset != selection.dataset:
            continue
        if selection.subject is not None and job.subject != selection.subject:
            continue
        if selection.fold is not None and job.fold != selection.fold:
            continue
        if selection.seed is not None and job.seed != selection.seed:
            continue
        jobs.append(job)
    if selection.scope == "job":
        expected = 1
    elif selection.scope == "subject":
        assert selection.dataset is not None
        expected = len(plan["datasets"][selection.dataset]["folds"]) * len(
            plan["seeds"]
        )
    elif selection.scope == "dataset":
        assert selection.dataset is not None
        expected = (
            len(plan["datasets"][selection.dataset]["subjects"])
            * len(plan["datasets"][selection.dataset]["folds"])
            * len(plan["seeds"])
        )
    else:
        expected = sum(
            len(plan["datasets"][dataset]["subjects"])
            * len(plan["datasets"][dataset]["folds"])
            * len(plan["seeds"])
            for dataset in plan["dataset_order"]
        )
    if len(jobs) != expected or len({job.job_id for job in jobs}) != expected:
        raise ReviewerReplayError("selected jobs differ from the frozen Cartesian subset")
    return tuple(jobs)


def build_replay_plan(
    plan: Mapping[str, Any],
    selection: Selection,
    *,
    schema: str = REPLAY_SCHEMA,
) -> dict[str, Any]:
    if schema not in {LEGACY_REPLAY_SCHEMA, REPLAY_SCHEMA}:
        raise ReviewerReplayError(f"unsupported replay-plan schema: {schema}")
    jobs = select_jobs(plan, selection)
    cache_keys = sorted(
        {
            full_grid._subject_identity_key(job.dataset, job.subject)
            for job in jobs
        }
    )
    split_keys = sorted(
        {
            full_grid._split_identity_key(job.dataset, job.subject, job.fold)
            for job in jobs
        }
    )
    source = source_compatibility(
        plan, legacy_v1_snapshot=schema == LEGACY_REPLAY_SCHEMA
    )
    payload: dict[str, Any] = {
        "schema": schema,
        "purpose": "post_publication_bounded_result_reproduction",
        "publication_authority": False,
        "reference_plan_sha256": plan["plan_sha256"],
        "reference_plan_file_sha256": _sha256_file(REFERENCE_ROOT / "plan.json"),
        "reference_analysis_manifest_sha256": _sha256_file(ANALYSIS_ROOT / "manifest.json"),
        "selection": selection.as_dict(),
        "job_count": len(jobs),
        "job_ids": [job.job_id for job in jobs],
        "jobs": [job.identity() for job in jobs],
        "seeds": list(plan["seeds"]),
        "train_config": plan["train_config"],
        "dataset_contracts": {
            dataset: plan["datasets"][dataset]
            for dataset in plan["dataset_order"]
            if any(job.dataset == dataset for job in jobs)
        },
        "cache_identity": {key: plan["cache_identity"][key] for key in cache_keys},
        "split_identity": {key: plan["split_identity"][key] for key in split_keys},
        "source_compatibility": source,
        "result_authority": {
            "job_metrics_sha256": _sha256_file(ANALYSIS_ROOT / "job_metrics.csv"),
            "subject_metrics_sha256": _sha256_file(ANALYSIS_ROOT / "subject_metrics.csv"),
            "dataset_summary_sha256": _sha256_file(ANALYSIS_ROOT / "dataset_summary.csv"),
            "overall_summary_sha256": _sha256_file(ANALYSIS_ROOT / "overall_summary.csv"),
            "input_checksum_ledger_sha256": _sha256_file(
                ANALYSIS_ROOT / "input_checksum_ledger.csv"
            ),
        },
        "retention": {
            "trial_rows": False,
            "probabilities": False,
            "scalar_metrics": ["accuracy", "balanced_accuracy"],
            "sufficient_statistic": "integer_confusion_matrix",
        },
        "comparison": {
            "strict_absolute_tolerance": STRICT_ABSOLUTE_TOLERANCE,
            "relative_tolerance": 0.0,
            "paper_display": "three_decimal_percent",
        },
    }
    if schema == REPLAY_SCHEMA:
        payload["namespace_boundary"] = {
            "active": "eeg_mi",
            "sealed_common_grid_v6": "isolated_historical_eeg_mi",
        }
    payload["replay_plan_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _safe_run_root(path: Path, *, create: bool = False) -> Path:
    root = Path(os.path.abspath(os.fspath(path)))
    if create:
        root.mkdir(mode=0o700, parents=False, exist_ok=True)
    observed = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(observed.st_mode):
        raise ReviewerReplayError("run root must be a real directory")
    current = root
    while current != current.parent:
        if current.is_symlink():
            raise ReviewerReplayError("run root has a symlinked ancestor")
        current = current.parent
    return root


def _safe_existing_directory(path: Path, *, label: str) -> Path:
    value = Path(os.path.abspath(os.fspath(path)))
    if not value.exists() or value.is_symlink() or not value.is_dir():
        raise ReviewerReplayError(f"{label} must be an existing real directory")
    current = value
    while current != current.parent:
        observed = current.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ReviewerReplayError(f"{label} has a symlinked/non-directory ancestor")
        current = current.parent
    return value


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _node_exists(path: Path) -> bool:
    """Return whether a directory entry exists without following a symlink."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def validate_external_paths(
    cache_root: Path,
    run_root: Path,
    *,
    create_run: bool,
) -> tuple[Path, Path]:
    cache = _safe_existing_directory(cache_root, label="cache root")
    run = Path(os.path.abspath(os.fspath(run_root)))
    if create_run and not _node_exists(run):
        parent = _safe_existing_directory(run.parent, label="run-root parent")
        if parent != run.parent:
            raise ReviewerReplayError("run-root parent identity changed")
    if _node_exists(run):
        run = _safe_run_root(run)
    project = Path(os.path.abspath(os.fspath(PROJECT_ROOT)))
    if (
        _contains(project, cache)
        or _contains(cache, project)
        or _contains(project, run)
        or _contains(run, project)
        or _contains(cache, run)
        or _contains(run, cache)
    ):
        raise ReviewerReplayError(
            "cache and replay roots must be external to the project and mutually disjoint"
        )
    return cache, run


def _fsync_directory(path: Path) -> None:
    descriptor = full_grid._open_absolute_directory(path)
    try:
        os.fsync(descriptor)
        full_grid._assert_directory_descriptor_path(path, descriptor)
    finally:
        os.close(descriptor)


def _ensure_real_directory(path: Path, *, label: str) -> Path:
    """Create one leaf, then bind it through a no-follow component walk."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        descriptor = full_grid._open_absolute_directory(
            absolute, create_leaf=True, mode=0o700
        )
    except (OSError, full_grid.FullGridError) as error:
        raise ReviewerReplayError(f"{label} is absent or unsafe: {absolute}") from error
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise ReviewerReplayError(f"{label} is not a directory: {absolute}")
        full_grid._assert_directory_descriptor_path(absolute, descriptor)
    finally:
        os.close(descriptor)
    return absolute


def _write_or_repair_pair(
    path: Path,
    sidecar: Path,
    payload: bytes,
) -> str:
    """Create or power-cut-repair one exact canonical payload/hash pair."""

    digest = _sha256_bytes(payload)
    digest_payload = f"{digest}\n".encode("ascii")
    path_exists = _node_exists(path)
    sidecar_exists = _node_exists(sidecar)
    if path_exists and _read_regular(path) != payload:
        raise ReviewerReplayError(f"existing immutable payload differs: {path}")
    if sidecar_exists and _read_regular(sidecar) != digest_payload:
        raise ReviewerReplayError(f"existing immutable checksum differs: {sidecar}")
    if not path_exists:
        _write_bytes_exclusive(path, payload)
    if not sidecar_exists:
        _write_bytes_exclusive(sidecar, digest_payload)
    _fsync_directory(path.parent)
    return digest


class _RunLock:
    def __init__(self, root: Path) -> None:
        self.path = root / ".worker.lock"
        self.descriptor = -1

    def __enter__(self) -> "_RunLock":
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        self.descriptor = os.open(self.path, flags, 0o600)
        observed = os.fstat(self.descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            os.close(self.descriptor)
            self.descriptor = -1
            raise ReviewerReplayError("worker lock is not a unique regular file")
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self.descriptor)
            self.descriptor = -1
            raise ReviewerReplayError("another replay worker owns this run root") from error
        return self

    def __exit__(self, *unused: Any) -> None:
        if self.descriptor >= 0:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = -1


def _write_bytes_exclusive(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_or_validate_replay_plan(run_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    root = _safe_run_root(run_root, create=True)
    plan_path = root / MANIFEST_NAME
    digest_path = root / MANIFEST_DIGEST_NAME
    payload = _canonical_bytes(expected) + b"\n"
    digest = _sha256_bytes(_canonical_bytes({k: v for k, v in expected.items() if k != "replay_plan_sha256"}))
    if expected.get("replay_plan_sha256") != digest:
        raise ReviewerReplayError("replay plan has an invalid internal digest")
    _write_or_repair_pair(plan_path, digest_path, payload)
    observed = _load_json(plan_path)
    if observed != dict(expected):
        raise ReviewerReplayError("existing replay plan differs from the requested selection")
    return dict(expected)


def load_replay_plan(run_root: Path) -> dict[str, Any]:
    root = _safe_run_root(run_root)
    value = _load_json(root / MANIFEST_NAME)
    internal = value.get("replay_plan_sha256")
    computed = _sha256_bytes(_canonical_bytes({k: v for k, v in value.items() if k != "replay_plan_sha256"}))
    file_digest = _sha256_bytes(_canonical_bytes(value) + b"\n")
    if (
        value.get("schema") not in {LEGACY_REPLAY_SCHEMA, REPLAY_SCHEMA}
        or internal != computed
        or _read_regular(root / MANIFEST_NAME) != _canonical_bytes(value) + b"\n"
        or _read_regular(root / MANIFEST_DIGEST_NAME)
        != f"{file_digest}\n".encode("ascii")
    ):
        raise ReviewerReplayError("immutable replay plan is invalid")
    reference = load_reference_plan()
    selection = Selection(**value["selection"])
    rebuilt = build_replay_plan(reference, selection, schema=str(value["schema"]))
    if rebuilt != value:
        raise ReviewerReplayError("replay plan differs from current sealed authority/source")
    return value


def _selection_from_args(args: argparse.Namespace) -> Selection:
    return Selection(
        scope=args.scope,
        model=args.model,
        dataset=getattr(args, "dataset", None),
        subject=getattr(args, "subject", None),
        fold=getattr(args, "fold", None),
        seed=getattr(args, "seed", None),
    )


def _jobs_from_replay_plan(value: Mapping[str, Any]) -> tuple[full_grid.Job, ...]:
    jobs = tuple(full_grid.Job(**row) for row in value["jobs"])
    if [job.job_id for job in jobs] != value["job_ids"] or len(jobs) != value["job_count"]:
        raise ReviewerReplayError("replay-plan job identities are inconsistent")
    return jobs


def _require_resumable_replay_plan(value: Mapping[str, Any]) -> None:
    if value.get("schema") != REPLAY_SCHEMA:
        raise ReviewerReplayError(
            "legacy v1 replay root is read-only after namespace migration; "
            "status and compare remain available, but run/resume requires a new root"
        )


def _load_csv_index(path: Path, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    with io.StringIO(_read_regular(path).decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(key not in reader.fieldnames for key in keys):
            raise ReviewerReplayError(f"CSV schema lacks keys {keys}: {path}")
        result: dict[tuple[str, ...], dict[str, str]] = {}
        for row in reader:
            identity = tuple(row[key] for key in keys)
            if identity in result:
                raise ReviewerReplayError(f"duplicate CSV identity {identity}: {path}")
            result[identity] = row
    return result


def _prediction_ledger() -> dict[tuple[str, ...], dict[str, str]]:
    return _load_csv_index(
        ANALYSIS_ROOT / "input_checksum_ledger.csv", ("job_id",)
    )


def _prediction_npz_sha256(rows: np.ndarray, probabilities: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        rows=np.ascontiguousarray(rows, dtype=np.int64),
        probabilities=np.ascontiguousarray(probabilities, dtype=np.float64),
    )
    return _sha256_bytes(buffer.getvalue())


def _confusion(y: np.ndarray, probabilities: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        labels.ndim != 1
        or values.shape != (len(labels), n_classes)
        or np.any(labels < 0)
        or np.any(labels >= n_classes)
    ):
        raise ReviewerReplayError("prediction/label shapes are invalid")
    predicted = np.argmax(values, axis=1)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    return matrix


def _validate_prediction_output(
    reference: Mapping[str, Any],
    job: full_grid.Job,
    rows: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_rows = np.asarray(rows)
    observed_probabilities = np.asarray(probabilities)
    split_key = full_grid._split_identity_key(job.dataset, job.subject, job.fold)
    planned_test = reference["split_identity"][split_key]["partitions"]["test"]
    n_classes = int(reference["datasets"][job.dataset]["n_classes"])
    if (
        observed_rows.dtype != np.dtype(np.int64)
        or observed_rows.ndim != 1
        or len(observed_rows) != int(planned_test["count"])
        or np.any(observed_rows < 0)
        or len(np.unique(observed_rows)) != len(observed_rows)
        or full_grid._rows_sha256(observed_rows) != planned_test["rows_sha256"]
        or observed_probabilities.dtype != np.dtype(np.float64)
        or observed_probabilities.shape != (len(observed_rows), n_classes)
        or not np.all(np.isfinite(observed_probabilities))
        or np.any(observed_probabilities < 0.0)
        or np.any(observed_probabilities > 1.0)
        or not np.allclose(observed_probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ReviewerReplayError(f"executor output violates the frozen split: {job.job_id}")
    return (
        np.ascontiguousarray(observed_rows),
        np.ascontiguousarray(observed_probabilities),
    )


def _metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or np.any(values < 0):
        raise ReviewerReplayError("confusion matrix is invalid")
    count = int(values.sum())
    supports = values.sum(axis=1)
    if count <= 0 or not np.any(supports > 0):
        raise ReviewerReplayError("confusion matrix is empty")
    recalls = np.diag(values)[supports > 0] / supports[supports > 0]
    return {
        "accuracy": float(np.trace(values) / count),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def _validate_selected_cache(
    reference: Mapping[str, Any], cache_root: Path, job: full_grid.Job
) -> dict[str, Any]:
    split_indices = importlib.import_module(
        f"{HISTORICAL_PRIVATE_PACKAGE}.data"
    ).split_indices

    data = full_grid._load_subject_cache_safely(job.dataset, job.subject, cache_root=cache_root)
    cache_key = full_grid._subject_identity_key(job.dataset, job.subject)
    if data["identity"] != reference["cache_identity"][cache_key]:
        raise ReviewerReplayError(f"cache identity differs for {cache_key}")
    train, validation, test = split_indices(
        job.dataset,
        data["y"],
        data["sessions"],
        data["runs"],
        fold=job.fold,
        subject=job.subject,
    )
    observed = full_grid._one_split_identity(
        dataset=job.dataset,
        subject=job.subject,
        fold=job.fold,
        cache_array_sha256=str(data["identity"]["array_sha256"]),
        trial_count=len(data["y"]),
        train_rows=train,
        validation_rows=validation,
        test_rows=test,
    )
    split_key = full_grid._split_identity_key(job.dataset, job.subject, job.fold)
    if observed != reference["split_identity"][split_key]:
        raise ReviewerReplayError(f"split identity differs for {split_key}")
    return data


def _record_directory(root: Path, job: full_grid.Job) -> Path:
    return root / "records" / job.job_id


def _validate_record(
    root: Path,
    replay_plan: Mapping[str, Any],
    job: full_grid.Job,
    *,
    prediction_ledger: Mapping[tuple[str, ...], Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    directory = _record_directory(root, job)
    if directory.is_symlink() or not directory.is_dir():
        raise ReviewerReplayError(f"record directory is absent/unsafe: {job.job_id}")
    directory_stat = directory.lstat()
    if directory_stat.st_mode & 0o222:
        raise ReviewerReplayError(f"record directory is writable: {job.job_id}")
    names = {path.name for path in directory.iterdir()}
    if names != RECORD_FILES:
        raise ReviewerReplayError(f"record directory has unexpected entries: {job.job_id}")
    record_path = directory / "record.json"
    digest_path = directory / "record.sha256"
    for path in (record_path, digest_path):
        observed = path.lstat()
        if observed.st_mode & 0o222:
            raise ReviewerReplayError(f"record artifact is writable: {path}")
    payload = _read_regular(record_path)
    digest = _sha256_bytes(payload)
    if _read_regular(digest_path) != f"{digest}\n".encode("ascii"):
        raise ReviewerReplayError(f"record checksum differs: {job.job_id}")
    record = _load_json(record_path)
    if payload != _canonical_bytes(record) + b"\n":
        raise ReviewerReplayError(f"record is noncanonical: {job.job_id}")
    required = {
        "schema",
        "replay_plan_sha256",
        "reference_plan_sha256",
        "job",
        "job_id",
        "cache_array_sha256",
        "split_identity_sha256",
        "test_count",
        "n_classes",
        "confusion_matrix",
        "accuracy",
        "balanced_accuracy",
        "replay_predictions_sha256",
        "published_predictions_sha256",
        "prediction_sha256_exact",
        "fit",
        "timing_seconds",
        "environment_sha256",
        "retained_trial_rows",
        "retained_probabilities",
    }
    if set(record) != required:
        raise ReviewerReplayError(f"record schema differs: {job.job_id}")
    if (
        record["schema"] != _artifact_schema(replay_plan, "record")
        or record["replay_plan_sha256"] != replay_plan["replay_plan_sha256"]
        or record["reference_plan_sha256"] != replay_plan["reference_plan_sha256"]
        or record["job"] != job.identity()
        or record["job_id"] != job.job_id
        or record["retained_trial_rows"] is not False
        or record["retained_probabilities"] is not False
    ):
        raise ReviewerReplayError(f"record identity differs: {job.job_id}")
    dataset_contract = replay_plan["dataset_contracts"].get(job.dataset)
    cache_key = full_grid._subject_identity_key(job.dataset, job.subject)
    split_key = full_grid._split_identity_key(job.dataset, job.subject, job.fold)
    expected_cache = replay_plan["cache_identity"].get(cache_key)
    expected_split = replay_plan["split_identity"].get(split_key)
    if not isinstance(dataset_contract, Mapping) or not isinstance(expected_cache, Mapping) or not isinstance(expected_split, Mapping):
        raise ReviewerReplayError(f"record is outside replay-plan identities: {job.job_id}")
    n_classes = dataset_contract["n_classes"]
    expected_test_count = expected_split["partitions"]["test"]["count"]
    expected_split_sha = _sha256_bytes(_canonical_bytes(expected_split))
    raw_matrix = record["confusion_matrix"]
    if (
        type(record["n_classes"]) is not int
        or record["n_classes"] != n_classes
        or type(record["test_count"]) is not int
        or record["test_count"] != expected_test_count
        or record["cache_array_sha256"] != expected_cache["array_sha256"]
        or record["split_identity_sha256"] != expected_split_sha
        or not isinstance(raw_matrix, list)
        or len(raw_matrix) != n_classes
        or any(
            not isinstance(row, list)
            or len(row) != n_classes
            or any(type(value) is not int or value < 0 for value in row)
            for row in raw_matrix
        )
    ):
        raise ReviewerReplayError(f"record cache/split/confusion identity differs: {job.job_id}")
    for name in (
        "cache_array_sha256",
        "split_identity_sha256",
        "replay_predictions_sha256",
        "published_predictions_sha256",
        "environment_sha256",
    ):
        if not isinstance(record[name], str) or HEX64_RE.fullmatch(record[name]) is None:
            raise ReviewerReplayError(f"record hash is invalid: {job.job_id}/{name}")
    if type(record["prediction_sha256_exact"]) is not bool:
        raise ReviewerReplayError(f"prediction hash status is invalid: {job.job_id}")
    ledger = _prediction_ledger() if prediction_ledger is None else prediction_ledger
    expected_published_sha = ledger.get((job.job_id,), {}).get("predictions_sha256")
    if (
        not isinstance(expected_published_sha, str)
        or HEX64_RE.fullmatch(expected_published_sha) is None
        or record["published_predictions_sha256"] != expected_published_sha
    ):
        raise ReviewerReplayError(
            f"record is not bound to the sealed prediction ledger: {job.job_id}"
        )
    environment_payload = _read_regular(root / ENVIRONMENT_NAME)
    environment_sha = _sha256_bytes(environment_payload)
    if (
        record["environment_sha256"] != environment_sha
        or _read_regular(root / ENVIRONMENT_DIGEST_NAME)
        != f"{environment_sha}\n".encode("ascii")
    ):
        raise ReviewerReplayError(f"record environment binding differs: {job.job_id}")
    matrix = np.asarray(raw_matrix, dtype=np.int64)
    metrics = _metrics_from_confusion(matrix)
    if (
        matrix.shape != (n_classes, n_classes)
        or int(matrix.sum()) != int(record["test_count"])
        or any(
            isinstance(record[name], bool)
            or not isinstance(record[name], (int, float))
            or not math.isfinite(float(record[name]))
            or not 0.0 <= float(record[name]) <= 1.0
            for name in ("accuracy", "balanced_accuracy")
        )
        or not math.isclose(float(record["accuracy"]), metrics["accuracy"], rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(
            float(record["balanced_accuracy"]),
            metrics["balanced_accuracy"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or record["prediction_sha256_exact"]
        is not (record["replay_predictions_sha256"] == record["published_predictions_sha256"])
    ):
        raise ReviewerReplayError(f"record statistics are inconsistent: {job.job_id}")
    fit = record["fit"]
    expected_fit_keys = {
        "initial_state_sha256",
        "selection_state_sha256",
        "refit_state_sha256",
        "selected_epoch_index",
        "selection_epochs_run",
        "refit_epochs_run",
        "parameter_count",
        "architecture",
    }
    if (
        not isinstance(fit, Mapping)
        or set(fit) != expected_fit_keys
        or any(
            not isinstance(fit[name], str) or HEX64_RE.fullmatch(fit[name]) is None
            for name in (
                "initial_state_sha256",
                "selection_state_sha256",
                "refit_state_sha256",
            )
        )
        or any(
            type(fit[name]) is not int or fit[name] < 0
            for name in (
                "selected_epoch_index",
                "selection_epochs_run",
                "refit_epochs_run",
            )
        )
        or type(fit["parameter_count"]) is not int
        or fit["parameter_count"] <= 0
        or not isinstance(fit["architecture"], Mapping)
        or fit["architecture"].get("requested_name") != job.model
        or fit["refit_epochs_run"] != fit["selected_epoch_index"] + 1
    ):
        raise ReviewerReplayError(f"record fit identity is invalid: {job.job_id}")
    timing = record["timing_seconds"]
    if (
        not isinstance(timing, Mapping)
        or set(timing)
        != {"selection_fit", "refit_fit", "test_inference", "job_total"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in timing.values()
        )
    ):
        raise ReviewerReplayError(f"record timing is invalid: {job.job_id}")
    return record


def _publish_record(root: Path, job: full_grid.Job, record: Mapping[str, Any]) -> None:
    records = _ensure_real_directory(root / "records", label="records root")
    staging = _ensure_real_directory(root / ".staging", label="staging root")
    _fsync_directory(root)
    destination = _record_directory(root, job)
    if _node_exists(destination):
        return
    stage = staging / f"{job.job_id}.{uuid.uuid4().hex}.stage"
    stage.mkdir(mode=0o700)
    payload = _canonical_bytes(record) + b"\n"
    _write_bytes_exclusive(stage / "record.json", payload)
    _write_bytes_exclusive(stage / "record.sha256", f"{_sha256_bytes(payload)}\n".encode("ascii"))
    _fsync_directory(stage)
    os.chmod(stage, 0o555)
    _fsync_directory(stage)
    source_descriptor = full_grid._open_absolute_directory(staging)
    destination_descriptor = full_grid._open_absolute_directory(records)
    stage_descriptor = full_grid._open_absolute_directory(stage)
    try:
        full_grid._publish_sealed_directory_noreplace_at(
            source_descriptor,
            stage.name,
            stage_descriptor,
            destination_descriptor,
            destination.name,
            expected_names=RECORD_FILES,
            require_readonly_regular_children=True,
        )
    except FileExistsError:
        os.chmod(stage, 0o700)
        for child in stage.iterdir():
            os.chmod(child, 0o600)
            child.unlink()
        stage.rmdir()
    finally:
        os.close(stage_descriptor)
        os.close(destination_descriptor)
        os.close(source_descriptor)
    _fsync_directory(records)
    _fsync_directory(staging)


def _write_or_validate_environment(
    root: Path,
    environment: Mapping[str, Any],
    *,
    selected_gpu_uuid: str | None = None,
    selected_gpu: Mapping[str, Any] | None = None,
    runtime_compatibility: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "schema": ENVIRONMENT_SCHEMA,
        "identity": dict(environment),
        "selected_physical_gpu_uuid": selected_gpu_uuid,
        "selected_physical_gpu": None if selected_gpu is None else dict(selected_gpu),
        "runtime_compatibility": (
            None if runtime_compatibility is None else dict(runtime_compatibility)
        ),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }
    body = _canonical_bytes(payload) + b"\n"
    path = root / ENVIRONMENT_NAME
    sidecar = root / ENVIRONMENT_DIGEST_NAME
    return _write_or_repair_pair(path, sidecar, body)


def _validate_runtime_environment(
    environment: Mapping[str, Any], gpu_uuid: str
) -> dict[str, Any]:
    """Validate CUDA identity and report, but do not forbid, stack drift.

    A reviewer may use a different supported machine or software stack.  That
    is a fixed-protocol replication, not an exact execution-identity replay;
    the distinction is persisted and surfaced by ``compare``.
    """

    if full_grid.GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        raise ReviewerReplayError("selected physical GPU UUID is invalid")
    lock = tomllib.loads(_read_regular(PROJECT_ROOT / "uv.lock").decode("utf-8"))
    raw_locked = lock.get("package")
    if not isinstance(raw_locked, list) or not raw_locked:
        raise ReviewerReplayError("uv.lock package inventory is invalid")

    def normalized_package_name(value: Any) -> str:
        if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
            raise ReviewerReplayError("package inventory contains an invalid name")
        return value.lower().replace("_", "-")

    locked_versions: dict[str, str] = {}
    locked_dependencies: dict[str, tuple[str, ...]] = {}
    for row in raw_locked:
        if not isinstance(row, Mapping):
            raise ReviewerReplayError("uv.lock package entry is invalid")
        name = normalized_package_name(row.get("name"))
        version = row.get("version")
        if name in locked_versions or not isinstance(version, str) or not version:
            raise ReviewerReplayError("uv.lock package inventory is ambiguous")
        raw_dependencies = row.get("dependencies", [])
        if not isinstance(raw_dependencies, list):
            raise ReviewerReplayError("uv.lock dependency inventory is invalid")
        dependencies: list[str] = []
        for dependency in raw_dependencies:
            if not isinstance(dependency, Mapping):
                raise ReviewerReplayError("uv.lock dependency entry is invalid")
            dependencies.append(normalized_package_name(dependency.get("name")))
        locked_versions[name] = version
        locked_dependencies[name] = tuple(dependencies)

    project_name = "benchmark"
    if project_name not in locked_dependencies:
        raise ReviewerReplayError("uv.lock lacks the project dependency root")
    runtime_closure: set[str] = set()
    pending = list(locked_dependencies[project_name])
    while pending:
        name = pending.pop()
        if name in runtime_closure:
            continue
        if name not in locked_dependencies:
            raise ReviewerReplayError(f"uv.lock dependency is unresolved: {name}")
        runtime_closure.add(name)
        pending.extend(locked_dependencies[name])

    raw_installed = environment.get("packages")
    if not isinstance(raw_installed, list):
        raise ReviewerReplayError("installed package inventory is invalid")
    installed: dict[str, str] = {}
    for item in raw_installed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ReviewerReplayError("installed package entry is invalid")
        name = normalized_package_name(item[0])
        version = item[1]
        if name in installed or not isinstance(version, str) or not version:
            raise ReviewerReplayError("installed package inventory is ambiguous")
        installed[name] = version

    missing_or_drifted_runtime = {
        name: {"locked": locked_versions[name], "installed": installed.get(name)}
        for name in sorted(runtime_closure)
        if installed.get(name) != locked_versions[name]
    }
    installed_version_drift = {
        name: {"locked": locked_versions.get(name), "installed": version}
        for name, version in sorted(installed.items())
        if locked_versions.get(name) != version
    }
    unlocked_installed = sorted(set(installed) - set(locked_versions))
    torch_identity = environment.get("torch")
    inventory = environment.get("nvidia_gpu_inventory")
    selected = next(
        (
            dict(row)
            for row in inventory
            if isinstance(row, Mapping) and row.get("uuid") == gpu_uuid
        ),
        None,
    ) if isinstance(inventory, list) else None
    if (
        selected is None
        or not isinstance(torch_identity, Mapping)
        or not isinstance(torch_identity.get("version"), str)
        or not isinstance(torch_identity.get("cuda"), str)
        or not isinstance(torch_identity.get("cudnn"), int)
        or int(torch_identity["cudnn"]) <= 0
        or environment.get("cublas_workspace_config")
        != full_grid.REQUIRED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise ReviewerReplayError(
            "runtime lacks the selected physical CUDA GPU, CUDA/cuDNN, or the "
            "required deterministic CUBLAS workspace configuration"
        )
    python_match = str(environment.get("python_version", "")).startswith("3.12.13")
    platform_value = str(environment.get("platform", ""))
    platform_match = platform_value.startswith("Linux-") and "-x86_64-" in platform_value
    uv_match = environment.get("uv_version") == "uv 0.12.0 (x86_64-unknown-linux-gnu)"
    torch_match = (
        torch_identity.get("version") == "2.6.0+cu124"
        and torch_identity.get("cuda") == "12.4"
    )
    return {
        "selected_physical_gpu": selected,
        "release_locked_stack_match": bool(
            python_match
            and platform_match
            and uv_match
            and torch_match
            and not missing_or_drifted_runtime
            and not installed_version_drift
            and not unlocked_installed
        ),
        "python_match": python_match,
        "linux_x86_64_platform_match": platform_match,
        "uv_match": uv_match,
        "torch_cuda_match": torch_match,
        "runtime_closure_packages": len(runtime_closure),
        "installed_packages": len(installed),
        "missing_or_drifted_runtime_packages": missing_or_drifted_runtime,
        "installed_version_drift": installed_version_drift,
        "unlocked_installed_packages": unlocked_installed,
    }


def _assert_plain_tree(path: Path) -> None:
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise ReviewerReplayError(f"unsafe replay directory tree: {path}")
    for child in path.iterdir():
        child_stat = child.lstat()
        if stat.S_ISLNK(child_stat.st_mode):
            raise ReviewerReplayError(f"symlink in replay directory tree: {child}")
        if stat.S_ISDIR(child_stat.st_mode):
            _assert_plain_tree(child)
        elif not stat.S_ISREG(child_stat.st_mode) or child_stat.st_nlink != 1:
            raise ReviewerReplayError(f"special/aliased replay artifact: {child}")


def _quarantine_node(root: Path, path: Path, *, reason: str) -> Path:
    """Move an invalid private replay tree aside and seal it for inspection."""

    _assert_plain_tree(path)
    quarantine = _ensure_real_directory(root / "quarantine", label="quarantine root")
    source_parent = full_grid._open_absolute_directory(path.parent)
    destination_parent = full_grid._open_absolute_directory(quarantine)
    source_descriptor = full_grid._open_absolute_directory(path)
    destination_name = f"{path.name}.{reason}.{uuid.uuid4().hex}"
    try:
        source_identity = os.fstat(source_descriptor)
        os.fchmod(source_descriptor, 0o700)
        os.fsync(source_descriptor)
        full_grid._atomic_rename_noreplace_at(
            source_parent,
            path.name,
            destination_parent,
            destination_name,
        )
        os.fsync(source_parent)
        os.fsync(destination_parent)
        moved = os.stat(
            destination_name, dir_fd=destination_parent, follow_symlinks=False
        )
        if (moved.st_dev, moved.st_ino) != (
            source_identity.st_dev,
            source_identity.st_ino,
        ):
            raise ReviewerReplayError("quarantine move changed artifact identity")
        full_grid._seal_forensic_node_at(destination_parent, destination_name)
        os.fsync(destination_parent)
    finally:
        os.close(source_descriptor)
        os.close(destination_parent)
        os.close(source_parent)
    return quarantine / destination_name


def _quarantine_stages(root: Path) -> None:
    staging = root / ".staging"
    if not _node_exists(staging):
        return
    _ensure_real_directory(staging, label="staging root")
    for path in sorted(staging.iterdir()):
        if path.is_symlink() or not path.is_dir() or not path.name.endswith(".stage"):
            raise ReviewerReplayError(f"unsafe staging artifact: {path}")
        _quarantine_node(root, path, reason="interrupted-stage")
    _fsync_directory(staging)


def _run_worker(cache_root: Path, run_root: Path, gpu_uuid: str) -> int:
    full_grid._require_virtual_environment()
    full_grid._require_cublas_determinism_environment()
    full_grid._configure_torch_cpu_threads(CPU_THREADS)
    full_grid._verify_visible_cuda_device(gpu_uuid)
    # Eagerly bind every lazily imported formal module before re-hashing the
    # source closure in load_replay_plan/build_replay_plan.
    for module_name in ("baselines", "benchmark", "data", "models", "training"):
        importlib.import_module(f"{HISTORICAL_PRIVATE_PACKAGE}.{module_name}")
    cache, root = validate_external_paths(cache_root, run_root, create_run=False)
    with _RunLock(root):
        _verify_results_bundle()
        replay_plan = load_replay_plan(root)
        _require_resumable_replay_plan(replay_plan)
        reference = load_reference_plan()
        jobs = _jobs_from_replay_plan(replay_plan)
        environment = full_grid._environment_identity()
        runtime_compatibility = _validate_runtime_environment(environment, gpu_uuid)
        selected_gpu = runtime_compatibility["selected_physical_gpu"]
        environment_sha = _write_or_validate_environment(
            root,
            environment,
            selected_gpu_uuid=gpu_uuid,
            selected_gpu=selected_gpu,
            runtime_compatibility=runtime_compatibility,
        )
        _quarantine_stages(root)

        ledger = _prediction_ledger()
        lease = project_gpu_leases.acquire_gpu_lease(
            project_root=PROJECT_ROOT,
            run_root=root,
            plan_sha256=replay_plan["replay_plan_sha256"],
            gpu_uuid=gpu_uuid,
            track_scope=TRACK_SCOPE,
        )
        try:
            for job_index, job in enumerate(jobs, start=1):
                destination = _record_directory(root, job)
                if _node_exists(destination):
                    try:
                        _validate_record(
                            root,
                            replay_plan,
                            job,
                            prediction_ledger=ledger,
                        )
                    except ReviewerReplayError:
                        quarantined = _quarantine_node(
                            root, destination, reason="invalid-record"
                        )
                        print(
                            f"[{job_index}/{len(jobs)}] recover-invalid "
                            f"{job.job_id} -> {quarantined.name}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{job_index}/{len(jobs)}] resume-skip {job.identity()}",
                            flush=True,
                        )
                        continue
                print(
                    f"[{job_index}/{len(jobs)}] start {job.identity()}",
                    flush=True,
                )
                project_gpu_leases.assert_gpu_lease(lease)
                status = full_grid.probe_gpu(gpu_uuid, allowed_pids=(os.getpid(),))
                if not status.safe:
                    raise ReviewerReplayError(
                        f"GPU is not safely available: {status.reason}"
                    )
                subject_data = _validate_selected_cache(reference, cache, job)
                metadata, rows, probabilities = full_grid.execute_benchmark_job(
                    job=job,
                    plan=reference,
                    cache_root=cache,
                    device="cuda:0",
                )
                rows, probabilities = _validate_prediction_output(
                    reference, job, rows, probabilities
                )
                project_gpu_leases.assert_gpu_lease(lease)
                post_resources = full_grid._wait_for_publication_resources(
                    gpu_probe=lambda: full_grid.probe_gpu(
                        gpu_uuid, allowed_pids=(os.getpid(),)
                    ),
                    disk_probe=lambda: full_grid.probe_disk(
                        root,
                        minimum_free_gib=full_grid.DEFAULT_MIN_FREE_GIB,
                    ),
                )
                if not post_resources.safe:
                    raise ReviewerReplayError(
                        "GPU or output filesystem became unsafe during the job: "
                        f"{post_resources.reason}"
                    )
                _verify_results_bundle()
                if load_replay_plan(root) != replay_plan:
                    raise ReviewerReplayError("replay authority changed during training")
                post_subject_data = _validate_selected_cache(reference, cache, job)
                if post_subject_data["identity"] != subject_data["identity"]:
                    raise ReviewerReplayError("selected cache changed during training")
                del post_subject_data
                post_environment = full_grid._environment_identity()
                if post_environment != environment:
                    raise ReviewerReplayError("runtime environment changed during training")
                labels = np.asarray(subject_data["y"], dtype=np.int64)[rows]
                n_classes = int(reference["datasets"][job.dataset]["n_classes"])
                matrix = _confusion(labels, probabilities, n_classes)
                metrics = _metrics_from_confusion(matrix)
                replay_prediction_sha = _prediction_npz_sha256(rows, probabilities)
                published_prediction_sha = ledger.get((job.job_id,), {}).get(
                    "predictions_sha256"
                )
                if published_prediction_sha is None:
                    raise ReviewerReplayError(
                        f"sealed checksum ledger lacks {job.job_id}"
                    )
                split_key = full_grid._split_identity_key(
                    job.dataset, job.subject, job.fold
                )
                fit = metadata["fit"]
                record = {
                "schema": RECORD_SCHEMA,
                "replay_plan_sha256": replay_plan["replay_plan_sha256"],
                "reference_plan_sha256": reference["plan_sha256"],
                "job": job.identity(),
                "job_id": job.job_id,
                "cache_array_sha256": subject_data["identity"]["array_sha256"],
                "split_identity_sha256": _sha256_bytes(
                    _canonical_bytes(reference["split_identity"][split_key])
                ),
                "test_count": int(matrix.sum()),
                "n_classes": n_classes,
                "confusion_matrix": matrix.tolist(),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "replay_predictions_sha256": replay_prediction_sha,
                "published_predictions_sha256": published_prediction_sha,
                "prediction_sha256_exact": replay_prediction_sha == published_prediction_sha,
                "fit": {
                    "initial_state_sha256": fit["initial_state_sha256"],
                    "selection_state_sha256": fit["selection_state_sha256"],
                    "refit_state_sha256": fit["refit_state_sha256"],
                    "selected_epoch_index": fit["source_selected_epoch"],
                    "selection_epochs_run": fit["selection_epochs_run"],
                    "refit_epochs_run": fit["refit_epochs_run"],
                    "parameter_count": fit["parameter_count"],
                    "architecture": fit["architecture"],
                },
                "timing_seconds": metadata["timing_seconds"],
                "environment_sha256": environment_sha,
                "retained_trial_rows": False,
                "retained_probabilities": False,
                }
                del probabilities, rows, labels, subject_data
                _publish_record(root, job, record)
                observed = _validate_record(
                    root,
                    replay_plan,
                    job,
                    prediction_ledger=ledger,
                )
                print(
                    f"[{job_index}/{len(jobs)}] complete {job.job_id} "
                    f"prediction_sha256_exact={observed['prediction_sha256_exact']}",
                    flush=True,
                )
        finally:
            project_gpu_leases.release_gpu_lease(lease)
    return 0


def _validate_optional_root_pair(
    root: Path, payload_name: str, digest_name: str, *, schema: str
) -> str | None:
    path = root / payload_name
    sidecar = root / digest_name
    present = (_node_exists(path), _node_exists(sidecar))
    if present == (False, False):
        return None
    if present != (True, True):
        raise ReviewerReplayError(f"half-published root pair: {payload_name}")
    payload = _read_regular(path)
    value = _load_json(path)
    digest = _sha256_bytes(payload)
    if (
        value.get("schema") != schema
        or payload != _canonical_bytes(value) + b"\n"
        or _read_regular(sidecar) != f"{digest}\n".encode("ascii")
    ):
        raise ReviewerReplayError(f"invalid root artifact pair: {payload_name}")
    return digest


def _assert_sealed_plain_tree(path: Path) -> None:
    _assert_plain_tree(path)
    if path.lstat().st_mode & 0o222:
        raise ReviewerReplayError(f"quarantined directory is writable: {path}")
    for child in path.iterdir():
        if child.is_dir():
            _assert_sealed_plain_tree(child)
        elif child.lstat().st_mode & 0o222:
            raise ReviewerReplayError(f"quarantined artifact is writable: {child}")


def status(run_root: Path) -> dict[str, Any]:
    root = _safe_run_root(run_root)
    replay_plan = load_replay_plan(root)
    jobs = _jobs_from_replay_plan(replay_plan)
    ledger = _prediction_ledger()
    complete: list[str] = []
    invalid: dict[str, str] = {}
    for job in jobs:
        if not _node_exists(_record_directory(root, job)):
            continue
        try:
            _validate_record(
                root, replay_plan, job, prediction_ledger=ledger
            )
        except Exception as error:
            invalid[job.job_id] = str(error)
        else:
            complete.append(job.job_id)
    expected_ids = set(replay_plan["job_ids"])
    extras: list[str] = []
    record_root = root / "records"
    if _node_exists(record_root):
        if record_root.is_symlink() or not record_root.is_dir():
            raise ReviewerReplayError("records root is unsafe")
        extras = sorted(path.name for path in record_root.iterdir() if path.name not in expected_ids)
    staging = root / ".staging"
    if _node_exists(staging) and (staging.is_symlink() or not staging.is_dir()):
        raise ReviewerReplayError("staging root is unsafe")
    partials: list[str] = []
    if _node_exists(staging):
        for path in staging.iterdir():
            if (
                path.is_symlink()
                or not path.is_dir()
                or re.fullmatch(r"job-[0-9a-f]{24}\.[0-9a-f]{32}\.stage", path.name)
                is None
            ):
                raise ReviewerReplayError(f"unsafe staging entry: {path}")
            partials.append(path.name)
        partials.sort()
    quarantine = root / "quarantine"
    quarantined: list[str] = []
    if _node_exists(quarantine):
        if quarantine.is_symlink() or not quarantine.is_dir():
            raise ReviewerReplayError("quarantine root is unsafe")
        for path in quarantine.iterdir():
            _assert_sealed_plain_tree(path)
            quarantined.append(path.name)
        quarantined.sort()
    root_artifact_errors: list[str] = []
    for payload_name, digest_name, schema in (
        (
            ENVIRONMENT_NAME,
            ENVIRONMENT_DIGEST_NAME,
            _artifact_schema(replay_plan, "environment"),
        ),
        (
            COMPARISON_NAME,
            COMPARISON_DIGEST_NAME,
            _artifact_schema(replay_plan, "comparison"),
        ),
    ):
        try:
            _validate_optional_root_pair(
                root, payload_name, digest_name, schema=schema
            )
        except ReviewerReplayError as error:
            root_artifact_errors.append(str(error))
    allowed_root = {
        MANIFEST_NAME,
        MANIFEST_DIGEST_NAME,
        ENVIRONMENT_NAME,
        ENVIRONMENT_DIGEST_NAME,
        COMPARISON_NAME,
        COMPARISON_DIGEST_NAME,
        "records",
        ".staging",
        "quarantine",
        ".worker.lock",
    }
    unexpected_root = sorted(path.name for path in root.iterdir() if path.name not in allowed_root)
    unsafe_root = sorted(
        path.name
        for path in root.iterdir()
        if path.is_symlink()
        or not (path.is_file() or path.is_dir())
    )
    result = {
        "schema": _artifact_schema(replay_plan, "status"),
        "replay_plan_schema": replay_plan["schema"],
        "scope": replay_plan["selection"]["scope"],
        "selection": replay_plan["selection"],
        "expected": len(jobs),
        "complete": len(complete),
        "missing": len(jobs) - len(complete) - len(invalid),
        "invalid": len(invalid),
        "extra": len(extras),
        "partials": len(partials),
        "unexpected_root_entries": len(unexpected_root),
        "unsafe_root_entries": len(unsafe_root),
        "root_artifact_errors": len(root_artifact_errors),
        "quarantined": len(quarantined),
        "exact_selected_complete": (
            len(complete) == len(jobs)
            and not invalid
            and not extras
            and not partials
            and not unexpected_root
            and not unsafe_root
            and not root_artifact_errors
        ),
        "details": {
            "invalid": invalid,
            "extra": extras,
            "partials": partials,
            "unexpected_root_entries": unexpected_root,
            "unsafe_root_entries": unsafe_root,
            "root_artifact_errors": root_artifact_errors,
            "quarantined": quarantined,
        },
    }
    return result


def _aggregate_records(
    replay_plan: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selection = Selection(**replay_plan["selection"])
    by_identity = {
        (
            str(row["job"]["dataset"]),
            int(row["job"]["subject"]),
            int(row["job"]["fold"]),
            int(row["job"]["seed"]),
        ): np.asarray(row["confusion_matrix"], dtype=np.int64)
        for row in records
    }
    reference = load_reference_plan()
    datasets = (
        [selection.dataset]
        if selection.dataset is not None
        else list(reference["dataset_order"])
    )
    dataset_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        assert dataset is not None
        contract = reference["datasets"][dataset]
        subjects = (
            [selection.subject]
            if selection.subject is not None
            else list(contract["subjects"])
        )
        for subject in subjects:
            assert subject is not None
            per_seed: list[dict[str, float]] = []
            for seed in reference["seeds"]:
                if selection.scope == "job" and seed != selection.seed:
                    continue
                folds = (
                    [selection.fold]
                    if selection.scope == "job"
                    else list(contract["folds"])
                )
                matrices = [
                    by_identity[(dataset, int(subject), int(fold), int(seed))]
                    for fold in folds
                ]
                metrics = _metrics_from_confusion(sum(matrices[1:], matrices[0].copy()))
                per_seed.append(metrics)
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "model": selection.model,
                        "subject": int(subject),
                        "seed": int(seed),
                        **metrics,
                    }
                )
            subject_metric = {
                name: float(np.mean([row[name] for row in per_seed]))
                for name in ("accuracy", "balanced_accuracy")
            }
            subject_rows.append(
                {
                    "dataset": dataset,
                    "model": selection.model,
                    "subject": int(subject),
                    **subject_metric,
                }
            )
        selected_subjects = [row for row in subject_rows if row["dataset"] == dataset]
        dataset_rows.append(
            {
                "dataset": dataset,
                "model": selection.model,
                "accuracy": float(np.mean([row["accuracy"] for row in selected_subjects])),
                "balanced_accuracy": float(
                    np.mean([row["balanced_accuracy"] for row in selected_subjects])
                ),
            }
        )
    overall = {
        "model": selection.model,
        "accuracy": float(np.mean([row["accuracy"] for row in dataset_rows])),
        "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in dataset_rows])),
    }
    return {
        "seed_rows": seed_rows,
        "subject_rows": subject_rows,
        "dataset_rows": dataset_rows,
        "overall": overall,
    }


def _comparison_row(
    level: str,
    identity: Mapping[str, Any],
    replay: Mapping[str, Any],
    published: Mapping[str, str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in ("accuracy", "balanced_accuracy"):
        observed = float(replay[name])
        expected = float(published[name])
        metrics[name] = {
            "replay": observed,
            "published": expected,
            "absolute_difference": abs(observed - expected),
            "strict_match": math.isclose(
                observed,
                expected,
                rel_tol=0.0,
                abs_tol=STRICT_ABSOLUTE_TOLERANCE,
            ),
            "replay_display": f"{observed * 100:.3f}%",
            "published_display": f"{expected * 100:.3f}%",
            "display_match": f"{observed * 100:.3f}%" == f"{expected * 100:.3f}%",
        }
    return {
        "level": level,
        "identity": dict(identity),
        "metrics": metrics,
        "strict_match": all(row["strict_match"] for row in metrics.values()),
        "display_match": all(row["display_match"] for row in metrics.values()),
    }


def compare(cache_root: Path, run_root: Path) -> dict[str, Any]:
    _verify_results_bundle()
    cache, root = validate_external_paths(cache_root, run_root, create_run=False)
    replay_plan = load_replay_plan(root)
    run_status = status(root)
    if not run_status["exact_selected_complete"]:
        raise ReviewerReplayError("selected replay is incomplete or contains invalid/extra state")
    jobs = _jobs_from_replay_plan(replay_plan)
    reference = load_reference_plan()
    validated_splits: set[tuple[str, int, int]] = set()
    for job in jobs:
        identity = (job.dataset, job.subject, job.fold)
        if identity in validated_splits:
            continue
        _validate_selected_cache(reference, cache, job)
        validated_splits.add(identity)
    ledger = _prediction_ledger()
    records = [
        _validate_record(
            root, replay_plan, job, prediction_ledger=ledger
        )
        for job in jobs
    ]
    aggregate = _aggregate_records(replay_plan, records)
    selection = Selection(**replay_plan["selection"])
    comparisons: list[dict[str, Any]] = []
    if selection.scope == "job":
        published = _load_csv_index(
            ANALYSIS_ROOT / "job_metrics.csv", ("job_id",)
        )[(jobs[0].job_id,)]
        comparisons.append(
            _comparison_row("job", jobs[0].identity(), records[0], published)
        )
    elif selection.scope == "subject":
        row = aggregate["subject_rows"][0]
        key = (str(selection.dataset), selection.model, str(selection.subject))
        published = _load_csv_index(
            ANALYSIS_ROOT / "subject_metrics.csv", ("dataset", "model", "subject")
        )[key]
        comparisons.append(_comparison_row("subject", dict(zip(("dataset", "model", "subject"), key)), row, published))
    else:
        dataset_index = _load_csv_index(
            ANALYSIS_ROOT / "dataset_summary.csv", ("dataset", "model")
        )
        for row in aggregate["dataset_rows"]:
            key = (str(row["dataset"]), selection.model)
            comparisons.append(
                _comparison_row(
                    "dataset",
                    {"dataset": key[0], "model": key[1]},
                    row,
                    dataset_index[key],
                )
            )
        if selection.scope == "model":
            published = _load_csv_index(
                ANALYSIS_ROOT / "overall_summary.csv", ("model",)
            )[(selection.model,)]
            comparisons.append(
                _comparison_row(
                    "overall",
                    {"model": selection.model},
                    aggregate["overall"],
                    published,
                )
            )
    prediction_matches = sum(bool(row["prediction_sha256_exact"]) for row in records)
    environment_wrapper = _load_json(root / ENVIRONMENT_NAME)
    environment = environment_wrapper.get("identity")
    if not isinstance(environment, Mapping):
        raise ReviewerReplayError("replay environment payload is invalid")
    selected_uuid = environment_wrapper.get("selected_physical_gpu_uuid")
    if not isinstance(selected_uuid, str):
        raise ReviewerReplayError("replay environment lacks selected physical GPU")
    compatibility = _validate_runtime_environment(environment, selected_uuid)
    if (
        environment_wrapper.get("selected_physical_gpu")
        != compatibility["selected_physical_gpu"]
        or environment_wrapper.get("runtime_compatibility") != compatibility
    ):
        raise ReviewerReplayError("stored runtime compatibility assessment differs")
    exact_sources = all(
        row.get("status") == "exact"
        for row in replay_plan["source_compatibility"]["formal_sources"]
    )
    exact_environment = full_grid._normalized_worker_environment_identity(
        environment
    ) == full_grid._normalized_worker_environment_identity(
        reference["environment_identity"]
    )
    exact_gpu = selected_uuid in full_grid._formal_gpu_uuid_roster(
        reference["environment_identity"]
    )
    result = {
        "schema": _artifact_schema(replay_plan, "comparison"),
        "replay_plan_sha256": replay_plan["replay_plan_sha256"],
        "reference_plan_sha256": replay_plan["reference_plan_sha256"],
        "selection": replay_plan["selection"],
        "jobs": len(records),
        "prediction_sha256_exact_jobs": prediction_matches,
        "prediction_sha256_mismatch_jobs": len(records) - prediction_matches,
        "comparisons": comparisons,
        "strict_score_match": all(row["strict_match"] for row in comparisons),
        "paper_display_match": all(row["display_match"] for row in comparisons),
        "bit_exact_prediction_match": prediction_matches == len(records),
        "release_locked_runtime_stack_match": compatibility[
            "release_locked_stack_match"
        ],
        "runtime_compatibility": compatibility,
        "exact_execution_identity_match": _exact_execution_identity_match(
            exact_sources=exact_sources,
            exact_environment=exact_environment,
            exact_gpu=exact_gpu,
            compatibility=compatibility,
        ),
        "fixed_protocol_replication": True,
        "interpretation": (
            "strict_score_match confirms the sealed accuracy/balanced-accuracy row; "
            "bit_exact_prediction_match is the stronger same-prediction check"
        ),
    }
    if replay_plan["schema"] == REPLAY_SCHEMA:
        result["replay_plan_schema"] = replay_plan["schema"]
    body = _canonical_bytes(result) + b"\n"
    path = root / COMPARISON_NAME
    sidecar = root / COMPARISON_DIGEST_NAME
    _write_or_repair_pair(path, sidecar, body)
    return result


def _exact_execution_identity_match(
    *,
    exact_sources: bool,
    exact_environment: bool,
    exact_gpu: bool,
    compatibility: Mapping[str, Any],
) -> bool:
    """Require both sealed identity equality and the release-locked stack."""

    return bool(
        exact_sources
        and exact_environment
        and exact_gpu
        and compatibility.get("release_locked_stack_match") is True
    )


def estimate(selection: Selection) -> dict[str, Any]:
    _verify_results_bundle()
    plan = load_reference_plan()
    jobs = select_jobs(plan, selection)
    selected = {job.job_id for job in jobs}
    seconds = 0.0
    found: set[str] = set()
    with io.StringIO(_read_regular(ANALYSIS_ROOT / "job_metrics.csv").decode("utf-8"), newline="") as handle:
        for row in csv.DictReader(handle):
            if row["job_id"] in selected:
                seconds += float(row["job_total_seconds"])
                found.add(row["job_id"])
    if found != selected:
        raise ReviewerReplayError("timing table does not cover the exact selection")
    return {
        "schema": ESTIMATE_SCHEMA,
        "selection": selection.as_dict(),
        "jobs": len(jobs),
        "historical_aggregate_gpu_hours": seconds / 3600.0,
        "hardware": "four-NVIDIA-RTX-A5000 formal workstation",
        "warning": "measured historical sum, not a wall-time promise",
    }


def _resolve_one_gpu(value: str) -> full_grid.GPUIdentity:
    inventory = full_grid._discover_gpu_identities()
    matches = [row for row in inventory if row.index == value or row.uuid == value]
    if len(matches) != 1:
        raise ReviewerReplayError(f"GPU must identify exactly one physical device: {value}")
    return matches[0]


def _spawn_worker(args: argparse.Namespace, replay_plan: Mapping[str, Any]) -> int:
    identity = _resolve_one_gpu(args.gpu)
    initial = full_grid.probe_gpu(identity.uuid)
    if not initial.safe:
        raise ReviewerReplayError(f"GPU is not safely available: {initial.reason}")
    command = [
        sys.executable,
        "-m",
        "benchmark.reviewer_replay",
        "_worker",
        "--cache-root",
        str(Path(args.cache_root).resolve()),
        "--run-root",
        str(Path(args.run_root).resolve()),
        "--gpu",
        identity.uuid,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = identity.uuid
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUBLAS_WORKSPACE_CONFIG"] = full_grid.REQUIRED_CUBLAS_WORKSPACE_CONFIG
    environment["FULL_GRID_GPU_UUID"] = identity.uuid
    environment["REVIEWER_REPLAY_PLAN_SHA256"] = str(replay_plan["replay_plan_sha256"])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    active_tcformer_root = environment.get("EEG_MI_TCFORMER_ROOT")
    if not isinstance(active_tcformer_root, str) or not active_tcformer_root:
        active_tcformer_root = str(PROJECT_ROOT / "third_party" / "TCFormer")
        environment["EEG_MI_TCFORMER_ROOT"] = active_tcformer_root
    environment["EEG_MI_TCFORMER_ROOT"] = active_tcformer_root
    uv_bin = environment.get("UV_BIN")
    if isinstance(uv_bin, str) and uv_bin:
        uv_path = Path(uv_bin)
        if uv_path.is_absolute():
            environment["PATH"] = os.pathsep.join(
                (str(uv_path.parent), environment.get("PATH", ""))
            )
    for name in THREAD_VARIABLES:
        environment[name] = str(CPU_THREADS)
    process = subprocess.Popen(command, env=environment)
    try:
        return process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--subject", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list the 43 exact common-grid model IDs")
    estimate_parser = subparsers.add_parser("estimate", help="show exact jobs and historical GPU time")
    _add_selection_arguments(estimate_parser)
    run_parser = subparsers.add_parser("run", help="initialize or resume one bounded replay")
    _add_selection_arguments(run_parser)
    run_parser.add_argument("--cache-root", type=Path, required=True)
    run_parser.add_argument("--run-root", type=Path, required=True)
    run_parser.add_argument("--gpu", required=True)
    status_parser = subparsers.add_parser("status", help="validate bounded replay progress")
    status_parser.add_argument("--run-root", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare", help="compare a completed replay to sealed tables")
    compare_parser.add_argument("--cache-root", type=Path, required=True)
    compare_parser.add_argument("--run-root", type=Path, required=True)
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--cache-root", type=Path, required=True)
    worker.add_argument("--run-root", type=Path, required=True)
    worker.add_argument("--gpu", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            plan = load_reference_plan()
            print(json.dumps({"models": plan["architectures"], "count": len(plan["architectures"])}, indent=2))
            return 0
        if args.command == "estimate":
            print(json.dumps(estimate(_selection_from_args(args)), indent=2, sort_keys=True))
            return 0
        if args.command == "status":
            result = status(args.run_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["exact_selected_complete"] else 1
        if args.command == "compare":
            result = compare(args.cache_root, args.run_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["strict_score_match"] else 1
        if args.command == "_worker":
            expected = os.environ.get("REVIEWER_REPLAY_PLAN_SHA256")
            value = load_replay_plan(args.run_root)
            _require_resumable_replay_plan(value)
            if expected != value["replay_plan_sha256"]:
                raise ReviewerReplayError("worker is not bound to its parent replay plan")
            return _run_worker(args.cache_root.resolve(), args.run_root.resolve(), args.gpu)
        if args.command == "run":
            full_grid._require_virtual_environment()
            _verify_results_bundle()
            cache, run_root = validate_external_paths(
                args.cache_root, args.run_root, create_run=True
            )
            args.cache_root = cache
            args.run_root = run_root
            reference = load_reference_plan()
            selection = _selection_from_args(args)
            if _node_exists(args.run_root / MANIFEST_NAME):
                existing = load_replay_plan(args.run_root)
                _require_resumable_replay_plan(existing)
            value = build_replay_plan(reference, selection)
            write_or_validate_replay_plan(args.run_root, value)
            code = _spawn_worker(args, value)
            if code != 0:
                return code
            result = compare(args.cache_root, args.run_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["strict_score_match"] else 1
        raise ReviewerReplayError(f"unknown command: {args.command}")
    except (
        ReviewerReplayError,
        full_grid.FullGridError,
        project_gpu_leases.ProjectGPULeaseError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"reviewer replay failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
