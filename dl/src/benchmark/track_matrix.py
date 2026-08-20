"""Read-only execution matrix for the 70-entry EEG-MI model registry.

The common raw-trial grid, author-faithful recipes, procedure-coupled models,
native-transfer experiments, and deterministic controls answer different
scientific questions.  This module makes that separation machine-readable.
It deliberately imports no training framework and exposes no training command.

``matrix_manifest()`` returns exactly one cell for every
``(registry stable_id, benchmark dataset)`` pair.  Eligible cells which are not
currently executable against the locked cache contract are recorded as
``missing_adapter_or_blocker`` rather than silently treated as missing common
grid results.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

from .model_registry import BENCHMARK_DATASETS, MODEL_REGISTRY, ModelRecord


MATRIX_SCHEMA: Final[str] = "eeg-mi-track-execution-matrix-v2"
FORMAL_SEEDS: Final[tuple[int, ...]] = (7, 17, 27, 37, 47)

COMMON_FULL_GRID: Final[str] = "common_full_grid"
SEPARATE_ELIGIBLE: Final[str] = "separate_eligible"
REGISTRY_NA: Final[str] = "registry_na"
MISSING_ADAPTER: Final[str] = "missing_adapter_or_blocker"
CELL_STATUSES: Final[frozenset[str]] = frozenset(
    {COMMON_FULL_GRID, SEPARATE_ELIGIBLE, REGISTRY_NA, MISSING_ADAPTER}
)

HARMONIZED_DIRECT: Final[str] = "harmonized_v2_direct"
HARMONIZED_DERIVED: Final[str] = "harmonized_v2_deterministic_derived_view"
NATIVE_REQUIRED: Final[str] = "native_v2_required"
CACHE_COMPATIBILITY: Final[frozenset[str]] = frozenset(
    {HARMONIZED_DIRECT, HARMONIZED_DERIVED, NATIVE_REQUIRED}
)


@dataclass(frozen=True)
class DatasetGrid:
    """One established benchmark dimension."""

    subjects: tuple[int, ...]
    folds: tuple[int, ...]
    seeds: tuple[int, ...] = FORMAL_SEEDS

    @property
    def stochastic_jobs(self) -> int:
        return len(self.subjects) * len(self.folds) * len(self.seeds)

    @property
    def deterministic_jobs(self) -> int:
        return len(self.subjects) * len(self.folds)


DATASET_GRIDS: Final[Mapping[str, DatasetGrid]] = {
    "local_exp4": DatasetGrid(
        subjects=(1, 3, 4, 5, 6, 7, 8, 10),
        folds=(0,),
    ),
    "bnci2014_001": DatasetGrid(subjects=tuple(range(1, 10)), folds=(0,)),
    "bnci2014_004": DatasetGrid(subjects=tuple(range(1, 10)), folds=(0,)),
    "cho2017": DatasetGrid(subjects=tuple(range(1, 53)), folds=tuple(range(5))),
    "physionet_mi": DatasetGrid(
        subjects=tuple(range(1, 55)), folds=tuple(range(3))
    ),
}

# Native transfer has a source/target subject boundary that must not be erased
# by substituting the larger common-grid Cho cohort.
NATIVE_TARGET_GRIDS: Final[Mapping[str, DatasetGrid]] = {
    "cho2017": DatasetGrid(subjects=tuple(range(16, 53)), folds=tuple(range(5))),
    "physionet_mi": DatasetGrid(
        subjects=tuple(range(1, 55)), folds=tuple(range(3))
    ),
}


@dataclass(frozen=True)
class ExecutionDescriptor:
    """Static execution contract for one non-common registry entry."""

    stable_id: str
    registry_implementation: str
    component_callable: str
    runtime_key: str
    input_representation: str
    class_constraint: str
    preprocessing_protocol: str
    training_protocol: str
    cache_compatibility: str
    harmonized_v2_safe: bool
    harmonized_v2_note: str
    grid_kind: str = "standard_stochastic"
    configuration_identity: str = ""


@dataclass(frozen=True)
class Cardinality:
    """Proposed atomic execution units and result records for one cell."""

    subjects: tuple[int, ...]
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    atomic_jobs: int
    result_records: int
    seed_policy: str


@dataclass(frozen=True)
class TrackCell:
    """One exact registry-by-dataset decision."""

    stable_id: str
    display_name: str
    dataset: str
    track: str
    status: str
    registry_eligible: bool
    reason: str
    registry_implementation: str
    component_callable: str | None
    adapter_callable: str | None
    runtime_key: str | None
    existing_result_coverage: tuple[str, ...]
    cache_compatibility: str
    harmonized_v2_safe: bool
    atomic_jobs: int
    result_records: int
    subjects: tuple[int, ...]
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    seed_policy: str


@dataclass(frozen=True)
class DriftIssue:
    """A registry-to-executor ambiguity that needs an explicit resolution."""

    code: str
    severity: str
    stable_ids: tuple[str, ...]
    detail: str
    required_action: str


@dataclass(frozen=True)
class BacklogItem:
    """Ordered implementation backlog; no item authorizes a training run."""

    priority: int
    key: str
    scientific_value: str
    complexity: str
    affected_stable_ids: tuple[str, ...]
    deliverable: str


_BINARY_CONSTRAINT = (
    "Exactly two labels (left/right encoded 0/1); the implementation's heads "
    "and reflection-label transformations are binary."
)
_FOUR_BAND_VIEW = (
    "Fit channel scaling only on the declared fitting rows; apply a frozen "
    "per-trial four-band transform; construct trace-shrunk SPD covariances "
    "without labels or cross-trial statistics."
)
_TWO_PHASE_BINARY = (
    "Phase A fits train rows and uses validation only for duration/declared "
    "route selection. Phase B resets, refits all learned preprocessing on "
    "train+validation, trains the selected duration, then predicts test once."
)
_NATIVE_TRANSFER = (
    "Use native-profile v2 caches and the byte-pinned source checkpoint. "
    "Target Phase A selects duration on train/validation; Phase B reconstructs "
    "the exact initialization, resets BatchNorm/scalers on train+validation, "
    "refits, and publishes raw test predictions atomically."
)
_REGISTRY_BY_ID: Final[Mapping[str, ModelRecord]] = {
    record.stable_id: record for record in MODEL_REGISTRY
}


def _descriptor(
    stable_id: str,
    component_callable: str,
    runtime_key: str,
    input_representation: str,
    class_constraint: str,
    preprocessing_protocol: str,
    training_protocol: str,
    cache_compatibility: str,
    harmonized_v2_safe: bool,
    harmonized_v2_note: str,
    *,
    grid_kind: str = "standard_stochastic",
    configuration_identity: str = "",
) -> ExecutionDescriptor:
    return ExecutionDescriptor(
        stable_id=stable_id,
        registry_implementation=_REGISTRY_BY_ID[stable_id].implementation,
        component_callable=component_callable,
        runtime_key=runtime_key,
        input_representation=input_representation,
        class_constraint=class_constraint,
        preprocessing_protocol=preprocessing_protocol,
        training_protocol=training_protocol,
        cache_compatibility=cache_compatibility,
        harmonized_v2_safe=harmonized_v2_safe,
        harmonized_v2_note=harmonized_v2_note,
        grid_kind=grid_kind,
        configuration_identity=configuration_identity,
    )


_DESCRIPTORS: list[ExecutionDescriptor] = [
    _descriptor(
        "architecture.geoadaptnet",
        "benchmark.research.engine.train_model",
        "geoadapt",
        "four-band SPD covariance tensors",
        _BINARY_CONSTRAINT,
        _FOUR_BAND_VIEW,
        _TWO_PHASE_BINARY,
        HARMONIZED_DERIVED,
        True,
        "The view is a deterministic model-side transform of harmonized v2 raw trials.",
        configuration_identity=(
            "GeoAdaptNet plus an explicit frozen TrainConfig; no all-dataset "
            "v2 adapter currently binds those values."
        ),
    ),
    _descriptor(
        "architecture.geoadaptnet_fb",
        "benchmark.research.filterbank_net.FilterBankSPDClassifier.fit",
        "geoadapt_fb",
        "broadband raw EEG",
        _BINARY_CONSTRAINT,
        "Use harmonized v2 raw trials; fit the wrapper's channel statistics on "
        "training rows only. The learnable Sinc bank constructs SPD views internally.",
        _TWO_PHASE_BINARY,
        HARMONIZED_DIRECT,
        True,
        "Raw shape and 128 Hz sampling satisfy the model, but a "
        "reset/fixed-refit adapter is absent.",
        configuration_identity="FilterBankSPDNet(reduced_dim=None).",
    ),
    _descriptor(
        "architecture.geoadaptnet_fbsp",
        "benchmark.research.filterbank_net.FilterBankSPDClassifier.fit",
        "geoadapt_fbsp",
        "broadband raw EEG",
        _BINARY_CONSTRAINT,
        "Use harmonized v2 raw trials; fit the wrapper's channel statistics on "
        "training rows only; learn a per-band BiMap before ReEig/LogEig.",
        _TWO_PHASE_BINARY,
        HARMONIZED_DIRECT,
        True,
        "Raw shape and 128 Hz sampling satisfy the model, but a "
        "reset/fixed-refit adapter is absent.",
        configuration_identity="FilterBankSPDNet(reduced_dim=8).",
    ),
]

for stable_id, model_key, component_callable, identity in (
    (
        "architecture.cameo",
        "cameo",
        "benchmark.research.cameo_net.CAMEOClassifier.fit",
        "Frozen CAMEOConfig and its declared mixture/rho candidate set.",
    ),
    (
        "architecture.hemiparity",
        "hemiparity",
        "benchmark.research.parity_net.HemiParityClassifier.fit",
        "Frozen ParityConfig.",
    ),
    (
        "architecture.parity_fuse",
        "parity_fuse",
        "benchmark.research.parity_fuse_net.ParityFuseClassifier.fit",
        "Frozen ParityFuseConfig.",
    ),
):
    _DESCRIPTORS.append(
        _descriptor(
            stable_id,
            component_callable,
            model_key,
            "paired raw/reflected EEG plus paired four-band SPD tangent features",
            _BINARY_CONSTRAINT,
            _FOUR_BAND_VIEW
            + " Construct the sagittal reflection permutation from the cached channel names.",
            _TWO_PHASE_BINARY,
            HARMONIZED_DERIVED,
            True,
            "The local v2 adapter proves compatibility; other datasets need a "
            "shape-general adapter.",
            configuration_identity=identity,
        )
    )

_DESCRIPTORS.extend(
    [
        _descriptor(
            "architecture.hemi_q_field",
            "benchmark.research.hemi_q_field_net.HemiQFieldClassifier.fit",
            "hemi_q_field",
            "8--30 Hz supplied-bipolar C3/Cz/C4 trials",
            _BINARY_CONSTRAINT,
            "Starting from BNCI2014-004 harmonized v2 only, apply the frozen "
            "per-trial 8--30 Hz transform and retain the supplied bipolar order; "
            "never interpret nominal coordinates as point electrodes.",
            _TWO_PHASE_BINARY,
            HARMONIZED_DERIVED,
            True,
            "The source trials are compatible, but the existing HemiQ runner "
            "owns a different legacy cache/protocol.",
            configuration_identity=(
                "One-time-confirmed frozen HemiQFieldConfig from "
                "predecessor-workspace/hemi_q/hemi_q_final_frozen_manifest.json."
            ),
        ),
        _descriptor(
            "architecture.cardinal_spline_dual_view",
            "benchmark.dual_view_training.fit_dual_view_target",
            "cardinal_spline_dual_view",
            "native raw EEG and a geometry-only spline-canonical view",
            _BINARY_CONSTRAINT,
            "Use native-profile v2 raw volts. Fit the spherical-spline operator "
            "from geometry and distinct native/canonical scalers on fitting rows only.",
            _NATIVE_TRANSFER,
            NATIVE_REQUIRED,
            False,
            "The harmonized profile has already discarded the native montage; "
            "it cannot prove native transfer.",
            grid_kind="native_target",
            configuration_identity=(
                "CardinalSplineDualViewNet with frozen auxiliary/preference "
                "weights; no production checkpoint/artifact runner exists."
            ),
        ),
    ]
)

for version in ("v1", "v2", "v3", "v4", "v5"):
    _DESCRIPTORS.append(
        _descriptor(
            f"architecture.orbit_{version}",
            "benchmark.research.orbit_transport_net.OrbitTransportClassifier.fit",
            f"orbit_{version}",
            "paired raw/reflected EEG plus paired four-band SPD anchor logits",
            _BINARY_CONSTRAINT,
            _FOUR_BAND_VIEW
            + " Construct reflection views from the exact cached channel order.",
            _TWO_PHASE_BINARY,
            HARMONIZED_DERIVED,
            True,
            (
                "ORBIT-v3 has an exact local v2 adapter. Other versions and "
                "datasets currently depend on legacy loaders/protocols."
            ),
            configuration_identity=(
                f"Version-specific OrbitTransportConfig for {version}; the "
                "registry implementation path alone does not resolve this config."
            ),
        )
    )

_DESCRIPTORS.extend(
    [
        _descriptor(
            "reference.tcformer",
            "benchmark.reference_training.fit_tcformer_reference",
            "tcformer",
            "harmonized broadband EEG trial",
            "Two or more classes; output width must equal the dataset class count.",
            "Fit the study channel scaler on the declared source rows only; "
            "use the pinned official TCFormer factory and harmonized 128 Hz trials.",
            "Train the released Adam + warmup/cosine + one-for-one 8-segment "
            "S&R recipe for the fixed 1000-epoch horizon on source rows; no "
            "validation checkpoint selection; predict the held-out test once.",
            HARMONIZED_DIRECT,
            True,
            "Both 256 and 320 sample windows divide exactly into eight S&R segments.",
            configuration_identity=(
                "Official TCFormer commit "
                "74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5 plus "
                "TCFormerReferenceConfig."
            ),
        ),
        _descriptor(
            "reference.fbcnet",
            "benchmark.reference_training.fit_fbcnet_reference",
            "fbcnet",
            "nine-band causal Chebyshev-II filtered harmonized EEG",
            "Two or more classes; output width must equal the dataset class count.",
            "Fit the study scaler on training rows only, then apply "
            "apply_fbcnet_cheby2_filterbank independently per epoch.",
            "Released Adam and relative-no-decrease Stage 1; restore best model "
            "and optimizer; Stage 2 trains train+validation until the released "
            "loss threshold or cap; predict test only afterward.",
            HARMONIZED_DERIVED,
            True,
            "The pinned causal filter bank is an explicit deterministic transform of v2 trials.",
            configuration_identity=(
                "Braindecode 1.6.1 PrefilteredFBCNetAdapter, official recipe "
                "commit de1bbdd8a54cb1e466830e3d47070e0e56761a37, and "
                "FBCNetReferenceConfig."
            ),
        ),
    ]
)

for stable_id, condition in (
    ("procedure.pretrained_cardinal_fbc", "pretrained_cardinal_fbc"),
    ("procedure.scratch_cardinal_fbc", "scratch_cardinal_fbc"),
    ("procedure.scratch_fbmsnet_legacy_transfer", "scratch_fbmsnet"),
    (
        "procedure.pretrained_indexed_fbcnet_spherical_spline",
        "pretrained_indexed_fbcnet_spherical_spline",
    ),
):
    _DESCRIPTORS.append(
        _descriptor(
            stable_id,
            "benchmark.native_transfer.run_record",
            condition,
            "native-montage raw EEG under the CardinalFBC source/target protocol",
            _BINARY_CONSTRAINT,
            "Load and validate native-profile v2 caches and the immutable "
            "CardinalFBC checkpoint; any spline map is geometry-only and fitted "
            "inside the authorized target record.",
            _NATIVE_TRANSFER,
            NATIVE_REQUIRED,
            False,
            "The procedure's scientific object is native montage transfer; "
            "harmonized-profile input is invalid.",
            grid_kind="native_target",
            configuration_identity=f"benchmark.native_transfer condition={condition!r}.",
        )
    )

for stable_id, condition in (
    ("procedure.pretrained_cardinal_fbms", "pretrained_cardinal_fbms"),
    (
        "procedure.scratch_cardinal_fbms_canonical_seeded",
        "scratch_cardinal_fbms_canonical_seeded",
    ),
    (
        "procedure.scratch_cardinal_fbms_native_projected",
        "scratch_cardinal_fbms_native_projected",
    ),
    ("procedure.scratch_fbmsnet_native", "scratch_fbmsnet_native"),
    (
        "procedure.pretrained_indexed_fbmsnet_spherical_spline",
        "pretrained_indexed_fbmsnet_spherical_spline",
    ),
):
    _DESCRIPTORS.append(
        _descriptor(
            stable_id,
            "benchmark.native_fbms_transfer.run_record",
            condition,
            "native-montage raw EEG under the CardinalFBMS source/target protocol",
            _BINARY_CONSTRAINT,
            "Load and validate native-profile v2 caches and the byte-pinned "
            "CardinalFBMS checkpoint; preserve the condition-specific initialization.",
            _NATIVE_TRANSFER,
            NATIVE_REQUIRED,
            False,
            "The complete audited transfer grid requires native-profile data, "
            "not harmonized-profile tensors.",
            grid_kind="native_target",
            configuration_identity=f"benchmark.native_fbms_transfer condition={condition!r}.",
        )
    )

for stable_id, component_callable, runtime_key, representation in (
    (
        "control.riemann",
        "benchmark.research.baselines.RiemannianTangentLogistic.fit",
        "riemann",
        "four-band SPD covariance tensors",
    ),
    (
        "control.tangent_anchor",
        "benchmark.research.tangent_anchor.TangentAnchorClassifier.fit",
        "tangent_anchor",
        "four-band SPD covariance tensors",
    ),
    (
        "control.ea_fbcsp",
        "benchmark.research.baselines.EAFilterBankCSP.fit",
        "ea_fbcsp",
        "four-band filtered EEG epochs",
    ),
):
    _DESCRIPTORS.append(
        _descriptor(
            stable_id,
            component_callable,
            runtime_key,
            representation,
            "Binary and multiclass labels supported by the estimator.",
            "Use train/refit-row channel scaling followed by "
            "benchmark.local_geometric_controls.fixed_filter_bank; use "
            "spd_covariances for tangent controls.",
            "Select the small prespecified hyperparameter grid by validation "
            "NLL, refit deterministically on train+validation, and predict test "
            "without transductive calibrate(). One subject/fold execution emits "
            "one seedless score-blind prediction record.",
            HARMONIZED_DERIVED,
            True,
            "The fixed FIR/covariance functions accept either v2 epoch length; "
            "the score-blind control grid binds all five opened datasets.",
            grid_kind="standard_deterministic",
            configuration_identity=(
                "Frozen hyperparameter grids in benchmark.local_geometric_controls."
            ),
        )
    )

EXECUTION_DESCRIPTORS: Final[tuple[ExecutionDescriptor, ...]] = tuple(
    _DESCRIPTORS
)
DESCRIPTOR_BY_ID: Final[Mapping[str, ExecutionDescriptor]] = {
    descriptor.stable_id: descriptor for descriptor in EXECUTION_DESCRIPTORS
}


# An entry here means that a cache/protocol-bound executor exists now.  An
# eligible descriptor without an entry remains a concrete proposed job, but is
# labelled as blocked in the matrix.
_ready_adapters: dict[tuple[str, str], str] = {
    ("architecture.cameo", "local_exp4"): (
        "benchmark.local_outer_refit_benchmark.run_benchmark"
    ),
    ("architecture.hemiparity", "local_exp4"): (
        "benchmark.local_outer_refit_benchmark.run_benchmark"
    ),
    ("architecture.parity_fuse", "local_exp4"): (
        "benchmark.local_outer_refit_benchmark.run_benchmark"
    ),
    ("architecture.orbit_v3", "local_exp4"): (
        "benchmark.local_outer_refit_benchmark.run_benchmark"
    ),
    ("architecture.hemi_q_field", "bnci2014_004"): (
        "benchmark.hemiq_v2_grid.execute_job"
    ),
}

for descriptor in EXECUTION_DESCRIPTORS:
    if (
        descriptor.stable_id.startswith("procedure.")
        and descriptor.component_callable
        != "benchmark.native_fbms_transfer.run_record"
    ):
        for dataset in NATIVE_TARGET_GRIDS:
            _ready_adapters[(descriptor.stable_id, dataset)] = (
                descriptor.component_callable
            )
    if descriptor.stable_id.startswith("control."):
        for dataset in DATASET_GRIDS:
            _ready_adapters[(descriptor.stable_id, dataset)] = (
                "benchmark.control_grid.main"
            )
_READY_ADAPTERS: Final[Mapping[tuple[str, str], str]] = dict(_ready_adapters)
del _ready_adapters


def _blocker_reason(record: ModelRecord, dataset: str) -> str:
    stable_id = record.stable_id
    if (
        DESCRIPTOR_BY_ID[stable_id].component_callable
        == "benchmark.native_fbms_transfer.run_record"
    ):
        return (
            "The completed native-FBMS v1 artifacts remain auditable, but their "
            "source-pinned writer was retired after the namespace migration. A "
            "new versioned writer and artifact schema are required for execution."
        )
    if stable_id.startswith("reference."):
        return (
            "The train-only author-recipe callable exists, but there is no "
            "cache/split-bound atomic runner that constructs the pinned factory, "
            "predicts once, records timing/provenance, and resumes safely."
        )
    if stable_id.startswith("control."):
        return "The score-blind control-grid contract does not include this dataset."
    if stable_id == "architecture.cardinal_spline_dual_view":
        return (
            "The model and leakage-safe in-memory fitters exist, but no frozen "
            "source checkpoint publisher, per-target atomic runner, or audited "
            "result schema has been promoted."
        )
    if stable_id == "architecture.hemi_q_field":
        return (
            "The separate all-nine-subject harmonized-v2 development adapter "
            "exists, but remains unlaunched until its independent no-edit audit."
        )
    if stable_id.startswith("architecture.orbit_"):
        if stable_id == "architecture.orbit_v3":
            return (
                f"The exact v2 refit adapter exists only for local_exp4; {dataset} "
                "still uses a legacy loader/split and cannot be merged with the "
                "current benchmark."
            )
        return (
            "This ORBIT ablation has only legacy version-specific artifacts. "
            "No stable-ID-to-frozen-config resolver and no harmonized-v2 "
            "selection/reset/refit adapter exists."
        )
    if stable_id in {
        "architecture.cameo",
        "architecture.hemiparity",
        "architecture.parity_fuse",
    }:
        return (
            f"The exact v2 refit adapter exists only for local_exp4; {dataset} "
            "requires a shape-general cache adapter and frozen configuration "
            "without importing legacy scores."
        )
    if stable_id.startswith("architecture.geoadaptnet"):
        return (
            "The architecture/fitter exists, but no harmonized-v2 atomic runner "
            "implements the common split identities, reset/refit contract, "
            "multidataset provenance, and one-shot test prediction."
        )
    return "No cache/protocol-bound executor is registered for this eligible cell."


def _grid_for_descriptor(
    descriptor: ExecutionDescriptor, dataset: str
) -> DatasetGrid:
    if descriptor.grid_kind == "native_target":
        return NATIVE_TARGET_GRIDS[dataset]
    return DATASET_GRIDS[dataset]


def proposed_cardinality(stable_id: str, dataset: str) -> Cardinality:
    """Return the exact planned counts for one eligible non-common cell."""

    if stable_id not in _REGISTRY_BY_ID:
        raise KeyError(f"unknown registry stable_id: {stable_id!r}")
    if dataset not in DATASET_GRIDS:
        raise KeyError(f"unknown benchmark dataset: {dataset!r}")
    record = _REGISTRY_BY_ID[stable_id]
    if record.track == "common":
        raise ValueError(
            f"{stable_id!r} is a common-grid entry, not a non-common proposal"
        )
    eligibility = record.eligibility_for(dataset)
    if not eligibility.eligible:
        raise ValueError(
            f"{stable_id}/{dataset} is registry-ineligible: {eligibility.reason}"
        )
    descriptor = DESCRIPTOR_BY_ID[stable_id]
    grid = _grid_for_descriptor(descriptor, dataset)
    if descriptor.grid_kind == "standard_deterministic":
        return Cardinality(
            subjects=grid.subjects,
            folds=grid.folds,
            seeds=(),
            atomic_jobs=grid.deterministic_jobs,
            result_records=grid.deterministic_jobs,
            seed_policy=(
                "one canonical deterministic execution per subject/fold; no "
                "seed dimension and no synthetic seed-identity records"
            ),
        )
    result_records = grid.stochastic_jobs
    return Cardinality(
        subjects=grid.subjects,
        folds=grid.folds,
        seeds=grid.seeds,
        atomic_jobs=result_records,
        result_records=result_records,
        seed_policy="one independently initialized fit per declared seed",
    )


def _common_cardinality(dataset: str) -> Cardinality:
    grid = DATASET_GRIDS[dataset]
    return Cardinality(
        subjects=grid.subjects,
        folds=grid.folds,
        seeds=grid.seeds,
        atomic_jobs=grid.stochastic_jobs,
        result_records=grid.stochastic_jobs,
        seed_policy="one independently initialized common-recipe fit per seed",
    )


def build_track_matrix() -> tuple[TrackCell, ...]:
    """Build the canonical registry-order, dataset-order 70x5 matrix."""

    cells: list[TrackCell] = []
    for record in MODEL_REGISTRY:
        for dataset in BENCHMARK_DATASETS:
            eligibility = record.eligibility_for(dataset)
            if record.track == "common":
                cardinality = _common_cardinality(dataset)
                cells.append(
                    TrackCell(
                        stable_id=record.stable_id,
                        display_name=record.display_name,
                        dataset=dataset,
                        track=record.track,
                        status=COMMON_FULL_GRID,
                        registry_eligible=True,
                        reason=(
                            "Execute only in the common raw-trial architecture "
                            "grid; do not merge author-recipe or procedure scores."
                        ),
                        registry_implementation=record.implementation,
                        component_callable=None,
                        adapter_callable="benchmark.full_grid.execute_benchmark_job",
                        runtime_key=record.common_roster_name,
                        existing_result_coverage=record.existing_result_coverage,
                        cache_compatibility=HARMONIZED_DIRECT,
                        harmonized_v2_safe=True,
                        atomic_jobs=cardinality.atomic_jobs,
                        result_records=cardinality.result_records,
                        subjects=cardinality.subjects,
                        folds=cardinality.folds,
                        seeds=cardinality.seeds,
                        seed_policy=cardinality.seed_policy,
                    )
                )
                continue
            descriptor = DESCRIPTOR_BY_ID[record.stable_id]
            if not eligibility.eligible:
                cells.append(
                    TrackCell(
                        stable_id=record.stable_id,
                        display_name=record.display_name,
                        dataset=dataset,
                        track=record.track,
                        status=REGISTRY_NA,
                        registry_eligible=False,
                        reason=str(eligibility.reason),
                        registry_implementation=descriptor.registry_implementation,
                        component_callable=descriptor.component_callable,
                        adapter_callable=None,
                        runtime_key=descriptor.runtime_key,
                        existing_result_coverage=record.existing_result_coverage,
                        cache_compatibility=descriptor.cache_compatibility,
                        harmonized_v2_safe=descriptor.harmonized_v2_safe,
                        atomic_jobs=0,
                        result_records=0,
                        subjects=(),
                        folds=(),
                        seeds=(),
                        seed_policy="not applicable",
                    )
                )
                continue
            cardinality = proposed_cardinality(record.stable_id, dataset)
            adapter = _READY_ADAPTERS.get((record.stable_id, dataset))
            status = SEPARATE_ELIGIBLE if adapter is not None else MISSING_ADAPTER
            reason = (
                "Executable only as a separately labelled "
                f"{record.track} result under its registered procedure."
                if adapter is not None
                else _blocker_reason(record, dataset)
            )
            cells.append(
                TrackCell(
                    stable_id=record.stable_id,
                    display_name=record.display_name,
                    dataset=dataset,
                    track=record.track,
                    status=status,
                    registry_eligible=True,
                    reason=reason,
                    registry_implementation=descriptor.registry_implementation,
                    component_callable=descriptor.component_callable,
                    adapter_callable=adapter,
                    runtime_key=descriptor.runtime_key,
                    existing_result_coverage=record.existing_result_coverage,
                    cache_compatibility=descriptor.cache_compatibility,
                    harmonized_v2_safe=descriptor.harmonized_v2_safe,
                    atomic_jobs=cardinality.atomic_jobs,
                    result_records=cardinality.result_records,
                    subjects=cardinality.subjects,
                    folds=cardinality.folds,
                    seeds=cardinality.seeds,
                    seed_policy=cardinality.seed_policy,
                )
            )
    return tuple(cells)


TRACK_MATRIX: Final[tuple[TrackCell, ...]] = build_track_matrix()


DRIFT_ISSUES: Final[tuple[DriftIssue, ...]] = (
    DriftIssue(
        code="shared-class-config-not-bound",
        severity="blocking",
        stable_ids=(
            "architecture.geoadaptnet_fb",
            "architecture.geoadaptnet_fbsp",
        ),
        detail=(
            "Both registry rows name FilterBankSPDNet; only reduced_dim=None "
            "versus reduced_dim=8 distinguishes them at runtime."
        ),
        required_action=(
            "Freeze stable-ID-specific constructors and reset/fixed-refit semantics "
            "before emitting a formal record."
        ),
    ),
    DriftIssue(
        code="orbit-version-aliases-share-one-class",
        severity="blocking",
        stable_ids=tuple(f"architecture.orbit_v{index}" for index in range(1, 6)),
        detail=(
            "Five stable IDs resolve to one OrbitTransportClassifier class, while "
            "their scientific identities live in different historical configs."
        ),
        required_action=(
            "Create a hash-pinned stable-ID-to-OrbitTransportConfig resolver; "
            "never infer a version from display text or an output filename."
        ),
    ),
    DriftIssue(
        code="native-registry-path-is-module",
        severity="resolved-by-matrix",
        stable_ids=tuple(
            descriptor.stable_id
            for descriptor in EXECUTION_DESCRIPTORS
            if descriptor.stable_id.startswith("procedure.")
        ),
        detail=(
            "Registry implementation values name modules rather than the run_record "
            "callable and condition string needed to execute one atomic result."
        ),
        required_action=(
            "Use this matrix's component_callable + runtime_key mapping and "
            "validate it against each module's condition tuple."
        ),
    ),
    DriftIssue(
        code="legacy-scratch-fbms-runtime-alias",
        severity="resolved-by-matrix",
        stable_ids=("procedure.scratch_fbmsnet_legacy_transfer",),
        detail=(
            "The stable ID includes '_legacy_transfer', but the native_transfer "
            "runtime condition is exactly 'scratch_fbmsnet'."
        ),
        required_action=(
            "Join results by stable ID after validating the explicit runtime key; "
            "do not join by condition or display name."
        ),
    ),
    DriftIssue(
        code="scratch-fbms-controls-are-not-duplicates",
        severity="warning",
        stable_ids=(
            "procedure.scratch_fbmsnet_legacy_transfer",
            "procedure.scratch_fbmsnet_native",
        ),
        detail=(
            "Both are scratch indexed FBMSNet controls, but they belong to "
            "different CardinalFBC/FBMS source protocols and frozen target recipes."
        ),
        required_action=(
            "Keep separate stable IDs and procedure columns; never average or deduplicate them."
        ),
    ),
    DriftIssue(
        code="dual-view-model-is-not-an-executor",
        severity="blocking",
        stable_ids=("architecture.cardinal_spline_dual_view",),
        detail=(
            "The registry points at CardinalSplineDualViewNet, while the usable "
            "leakage-safe fitting boundary is fit_dual_view_target and no artifact "
            "publisher is present."
        ),
        required_action="Freeze/publish a source checkpoint and add an audited target runner.",
    ),
    DriftIssue(
        code="hemiq-legacy-cache-protocol",
        severity="resolved-by-adapter-awaiting-audit",
        stable_ids=("architecture.hemi_q_field",),
        detail=(
            "The historical runner still owns the S1-4 development/S5-9 "
            "one-time-confirmation legacy protocol. A separate score-blind "
            "all-nine-subject harmonized-v2 development runner now exists."
        ),
        required_action=(
            "Complete the independent no-edit audit before freezing or launching "
            "the new adapter; never rewrite the historical receipt."
        ),
    ),
    DriftIssue(
        code="common-versus-author-recipe-family-aliases",
        severity="warning",
        stable_ids=(
            "common.tcformer",
            "reference.tcformer",
            "common.fbcnet",
            "reference.fbcnet",
        ),
        detail=(
            "Each architecture appears once in the common recipe and once as a "
            "training-procedure reproduction. These are intentional procedure aliases."
        ),
        required_action="Report them in separate columns/sections, not as four architectures.",
    ),
)


EXECUTION_BACKLOG: Final[tuple[BacklogItem, ...]] = (
    BacklogItem(
        priority=1,
        key="generic-v2-controls",
        scientific_value="very high",
        complexity="low",
        affected_stable_ids=(
            "control.riemann",
            "control.tangent_anchor",
            "control.ea_fbcsp",
        ),
        deliverable=(
            "Execute and audit the score-blind control grid across all five "
            "split contracts. Preserve exactly one deterministic execution and "
            "one prediction record per subject/fold, with no seed aliases."
        ),
    ),
    BacklogItem(
        priority=2,
        key="author-faithful-atomic-runner",
        scientific_value="very high",
        complexity="medium",
        affected_stable_ids=("reference.tcformer", "reference.fbcnet"),
        deliverable=(
            "Bind pinned factories, source-only scaling/filtering, one-shot "
            "prediction, metrics/timing, cache hashes, and resumable atomic records."
        ),
    ),
    BacklogItem(
        priority=3,
        key="import-audit-completed-native-fbms-grid",
        scientific_value="high",
        complexity="low",
        affected_stable_ids=tuple(
            descriptor.stable_id
            for descriptor in EXECUTION_DESCRIPTORS
            if descriptor.stable_id.startswith("procedure.")
            and "fbms_transfer" in descriptor.component_callable
        ),
        deliverable=(
            "Audit and ingest the existing 8,675 raw-prediction records into a "
            "separate native-transfer table; do not rerun or call them common-grid scores."
        ),
    ),
    BacklogItem(
        priority=4,
        key="generic-v2-strong-procedures",
        scientific_value="medium-high",
        complexity="high",
        affected_stable_ids=(
            "architecture.cameo",
            "architecture.hemiparity",
            "architecture.parity_fuse",
            "architecture.orbit_v3",
        ),
        deliverable=(
            "Generalize the proven local reset/refit runner to BNCI2014-004, "
            "Cho2017, and PhysioNet without importing legacy split/results."
        ),
    ),
    BacklogItem(
        priority=5,
        key="hemiq-v2-development-adapter",
        scientific_value="medium",
        complexity="medium",
        affected_stable_ids=("architecture.hemi_q_field",),
        deliverable=(
            "Independently audit and freeze the implemented v2 8--30 Hz "
            "all-nine-subject development-only runner while preserving the old "
            "confirmation receipt unchanged."
        ),
    ),
    BacklogItem(
        priority=6,
        key="legacy-native-fbc-transfer-grid",
        scientific_value="medium",
        complexity="medium-high",
        affected_stable_ids=tuple(
            descriptor.stable_id
            for descriptor in EXECUTION_DESCRIPTORS
            if descriptor.stable_id.startswith("procedure.")
            and descriptor.component_callable
            == "benchmark.native_transfer.run_record"
        ),
        deliverable=(
            "Add a score-blind full-grid operations/audit layer for the four "
            "legacy conditions if their separate comparison remains necessary."
        ),
    ),
    BacklogItem(
        priority=7,
        key="dual-view-promotion",
        scientific_value="exploratory",
        complexity="high",
        affected_stable_ids=("architecture.cardinal_spline_dual_view",),
        deliverable=(
            "Only after a frozen development gate: publish the source checkpoint, "
            "target record writer, auditor, and native-grid operations contract."
        ),
    ),
    BacklogItem(
        priority=8,
        key="geoadapt-v2-adapters",
        scientific_value="exploratory/ablation",
        complexity="medium",
        affected_stable_ids=(
            "architecture.geoadaptnet",
            "architecture.geoadaptnet_fb",
            "architecture.geoadaptnet_fbsp",
        ),
        deliverable=(
            "Freeze constructors and implement exact reset/refit adapters before "
            "spending the full 6,585 stochastic jobs."
        ),
    ),
    BacklogItem(
        priority=9,
        key="orbit-ablation-backfill",
        scientific_value="low unless ORBIT-v3 remains competitive",
        complexity="very high",
        affected_stable_ids=(
            "architecture.orbit_v1",
            "architecture.orbit_v2",
            "architecture.orbit_v4",
            "architecture.orbit_v5",
        ),
        deliverable=(
            "Resolve and pin historical configs, then run only if a prespecified "
            "ablation question justifies 8,780 additional stochastic jobs."
        ),
    ),
)


def _source_symbol_exists(dotted_path: str) -> bool:
    """Resolve a local dotted symbol through source AST without importing it."""

    root = Path(__file__).resolve().parents[1]
    parts = dotted_path.split(".")
    module_file: Path | None = None
    symbol_parts: list[str] = []
    for split in range(len(parts), 0, -1):
        candidate = root.joinpath(*parts[:split]).with_suffix(".py")
        if candidate.is_file():
            module_file = candidate
            symbol_parts = parts[split:]
            break
        package = root.joinpath(*parts[:split], "__init__.py")
        if package.is_file():
            module_file = package
            symbol_parts = parts[split:]
            break
    if module_file is None or not symbol_parts:
        return False
    tree = ast.parse(module_file.read_text(encoding="utf-8"), filename=str(module_file))
    body = tree.body
    for name in symbol_parts:
        matches = [
            node
            for node in body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        if len(matches) != 1:
            return False
        node = matches[0]
        body = node.body if isinstance(node, ast.ClassDef) else []
    return True


def _static_value(node: ast.AST, values: Mapping[str, object]) -> object:
    """Evaluate the small literal subset used by frozen runner constants."""

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError(node.id)
        return values[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_static_value(item, values) for item in node.elts)
    if isinstance(node, ast.List):
        return [_static_value(item, values) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _static_value(key, values): _static_value(value, values)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_static_value(item, values) for item in node.args]
        if node.func.id == "range":
            return range(*(int(item) for item in arguments))
        if node.func.id == "tuple" and len(arguments) == 1:
            return tuple(arguments[0])
        if node.func.id == "MappingProxyType" and len(arguments) == 1:
            return dict(arguments[0])
    raise ValueError(ast.dump(node, include_attributes=False))


def _source_constants(relative_path: str) -> dict[str, object]:
    """Read statically evaluable top-level constants without importing code."""

    path = Path(__file__).resolve().parents[2] / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        name: str | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name, value = target.id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        if name is None or value is None:
            continue
        try:
            values[name] = _static_value(value, values)
        except (TypeError, ValueError):
            continue
    return values


def duplicate_callable_groups() -> dict[str, tuple[str, ...]]:
    """Return shared non-common component callables, including aliases."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for descriptor in EXECUTION_DESCRIPTORS:
        if (
            descriptor.registry_implementation
            != _REGISTRY_BY_ID[descriptor.stable_id].implementation
        ):
            raise ValueError(
                f"{descriptor.stable_id}: registry implementation path drifted"
            )
        grouped[descriptor.component_callable].append(descriptor.stable_id)
    return {
        path: tuple(ids)
        for path, ids in sorted(grouped.items())
        if len(ids) > 1
    }


def validate_track_matrix() -> None:
    """Fail closed on registry, cell, cardinality, or source drift."""

    if tuple(DATASET_GRIDS) != BENCHMARK_DATASETS:
        raise ValueError("dataset grid order differs from the registry")
    if len(MODEL_REGISTRY) != 70:
        raise ValueError(f"expected 70 registry rows, found {len(MODEL_REGISTRY)}")
    non_common = tuple(record for record in MODEL_REGISTRY if record.track != "common")
    if len(non_common) != 27:
        raise ValueError(f"expected 27 non-common rows, found {len(non_common)}")
    if set(DESCRIPTOR_BY_ID) != {record.stable_id for record in non_common}:
        missing = sorted(
            {record.stable_id for record in non_common} - set(DESCRIPTOR_BY_ID)
        )
        extra = sorted(
            set(DESCRIPTOR_BY_ID) - {record.stable_id for record in non_common}
        )
        raise ValueError(f"descriptor/registry drift: missing={missing}, extra={extra}")
    if len(EXECUTION_DESCRIPTORS) != len(DESCRIPTOR_BY_ID):
        raise ValueError("duplicate non-common execution descriptor")
    for descriptor in EXECUTION_DESCRIPTORS:
        if descriptor.cache_compatibility not in CACHE_COMPATIBILITY:
            raise ValueError(
                f"{descriptor.stable_id}: invalid cache compatibility"
            )
        if not _source_symbol_exists(descriptor.component_callable):
            raise ValueError(
                f"{descriptor.stable_id}: component callable does not resolve: "
                f"{descriptor.component_callable}"
            )
    if not _source_symbol_exists("benchmark.full_grid.execute_benchmark_job"):
        raise ValueError("common full-grid executor does not resolve")
    for adapter in set(_READY_ADAPTERS.values()):
        if not _source_symbol_exists(adapter):
            raise ValueError(f"ready adapter does not resolve: {adapter}")

    full_grid_constants = _source_constants("src/benchmark/full_grid.py")
    if tuple(full_grid_constants.get("OPENED_DATASETS", ())) != BENCHMARK_DATASETS:
        raise ValueError("full-grid dataset order differs from the track matrix")
    if full_grid_constants.get("DATASET_FOLDS") != {
        dataset: grid.folds for dataset, grid in DATASET_GRIDS.items()
    }:
        raise ValueError("full-grid fold contract differs from the track matrix")
    if tuple(full_grid_constants.get("FORMAL_SEEDS", ())) != FORMAL_SEEDS:
        raise ValueError("full-grid seed contract differs from the track matrix")

    native_constants = _source_constants("src/benchmark/native_transfer.py")
    if native_constants.get("NATIVE_TARGET_DEVELOPMENT_COHORTS") != {
        dataset: grid.subjects for dataset, grid in NATIVE_TARGET_GRIDS.items()
    }:
        raise ValueError("native target cohort differs from the track matrix")
    if native_constants.get("NATIVE_TARGET_FOLDS") != {
        dataset: grid.folds for dataset, grid in NATIVE_TARGET_GRIDS.items()
    }:
        raise ValueError("native target folds differ from the track matrix")
    if tuple(native_constants.get("FROZEN_DEVELOPMENT_SEEDS", ())) != FORMAL_SEEDS:
        raise ValueError("native target seeds differ from the track matrix")
    legacy_conditions = {
        descriptor.runtime_key
        for descriptor in EXECUTION_DESCRIPTORS
        if descriptor.component_callable == "benchmark.native_transfer.run_record"
    }
    if set(native_constants.get("TRANSFER_CONDITIONS", ())) != legacy_conditions:
        raise ValueError("legacy native-transfer conditions differ from the matrix")
    fbms_constants = _source_constants("src/benchmark/native_fbms_transfer.py")
    fbms_conditions = {
        descriptor.runtime_key
        for descriptor in EXECUTION_DESCRIPTORS
        if descriptor.component_callable
        == "benchmark.native_fbms_transfer.run_record"
    }
    if set(fbms_constants.get("TRANSFER_CONDITIONS", ())) != fbms_conditions:
        raise ValueError("FBMS native-transfer conditions differ from the matrix")
    local_outer_constants = _source_constants(
        "src/benchmark/local_outer_refit_benchmark.py"
    )
    if set(local_outer_constants.get("MODEL_NAMES", ())) != {
        "cameo",
        "hemiparity",
        "parity_fuse",
        "orbit_v3",
    }:
        raise ValueError("local exact procedure roster differs from the matrix")
    control_constants = _source_constants("src/benchmark/control_grid.py")
    if tuple(control_constants.get("OPENED_DATASETS", ())) != BENCHMARK_DATASETS:
        raise ValueError("control-grid dataset order differs from the matrix")
    if control_constants.get("DATASET_SUBJECTS") != {
        dataset: grid.subjects for dataset, grid in DATASET_GRIDS.items()
    }:
        raise ValueError("control-grid subject contract differs from the matrix")
    if control_constants.get("DATASET_FOLDS") != {
        dataset: grid.folds for dataset, grid in DATASET_GRIDS.items()
    }:
        raise ValueError("control-grid fold contract differs from the matrix")
    control_ids = tuple(
        descriptor.stable_id
        for descriptor in EXECUTION_DESCRIPTORS
        if descriptor.grid_kind == "standard_deterministic"
    )
    if tuple(control_constants.get("CONTROLS", ())) != control_ids:
        raise ValueError("control-grid roster differs from the matrix")
    if len(TRACK_MATRIX) != len(MODEL_REGISTRY) * len(BENCHMARK_DATASETS):
        raise ValueError("track matrix is not exactly 70x5")
    keys = [(cell.stable_id, cell.dataset) for cell in TRACK_MATRIX]
    if len(keys) != len(set(keys)):
        raise ValueError("track matrix contains duplicate cells")
    expected_keys = {
        (record.stable_id, dataset)
        for record in MODEL_REGISTRY
        for dataset in BENCHMARK_DATASETS
    }
    if set(keys) != expected_keys:
        raise ValueError("track matrix does not cover the registry Cartesian product")
    records = {record.stable_id: record for record in MODEL_REGISTRY}
    for cell in TRACK_MATRIX:
        if cell.status not in CELL_STATUSES:
            raise ValueError(f"{cell.stable_id}/{cell.dataset}: invalid status")
        eligibility = records[cell.stable_id].eligibility_for(cell.dataset)
        if cell.registry_eligible != eligibility.eligible:
            raise ValueError(
                f"{cell.stable_id}/{cell.dataset}: eligibility differs from registry"
            )
        if cell.status == REGISTRY_NA:
            if cell.reason != eligibility.reason:
                raise ValueError(
                    f"{cell.stable_id}/{cell.dataset}: N/A reason drifted"
                )
            if cell.atomic_jobs or cell.result_records:
                raise ValueError(
                    f"{cell.stable_id}/{cell.dataset}: N/A cell has jobs"
                )
        elif not cell.registry_eligible:
            raise ValueError(
                f"{cell.stable_id}/{cell.dataset}: ineligible non-N/A cell"
            )
        if cell.status == COMMON_FULL_GRID and cell.track != "common":
            raise ValueError("non-common entry leaked into common grid")
        if cell.track == "common" and cell.status != COMMON_FULL_GRID:
            raise ValueError("common entry left the common grid")
        if cell.status == SEPARATE_ELIGIBLE and cell.adapter_callable is None:
            raise ValueError(
                f"{cell.stable_id}/{cell.dataset}: ready cell lacks adapter"
            )
        if cell.status == MISSING_ADAPTER and cell.adapter_callable is not None:
            raise ValueError(
                f"{cell.stable_id}/{cell.dataset}: blocked cell has adapter"
            )
        if cell.status != REGISTRY_NA:
            seed_multiplier = len(cell.seeds) if cell.seeds else 1
            expected_records = (
                len(cell.subjects) * len(cell.folds) * seed_multiplier
            )
            if cell.result_records != expected_records:
                raise ValueError(
                    f"{cell.stable_id}/{cell.dataset}: result cardinality drift"
                )
            if not (0 < cell.atomic_jobs <= cell.result_records):
                raise ValueError(
                    f"{cell.stable_id}/{cell.dataset}: atomic cardinality invalid"
                )
            descriptor = DESCRIPTOR_BY_ID.get(cell.stable_id)
            if (
                descriptor is not None
                and descriptor.grid_kind == "standard_deterministic"
            ):
                if cell.seeds:
                    raise ValueError(
                        f"{cell.stable_id}/{cell.dataset}: deterministic control "
                        "must not have a seed dimension"
                    )
                if cell.atomic_jobs != cell.result_records:
                    raise ValueError(
                        f"{cell.stable_id}/{cell.dataset}: deterministic control "
                        "must emit one record per atomic execution"
                    )


def matrix_manifest() -> dict[str, object]:
    """Return a JSON-serializable exact execution plan and audit."""

    validate_track_matrix()
    status_counts = Counter(cell.status for cell in TRACK_MATRIX)
    common_cells = tuple(cell for cell in TRACK_MATRIX if cell.track == "common")
    non_common_cells = tuple(
        cell for cell in TRACK_MATRIX if cell.track != "common"
    )

    def cardinality_totals(cells: Sequence[TrackCell]) -> dict[str, int]:
        return {
            "atomic_execution_units": sum(cell.atomic_jobs for cell in cells),
            "result_records": sum(cell.result_records for cell in cells),
        }

    by_status = {
        status: cardinality_totals(
            tuple(cell for cell in TRACK_MATRIX if cell.status == status)
        )
        for status in sorted(CELL_STATUSES)
    }
    return {
        "schema": MATRIX_SCHEMA,
        "shape": [len(MODEL_REGISTRY), len(BENCHMARK_DATASETS)],
        "cell_count": len(TRACK_MATRIX),
        "registry_order": [record.stable_id for record in MODEL_REGISTRY],
        "dataset_order": list(BENCHMARK_DATASETS),
        "dataset_grids": {
            dataset: asdict(grid) | {"stochastic_jobs": grid.stochastic_jobs}
            for dataset, grid in DATASET_GRIDS.items()
        },
        "native_target_grids": {
            dataset: asdict(grid) | {"stochastic_jobs": grid.stochastic_jobs}
            for dataset, grid in NATIVE_TARGET_GRIDS.items()
        },
        "status_counts": dict(sorted(status_counts.items())),
        "proposed_cardinality_totals": {
            "common": cardinality_totals(common_cells),
            "non_common": cardinality_totals(non_common_cells),
            "all": cardinality_totals(TRACK_MATRIX),
            "by_status": by_status,
        },
        "cells": [asdict(cell) for cell in TRACK_MATRIX],
        "non_common_descriptors": [
            asdict(descriptor) for descriptor in EXECUTION_DESCRIPTORS
        ],
        "duplicate_callable_groups": duplicate_callable_groups(),
        "drift_issues": [asdict(issue) for issue in DRIFT_ISSUES],
        "execution_backlog": [asdict(item) for item in EXECUTION_BACKLOG],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the matrix and optionally emit its complete JSON manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete 350-cell JSON manifest",
    )
    arguments = parser.parse_args(argv)
    manifest = matrix_manifest()
    if arguments.json:
        print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            json.dumps(
                {
                    "schema": manifest["schema"],
                    "shape": manifest["shape"],
                    "cell_count": manifest["cell_count"],
                    "status_counts": manifest["status_counts"],
                    "validation": "passed",
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_DATASETS",
    "CELL_STATUSES",
    "COMMON_FULL_GRID",
    "DATASET_GRIDS",
    "DESCRIPTOR_BY_ID",
    "DRIFT_ISSUES",
    "EXECUTION_BACKLOG",
    "EXECUTION_DESCRIPTORS",
    "MATRIX_SCHEMA",
    "MISSING_ADAPTER",
    "NATIVE_TARGET_GRIDS",
    "REGISTRY_NA",
    "SEPARATE_ELIGIBLE",
    "TRACK_MATRIX",
    "BacklogItem",
    "Cardinality",
    "DatasetGrid",
    "DriftIssue",
    "ExecutionDescriptor",
    "TrackCell",
    "build_track_matrix",
    "duplicate_callable_groups",
    "matrix_manifest",
    "proposed_cardinality",
    "validate_track_matrix",
]
