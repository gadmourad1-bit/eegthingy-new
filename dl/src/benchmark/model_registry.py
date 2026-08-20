"""Canonical identity and benchmark-track registry for EEG-MI models.

This module is intentionally metadata-only.  Importing it does not import
PyTorch, Braindecode, a dataset loader, or an experiment runner.  The registry
separates four concepts that were historically easy to conflate:

* an architecture family;
* one concrete configuration of that family;
* a training/transfer procedure applied to an architecture; and
* a classical (non-neural) control.

The 43 ``common`` entries exactly cover
``benchmark.local_model_tournament.ARCHITECTURES``.  Procedure-only and classical
entries are present for provenance, but they are not silently promoted into the
common raw-trial comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


BENCHMARK_DATASETS: tuple[str, ...] = (
    "local_exp4",
    "bnci2014_001",
    "bnci2014_004",
    "cho2017",
    "physionet_mi",
)

TRACKS = frozenset({"common", "author_faithful", "procedure", "control"})
IDENTITY_LEVELS = frozenset(
    {
        "distinct_architecture",
        "family_configuration",
        "procedure_configuration",
        "classical_control",
    }
)
CLASS_SUPPORT = frozenset({"binary", "binary_and_multiclass"})


@dataclass(frozen=True)
class DatasetEligibility:
    """One explicit dataset decision.

    ``reason`` is mandatory for an ineligible entry and absent for an eligible
    one.  Eligibility describes the registered model/procedure contract, not a
    claim that a completed result already exists; the latter is recorded
    separately in :class:`ModelRecord.existing_result_coverage`.
    """

    dataset: str
    eligible: bool
    reason: str | None = None


@dataclass(frozen=True)
class ModelRecord:
    """Stable metadata for one architecture, configuration, or procedure."""

    stable_id: str
    display_name: str
    family: str
    identity_level: str
    ownership: str
    track: str
    implementation: str
    input_representation: str
    class_support: str
    dataset_eligibility: tuple[DatasetEligibility, ...]
    methodology: str
    existing_result_coverage: tuple[str, ...]
    documentation_link: str
    common_roster_name: str | None = None
    citation: str | None = None
    citation_url: str | None = None
    source_url: str | None = None
    source_pin: str | None = None
    license: str = "unknown/not recorded"

    def eligibility_for(self, dataset: str) -> DatasetEligibility:
        """Return this entry's decision for ``dataset``."""

        for decision in self.dataset_eligibility:
            if decision.dataset == dataset:
                return decision
        raise KeyError(dataset)


def _all_eligible() -> tuple[DatasetEligibility, ...]:
    return tuple(DatasetEligibility(dataset, True) for dataset in BENCHMARK_DATASETS)


def _binary_eligible(
    *eligible: str,
    unavailable_reason: str | None = None,
) -> tuple[DatasetEligibility, ...]:
    allowed = set(eligible)
    unknown = allowed - set(BENCHMARK_DATASETS)
    if unknown:
        raise ValueError(f"unknown eligible datasets: {sorted(unknown)}")
    decisions: list[DatasetEligibility] = []
    for dataset in BENCHMARK_DATASETS:
        if dataset in allowed:
            decisions.append(DatasetEligibility(dataset, True))
        elif dataset == "bnci2014_001":
            decisions.append(
                DatasetEligibility(
                    dataset,
                    False,
                    "The registered task is four-class, while this entry is a "
                    "binary left-versus-right model/procedure.",
                )
            )
        else:
            decisions.append(
                DatasetEligibility(
                    dataset,
                    False,
                    unavailable_reason
                    or "No registered implementation contract exists for this dataset.",
                )
            )
    return tuple(decisions)


COMMON_RESULT = (
    "local_exp4: five-seed common-recipe tournament "
    "(src/benchmark/results/local_exp4_allmodels_20260721/neural/final_ranking.json)"
)
CARDINAL_SCREEN_RESULT = (
    "local_exp4, bnci2014_001, bnci2014_004, cho2017 S1--15: "
    "single-seed architecture screens summarized in MODEL_ACCURACY_INVENTORY.md"
)
PUBLIC_SOURCE = "https://github.com/braindecode/braindecode"
PUBLIC_PIN = "braindecode==1.6.1"
PUBLIC_LICENSE = "BSD-3-Clause (Braindecode implementation)"


_PUBLIC_CITATIONS: dict[str, tuple[str, str, str | None, str]] = {
    "eegnet": (
        "Lawhern et al. (2018), EEGNet: a compact convolutional neural network "
        "for EEG-based brain-computer interfaces.",
        "https://braindecode.org/stable/generated/braindecode.models.EEGNet.html",
        None,
        "EEGNet",
    ),
    "schirrmeister": (
        "Schirrmeister et al. (2017), Deep learning with convolutional neural "
        "networks for EEG decoding and visualization.",
        "https://doi.org/10.1002/hbm.23730",
        None,
        "ShallowFBCSPNet/Deep4Net",
    ),
    "eegconformer": (
        "Song et al. (2022/2023), EEG Conformer: Convolutional Transformer for "
        "EEG Decoding and Visualization.",
        "https://doi.org/10.1109/TNSRE.2022.3230250",
        None,
        "EEGConformer",
    ),
    "atcnet": (
        "Altaheri et al. (2022), Physics-informed attention temporal "
        "convolutional network for EEG-based motor imagery classification.",
        "https://doi.org/10.1109/TII.2022.3197419",
        "https://github.com/Altaheri/EEG-ATCNet",
        "ATCNet",
    ),
    "fbcnet": (
        "Mane et al. (2021), FBCNet: A multi-view convolutional neural network "
        "for brain-computer interface.",
        "https://arxiv.org/abs/2104.01233",
        "https://github.com/ravikiran-mane/FBCNet",
        "FBCNet",
    ),
    "eegtcnet": (
        "Ingolfsson et al. (2020), EEG-TCNet: An accurate temporal "
        "convolutional network for embedded motor-imagery brain-machine interfaces.",
        "https://doi.org/10.48550/arXiv.2006.00622",
        None,
        "EEGTCNet",
    ),
    "fbmsnet": (
        "Liu et al. (2022), FBMSNet: A filter-bank multi-scale convolutional "
        "neural network for EEG-based motor imagery decoding.",
        "https://braindecode.org/stable/generated/braindecode.models.FBMSNet.html",
        "https://github.com/Want2Vanish/FBMSNet",
        "FBMSNet",
    ),
    "ctnet": (
        "Zhao et al. (2024), CTNet: a convolutional transformer network for "
        "EEG-based motor imagery classification.",
        "https://braindecode.org/stable/generated/braindecode.models.CTNet.html",
        "https://github.com/snailpt/CTNet",
        "CTNet",
    ),
    "eegsym": (
        "Perez-Velasco et al. (2022), EEGSym: Overcoming inter-subject "
        "variability in motor imagery based BCIs with deep learning.",
        "https://braindecode.org/stable/generated/braindecode.models.EEGSym.html",
        "https://github.com/Serpeve/EEGSym",
        "EEGSym",
    ),
    "tcformer": (
        "Altaheri et al. (2025), Temporal convolutional transformer for EEG "
        "based motor imagery decoding.",
        "https://doi.org/10.1038/s41598-025-16219-7",
        "https://github.com/Altaheri/TCFormer",
        "TCFormer",
    ),
}


def _common_in_house(
    name: str,
    display_name: str,
    family: str,
    identity_level: str,
    implementation: str,
    representation: str,
    methodology: str,
    documentation_anchor: str,
    *,
    derivative_citation: str | None = None,
    cross_dataset_screen: bool = False,
) -> ModelRecord:
    citation = None
    citation_url = None
    source_url = None
    if derivative_citation is not None:
        citation, citation_url, source_url, _ = _PUBLIC_CITATIONS[derivative_citation]
    coverage = (COMMON_RESULT,)
    if cross_dataset_screen:
        coverage += (CARDINAL_SCREEN_RESULT,)
    return ModelRecord(
        stable_id=f"common.{name}",
        common_roster_name=name,
        display_name=display_name,
        family=family,
        identity_level=identity_level,
        ownership="in_house",
        track="common",
        implementation=implementation,
        input_representation=representation,
        class_support="binary_and_multiclass",
        dataset_eligibility=_all_eligible(),
        methodology=methodology,
        existing_result_coverage=coverage,
        documentation_link=f"docs/MODEL_REGISTRY.md#{documentation_anchor}",
        citation=citation,
        citation_url=citation_url,
        source_url=source_url,
        source_pin="repository source hashes are captured in benchmark artifacts",
        license="project license not declared in repository",
    )


def _common_public(
    name: str,
    display_name: str,
    family: str,
    identity_level: str,
    citation_key: str,
    methodology: str,
    documentation_anchor: str,
) -> ModelRecord:
    citation, citation_url, original_source, implementation_name = _PUBLIC_CITATIONS[
        citation_key
    ]
    source_url = PUBLIC_SOURCE
    source_pin = PUBLIC_PIN
    license_name = PUBLIC_LICENSE
    ownership = "public_implementation"
    if identity_level == "family_configuration":
        ownership = "public_architecture_local_configuration"
    if citation_key == "tcformer":
        source_url = original_source
        source_pin = "commit 74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5"
        license_name = "MIT (official TCFormer repository)"
        ownership = "public_official_source_local_adapter"
    return ModelRecord(
        stable_id=f"common.{name}",
        common_roster_name=name,
        display_name=display_name,
        family=family,
        identity_level=identity_level,
        ownership=ownership,
        track="common",
        implementation=(
            "third_party/TCFormer/models/tcformer.py"
            if citation_key == "tcformer"
            else f"braindecode.models.{implementation_name}"
        ),
        input_representation="harmonized broadband EEG trial (channels x time)",
        class_support="binary_and_multiclass",
        dataset_eligibility=_all_eligible(),
        methodology=methodology,
        existing_result_coverage=(COMMON_RESULT,),
        documentation_link=f"docs/MODEL_REGISTRY.md#{documentation_anchor}",
        citation=citation,
        citation_url=citation_url,
        source_url=source_url,
        source_pin=source_pin,
        license=license_name,
    )


_COMMON_IN_HOUSE: tuple[ModelRecord, ...] = (
    _common_in_house(
        "scope",
        "SCOPE-Net",
        "SCOPE",
        "distinct_architecture",
        "benchmark.models.ScopeNet",
        "raw EEG plus electrode coordinates",
        "Ordered physical-frequency Sinc filters feed continuous spherical scalp "
        "fields; multi-resolution moments and an axial source/frequency mixer "
        "form a residual over a direct log-energy floor.",
        "family-scope",
    ),
    _common_in_house(
        "free_scope",
        "FreeSCOPE",
        "SCOPE",
        "family_configuration",
        "benchmark.models.FreeScopeNet",
        "raw EEG in a fixed channel order",
        "Channel-indexed spatial-filter control for SCOPE with the same temporal "
        "filter and energy/mixer decoder but without coordinate continuity.",
        "family-scope",
    ),
    _common_in_house(
        "cardinal",
        "CardinalField",
        "CardinalField",
        "distinct_architecture",
        "benchmark.models.CardinalFieldNet",
        "raw EEG plus electrode coordinates",
        "Ordered Sinc bands are projected through a cardinal-RBF scalp field and "
        "decoded by segmented log variance.",
        "family-cardinalfield",
    ),
    _common_in_house(
        "free_cardinal",
        "FreeCardinal",
        "CardinalField",
        "family_configuration",
        "benchmark.models.FreeCardinalFieldNet",
        "raw EEG in a fixed channel order",
        "Channel-indexed control retaining CardinalField's filter bank and "
        "segmented log-variance decoder.",
        "family-cardinalfield",
    ),
    _common_in_house(
        "cardinal_dynamics",
        "CardinalDynamics",
        "CardinalDynamics",
        "distinct_architecture",
        "benchmark.models.CardinalDynamicsNet",
        "raw EEG plus electrode coordinates",
        "A full-rank continuous spatiotemporal field supplies log-energy and "
        "signed temporal-dynamics summaries to one fixed classifier.",
        "family-cardinaldynamics",
    ),
    _common_in_house(
        "cardinal_dynamics_compact",
        "CardinalDynamics compact",
        "CardinalDynamics",
        "family_configuration",
        "benchmark.models.CardinalDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Reduced 24-filter/24-source/12-dynamics-channel CardinalDynamics configuration.",
        "family-cardinaldynamics",
    ),
    _common_in_house(
        "cardinal_dynamics_extended",
        "CardinalDynamics extended",
        "CardinalDynamics",
        "family_configuration",
        "benchmark.models.CardinalDynamicsNet",
        "raw EEG plus electrode coordinates",
        "CardinalDynamics sampled from the 31-anchor union atlas instead of 21 anchors.",
        "family-cardinaldynamics",
    ),
    _common_in_house(
        "cardinal_dynamics_sinc",
        "CardinalDynamics Sinc",
        "CardinalDynamics",
        "family_configuration",
        "benchmark.models.CardinalDynamicsNet",
        "raw EEG plus electrode coordinates",
        "CardinalDynamics with 16 constrained ordered Sinc filters in place of "
        "the unconstrained temporal convolution.",
        "family-cardinaldynamics",
    ),
    _common_in_house(
        "cardinal_dynamics_sinc_residual",
        "CardinalDynamics Sinc residual",
        "CardinalDynamics",
        "family_configuration",
        "benchmark.models.CardinalDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Sinc CardinalDynamics plus a gated low-rank multiscale temporal residual.",
        "family-cardinaldynamics",
    ),
    _common_in_house(
        "cardinal_dynamics_sinc_extended",
        "CardinalDynamics Sinc extended",
        "CardinalDynamics",
        "family_configuration",
        "benchmark.models.CardinalDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Ordered-Sinc CardinalDynamics using the 31-anchor union atlas.",
        "family-cardinaldynamics",
    ),
    _common_in_house(
        "cardinal_fbc",
        "CardinalFBC",
        "CardinalFBC",
        "distinct_architecture",
        "benchmark.models.CardinalFBCNet",
        "raw EEG plus electrode coordinates",
        "Function-preserving replacement of FBCNet's channel-indexed grouped "
        "spatial convolution by a cardinal-RBF field.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_extended",
        "CardinalFBC extended",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCNet",
        "raw EEG plus electrode coordinates",
        "CardinalFBC with a 31-anchor union atlas.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_corr",
        "CardinalFBC correlation",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCCorrelationNet",
        "raw EEG plus electrode coordinates",
        "CardinalFBC floor plus zero-started low-rank, shrinkage Fisher-z "
        "correlations of signed spatial sources.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_corr_extended",
        "CardinalFBC correlation extended",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCCorrelationNet",
        "raw EEG plus electrode coordinates",
        "Correlation continuation with the 31-anchor union atlas.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_compactdyn",
        "CardinalFBC compact dynamics",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCCompactDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Exact CardinalFBC floor with a zero-started, full compact "
        "CardinalDynamics feature continuation in one max-norm head.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_compactdyn_extended",
        "CardinalFBC compact dynamics extended",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCCompactDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Compact-dynamics continuation with the 31-anchor union atlas.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_compactdyn_scale010",
        "CardinalFBC compact dynamics scale 0.10",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCCompactDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Compact-dynamics continuation multiplied by 0.10 before the shared head.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_compactdyn_scale010_extended",
        "CardinalFBC compact dynamics scale 0.10 extended",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCCompactDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Scale-0.10 compact continuation with 31 anchors.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_compactdyn_scale025",
        "CardinalFBC compact dynamics scale 0.25",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCCompactDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Compact-dynamics continuation multiplied by 0.25 before the shared head.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_compactdyn_scale025_extended",
        "CardinalFBC compact dynamics scale 0.25 extended",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCCompactDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Scale-0.25 compact continuation with 31 anchors.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_micro",
        "CardinalFBC micro dynamics",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCMicroDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Exact CardinalFBC floor plus a zero-started compact learnable-Gabor "
        "energy/dynamics continuation.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_micro_extended",
        "CardinalFBC micro dynamics extended",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCMicroDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Micro-dynamics continuation with the 31-anchor union atlas.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_physical",
        "CardinalFBC ordered physical dynamics",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCPhysicalDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Exact CardinalFBC floor plus a zero-started ordered-Sinc "
        "energy/dynamics continuation.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbc_physical_extended",
        "CardinalFBC ordered physical dynamics extended",
        "CardinalFBC",
        "family_configuration",
        "benchmark.models.CardinalFBCPhysicalDynamicsNet",
        "raw EEG plus electrode coordinates",
        "Ordered-physical continuation with the 31-anchor union atlas.",
        "family-cardinalfbc",
        derivative_citation="fbcnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbms",
        "CardinalFBMS",
        "CardinalFBMS",
        "distinct_architecture",
        "benchmark.models.CardinalFBMSNet",
        "raw EEG plus electrode coordinates",
        "Function-preserving coordinate-field replacement of FBMSNet's grouped "
        "channel-indexed spatial convolution.",
        "family-cardinalfbms",
        derivative_citation="fbmsnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_fbms_extended",
        "CardinalFBMS extended",
        "CardinalFBMS",
        "family_configuration",
        "benchmark.models.CardinalFBMSNet",
        "raw EEG plus electrode coordinates",
        "CardinalFBMS with a 31-anchor union atlas.",
        "family-cardinalfbms",
        derivative_citation="fbmsnet",
        cross_dataset_screen=True,
    ),
    _common_in_house(
        "cardinal_mix",
        "CardinalMixedTemporal",
        "CardinalMixedTemporal",
        "distinct_architecture",
        "benchmark.models.CardinalMixedTemporalNet",
        "raw EEG plus electrode coordinates",
        "FBMSNet filter/mixed-temporal views followed by a continuous cardinal "
        "spatial field and a single segmented log-variance head.",
        "family-cardinalmixedtemporal",
        derivative_citation="fbmsnet",
    ),
    _common_in_house(
        "cardinal_mix_drop",
        "CardinalMixedTemporal dropout",
        "CardinalMixedTemporal",
        "family_configuration",
        "benchmark.models.CardinalMixedTemporalNet",
        "raw EEG plus electrode coordinates",
        "CardinalMixedTemporal with classifier-path dropout 0.25.",
        "family-cardinalmixedtemporal",
        derivative_citation="fbmsnet",
    ),
)


_COMMON_PUBLIC: tuple[ModelRecord, ...] = (
    _common_public(
        "eegnet",
        "EEGNet",
        "EEGNet",
        "distinct_architecture",
        "eegnet",
        "Compact temporal, depthwise spatial, and separable convolutions.",
        "public-models",
    ),
    _common_public(
        "shallow",
        "ShallowFBCSPNet",
        "ShallowFBCSPNet",
        "distinct_architecture",
        "schirrmeister",
        "Shallow temporal/spatial convolution followed by square, mean pooling, "
        "log transform, and a classifier.",
        "public-models",
    ),
    _common_public(
        "deep4",
        "Deep4Net",
        "Deep4Net",
        "distinct_architecture",
        "schirrmeister",
        "Four-block temporal/spatial convolutional network.",
        "public-models",
    ),
    _common_public(
        "eegconformer",
        "EEG-Conformer",
        "EEGConformer",
        "distinct_architecture",
        "eegconformer",
        "Shallow convolutional tokenization followed by Transformer self-attention.",
        "public-models",
    ),
    _common_public(
        "eegconformer_compact",
        "EEG-Conformer compact",
        "EEGConformer",
        "family_configuration",
        "eegconformer",
        "Local four-layer configuration of EEGConformer (six layers in the "
        "non-compact common entry).",
        "public-models",
    ),
    _common_public(
        "atcnet",
        "ATCNet",
        "ATCNet",
        "distinct_architecture",
        "atcnet",
        "EEG convolutional stem, sliding-window attention, and temporal "
        "convolutional residual blocks.",
        "public-models",
    ),
    _common_public(
        "atcnet_aggressive_pool",
        "ATCNet aggressive pooling",
        "ATCNet",
        "family_configuration",
        "atcnet",
        "Local ATCNet pooling configuration using a second pool size of 4 "
        "instead of 7.",
        "public-models",
    ),
    _common_public(
        "fbcnet",
        "FBCNet",
        "FBCNet",
        "distinct_architecture",
        "fbcnet",
        "Nine-band filtering, bandwise depthwise spatial convolution, "
        "segmented log variance, and a constrained classifier.",
        "public-models",
    ),
    _common_public(
        "eegtcnet",
        "EEG-TCNet",
        "EEGTCNet",
        "distinct_architecture",
        "eegtcnet",
        "EEGNet-style front end followed by a compact temporal convolutional network.",
        "public-models",
    ),
    _common_public(
        "fbmsnet",
        "FBMSNet",
        "FBMSNet",
        "distinct_architecture",
        "fbmsnet",
        "Filter bank, mixed-scale temporal convolution, grouped spatial "
        "convolution, temporal log variance, and constrained classification.",
        "public-models",
    ),
    _common_public(
        "ctnet",
        "CTNet (corrected wrapper)",
        "CTNet",
        "distinct_architecture",
        "ctnet",
        "Convolutional patch encoder and Transformer stack; the local wrapper "
        "restores configured pre-classifier dropout.",
        "public-models",
    ),
    _common_public(
        "ctnet_compact",
        "CTNet compact (corrected wrapper)",
        "CTNet",
        "family_configuration",
        "ctnet",
        "Local 8-filter/16-embedding/two-head CTNet configuration with the same "
        "dropout-restoring wrapper.",
        "public-models",
    ),
    _common_public(
        "eegsym",
        "EEGSym",
        "EEGSym",
        "distinct_architecture",
        "eegsym",
        "Hemispheric branches and multi-scale temporal processing; explicit "
        "left/right channel pairs are supplied by the benchmark.",
        "public-models",
    ),
    _common_public(
        "eegsym_wide",
        "EEGSym wide",
        "EEGSym",
        "family_configuration",
        "eegsym",
        "Local EEGSym configuration with 24 filters per branch.",
        "public-models",
    ),
    _common_public(
        "tcformer",
        "TCFormer",
        "TCFormer",
        "distinct_architecture",
        "tcformer",
        "Official multi-kernel CNN, grouped-query Transformer/RoPE, and TCN "
        "implementation loaded from a clean pinned checkout.",
        "public-models",
    ),
)


_BINARY_FOUR = _binary_eligible("local_exp4", "bnci2014_004", "cho2017", "physionet_mi")
GEOADAPT_BNCI001_NA_REASON = (
    "hard binary main heads are incompatible with the registered four-class "
    "BNCI2014-001 task"
)
GEOADAPT_BNCI004_NA_REASON = (
    "frozen GeoAdaptNet reduced_dim=8 exceeds the three supplied bipolar "
    "channels and the constructor requires reduced_dim<=channels; a "
    "dataset-conditioned reduced_dim=3 model would be a new variant"
)
GEOADAPT_FBSP_BNCI004_NA_REASON = (
    "frozen reduced_dim=8 exceeds the three supplied bipolar channels; "
    "clamping would collapse/alias FBSP to FB and is forbidden"
)


def _geoadapt_eligibility(
    *, bnci004_reason: str | None
) -> tuple[DatasetEligibility, ...]:
    """Return the literal harmonized-v2 GeoAdapt procedure matrix."""

    decisions: list[DatasetEligibility] = []
    for dataset in BENCHMARK_DATASETS:
        if dataset == "bnci2014_001":
            decisions.append(
                DatasetEligibility(dataset, False, GEOADAPT_BNCI001_NA_REASON)
            )
        elif dataset == "bnci2014_004" and bnci004_reason is not None:
            decisions.append(DatasetEligibility(dataset, False, bnci004_reason))
        else:
            decisions.append(DatasetEligibility(dataset, True))
    return tuple(decisions)


_GEOADAPT_FIXED_ELIGIBILITY = _geoadapt_eligibility(
    bnci004_reason=GEOADAPT_BNCI004_NA_REASON
)
_GEOADAPT_FB_ELIGIBILITY = _geoadapt_eligibility(bnci004_reason=None)
_GEOADAPT_FBSP_ELIGIBILITY = _geoadapt_eligibility(
    bnci004_reason=GEOADAPT_FBSP_BNCI004_NA_REASON
)
_LEGACY_BINARY_RESULTS = (
    "Legacy deepnet result artifacts use local Exp4 and/or Cho2017; some "
    "parity-family artifacts use a binary-only BNCI2014-001 protocol that is "
    "not the registered four-class common benchmark."
)


_ADDITIONAL_ARCHITECTURES: tuple[ModelRecord, ...] = (
    ModelRecord(
        stable_id="architecture.geoadaptnet",
        display_name="GeoAdaptNet",
        family="GeoAdaptNet",
        identity_level="distinct_architecture",
        ownership="in_house",
        track="procedure",
        implementation="benchmark.research.model.GeoAdaptNet",
        input_representation="four-band SPD covariance tensors",
        class_support="binary",
        dataset_eligibility=_GEOADAPT_FIXED_ELIGIBILITY,
        methodology="A full tangent-space linear anchor is protected by a "
        "near-zero gated residual of learned per-band BiMap/ReEig/LogEig "
        "features with band attention.",
        existing_result_coverage=(
            "local_exp4 and Cho2017 architecture/deployment artifacts in predecessor-workspace/research-results",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#geoadaptnet-family",
        source_pin="repository source hashes are captured in result artifacts",
        license="project license not declared in repository",
    ),
    ModelRecord(
        stable_id="architecture.geoadaptnet_fb",
        display_name="GeoAdaptNet-FB",
        family="GeoAdaptNet-FB",
        identity_level="distinct_architecture",
        ownership="in_house",
        track="procedure",
        implementation="benchmark.research.filterbank_net.FilterBankSPDNet",
        input_representation="broadband raw EEG",
        class_support="binary",
        dataset_eligibility=_GEOADAPT_FB_ELIGIBILITY,
        methodology="Sequential-logit ordered Sinc filters form unit-trace "
        "full spatial covariances with 0.05 isotropic shrinkage and a declared "
        "diagonal ramp; identity-reference matrix-log features feed a compact "
        "linear head.",
        existing_result_coverage=(
            "local_exp4 and Cho2017 diagnostic artifacts in predecessor-workspace/research-results",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#geoadaptnet-family",
        citation=(
            "Ravanelli and Bengio (2018) SincNet is cited in the implementation "
            "as the filter parameterization; exact URL not recorded locally."
        ),
        source_pin="repository source hashes are captured in result artifacts",
        license="project license not declared in repository",
    ),
    ModelRecord(
        stable_id="architecture.geoadaptnet_fbsp",
        display_name="GeoAdaptNet-FBSP",
        family="GeoAdaptNet-FB",
        identity_level="family_configuration",
        ownership="in_house",
        track="procedure",
        implementation="benchmark.research.filterbank_net.FilterBankSPDNet",
        input_representation="broadband raw EEG",
        class_support="binary",
        dataset_eligibility=_GEOADAPT_FBSP_ELIGIBILITY,
        methodology="GeoAdaptNet-FB with one learned SPD BiMap per band before "
        "ReEig/LogEig tangent vectorization.",
        existing_result_coverage=(
            "Cho2017 30-subject diagnostic artifact in predecessor-workspace/research-results/dev",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#geoadaptnet-family",
        source_pin="repository source hashes are captured in result artifacts",
        license="project license not declared in repository",
    ),
    ModelRecord(
        stable_id="architecture.cameo",
        display_name="CAMEO-Net",
        family="CAMEO",
        identity_level="distinct_architecture",
        ownership="in_house",
        track="procedure",
        implementation="benchmark.research.cameo_net.CAMEOClassifier",
        input_representation="raw EEG plus four-band SPD covariances",
        class_support="binary",
        dataset_eligibility=_BINARY_FOUR,
        methodology="Frozen source-only convex tangent anchor plus shared raw "
        "log-energy and signed-dynamics experts; validation chooses a "
        "counterfactual subtraction and expert mixture.",
        existing_result_coverage=(
            "local_exp4 and Cho2017 CAMEO artifacts; exact local outer-refit result",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#cameo",
        source_pin="frozen configurations are hash-pinned by benchmark artifacts",
        license="project license not declared in repository",
    ),
    ModelRecord(
        stable_id="architecture.hemiparity",
        display_name="HemiParityNet",
        family="HemiParity",
        identity_level="distinct_architecture",
        ownership="in_house",
        track="procedure",
        implementation="benchmark.research.parity_net.HemiParityClassifier",
        input_representation="paired raw/reflected EEG plus paired SPD tangent features",
        class_support="binary",
        dataset_eligibility=_BINARY_FOUR,
        methodology="Internal Z2 even/odd decomposition with conditional odd "
        "raw and tangent readouts and an invariant per-sample fusion gate.",
        existing_result_coverage=(
            _LEGACY_BINARY_RESULTS,
            "exact local outer-refit result",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#hemiparity-and-parity-fuse",
        source_pin="frozen configurations are hash-pinned by benchmark artifacts",
        license="project license not declared in repository",
    ),
    ModelRecord(
        stable_id="architecture.parity_fuse",
        display_name="PARITY-Fuse",
        family="PARITY-Fuse",
        identity_level="distinct_architecture",
        ownership="in_house",
        track="procedure",
        implementation="benchmark.research.parity_fuse_net.ParityFuseClassifier",
        input_representation="paired raw/reflected EEG plus paired SPD tangent features",
        class_support="binary",
        dataset_eligibility=_BINARY_FOUR,
        methodology="Deterministic shared encoders produce even/odd raw and "
        "tangent latents; invariant coordinate-wise fusion and a zero-started "
        "odd residual preserve exact reflection anti-equivariance.",
        existing_result_coverage=(
            _LEGACY_BINARY_RESULTS,
            "exact local outer-refit result",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#hemiparity-and-parity-fuse",
        source_pin="frozen configurations are hash-pinned by benchmark artifacts",
        license="project license not declared in repository",
    ),
    ModelRecord(
        stable_id="architecture.hemi_q_field",
        display_name="HemiQ-FieldNet",
        family="HemiQ-Field",
        identity_level="distinct_architecture",
        ownership="in_house",
        track="procedure",
        implementation="benchmark.research.hemi_q_field_net.HemiQFieldClassifier",
        input_representation="8--30 Hz supplied-bipolar C3/Cz/C4 EEG",
        class_support="binary",
        dataset_eligibility=_binary_eligible(
            "bnci2014_004",
            unavailable_reason=(
                "The frozen model contract is specific to BNCI2014-004's "
                "supplied-bipolar C3/Cz/C4 derivations; point-electrode transfer "
                "would be a new, unvalidated procedure."
            ),
        ),
        methodology="Exact sagittal anti-equivariance built from even "
        "hemispheric/midline signals, an odd C3-C4 signal, constrained "
        "quadrature Gabor filters, invariant context, odd coupling fields, and "
        "bias-free odd readouts.",
        existing_result_coverage=(
            "BNCI2014-004 development and sealed-confirmation artifacts in "
            "predecessor-workspace/research-results/hemi_q",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#hemiq-fieldnet",
        source_pin="frozen manifest and one-time confirmation receipt are stored",
        license="project license not declared in repository",
    ),
    ModelRecord(
        stable_id="architecture.cardinal_spline_dual_view",
        display_name="CardinalSplineDualView",
        family="CardinalSplineDualView",
        identity_level="distinct_architecture",
        ownership="in_house",
        track="procedure",
        implementation="benchmark.dual_view.CardinalSplineDualViewNet",
        input_representation=(
            "native raw EEG and a geometry-only spherical-spline canonical view"
        ),
        class_support="binary",
        dataset_eligibility=_binary_eligible(
            "cho2017",
            "physionet_mi",
            unavailable_reason=(
                "The registered prototype belongs to the native-transfer target "
                "procedure, currently defined only for Cho2017 and PhysioNet."
            ),
        ),
        methodology="One shared CardinalFBC backbone evaluates native and "
        "spline-canonical views; a tiny detached-disagreement gate mixes their "
        "logits and initializes at exact 50/50 fusion.",
        existing_result_coverage=(
            "unit-tested bounded development prototype; no formal aggregate "
            "result identified in the repository",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#cardinalsplinedualview",
        citation=_PUBLIC_CITATIONS["fbcnet"][0],
        citation_url=_PUBLIC_CITATIONS["fbcnet"][1],
        source_pin="repository implementation and tests",
        license="project license not declared in repository",
    ),
)


def _orbit_record(
    version: str,
    display: str,
    identity_level: str,
    methodology: str,
    coverage: tuple[str, ...],
) -> ModelRecord:
    return ModelRecord(
        stable_id=f"architecture.orbit_{version}",
        display_name=display,
        family="OrbitTransportNet",
        identity_level=identity_level,
        ownership="in_house",
        track="procedure",
        implementation="benchmark.research.orbit_transport_net.OrbitTransportClassifier",
        input_representation="paired raw/reflected EEG plus paired SPD anchor logits",
        class_support="binary",
        dataset_eligibility=_BINARY_FOUR,
        methodology=methodology,
        existing_result_coverage=coverage,
        documentation_link="docs/MODEL_REGISTRY.md#orbittransportnet",
        source_pin="configuration and source hashes are stored in each artifact",
        license="project license not declared in repository",
    )


_ORBIT_CONFIGURATIONS: tuple[ModelRecord, ...] = (
    _orbit_record(
        "v1",
        "ORBIT-v1",
        "distinct_architecture",
        "Group-normalized paired energy/dynamics experts are converted to exact "
        "odd logits by learned orientation transport and fused with an odd "
        "source-only tangent anchor.",
        ("local_exp4, Cho2017 S1--8, and legacy binary BNCI2014-001 screens",),
    ),
    _orbit_record(
        "v2",
        "ORBIT-v2 batch",
        "family_configuration",
        "ORBIT-v1 with a view-symmetric BatchNorm pass for the paired raw experts.",
        ("Cho2017 S1--8 development screen",),
    ),
    _orbit_record(
        "v3",
        "ORBIT-v3 batch auxiliary",
        "family_configuration",
        "Batch-normalized ORBIT with zero orientation penalty and stronger "
        "transport/view auxiliary losses; this is the frozen procedure used in "
        "the exact local outer-refit comparison.",
        (
            "local_exp4, Cho2017, and legacy binary BNCI2014-001 artifacts",
            "exact local outer-refit result",
        ),
    ),
    _orbit_record(
        "v4",
        "ORBIT-v4 transport-only",
        "family_configuration",
        "ORBIT-v3 ablation with the view auxiliary loss removed.",
        ("Cho2017 S1--8 development screen",),
    ),
    _orbit_record(
        "v5",
        "ORBIT-v5 raw-mean-only",
        "family_configuration",
        "ORBIT-v4 procedure restricted to the predeclared raw-mean candidate.",
        ("Cho2017 S1--8 development screen",),
    ),
)


def _reference_record(
    model: str,
    display_name: str,
    citation_key: str,
    methodology: str,
    source_pin: str,
) -> ModelRecord:
    citation, citation_url, original_source, implementation_name = _PUBLIC_CITATIONS[
        citation_key
    ]
    return ModelRecord(
        stable_id=f"reference.{model}",
        display_name=display_name,
        family=display_name.split()[0],
        identity_level="procedure_configuration",
        ownership="public_architecture_local_reference_recipe",
        track="author_faithful",
        implementation=(
            "benchmark.reference_training."
            + (
                "fit_tcformer_reference"
                if model == "tcformer"
                else "fit_fbcnet_reference"
            )
        ),
        input_representation="harmonized source-partition EEG",
        class_support="binary_and_multiclass",
        dataset_eligibility=_all_eligible(),
        methodology=methodology,
        existing_result_coverage=(
            "reference-recipe implementation and determinism tests; no completed "
            "formal all-dataset result identified",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#author-faithful-reference-recipes",
        citation=citation,
        citation_url=citation_url,
        source_url=original_source,
        source_pin=source_pin,
        license=(
            "MIT (official source)"
            if model in {"tcformer", "fbcnet"}
            else "unknown/not recorded"
        ),
    )


_REFERENCE_PROCEDURES: tuple[ModelRecord, ...] = (
    _reference_record(
        "tcformer",
        "TCFormer author-faithful recipe",
        "tcformer",
        "Study-adapted input with the released Adam, linear-warmup/cosine "
        "schedule, segmentation-and-reconstruction augmentation, and fixed "
        "final-epoch horizon; explicitly not a bit-for-bit reproduction.",
        "official commit 74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5",
    ),
    _reference_record(
        "fbcnet",
        "FBCNet author-faithful recipe",
        "fbcnet",
        "Braindecode FBCNet behind the original causal Chebyshev-II filter bank "
        "and released Adam/two-stage stopping procedure; explicitly not a "
        "bit-for-bit reproduction.",
        "official recipe commit de1bbdd8a54cb1e466830e3d47070e0e56761a37; "
        "braindecode==1.6.1",
    ),
)


def _transfer_record(
    stable_id: str,
    display_name: str,
    family: str,
    implementation: str,
    methodology: str,
    coverage: str,
    *,
    ownership: str = "in_house",
    public_citation: str | None = None,
) -> ModelRecord:
    citation = None
    citation_url = None
    source_url = None
    license_name = "project license not declared in repository"
    if public_citation is not None:
        citation, citation_url, source_url, _ = _PUBLIC_CITATIONS[public_citation]
        ownership = "public_control_with_local_transfer_procedure"
        license_name = PUBLIC_LICENSE
    return ModelRecord(
        stable_id=f"procedure.{stable_id}",
        display_name=display_name,
        family=family,
        identity_level="procedure_configuration",
        ownership=ownership,
        track="procedure",
        implementation=implementation,
        input_representation="native-montage raw EEG under source-pretrain/target-finetune protocol",
        class_support="binary",
        dataset_eligibility=_binary_eligible(
            "cho2017",
            "physionet_mi",
            unavailable_reason=(
                "The frozen native-transfer procedure is defined only for "
                "Cho2017 S16--52 and PhysioNet S1--54."
            ),
        ),
        methodology=methodology,
        existing_result_coverage=(coverage,),
        documentation_link="docs/MODEL_REGISTRY.md#native-transfer-procedures",
        citation=citation,
        citation_url=citation_url,
        source_url=source_url,
        source_pin="condition tuple, source checkpoint, and artifact schemas are frozen",
        license=license_name,
    )


_NATIVE_TRANSFER_PROCEDURES: tuple[ModelRecord, ...] = (
    _transfer_record(
        "pretrained_cardinal_fbc",
        "Pretrained CardinalFBC",
        "CardinalFBC transfer",
        "benchmark.native_transfer",
        "Canonical CardinalFBC source checkpoint fine-tuned on a native target montage.",
        "Cho2017 and PhysioNet fold-0/seed-7 development screen",
    ),
    _transfer_record(
        "scratch_cardinal_fbc",
        "Scratch CardinalFBC",
        "CardinalFBC transfer",
        "benchmark.native_transfer",
        "CardinalFBC trained from scratch under the matched target-stage recipe.",
        "Cho2017 and PhysioNet fold-0/seed-7 development screen",
    ),
    _transfer_record(
        "scratch_fbmsnet_legacy_transfer",
        "Scratch FBMSNet (legacy FBC transfer control)",
        "CardinalFBC transfer",
        "benchmark.native_transfer",
        "Native indexed FBMSNet scratch control in the predecessor FBC transfer screen.",
        "Cho2017 and PhysioNet fold-0/seed-7 development screen",
        public_citation="fbmsnet",
    ),
    _transfer_record(
        "pretrained_indexed_fbcnet_spherical_spline",
        "Pretrained indexed FBCNet + spherical spline",
        "CardinalFBC transfer",
        "benchmark.native_transfer",
        "Checkpoint-derived indexed FBCNet evaluated after a fixed label-free "
        "Perrin spherical-spline transport to the canonical montage.",
        "Cho2017 and PhysioNet fold-0/seed-7 development screen",
        public_citation="fbcnet",
    ),
    _transfer_record(
        "pretrained_cardinal_fbms",
        "Pretrained CardinalFBMS",
        "CardinalFBMS transfer",
        "benchmark.native_fbms_transfer",
        "Byte-pinned canonical CardinalFBMS source checkpoint fine-tuned on "
        "native target montages.",
        "complete five-seed/all-fold Cho2017 S16--52 and PhysioNet S1--54 grid",
    ),
    _transfer_record(
        "scratch_cardinal_fbms_canonical_seeded",
        "Scratch CardinalFBMS, canonical seeded",
        "CardinalFBMS transfer",
        "benchmark.native_fbms_transfer",
        "Scratch target model initialized from the canonical coordinate-field "
        "parameterization without learned source weights.",
        "complete five-seed/all-fold Cho2017 S16--52 and PhysioNet S1--54 grid",
    ),
    _transfer_record(
        "scratch_cardinal_fbms_native_projected",
        "Scratch CardinalFBMS, native projected",
        "CardinalFBMS transfer",
        "benchmark.native_fbms_transfer",
        "Scratch target model with the fresh indexed native spatial layer "
        "projected into the coordinate-field parameterization.",
        "complete five-seed/all-fold Cho2017 S16--52 and PhysioNet S1--54 grid",
    ),
    _transfer_record(
        "scratch_fbmsnet_native",
        "Scratch native indexed FBMSNet",
        "CardinalFBMS transfer",
        "benchmark.native_fbms_transfer",
        "Fresh native channel-indexed FBMSNet target control.",
        "complete five-seed/all-fold Cho2017 S16--52 and PhysioNet S1--54 grid",
        public_citation="fbmsnet",
    ),
    _transfer_record(
        "pretrained_indexed_fbmsnet_spherical_spline",
        "Pretrained indexed FBMSNet + spherical spline",
        "CardinalFBMS transfer",
        "benchmark.native_fbms_transfer",
        "Checkpoint-matched indexed FBMSNet after fixed label-free "
        "spherical-spline transport; this is a transfer control, not an "
        "author-faithful FBMSNet reproduction.",
        "complete five-seed/all-fold Cho2017 S16--52 and PhysioNet S1--54 grid",
        public_citation="fbmsnet",
    ),
)


def _control_record(
    stable_id: str,
    display_name: str,
    implementation: str,
    representation: str,
    methodology: str,
) -> ModelRecord:
    return ModelRecord(
        stable_id=f"control.{stable_id}",
        display_name=display_name,
        family=display_name,
        identity_level="classical_control",
        ownership="in_house_implementation_of_classical_method",
        track="control",
        implementation=implementation,
        input_representation=representation,
        class_support="binary_and_multiclass",
        dataset_eligibility=_all_eligible(),
        methodology=methodology,
        existing_result_coverage=(
            "local_exp4 five-seed-identity deterministic-control tournament "
            "(identical deterministic fits are not independent seeds)",
        ),
        documentation_link="docs/MODEL_REGISTRY.md#classical-controls",
        source_pin="repository source hashes are captured in control artifacts",
        license="project license not declared in repository; dependency licenses separate",
    )


_CLASSICAL_CONTROLS: tuple[ModelRecord, ...] = (
    _control_record(
        "riemann",
        "Riemannian tangent-space logistic regression",
        "benchmark.research.baselines.RiemannianTangentLogistic",
        "four-band SPD covariance tensors",
        "Per-band affine-invariant Riemannian source whitening and tangent-space "
        "features, standardized before L2 logistic regression.",
    ),
    _control_record(
        "tangent_anchor",
        "Log-Euclidean tangent anchor",
        "benchmark.research.tangent_anchor.TangentAnchorClassifier",
        "four-band SPD covariance tensors",
        "Frozen train-only log-Euclidean reference, exact congruence recentering, "
        "matrix log and upper-triangle vectorization, then standardized convex "
        "L2 logistic regression.",
    ),
    _control_record(
        "ea_fbcsp",
        "EA-FBCSP",
        "benchmark.research.baselines.EAFilterBankCSP",
        "four-band filtered EEG epochs",
        "Train-only Euclidean alignment, per-band shrinkage CSP, concatenated "
        "log-power features, and shrinkage LDA.",
    ),
)


_COMMON_ROSTER_ORDER: tuple[str, ...] = (
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
_COMMON_BY_NAME = {
    record.common_roster_name: record for record in (*_COMMON_IN_HOUSE, *_COMMON_PUBLIC)
}
COMMON_RECORDS: tuple[ModelRecord, ...] = tuple(
    _COMMON_BY_NAME[name] for name in _COMMON_ROSTER_ORDER
)

MODEL_REGISTRY: tuple[ModelRecord, ...] = (
    *COMMON_RECORDS,
    *_ADDITIONAL_ARCHITECTURES,
    *_ORBIT_CONFIGURATIONS,
    *_REFERENCE_PROCEDURES,
    *_NATIVE_TRANSFER_PROCEDURES,
    *_CLASSICAL_CONTROLS,
)

COMMON_ROSTER_NAMES: tuple[str, ...] = tuple(
    record.common_roster_name
    for record in COMMON_RECORDS
    if record.common_roster_name is not None
)
IN_HOUSE_COMMON_RECORDS: tuple[ModelRecord, ...] = _COMMON_IN_HOUSE
PUBLIC_COMMON_RECORDS: tuple[ModelRecord, ...] = _COMMON_PUBLIC


def validate_registry(records: Iterable[ModelRecord] = MODEL_REGISTRY) -> None:
    """Raise ``ValueError`` if core identity/provenance invariants are broken."""

    materialized = tuple(records)
    stable_ids = [record.stable_id for record in materialized]
    if len(stable_ids) != len(set(stable_ids)):
        duplicates = sorted(
            value for value in set(stable_ids) if stable_ids.count(value) > 1
        )
        raise ValueError(f"duplicate stable model IDs: {duplicates}")
    for record in materialized:
        if not record.stable_id or not record.display_name or not record.family:
            raise ValueError("stable_id, display_name, and family must be nonempty")
        if record.track not in TRACKS:
            raise ValueError(f"{record.stable_id}: unknown track {record.track!r}")
        if record.identity_level not in IDENTITY_LEVELS:
            raise ValueError(
                f"{record.stable_id}: unknown identity level {record.identity_level!r}"
            )
        if record.class_support not in CLASS_SUPPORT:
            raise ValueError(
                f"{record.stable_id}: unknown class support {record.class_support!r}"
            )
        if not record.documentation_link.startswith("docs/MODEL_REGISTRY.md#"):
            raise ValueError(f"{record.stable_id}: invalid documentation link")
        decisions = record.dataset_eligibility
        decision_names = tuple(decision.dataset for decision in decisions)
        if decision_names != BENCHMARK_DATASETS:
            raise ValueError(
                f"{record.stable_id}: dataset decisions must exactly follow "
                f"{BENCHMARK_DATASETS}"
            )
        for decision in decisions:
            if decision.eligible and decision.reason is not None:
                raise ValueError(
                    f"{record.stable_id}/{decision.dataset}: eligible entries "
                    "must not carry an N/A reason"
                )
            if not decision.eligible and not decision.reason:
                raise ValueError(
                    f"{record.stable_id}/{decision.dataset}: ineligible entries "
                    "must carry an N/A reason"
                )
        if record.track == "common" and record.common_roster_name is None:
            raise ValueError(
                f"{record.stable_id}: common entry lacks common_roster_name"
            )
        if record.track != "common" and record.common_roster_name is not None:
            raise ValueError(
                f"{record.stable_id}: non-common entry has common_roster_name"
            )


def record_by_id(stable_id: str) -> ModelRecord:
    """Return exactly one record by stable ID."""

    for record in MODEL_REGISTRY:
        if record.stable_id == stable_id:
            return record
    raise KeyError(stable_id)


def records_for_track(track: str) -> tuple[ModelRecord, ...]:
    """Return all records assigned to ``track`` in canonical order."""

    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")
    return tuple(record for record in MODEL_REGISTRY if record.track == track)


validate_registry()


__all__ = [
    "BENCHMARK_DATASETS",
    "COMMON_RECORDS",
    "COMMON_ROSTER_NAMES",
    "DatasetEligibility",
    "GEOADAPT_BNCI001_NA_REASON",
    "GEOADAPT_BNCI004_NA_REASON",
    "GEOADAPT_FBSP_BNCI004_NA_REASON",
    "IDENTITY_LEVELS",
    "IN_HOUSE_COMMON_RECORDS",
    "MODEL_REGISTRY",
    "ModelRecord",
    "PUBLIC_COMMON_RECORDS",
    "TRACKS",
    "record_by_id",
    "records_for_track",
    "validate_registry",
]
