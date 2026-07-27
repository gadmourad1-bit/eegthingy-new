"""One factory for proposed models and rerun neural baselines."""

from __future__ import annotations

import math
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
from torch import nn

from .models import (
    CardinalFieldNet,
    CardinalFieldConfig,
    CardinalFBMSNet,
    CardinalFBCCompactDynamicsNet,
    CardinalFBCCorrelationNet,
    CardinalFBCNet,
    CardinalFBCMicroDynamicsNet,
    CardinalFBCPhysicalDynamicsNet,
    CardinalMixedTemporalNet,
    CardinalDynamicsConfig,
    CardinalDynamicsNet,
    FreeCardinalFieldNet,
    FreeScopeNet,
    ScopeConfig,
    ScopeNet,
)


class CorrectedCTNet(nn.Module):
    """Braindecode CTNet with its configured pre-classifier dropout restored."""

    uses_positions = False

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.base.ensuredim(x)
        cnn = self.base.cnn(values) * math.sqrt(self.base.embed_dim)
        cnn = self.base.position(cnn)
        features = cnn + self.base.trans(cnn)
        return self.base.final_layer(self.base.flatten_drop_layer(features))


class DeterministicHalfPool3d(nn.Module):
    """Exact AvgPool3d((1, 2, 1), stride=(1, 2, 1)) via reshape/mean."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height = x.shape[-2]
        if height % 2:
            x = x[..., : height - 1, :]
            height -= 1
        shape = (*x.shape[:-2], height // 2, 2, x.shape[-1])
        return x.reshape(shape).mean(dim=-2)


def _replace_eegsym_pools(model: nn.Module) -> None:
    for module in model.modules():
        for name, child in tuple(module.named_children()):
            if isinstance(child, nn.AvgPool3d):
                if child.kernel_size != (1, 2, 1) or child.stride != (1, 2, 1):
                    raise RuntimeError(f"unsupported EEGSym pooling layer {child}")
                setattr(module, name, DeterministicHalfPool3d())


def _load_module_from_path(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _official_tcformer_module() -> type[nn.Module]:
    """Load only TCFormerModule from a pinned official checkout.

    This bypasses the repository package initializer, which imports its
    Lightning training stack.  The benchmark uses the pure PyTorch module and
    the common optimizer, while leaving the MIT-licensed source unmodified.
    """

    root = Path(
        os.environ.get(
            "IEEE_MI_TCFORMER_ROOT",
            str(Path.cwd() / "third_party" / "TCFormer"),
        )
    ).resolve()
    expected_commit = "74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5"
    if not (root / "models" / "tcformer.py").exists():
        raise FileNotFoundError(
            f"official TCFormer checkout is missing at {root}; expected commit {expected_commit}"
        )
    cached = sys.modules.get("_ieee_tcformer.models.tcformer")
    if cached is not None:
        return cached.TCFormerModule  # type: ignore[attr-defined, no-any-return]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not verify official TCFormer checkout at {root}") from exc
    if commit != expected_commit:
        raise RuntimeError(
            f"TCFormer checkout is {commit}; expected pinned commit {expected_commit}"
        )
    if dirty:
        raise RuntimeError("official TCFormer checkout has tracked modifications")

    package = types.ModuleType("_ieee_tcformer")
    package.__path__ = [str(root)]  # type: ignore[attr-defined]
    models_package = types.ModuleType("_ieee_tcformer.models")
    models_package.__path__ = [str(root / "models")]  # type: ignore[attr-defined]
    sys.modules[package.__name__] = package
    sys.modules[models_package.__name__] = models_package

    classification = types.ModuleType("_ieee_tcformer.models.classification_module")
    classification.ClassificationModule = nn.Module  # type: ignore[attr-defined]
    sys.modules[classification.__name__] = classification
    _load_module_from_path(
        "_ieee_tcformer.models.modules", root / "models" / "modules.py"
    )
    _load_module_from_path(
        "_ieee_tcformer.models.channel_group_attention",
        root / "models" / "channel_group_attention.py",
    )

    utility_names = ("utils", "utils.weight_initialization", "utils.latency")
    previous = {name: sys.modules.get(name) for name in utility_names}
    try:
        utilities = types.ModuleType("utils")
        utilities.__path__ = [str(root / "utils")]  # type: ignore[attr-defined]
        sys.modules["utils"] = utilities
        _load_module_from_path(
            "utils.weight_initialization", root / "utils" / "weight_initialization.py"
        )
        latency = types.ModuleType("utils.latency")
        latency.measure_latency = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[attr-defined]
            RuntimeError("latency helper is not part of the unified training harness")
        )
        sys.modules["utils.latency"] = latency
        module = _load_module_from_path(
            "_ieee_tcformer.models.tcformer", root / "models" / "tcformer.py"
        )
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    module.TCFormerModule._ieee_provenance = {  # type: ignore[attr-defined]
        "repository": str(root),
        "commit": commit,
        "tracked_dirty": False,
    }
    return module.TCFormerModule  # type: ignore[attr-defined, no-any-return]


def make_model(
    name: str,
    *,
    n_channels: int,
    n_outputs: int,
    n_times: int = 320,
    sfreq: float = 128.0,
    scope_config: ScopeConfig = ScopeConfig(),
    cardinal_config: CardinalFieldConfig = CardinalFieldConfig(),
    channel_names: tuple[str, ...] | None = None,
    channel_positions: torch.Tensor | None = None,
) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key == "scope":
        return ScopeNet(n_outputs=n_outputs, sfreq=sfreq, config=scope_config)
    if key == "free_scope":
        return FreeScopeNet(
            n_channels=n_channels,
            n_outputs=n_outputs,
            sfreq=sfreq,
            config=scope_config,
        )
    if key == "cardinal":
        return CardinalFieldNet(
            n_outputs=n_outputs, sfreq=sfreq, config=cardinal_config
        )
    if key == "free_cardinal":
        return FreeCardinalFieldNet(
            n_channels=n_channels,
            n_outputs=n_outputs,
            sfreq=sfreq,
            config=cardinal_config,
        )
    if key in {
        "cardinal_dynamics",
        "cardinal_dynamics_compact",
        "cardinal_dynamics_extended",
        "cardinal_dynamics_sinc",
        "cardinal_dynamics_sinc_residual",
        "cardinal_dynamics_sinc_extended",
    }:
        if key.endswith("_compact"):
            config = CardinalDynamicsConfig(
                temporal_filters=24,
                latent_sources=24,
                dynamics_channels=12,
                dropout=0.35,
            )
        elif key == "cardinal_dynamics_sinc":
            config = CardinalDynamicsConfig(
                temporal_filters=16,
                temporal_kernel=65,
                physical_filter_bank=True,
                latent_sources=24,
                pool_kernel=64,
                pool_stride=16,
                dynamics_channels=12,
                dropout=0.35,
            )
        elif key == "cardinal_dynamics_sinc_residual":
            config = CardinalDynamicsConfig(
                temporal_filters=16,
                temporal_kernel=65,
                physical_filter_bank=True,
                spectral_residual=True,
                latent_sources=24,
                pool_kernel=64,
                pool_stride=16,
                dynamics_channels=12,
                dropout=0.35,
            )
        elif key == "cardinal_dynamics_sinc_extended":
            config = CardinalDynamicsConfig(
                temporal_filters=16,
                temporal_kernel=65,
                physical_filter_bank=True,
                latent_sources=24,
                pool_kernel=64,
                pool_stride=16,
                dynamics_channels=12,
                dropout=0.35,
                extended_atlas=True,
            )
        elif key.endswith("_extended"):
            config = CardinalDynamicsConfig(extended_atlas=True)
        else:
            config = CardinalDynamicsConfig()
        return CardinalDynamicsNet(
            n_outputs=n_outputs,
            n_times=n_times,
            config=config,
        )

    from braindecode import models

    if key == "eegnet":
        model = models.EEGNet(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
        )
    elif key == "shallow":
        model = models.ShallowFBCSPNet(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            filter_time_length=13,
            pool_time_length=38,
            pool_time_stride=8,
        )
    elif key == "deep4":
        model = models.Deep4Net(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            filter_time_length=5,
            filter_length_2=5,
            filter_length_3=5,
            filter_length_4=5,
        )
    elif key in {"eegconformer", "eegconformer_compact"}:
        model = models.EEGConformer(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            filter_time_length=13,
            pool_time_length=38,
            pool_time_stride=8,
            num_layers=4 if key.endswith("_compact") else 6,
        )
    elif key in {"atcnet", "atcnet_aggressive_pool"}:
        model = models.ATCNet(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            conv_block_kernel_length_1=33,
            conv_block_kernel_length_2=8,
            conv_block_pool_size_1=4,
            conv_block_pool_size_2=(
                4 if key.endswith("_aggressive_pool") else 7
            ),
        )
    elif key == "fbcnet":
        model = models.FBCNet(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
        )
    elif key in {
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
    }:
        base = models.FBCNet(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
        )
        if "_corr" in key:
            model_type = CardinalFBCCorrelationNet
        elif "_compactdyn" in key:
            model_type = CardinalFBCCompactDynamicsNet
        elif "_micro" in key:
            model_type = CardinalFBCMicroDynamicsNet
        elif "_physical" in key:
            model_type = CardinalFBCPhysicalDynamicsNet
        else:
            model_type = CardinalFBCNet
        model_kwargs: dict[str, object] = {}
        if "_scale010" in key:
            model_kwargs["continuation_scale"] = 0.10
        elif "_scale025" in key:
            model_kwargs["continuation_scale"] = 0.25
        if model_type in {
            CardinalFBCMicroDynamicsNet,
            CardinalFBCPhysicalDynamicsNet,
        }:
            model_kwargs["sfreq"] = sfreq
        return model_type(
            spectral_filtering=base.spectral_filtering,
            batch_norm=base.spatial_conv[1],
            activation=base.spatial_conv[2],
            padding_layer=base.padding_layer,
            temporal_layer=base.temporal_layer,
            flatten_layer=base.flatten_layer,
            final_layer=base.final_layer,
            n_times=n_times,
            n_bands=base.n_bands,
            n_sources=base.n_filters_spat,
            stride_factor=base.stride_factor,
            extended_atlas=key.endswith("_extended"),
            indexed_spatial_weights=(
                base.spatial_conv[0]
                .weight.detach()
                .reshape(base.n_bands, base.n_filters_spat, n_channels)
                if channel_positions is not None
                else None
            ),
            indexed_spatial_bias=(
                base.spatial_conv[0]
                .bias.detach()
                .reshape(base.n_bands, base.n_filters_spat)
                if channel_positions is not None
                else None
            ),
            observed_positions=channel_positions,
            **model_kwargs,
        )
    elif key == "eegtcnet":
        model = models.EEGTCNet(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            kern_length=33,
        )
    elif key in {"fbmsnet", "cardinal_fbms", "cardinal_fbms_extended"}:
        base = models.FBMSNet(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            n_bands=9,
            n_filters_spat=36,
            temporal_layer="LogVarLayer",
            stride_factor=4,
            dilatability=8,
            kernels_weights=(7, 15, 31, 63),
        )
        if key == "fbmsnet":
            model = base
        else:
            spatial_conv = base.spatial_conv[0]
            return CardinalFBMSNet(
                spectral_filtering=base.spectral_filtering,
                mix_conv=base.mix_conv,
                batch_norm=base.spatial_conv[1],
                activation=base.spatial_conv[2],
                padding_layer=base.padding_layer,
                temporal_layer=base.temporal_layer,
                flatten_layer=base.flatten_layer,
                final_layer=base.final_layer,
                n_times=n_times,
                n_temporal_views=base.n_filters_spat,
                dilatability=base.dilatability,
                stride_factor=base.stride_factor,
                extended_atlas=key.endswith("_extended"),
                indexed_spatial_weights=(
                    spatial_conv.weight.detach().reshape(
                        base.n_filters_spat, base.dilatability, n_channels
                    )
                    if channel_positions is not None
                    else None
                ),
                indexed_spatial_bias=(
                    spatial_conv.bias.detach().reshape(
                        base.n_filters_spat, base.dilatability
                    )
                    if channel_positions is not None
                    else None
                ),
                observed_positions=channel_positions,
                spatial_max_norm=float(spatial_conv.max_norm),
            )
    elif key in {"cardinal_mix", "cardinal_mix_drop"}:
        base = models.FBMSNet(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            n_bands=9,
            n_filters_spat=36,
            temporal_layer="LogVarLayer",
            stride_factor=4,
            dilatability=8,
            kernels_weights=(7, 15, 31, 63),
        )
        return CardinalMixedTemporalNet(
            spectral_filtering=base.spectral_filtering,
            mixed_temporal=base.mix_conv,
            n_outputs=n_outputs,
            dropout=0.25 if key.endswith("_drop") else 0.0,
        )
    elif key in {"ctnet", "ctnet_compact"}:
        compact = key == "ctnet_compact"
        base = models.CTNet(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            n_filters_time=8 if compact else 20,
            embed_dim=16 if compact else 40,
            num_heads=2 if compact else 4,
            num_layers=6,
            kernel_size=33,
            pool_size_1=4,
            pool_size_2=8,
            cnn_drop_prob=0.5,
            att_positional_drop_prob=0.1,
            final_drop_prob=0.5,
        )
        return CorrectedCTNet(base)
    elif key in {"eegsym", "eegsym_wide"}:
        if channel_names is None or len(channel_names) != n_channels:
            raise ValueError("EEGSym requires channel_names in input order")
        from .config import REFLECTION_PAIRS

        available = set(channel_names)
        pairs = [
            (left, right)
            for left, right in REFLECTION_PAIRS
            if left in available and right in available
        ]
        paired = {name for pair in pairs for name in pair}
        middle = [name for name in channel_names if name not in paired]
        if len(paired) + len(middle) != n_channels:
            raise RuntimeError("EEGSym channel partition is incomplete")
        eegsym_kwargs: dict[str, object] = {}
        if key.endswith("_wide"):
            eegsym_kwargs["filters_per_branch"] = 24
        model = models.EEGSym(
            n_chans=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            chs_info=[{"ch_name": name} for name in channel_names],
            scales_time=(500, 250, 125),
            drop_prob=0.4,
            spatial_resnet_repetitions=1,
            left_right_chs=pairs,
            middle_chs=middle,
            **eegsym_kwargs,
        )
        _replace_eegsym_pools(model)
    elif key == "tcformer":
        tcformer = _official_tcformer_module()
        model = tcformer(
            n_channels=n_channels,
            n_classes=n_outputs,
            F1=32,
            temp_kernel_lengths=(10, 16, 33),
            pool_length_1=4,
            pool_length_2=7,
            D=2,
            dropout_conv=0.4,
            d_group=16,
            tcn_depth=2,
            kernel_length_tcn=4,
            dropout_tcn=0.3,
            use_group_attn=True,
            q_heads=4,
            kv_heads=2,
            trans_depth=2,
            trans_dropout=0.4,
            drop_path_max=0.25,
        )
    else:
        raise ValueError(f"unknown model {name!r}")
    model.uses_positions = False
    return model
