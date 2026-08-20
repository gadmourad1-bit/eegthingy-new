# Local procedure configuration boundary

`benchmark.local_outer_refit_benchmark` uses four immutable, configuration-only
JSON files:

| Procedure | Configuration file | Bytes | SHA-256 |
|---|---|---:|---|
| CAMEO v1 | `configs/local_procedures/cameo_v1.json` | 830 | `23d07dcfc03bac8ac1e0cb076621850d5738df5a7d31fa7ea89ba42b3dfb2336` |
| HemiParity v1 | `configs/local_procedures/hemiparity_v1.json` | 570 | `0978f53bbc603378052d4ea97355641fe454d250f2745593ead239df4eb463cc` |
| PARITY-Fuse v1 | `configs/local_procedures/parity_fuse_v1.json` | 629 | `c5a2e84163b49f2da7c47c359418aa790f7a9d6490c7e0bc48c9aa8be3f9c6fb` |
| ORBIT v3 | `configs/local_procedures/orbit_v3.json` | 849 | `95edd8a374892e4f28a8feb4e3dfe4ffb32e415b4c3cca2d1ea4a6f89a72b456` |

These files preserve the exact architecture and training hyperparameters used
by the four local procedures. They are not result artifacts and contain no
participant identity, split, seed, device, labels, predictions, probabilities,
metrics, scores, histories, summaries, or repository/environment metadata.
Seed and device are explicit runtime overrides and are recorded in each run's
contract.

## Exact schema

Every file has exactly five top-level fields:

```json
{
  "schema": "eeg-mi-local-procedure-config-v2",
  "version": 1,
  "model": "architecture.registry_stable_id",
  "architecture": {},
  "training": {}
}
```

The `model` value is the exact registry stable ID (`architecture.cameo`,
`architecture.hemiparity`, `architecture.parity_fuse`, or
`architecture.orbit_v3`). The adapter freezes an exact architecture-field and
training-field allowlist for each procedure. It also verifies that the union
of those fields plus the runtime-only `seed` and `device` fields exactly
equals the corresponding configuration dataclass. Adding a dataclass field
therefore fails closed until the configuration schema is reviewed and
deliberately versioned.

Before constructing a model, the loader:

1. parses strict JSON, rejecting duplicate keys and non-finite constants;
2. recursively rejects outcome, cohort, split, seed, device, and other runtime
   keys;
3. validates the exact schema, version, model identity, and per-procedure field
   allowlists;
4. validates that every setting is a finite JSON scalar or a list of such
   scalars;
5. verifies the complete file byte count and SHA-256;
6. constructs the typed configuration with runtime seed/device overrides; and
7. requires the typed configuration to round-trip to the same settings.

The run contract records both the file hash and a canonical hash of the
flattened settings. `run_one` reopens and revalidates the file and compares all
immutable configuration-identity fields before fitting. The runner also
requires the loaded configuration digest to equal that file's entry in the
source manifest.

## Source identity

The procedure source manifest includes:

- `src/benchmark/research/config.py`;
- the adapter and all model, augmentation, SPD, data, and harmonized-cache
  source dependencies; and
- all four configuration-only JSON files.

No path below `output/research/` is part of the runtime source closure.
Changing code or any configuration file changes the experiment contract and
prevents an old partial result from resuming under a different procedure.

## Verification

Run the focused tests in the project's isolated UV environment:

```bash
MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/eegthingy-mpl \
UV_CACHE_DIR=/tmp/eegthingy-uv-cache \
uv run --frozen --no-sync \
python -m pytest -q tests/core/test_local_outer_refit_benchmark.py
```

The configuration tests assert every preserved hyperparameter, source-manifest
closure, absence of result paths, recursive rejection of outcome/runtime keys,
missing and unknown setting rejection, settings-only hash tampering, and
duplicate-key and source-manifest mismatch rejection. The remaining module
tests cover the local split, covariance transform, selection/refit boundary,
deterministic reset, fixed-duration refit, prediction-only test access, and
resumable artifact validation.
