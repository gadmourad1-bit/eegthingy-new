# ORBIT configuration provenance boundary

This is a read-only inventory. It does not promote, extract, normalize, or
reconstruct a historical configuration. `ieee_mi.procedure_grid inventory`
produces the matching machine-readable view without opening any file below a
`results/` directory.

## Decision

Only `architecture.orbit_v3` has an exact outcome-free configuration file:

| Path | Bytes | SHA-256 |
|---|---:|---|
| `configs/local_procedures/orbit_v3.json` | 849 | `95edd8a374892e4f28a8feb4e3dfe4ffb32e415b4c3cca2d1ea4a6f89a72b456` |

ORBIT-v1/v2/v4/v5 remain **blocked historical/post-hoc variants** in the
harmonized-v2 procedure runner. Their stable IDs are real registry entries and
their result artifacts contain useful provenance, but result artifacts cannot
become a formal runtime configuration dependency after their outcomes have
already been observed.

## Variant inventory

### `architecture.orbit_v1`

- Historical artifact:
  `deepnet/results/parity/orbit_v1_cho_dev8_formal.json`
- Artifact SHA-256:
  `b877b6ec6b758c1ee4f16d330d89e1febfe47561c99704b59a2075fa75c08b34`
- Recorded command:
  `/home/user/Desktop/eegthingy_codex_arch_20260718/deepnet/orbit_transport_benchmark.py --mode cho-dev --subjects 1-8 --folds 5 --seeds 7 --output deepnet/results/parity/orbit_v1_cho_dev8_formal.json`
- Stable-ID evidence:
  `ieee_mi.model_registry._ORBIT_CONFIGURATIONS` and
  `docs/MODEL_REGISTRY.md` identify v1 as the GroupNorm configuration.
- Complete current config object in the artifact: **no**. The artifact settings
  omit the current explicit `normalization` field.
- Deterministic config-only extraction without inference: **no**. Supplying
  `normalization="group"` would rely on historical default/registry evidence,
  which is still an inference during a post-outcome reconstruction.
- Runner decision: **blocked**.

### `architecture.orbit_v2`

- Historical artifact:
  `deepnet/results/parity/orbit_v2_batch_cho_dev8.json`
- Artifact SHA-256:
  `626b65a3463e7ead4cffb04f79bc79d9d5e75c25b86592c707f7e64291673b51`
- Recorded command:
  `/home/user/Desktop/eegthingy_codex_arch_20260718/deepnet/orbit_transport_benchmark.py --mode cho-dev --subjects 1-8 --folds 5 --seeds 7 --normalization batch --output deepnet/results/parity/orbit_v2_batch_cho_dev8.json`
- Stable-ID evidence:
  `ieee_mi.model_registry._ORBIT_CONFIGURATIONS` identifies v2 as the
  view-symmetric BatchNorm family configuration.
- Complete current config object in the artifact: **yes, apparently**.
- Deterministic field extraction without guessing values: **mechanically yes**,
  after removing runtime seed/device.
- Scientific eligibility of extracting it now: **no**. The only object is
  nested in an outcome-bearing result created before this config-only
  boundary. Creating a new config from it now would be post hoc.
- Runner decision: **blocked**.

### `architecture.orbit_v4`

- Historical artifact:
  `deepnet/results/parity/orbit_v4_transport_only_cho_dev8.json`
- Artifact SHA-256:
  `c337f8f3d91a2b6781b7cc59fb6bfde7687cb122fb9ed50d56c1e360b71cf6b5`
- Recorded command:
  `/home/user/Desktop/eegthingy_codex_arch_20260718/deepnet/orbit_transport_benchmark.py --mode cho-dev --subjects 1-8 --folds 5 --seeds 7 --normalization batch --transport-auxiliary-weight 0.35 --view-auxiliary-weight 0 --orientation-penalty 0 --output deepnet/results/parity/orbit_v4_transport_only_cho_dev8.json`
- Stable-ID evidence:
  `ieee_mi.model_registry._ORBIT_CONFIGURATIONS` identifies v4 as the v3
  transport-only ablation.
- Complete current config object in the artifact: **yes, apparently**.
- Deterministic field extraction without guessing values: **mechanically yes**.
- Scientific eligibility of extracting it now: **no**, because the source is
  outcome-bearing and the promotion would be post hoc.
- Runner decision: **blocked**.

### `architecture.orbit_v5`

- Historical artifact:
  `deepnet/results/parity/orbit_v5_rawmean_only_cho_dev8.json`
- Artifact SHA-256:
  `b22597098cf127070a295f1338c33c7231fe8d15c10471ddb22388fd532a79af`
- Recorded command: `-c`
- Stable-ID evidence:
  `ieee_mi.model_registry._ORBIT_CONFIGURATIONS` identifies v5 as the v4
  procedure restricted to the `raw_mean` candidate.
- Complete current config object in the artifact: **yes, apparently**.
- Deterministic field extraction without guessing values: **mechanically yes**.
- Command-level reproducibility: weaker than v2/v4 because the stored receipt
  is only `-c`.
- Scientific eligibility of extracting it now: **no**, because the source is
  outcome-bearing and the promotion would be post hoc.
- Runner decision: **blocked**.

## Why “mechanically extractable” is not “frozen before outcomes”

An exact JSON transformation can answer whether values can be copied without
guessing. It cannot establish that the copied configuration was separated from
the result when the experiment was designed. The harmonized-v2 runner therefore
requires both:

1. a complete exact mapping to the current typed configuration; and
2. an outcome-free, byte-pinned config file approved as a runtime dependency.

v2/v4/v5 satisfy only the first property inside historical results. v1 does not
even satisfy that property without resolving the missing normalization field.
None of the four is executed by this runner revision.

The historical paths and commands are static strings in the provenance
inventory. They are not included in `SOURCE_FILES`, never parsed by
`_config_inventory`, and cannot contribute a metric, selected epoch, score, or
prediction to a runtime plan.

## Future unblocking rule

Do not create config-only files for these variants merely to fill the table.
Unblocking would require a separately reviewed protocol decision that:

- explicitly labels the configuration reconstruction post hoc;
- documents its source and every transformation;
- decides whether it is appropriate only for exploratory replication;
- versions a new config schema and stable hash; and
- freezes that decision before any new harmonized-v2 outputs are opened.

Until then, the correct table entry is “not run - no pre-outcome exact config,”
not a guessed score and not an alias of ORBIT-v3.

