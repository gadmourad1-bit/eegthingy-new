# Research layer

`benchmark.research` contains exploratory and earlier local-method
implementations, including GeoAdapt, parity, CAMEO, ORBIT, HemiQ, filter-bank
models, and classical controls. It is part of the single
`benchmark` namespace; it is not a second package or a second formal
benchmark implementation.

The benchmark layer may reuse these model implementations. This layer does
not own formal dataset splits, publication aggregation, the 43-model roster,
or leaderboard claims, and it never imports the benchmark layer. Reusable
SPD algebra, augmentation, CSP initialization, and metric helpers have been
moved to `benchmark.shared` so both higher layers can use them without
a dependency cycle.

## Supported use

From the project root, use:

```bash
scripts/reproduce.sh setup runtime
scripts/reproduce.sh setup test
scripts/reproduce.sh test
```

Direct research-module commands are internal unless a current procedure
document explicitly prescribes one. Caches, checkpoints, raw EEG, and result
trees must remain outside the source directory.

## Contents

| Area | Responsibility |
| --- | --- |
| `config.py`, `data.py`, `protocols.py` | Local-cohort and preprocessing contracts |
| `model.py`, `filterbank_net.py`, `hemi_q_field_net.py` | Geometric and filter-bank implementations |
| `cameo_net.py`, `parity_net.py`, `parity_fuse_net.py`, `orbit_transport_net.py` | Exploratory in-house families |
| `baselines.py`, `tangent_anchor.py`, `dnn_baselines.py` | Classical and neural comparators |
| `*_benchmark.py`, `experiment.py`, `loso.py` | Auxiliary procedures, not the formal public command surface |
| `deployment.py`, `recenter.py` | Experimental causal decoding and adaptation helpers |

Tests mirror this layer under `tests/research/`; shared numerical regressions
live under `tests/shared/`.

Results from historical or auxiliary research protocols must not be inserted
into the common-grid leaderboard. See
[`docs/PROTOCOL_BOUNDARIES.md`](../../../docs/PROTOCOL_BOUNDARIES.md) and
[`docs/PROJECT_STRUCTURE.md`](../../../docs/PROJECT_STRUCTURE.md).

This is experimental research software, not a medical device. The included
evidence does not establish clinical safety or benefit.
