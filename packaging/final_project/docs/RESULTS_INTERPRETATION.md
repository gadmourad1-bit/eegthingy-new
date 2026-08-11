# Interpretation of the verified results

## Bottom line

The exact 43-model common grid is complete. Under its frozen shared recipe,
the observed equal-dataset balanced-accuracy leader was the in-house FBCNet
derivative/configuration `cardinal_fbc_compactdyn_scale025_extended` at
**73.9900%**. The runner-up was
`cardinal_fbc_compactdyn_extended` at **73.7571%**. The prespecified external
context comparator, TCFormer, ranked 12th at **73.3163%**.

The selected leader exceeded TCFormer by **0.6737 percentage point**. Its
descriptive fixed-suite 95% participant-cluster bootstrap interval was
**[-0.427, +1.899] points**. Because the interval includes zero and the leader
was selected from the same 43-model outcome table, the result does not support
a confirmatory superiority claim.

All five cohorts were opened development cohorts. The result is not global
SOTA, independent confirmation, clinical validation, or evidence of benefit
for disabled or paralyzed people.

## Integrity of the result package

The final score-blind audit reports:

| Audit field | Value |
|---|---:|
| Expected jobs | 96,320 |
| Complete jobs | 96,320 |
| Missing / extra / failed | 0 / 0 / 0 |
| Live / stale claims | 0 / 0 |
| Partials | 0 |
| Unsafe paths / unexpected root entries | 0 / 0 |
| Preflight attestation valid | Yes |
| Resolved forensic artifacts | 6 of 6 |

The six forensic artifacts are preserved stale-claim evidence from interrupted
execution, not failed or duplicated predictions. The canonical files are:

- `results/common_grid_v6/final_audit.json` — SHA-256
  `010b44542157efb6ba5a95376198b9b691e9803ce3fe605e77cb5cdb9739d063`;
- `results/common_grid_v6/analysis/manifest.json` — SHA-256
  `960ca912d058a2ad3b60def4116dbba436f8ae9162916cde42a4ab265bf01f5d`;
  and
- `results/common_grid_v6/analysis/RESULTS.md` — analyzer-generated complete
  43-model table.

The sealed publication also contains `job_metrics.csv`,
`subject_seed_metrics.csv`, and `subject_metrics.csv` with pseudonymous
participant keys and derived outcomes. They are not raw EEG or trial
predictions, but they remain participant-derived and require privacy/Local
Exp4 governance approval before public disclosure.

## Primary ranking

The leading common configurations were:

| Rank | Model | Equal-dataset balanced accuracy |
|---:|---|---:|
| 1 | `cardinal_fbc_compactdyn_scale025_extended` | 73.9900% |
| 2 | `cardinal_fbc_compactdyn_extended` | 73.7571% |
| 3 | `cardinal_dynamics_sinc_extended` | 73.6674% |
| 4 | `cardinal_dynamics_extended` | 73.6314% |
| 5 | `cardinal_fbc_compactdyn_scale010_extended` | 73.5382% |
| 6 | `cardinal_fbc_physical_extended` | 73.5349% |
| 7 | `cardinal_fbc_compactdyn` | 73.4642% |
| 8 | `cardinal_fbc_compactdyn_scale025` | 73.4204% |
| 9 | `cardinal_fbms_extended` | 73.3652% |
| 10 | `cardinal_dynamics_compact` | 73.3180% |
| 11 | `fbmsnet` | 73.3173% |
| 12 | `tcformer` | 73.3163% |

Ranks 10–12 differ by only 0.0017 percentage point. More broadly, many of the
top configurations have overlapping descriptive intervals against TCFormer.
Rank order should not be interpreted as a clinically or statistically
meaningful separation for every adjacent pair.

The full-precision authority is
`results/common_grid_v6/analysis/model_ranking.csv`; rounded percentages here
are presentation values.

## Dataset-specific behavior

The overall winner was not the winner on every dataset:

| Dataset | Observed dataset leader | Leader BA | Overall leader BA | TCFormer BA |
|---|---|---:|---:|---:|
| Local Exp4 | `cardinal_dynamics_sinc_extended` | 91.6250% | 91.2500% | 91.2083% |
| BNCI2014-001 | `cardinal_fbc_compactdyn_extended` | 75.7330% | 75.4630% | 72.3843% |
| BNCI2014-004 | `cardinal_dynamics_sinc_extended` | 78.9623% | 77.9236% | 78.0000% |
| Cho2017 | `cardinal_fbc_compactdyn_scale025_extended` | 71.2455% | 71.2455% | 69.4657% |
| PhysioNet MI S1–S54 | `cardinal_dynamics` | 58.8414% | 54.0679% | 55.5232% |

The last two comparison columns are what the analyzer-generated
`RESULTS.md` calls “descriptive leader and TCFormer”: “leader” there means the
overall fixed-suite leader, not the per-dataset winner. The per-dataset winner
column above is derived from the sealed `dataset_summary.csv`.

This heterogeneity is central. The overall leader gained most clearly on
BNCI2014-001 and Cho2017, was close locally, trailed TCFormer slightly on
BNCI2014-004, and trailed it on PhysioNet. Equal-dataset averaging prevents the
large Cho/PhysioNet cohorts from dominating, but it does not establish
transportability.

## Accuracy is not the only outcome

For the overall leader versus TCFormer, the equal-dataset aggregate was:

| Metric | Overall leader | TCFormer | Interpretation |
|---|---:|---:|---|
| Accuracy | 74.02% | 73.36% | Similar direction to balanced accuracy |
| Balanced accuracy | 73.99% | 73.32% | Primary descriptive outcome |
| Macro F1 | 72.59% | 70.59% | Secondary outcome |
| AUROC | 82.09% | 80.72% | Secondary; averaged only where defined |
| Negative log likelihood | 0.6261 | 0.5032 | Lower is better; TCFormer was better |
| Multiclass Brier | 0.3692 | 0.3212 | Lower is better; TCFormer was better |
| ECE (15 bins) | 14.29% | 6.98% | Lower is better; TCFormer was substantially better calibrated |

The observed accuracy leader was not the calibration leader. Any application
that uses confidence thresholds, abstention, or command activation must not
choose a model from balanced accuracy alone. Calibration should be evaluated
prospectively in the intended population and deployment stream.

## Why this is descriptive rather than confirmatory

1. All five datasets and all Local Exp4 participants had already informed the
   broader development program.
2. The reported leader was selected from 43 configurations using these same
   outcomes.
3. The fixed-suite bootstrap preserves participant clustering, but it is not
   adjusted for winner selection.
4. Only five dataset environments are represented, so an equal-dataset mean
   is not a guarantee over the global population of datasets, montages, or
   users.
5. The target clinical population was not evaluated.

The analysis intentionally reports no p-values or null-hypothesis decisions.
An interval crossing zero is not proof of equivalence; equivalence and
non-inferiority require prespecified margins and a suitable confirmation
design.

## Local-data winner

On the eight-participant Local Exp4 development cohort,
`cardinal_dynamics_sinc_extended` achieved the highest participant-equal,
five-seed balanced accuracy at 91.6250%. The overall leader achieved 91.2500%
and TCFormer 91.2083%.

Those differences are small, and the local held-out recording was part of the
opened development stream. It is reasonable to call
`cardinal_dynamics_sinc_extended` the **observed local development winner under
this common recipe**. It is not reasonable to call it independently validated,
clinically superior, or the best model for future local users without a new
prospective evaluation.

## Gauge is a failed gate, not a hidden winner

GaugeQuotientCrossMomentNet was evaluated later under a separate prespecified
17-subject Gate‑1 protocol. It achieved 75.0704% equal-dataset balanced
accuracy versus 74.3988% for its CardinalFBC-micro-extended primary reference,
a +0.6716-point margin, and won 11/17 subjects.

However, the candidate had nonnegative dataset deltas on only two of five
datasets. The frozen rule required at least three. Its negative deltas were
Local Exp4 -1.6667 points, BNCI2014-001 -1.3889, and Cho2017 -0.6250; positive
deltas were BNCI2014-004 +1.4583 and PhysioNet +5.5804. The persisted outcome
is `passed=false` with failure action
`kill_candidate_before_disjoint_gate`.

Therefore:

- Gauge is not a 44th common-grid row;
- its 75.0704% subset value cannot be ranked against the 96,320-job common
  aggregate as if the protocols were identical;
- Gate 2, tuning, renamed retries, and promotion are forbidden by the frozen
  decision; and
- the honest result is a numerically interesting but failed, development-only
  falsification gate.

The canonical Gate‑1 analysis is `results/gauge_gate1/analysis.json`, SHA-256
`d89480531b4f37f98aa1faae27f976038ec5e09322368d02a9626bfec9d29eff`.

## Other tracks

The independent verification under
`results/cardinal_fbms_transfer/independent_verification_lab.json` checks a
separate 8,675-record native-transfer artifact. Its report SHA-256 is
`7c1302f17f22a4e895c7e42257f5f0a8ab1badf0d59b93ea6b4ee314489f2562`.
That artifact answers a checkpoint/native-montage transfer question and must
not be inserted into the common architecture ranking.

Historical HemiQ confirmation, local binary procedures, GeoAdapt screens,
CHSD screens, author-recipe references, deterministic controls, and incomplete
CardinalFBC transfer work have their own protocols and privacy boundaries.
Some are unrun under the harmonized release contract, some are negative, and
some contain participant/trial material excluded from this clean bundle. See
`docs/PROTOCOL_BOUNDARIES.md` rather than using the nearest available number.

## Defensible paper language

A defensible summary is:

> In an exploratory, frozen shared-recipe benchmark spanning five previously
> opened motor-imagery EEG cohorts, an in-house CardinalFBC continuation
> configuration was the observed equal-dataset balanced-accuracy leader. Its
> descriptive advantage over the prespecified TCFormer comparator was small
> and uncertain, calibration was worse, and independent confirmation remains
> necessary.

Avoid “state of the art,” “top globally,” “significantly superior,” “proven
novel,” “montage invariant,” “reference invariant,” “clinical,” and “works for
paralyzed people.” None is established by the distributed evidence.
