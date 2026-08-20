# CHSDNet novelty audit

Status: bounded literature search and independent red-team completed
2026-07-29; the fixed conditioned candidate subsequently failed its
predeclared disjoint-subject robustness gate and is not promoted.  
Scope: motor-imagery EEG decoding, complex cross-spectral representations,
Hermitian positive-definite geometry, time-varying covariance/spectral
surfaces, and neural temporal/frequency decoders.

## Audit conclusion

The proposed model is **CHSDNet: Complex-Hermitian Surface-Differential
Network**. Its proposed contribution is the conjunction of:

1. a trial represented as a time-by-frequency surface of regularized complex
   coherency matrices;
2. a Hermitian matrix-log state that retains cross-channel phase while
   separating scale into a distinct log-power branch;
3. explicit identity- or training-reference-anchored finite-difference fields
   over time, frequency, and their mixed difference; and
4. a learned factorized decoder of the state and differential fields.

The search found close precedents for every individual ingredient and for
several three-way combinations, but did not find an MI decoder combining all
four items. The novelty assessment is therefore **amber**: a narrow
compositional claim may be defensible, but a claim of new geometry, the first
complex/Riemannian MI network, or the first time-frequency connectivity
decoder is not. The bounded search cannot prove that no unpublished,
unindexed, differently worded, or later work exists.

The core novelty is the **Hermitian Surface Differential (HSD) representation
and layer**. Coordinate-based montage projection, a TCN/attention decoder,
filter banks, shrinkage, and a power branch are supporting engineering and must
not be presented as independently novel.

### Empirical disposition

The fixed `chsdnet_conditioned_005` hybrid was evaluated on 115 previously
unused opened-development participants across five datasets, alongside
CardinalFBC micro extended, FBCNet, and TCFormer. Its equal-dataset balanced
accuracy was 73.543%, versus 74.087%, 73.628%, and 72.862%, respectively.
Although it was within 0.544 percentage point of the strongest reference and
exceeded TCFormer, it failed the predeclared dataset-breadth and paired-subject
win-rate requirements. The immutable decision was
`stop_chsd_promotion`.

Accordingly, the material below documents a defensible narrow representation
idea and a negative experiment. It is not a selected model, a formal-grid
candidate, confirmation evidence, or support for a state-of-the-art claim.

## Safe claim language

> Based on the literature reviewed, CHSDNet appears to be the first
> motor-imagery EEG decoding framework to combine a regularized
> complex-coherency HPD time-frequency grid with an explicitly structured set
> of state, temporal first-difference, spectral first-difference, and mixed
> time-frequency finite-difference channels computed in one shared
> Log-Euclidean Hermitian coordinate chart. The contribution is this
> task-specific representation/decoder combination, not a new Riemannian
> metric, matrix logarithm, complex connectivity estimator, or intrinsic
> differential geometry.

The claim must be revised or withdrawn if a closer method is found before
submission.

## Terminology boundary

For a shared Hermitian log chart with state \(X_{b,t}\), the implemented
quantities are:

\[
\Delta_t X_{b,t}
  = \frac{X_{b,t} - X_{b,t-1}}{t_t-t_{t-1}},
\qquad
\Delta_f X_{b,t}
  = \frac{X_{b,t} - X_{b-1,t}}{f_b-f_{b-1}},
\]

\[
\Delta_{tf}X_{b,t}
  = \frac{\Delta_tX_{b,t}-\Delta_tX_{b-1,t}}{f_b-f_{b-1}}.
\]

These are:

- shared-chart or reference-anchored log-domain differences;
- tangent-coordinate displacements; and
- finite-difference fields (or, more conservatively, inter-window and
  inter-band contrasts).

They are **not**:

- Riemannian parallel transport;
- intrinsic or geodesic velocity;
- a Koopman operator;
- curvature; or
- proof of a biophysical causal flow.

Those stronger terms require different mathematics and numerical verification.

## Closest prior art

| Work | Status | Material overlap | Remaining distinction |
|---|---|---|---|
| Li, Wong, and deBruin, “EEG Signal Classification Based on a Riemannian Distance Measure,” [DOI 10.1109/TIC-STH.2009.5444491](https://doi.org/10.1109/TIC-STH.2009.5444491) | Peer-reviewed conference, 2009 | EEG as curves of complex HPD spectral matrices with a Riemannian distance | Non-neural sleep-stage method; no MI time-frequency surface differential layer or separated ERD branch |
| Augmented complex CSP paper (2014) | Peer-reviewed, 2014 | Complex Hermitian analytic covariance and phase for MI | Whole-trial CSP rather than a learned band-by-time HPD surface and finite-difference decoder |
| Xu et al., “Feature Extraction from the Hermitian Manifold for Brain-Computer Interfaces,” [DOI 10.1109/NER.2019.8717011](https://doi.org/10.1109/NER.2019.8717011) | Peer-reviewed NER, 2019 | Analytic-signal complex covariance HPD features across BCI paradigms | Static Hermitian features rather than a trial-local time-frequency coherency surface |
| Kalaganis et al., “A complex-valued functional brain connectivity descriptor amenable to Riemannian geometry,” [DOI 10.1088/1741-2552/ab8130](https://doi.org/10.1088/1741-2552/ab8130) | Peer-reviewed JNE, 2020 | Complex phase-connectivity HPD matrices, AIRM/matrix-log distances, and EEG classification | No explicit two-axis state/difference grid or learned MI decoder |
| Mishuhina and Jiang, “Complex common spatial patterns on time-frequency decomposed EEG for BCI,” [DOI 10.1016/j.patcog.2021.107918](https://doi.org/10.1016/j.patcog.2021.107918) | Peer-reviewed Pattern Recognition, 2021 | Complex CSP applied in time-stage by frequency cells on multiple MI datasets | No coherency-HPD shared-log state or explicit directional/mixed difference decoder |
| FUCONE, [DOI 10.1109/TBME.2022.3154885](https://doi.org/10.1109/TBME.2022.3154885) | Peer-reviewed TBME, 2022 | Connectivity/coherence estimators and Riemannian/tangent classifiers across eight MI datasets | Static estimator fusion rather than a learned complex surface-difference representation |
| Tensor-CSPNet, [arXiv:2202.02472](https://arxiv.org/abs/2202.02472) | Preprint, 2022 | Time/spatial/frequency tensors of real SPD covariance for MI | No complex coherency phase and no explicit shared-chart differential fields |
| MAtt, [NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/c981fd12b1d5703f19bd8289da9fc996-Abstract-Conference.html) | Peer-reviewed, 2022 | End-to-end attention on EEG SPD manifolds | Real SPD attention, not complex spectral coherency with a state/difference decomposition |
| Chau and von Sachs, time-varying spectral-matrix estimation, [DOI 10.1016/j.csda.2022.107477](https://doi.org/10.1016/j.csda.2022.107477) | Peer-reviewed CSDA, 2022 | Strongest mathematical prior: a 2-D time-frequency surface of complex HPD spectral matrices with intrinsic wavelet detail coefficients represented at the identity tangent space | Statistical estimation for seizure EEG, not coherency-normalized MI classification or explicitly separated shared-chart state, time, frequency, and mixed fields |
| HarMNqEEG, [DOI 10.1016/j.neuroimage.2022.119190](https://doi.org/10.1016/j.neuroimage.2022.119190) | Peer-reviewed NeuroImage, 2022 | Frequency-indexed complex Hermitian cross-spectral matrices treated as a tensor and Riemannian-vectorized | General quantitative EEG rather than a trial-local learned MI surface decoder |
| KCS-FCNet, [DOI 10.3390/diagnostics13061122](https://doi.org/10.3390/diagnostics13061122) | Peer-reviewed, 2023 | Cross-spectral connectivity and a shallow CNN for binary MI | Connectivity maps are not constrained/processed as an HPD log surface and have no explicit differential fields |
| Fan et al., temporal-frequency-phase 3-D CNN, [DOI 10.3389/fnins.2023.1250991](https://doi.org/10.3389/fnins.2023.1250991) | Peer-reviewed Frontiers in Neuroscience, 2023 | Sliding-window PCC/coherence/PLV connectivity tensors decoded by a 3-D CNN | Does not form a regularized complex-coherency HPD log chart or explicit directional/mixed fields |
| Coherence-based GCN after spinal-cord injury, [DOI 10.3389/fnins.2022.1097660](https://doi.org/10.3389/fnins.2022.1097660) | Peer-reviewed Frontiers in Neuroscience, 2023 | Magnitude-squared coherence graph decoding in the target disabled population | Phase is discarded and no HPD log-surface differential representation is used |
| Graph-CSPNet, [DOI 10.1109/TNNLS.2023.3307470](https://doi.org/10.1109/TNNLS.2023.3307470) | Peer-reviewed TNNLS, 2024 | Manifold graph convolution over time-frequency real-SPD covariance tokens on five MI datasets | Does not retain complex phase/coherency or decompose shared-chart state and time/frequency differences |
| STaRNet, [DOI 10.1016/j.neunet.2024.106471](https://doi.org/10.1016/j.neunet.2024.106471) | Peer-reviewed Neural Networks, 2024 | CNN features, covariance, matrix log, and MI classification | Learned real covariance rather than a complex spectral HPD surface and HSD |
| “Averaging trajectories on the manifold of SPD matrices,” [EUSIPCO 2024 paper](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0001182.pdf) | Peer-reviewed conference, 2024 | Sliding-window real covariance trajectories on six MI datasets | Prototype/DTW classification, not learned complex spectral state and differential fields |
| DFBRTS, [DOI 10.1016/j.bspc.2024.106797](https://doi.org/10.1016/j.bspc.2024.106797) | Peer-reviewed BSPC, 2025 | Filter bank, real Riemannian tangent features, CNN, and cross-band interaction | Does not use phase-preserving complex coherency or an explicit two-axis surface differential |
| CGNet/FBCGNet, [DOI 10.1016/j.neunet.2025.107795](https://doi.org/10.1016/j.neunet.2025.107795) | Peer-reviewed Neural Networks, 2025 | Complex-valued convolutions, amplitude/phase, attention, and a dynamic graph for MI | Complex Euclidean graph model without HPD log geometry or explicit HSD fields |
| CorAtt, [IJCAI 2025 paper](https://www.ijcai.org/proceedings/2025/598) | Peer-reviewed, 2025 | Scale-invariant correlation-manifold attention for EEG | Real correlation matrices, not a complex phase-preserving time-frequency differential surface |
| Cortical-SSM, [DOI 10.1088/1741-2552/ae89e8](https://doi.org/10.1088/1741-2552/ae89e8) | Peer-reviewed J Neural Engineering, 2026 | Frequency-, channel-, and time-aware state-space MI decoding | Wavelet/convolutional features rather than complex HPD coherency; an SSM/CDE backbone is therefore not a novelty claim |
| Unified SPD Token Transformer, [arXiv:2601.21521](https://arxiv.org/abs/2601.21521) | Preprint, 2026 | Multiple SPD embeddings and Transformer token processing | Real static SPD tokens, not complex coherency or explicit surface differences |
| Nilsson and Bernhardsson, analytic covariance HPD classification, [DOI 10.1016/j.sigpro.2026.110786](https://doi.org/10.1016/j.sigpro.2026.110786) | Peer-reviewed/in press Signal Processing, 2026 | Complex HPD analytic covariance and AIRM classifiers on BCI IV-2a | Static wide-band covariance and a classical classifier; no band-by-time surface or learned HSD |

## Directions rejected as the core contribution

The audit found the following areas too occupied to support the paper's main
novelty:

- another CNN plus Transformer, Mamba, SSM, GRU, or attention block;
- generic real-SPD tokens or covariance trajectories;
- a complex-valued CNN or graph alone;
- coordinate-aware montage projection;
- midsagittal/mirror equivariance;
- generic dynamic graphs;
- persistent-homology feature fusion;
- path signatures;
- uncertainty, evidential, or calibration heads; and
- domain adaptation/generalization without a new representation.

These may appear as implementation support or controls, with the relevant
prior art acknowledged.

## Numerical contract

1. Build every CSD from a common multivariate estimator:
   \(S=\sum_j w_j z_jz_j^H/\sum_jw_j\). Pairwise independently estimated
   coherence values are forbidden because the assembled matrix need not be
   positive semidefinite.
2. Re-Hermitianize with \((S+S^H)/2\).
3. Normalize to coherency with the CSD diagonal, then shrink:
   \(R_\lambda=(1-\lambda)R+\lambda I\), where \(0<\lambda<1\).
4. Never use an elementwise logarithm. The trainable path uses a
   solve-and-multiply matrix-atanh series whose truncation bound is validated
   from matrix size, shrinkage, and term count. A spectral Hermitian
   eigendecomposition remains an independent forward audit.
5. In audit code, record minimum/maximum eigenvalues, condition numbers,
   Hermitian residuals, the series-versus-spectral discrepancy, and the
   reconstruction residual after matrix exponentiation. Positive shrinkage,
   rather than an eigenvalue clamp hidden in the training path, establishes
   the supported HPD lower bound.
6. Vectorize an \(r\times r\) Hermitian matrix into exactly \(r^2\) reals:
   real diagonal, \(\sqrt{2}\) times the real strict upper triangle, and
   \(\sqrt{2}\) times the imaginary strict upper triangle.
7. Fit any data-derived reference on outer-training rows only. Freeze it for
   validation/test. Refit it from train+validation only during the final reset
   and refit phase.
8. Never center on a prediction batch or use a class-specific reference.
9. Keep the log-power branch separate because coherency deliberately removes
   amplitude scale.
10. Keep the preprocessing/reference/filter-phase convention identical across
    every compared model.

## Required ablations

All ablations must use identical outer rows, seeds, optimizer budget, and a
comparable decoder capacity.

1. **Representation:** real covariance; static analytic HPD; complex
   coherency state sequence; full HSD.
2. **Information:** power only; coherency only; fused.
3. **Phase:** complex; real part; magnitude; imaginary part; phase randomized.
4. **Geometry:** Euclidean flattening; identity log chart; train-only
   log-Euclidean reference; optional AIRM/Karcher reference.
5. **Differentials:** static; state only; state plus time; plus frequency; full
   mixed term.
6. **Decoder:** parameter-matched TCN, GRU, and optional Neural CDE.
7. **Reliability:** none; rank/condition; taper-jackknife reliability.
8. **Montage support:** native/dataset-specific adapter versus coordinate
   projection, explicitly treated as robustness rather than novelty.
9. **Robustness:** channel dropout, colored noise, missing windows, session
   shift, calibration error, and latency.

## Falsification/kill criteria

The architecture is not promoted merely because it is novel.

- Do not claim a phase/complex advantage unless full complex features beat
  real-only, magnitude-only, and power-only controls in paired subject-level
  analyses.
- Do not claim a surface-differential advantage unless at least the temporal
  term and one spectral term improve over state-only on more than one dataset.
- Withdraw the representation claim if an equal-capacity ordinary 2-D decoder
  on log-coherency state alone matches the complete directional/mixed field
  model.
- Remove the reliability gate if it does not improve calibration or
  robustness.
- Do not claim continuous-time modeling unless a parameter-matched Neural CDE
  actually outperforms the discrete decoder.
- A development target is at least a 1.5-point mean gain over the strongest
  reproduced baseline on three datasets, with no regression greater than one
  point on another dataset and a pooled paired interval excluding zero. This
  is a project decision rule, not a universal publication threshold.

Regardless of outcome, all attempts, negative ablations, per-subject values,
confidence intervals, and failure reasons must be retained.
