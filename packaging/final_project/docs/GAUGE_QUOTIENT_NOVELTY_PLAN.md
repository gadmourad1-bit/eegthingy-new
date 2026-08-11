# GaugeQuotientCrossMomentNet: bounded novelty and falsification plan

## Status

`GaugeQuotientCrossMomentNet` (`gauge_quotient_crossmoment_v1`) is one
scratch-only experimental configuration. It has not been entered into a formal
model registry, no score plan has been created, and no score run has been
launched. A passing result would justify further evaluation; it would not by
itself establish state of the art, clinical utility, or suitability for an
assistive device.

The candidate is deliberately attached to the already strong
`cardinal_fbc_micro_extended` model through a function-preserving head
extension. The new representation is mathematically testable before any
outcome is observed.

## Claim boundary in one paragraph

The candidate contribution is a compact continuation that combines:

1. learned continuous spatial potentials evaluated at supplied channel
   coordinates;
2. a valid-channel, mask-dependent projection into the quotient by the
   spatially constant voltage mode;
3. fixed band/window log-variance features; and
4. shrinkage-normalized, multi-lag antisymmetric cross-covariance ("lag
   wedge") features in the same zero-extended constrained classifier.

The checked literature contains every broad ingredient in nearby forms:
coordinate-aware arbitrary-montage EEG models, reference-free differences and
surface Laplacians, graph MI networks, hemispheric differences, filter-bank
variance networks, delay-augmented covariance, and phase-locking component
pairs. We therefore make only a bounded **candidate compositional novelty**
claim. We do not claim that the quotient projector, lag covariance, or
cross-channel differences are new mathematical objects, and we do not claim
priority without a systematic database search by the authors and a qualified
information specialist.

## Frozen configuration

There is one candidate, not a hyperparameter family.

| Item | Frozen value |
|---|---|
| Native predictor | `cardinal_fbc_micro_extended` |
| Model input sample rate | 128 Hz only |
| Continuation bands | [4,8), [8,12), [12,16), [16,22), [22,30), [30,40) Hz |
| FIR design | deterministic 65-sample Hamming-windowed sinc; float32 coefficients are residual-corrected to absolute DC sum <=1e-7 and have unit norm within 1e-6; not trainable |
| Spatial potential | regularized cardinal RBF over the fixed extended 31-position atlas |
| Quotient sources | 4 per band |
| Windows | 4 equal contiguous windows |
| Lags | 1, 2, 4, 8 samples (7.8125, 15.625, 31.25, 62.5 ms) |
| Covariance shrinkage | 0.10 toward the per-window mean variance |
| Variance floor | 1e-6 |
| Normalized moment clip | +/-0.9999 before `atanh` |
| New feature count | 96 log-variance + 576 lag-wedge = 672 |
| Classifier integration | one existing max-norm head extended with 672 exactly zero columns |
| Parameter ceiling | 30,000 trainable parameters including the native predictor |
| Model selection | none inside the model; no labels, split identity, subject identity, or test state |

Before any native filter or batch-normalization call, the candidate rejects a
non-rank-3 or empty EEG tensor, a non-floating or nonfinite EEG value, positions
with the wrong rank/shape, non-floating/nonfinite/zero positions, a non-Boolean
or wrongly shaped mask, an empty valid-channel set, an epoch length not
divisible by four, and a time window that is not longer than every frozen lag.
These rejected train-mode calls are required to leave every parameter, every
buffer (including batch-normalization counters), native hooks, and PyTorch RNG
state unchanged. One valid channel is allowed, but its quotient is the trivial
zero space.

## Mathematical definition

Let `X_b` be a model-input trial with channels by time, `m_b` its Boolean valid
channel mask, and `r_c` the supplied position of channel `c`. Shared fixed
filter `H_f` produces:

`U_b,c,f(t) = (H_f * X_b,c)(t)`.

For band `f` and source `s`, a learned continuous potential
`g_f,s(r; theta)` is evaluated at every supplied position. Define:

`P_m = diag(m) - m m^T / (m^T m)`.

The normalized spatial weight and source are:

`q_b,f,s = P_m g_f,s / max(||P_m g_f,s||_2, eps)`

`z_b,f,s(t) = sum_c q_b,f,s,c U_b,c,f(t)`.

Every invalid-channel projected weight is zero and
`sum_c q_b,f,s,c = 0` up to floating-point roundoff. The stored trainable RBF
potential coefficients are **not** themselves constrained to sum to zero;
centering is performed on the evaluated, mask-specific weights. Consequently,
for any time-varying scalar `a_b(t)` added to every valid **model-input**
channel, `z(X + m a) = z(X)`.

Before normalization, the same response has the complete-graph difference
form:

`g^T P_m x = (1/n) sum_(i<j, valid) (g_i-g_j)(x_i-x_j)`.

This is an identity, not a new graph theorem.

Within each band and window, the candidate records source log variances. For
each fixed positive lag `l`, it also computes:

`A_ij(l) = 0.5 E[z_i(t)z_j(t+l) - z_j(t)z_i(t+l)]`.

`A(l)` is skew-symmetric, is identically zero at lag zero, and changes sign
under time reversal. Its upper-triangular entries are divided by a
shrinkage-stabilized variance scale, clipped, transformed with `atanh`, and
flattened in the frozen band-window-lag-pair order. This explicit
time-reversal-odd component is the nonredundant hypothesis relative to the
candidate's same-time log-variance features.

Global voltage sign inversion is different from time reversal. Because the
wedge is bilinear, `A_l(-z) = A_l(z)`; the continuation is globally
sign-invariant, not sign-odd. This statement applies only to the quotient
continuation: the native CardinalFBC path is unconstrained, so the complete
predictor is not claimed to be globally sign-invariant. Time reversal preserves
a window's complete zero-lag covariance but negates any nonzero lag wedge.
Therefore the lag wedge cannot be recovered as a function of the zero-lag
second moment alone. This is a mathematical nonrecoverability statement, not
evidence that the feature is superior or a claim that the lag wedge is a new
statistic.

The four learned source indices are ordered parameters and their flattened
feature columns are bound to ordered classifier columns. The model does not
claim invariance to permuting only its latent sources. A simultaneous source
permutation, pair-orientation conjugation, and matching classifier-column
permutation describes the same function, but that reparameterization is not a
runtime symmetry exposed by the implementation.

The native feature vector and the 672 new features enter one classifier:

`logits = W_expanded [native_features ; quotient_features] + b`.

At construction, the native columns and bias are copied exactly and every new
column is zero. The expanded max-norm operation has the same native row norm
because appending zeros cannot change that norm. The candidate therefore
represents exactly the native predictor before training, apart from possible
floating-point kernel scheduling differences in a larger matrix multiply.

## Important structural redundancy and preprocessing limit

For exact common-average-referenced point-electrode data, `1^T x = 0` and
`P x = x`. Therefore:

`g^T x = (P g)^T x`.

On such data an unconstrained linear spatial filter has an equivalent sum-zero
filter. The quotient constraint adds no linear representational power. It can
only act as a parameterization/regularization and missing-channel deployment
bias.

The project's point-electrode caches are common-average referenced, but
source-fitted scaling subsequently uses a separate mean and standard deviation
for each channel. If a raw common shift `a(t)` is applied before scaling, its
standardized contribution is `a(t)/sigma_c`, which is generally not common
across channels. Thus:

- the continuation is exactly invariant to a common offset at its **model
  input**;
- the current preprocessing pipeline is not proven invariant to an arbitrary
  raw-voltage reference change;
- the complete predictor is not invariant because the preserved CardinalFBC
  floor is unconstrained; and
- an accuracy improvement must not be attributed to "new information" created
  by quotienting CAR data.

The scientifically meaningful hypothesis is narrower: the constrained
coordinate field may regularize montage/mask transfer, while the lag wedges
may add compact lag-orientation-sensitive cross-moment information not present
in same-time log variance. A nonzero temporal asymmetry does not establish
causality, directed information transfer, directed neural flow, or physiological
signal propagation. Both hypotheses can fail.

## BNCI2014-004 semantic boundary

BNCI2014-004 supplies three bipolar derivation signals. The nominal C3, Cz,
and C4 metadata used by the common model interface must not be described as
three point-electrode voltage measurements. On this dataset the proven
property is only **derivation-space common-offset invariance**: adding the same
time series to all three supplied derivation channels leaves the quotient
continuation unchanged. It is not a statement about physical re-referencing
of the original electrodes, which are unavailable through this representation.

No scalp-field figure or interpretation may depict the three learned values as
potentials measured at C3, Cz, and C4.

## Closest-art collision audit

This is a targeted primary-source audit, not a systematic review. "Difference"
below means a difference in the method as described by the cited source, not
proof that no equivalent method exists elsewhere.

| Work | Relevant overlap | Remaining difference and collision risk |
|---|---|---|
| [FBCNet (Mane et al.)](https://arxiv.org/abs/2104.01233) | fixed filter-bank spatial features and segmented variance for MI | It is the native conceptual floor; our filter-bank/log-variance component is not novel. |
| [Learning Topology-Agnostic EEG Representations / MMM, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a8c893712cb7858e49631fb03c941f8d-Abstract-Conference.html) | geometry-aware encoding and heterogeneous channel selections | MMM maps montages to a unified topology. The checked description does not establish a mask-dependent sum-zero voltage quotient or the fixed lag-wedge readout. Collision risk: medium. |
| [REVE, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html) | arbitrary electrode arrangements through 4D positional encoding | Much larger pretrained foundation model. No priority claim is made over its spatial representation. Collision risk: medium. |
| [Neural Brain Fields](https://arxiv.org/abs/2601.00012) | continuous coordinate-conditioned EEG field and nonexistent-electrode rendering | Very close to continuous field evaluation. Our quotient projection and discriminative lag-wedge continuation differ, but "continuous EEG field" is not novel here. Collision risk: high. |
| [Montage-agnostic frozen-model adapter, JNE 2026](https://doi.org/10.1088/1741-2552/ae6142) | learned coordinate interpolation for arbitrary montages | Close adapter concept; our continuation is a task-specific invariant spatial functional rather than embedding interpolation. Collision risk: high. |
| [GCNs-Net, IEEE TNNLS](https://doi.org/10.1109/TNNLS.2022.3202569) | graph filtering for MI | General graph overlap. Complete-graph edge differences are an algebraic interpretation here, not evidence of graph-method novelty. |
| [SF-TGCN](https://doi.org/10.1016/j.eswa.2023.121915) | graph-Laplacian spatial filtering and temporal graph convolution for MI | A Laplacian also removes constant graph modes. Our learned coordinate potential followed by a complete-graph quotient is a different parameterization, not a new invariance principle. Collision risk: high. |
| [DST-GNN (Zeng and Liu), Scientific Reports 2026](https://doi.org/10.1038/s41598-026-58803-5) | MI decoding with learned physical and functional adjacency, spatiotemporal graph convolution, and graph readout | Dynamic spatial/temporal channel interaction learning is close in purpose. The checked primary method does not show this candidate's mask-dependent voltage quotient or fixed skew-lag continuation, but those broad interaction claims are occupied. Collision risk: medium-high. |
| [Hemispheric-difference dual CNN](https://doi.org/10.3389/fnins.2022.865594) | explicit left-right channel differences for MI | Pairwise differences and hemispheric contrast are established. Our all-pairs identity and learned continuous potentials are broader, but difference features themselves are not novel. |
| [Local spatial analysis](https://doi.org/10.1152/jn.00560.2019) | average-reference, Laplacian, and contralateral-difference spatial filters | Establishes close signal-processing precedent and limits any reference-free claim. |
| [Carrara and Papadopoulo, Classification of BCI-EEG Based on the Augmented Covariance Matrix, IEEE TBME 2024](https://doi.org/10.1109/TBME.2024.3386219) and [Block-Toeplitz/Siegel follow-up](https://arxiv.org/abs/2406.16909) | delay/phase-space embedding and augmented spatial-temporal covariance for MI | If `C_l = E[z(t)z(t+l)^T]` is an off-diagonal block of the augmented covariance, this candidate's unnormalized wedge is exactly `(C_l-C_l^T)/2`. The proposed statistic is therefore a fixed linear extraction from already established augmented covariance, not a new covariance object. Collision risk: direct and very high. |
| [CTCD based on Augmented Covariance Networks (Su, Xie, Yang et al.), Neurocomputing 634 (2025) 129911](https://doi.org/10.1016/j.neucom.2025.129911) | MI classification with augmented-covariance neural/tensor coupling | This is a close neural continuation of augmented spatiotemporal covariance and materially narrows any neural-representation claim for delayed second-order structure. The candidate's explicit signed skew extraction, mask quotient, and zero-start attachment differ, but the augmented-covariance neural territory is occupied. Collision risk: direct and very high. |
| [Phase-SPDNet: Carrara et al., Geometric neural network based on phase space for BCI-EEG decoding](https://doi.org/10.1088/1741-2552/ad88a2) | phase-space/temporal augmentation of EEG covariance followed by an SPD neural network; evaluated for BCI decoding | This is a particularly close neural collision for learning from delay-augmented second-order EEG structure. Our signed skew-block extraction and zero-start scalar head differ from SPD-manifold processing, but neither temporal augmentation nor neural use of its covariance blocks is new. Collision risk: direct and very high. |
| [ST-MA-SENet (Wu), Scientific Reports 2026](https://doi.org/10.1038/s41598-026-58874-4) | Euclidean and Riemannian covariance representations with spatiotemporal/spectral dynamic-attention fusion for MI | This is nearby covariance-neural representation and multi-domain dynamic fusion. The checked primary page does not show an equation-level mask quotient or signed skew-lag extraction, but covariance-attention novelty language must remain narrow. Collision risk: high. |
| [Mijalkov et al., Directed Brain Connectivity Identifies Widespread Functional Network Abnormalities in Parkinson's Disease](https://doi.org/10.1093/cercor/bhab237) | lagged connectivity split into symmetric and antisymmetric analyses across temporal lags | This is an equation-level concept collision for using the antisymmetric part of lagged cross-relations to encode temporal orientation. It also demonstrates why a signed lag asymmetry must not be promoted here as a new statistic. Their directed-connectivity interpretation does not license causal or directed-flow claims for this discriminative EEG feature. Collision risk: direct and very high. |
| [Hindriks et al., Building blocks of functional connectivity measures for aperiodic electrophysiological brain signals](https://pmc.ncbi.nlm.nih.gov/articles/PMC12746282/) | derives the exact bivariate exterior/Pluecker coordinate `x_i y_j - y_i x_j` for EEG, then constructs a temporal irreversibility index from its lagged diagonals, equivalently the positive-minus-negative-lag cross-correlation | This is a direct equation-level collision with both the candidate's wedge terminology and its antisymmetric lag product. The candidate retains separate learned-source, band, window, lag, and pair coordinates and feeds them to a discriminative zero-start head rather than collapsing them into the paper's normalized connectivity index; that is an implementation composition, not a new exterior statistic or irreversibility principle. Collision risk: exact and decisive for the statistic-level claim. |
| [Self-organized phase-locked component-pair learning](https://doi.org/10.1016/j.asoc.2025.114250) | end-to-end pairwise phase synchronization plus spectral power for MI | Directional/phase coupling is close in purpose. Lag wedges are real-valued signed delayed cross-moments rather than phase-lock values, but both target inter-component timing. Collision risk: high. |
| [Covariance Density Neural Networks, TMLR 2026](https://openreview.net/forum?id=TwCkGi5XFB) | covariance-derived neural graph operator and subject-independent MI | Same-time/global covariance-density modeling differs from per-trial band/window lag wedges, but covariance-based MI networks are established. Collision risk: medium-high. |
| [MCFANet, Frontiers in Human Neuroscience 2026](https://doi.org/10.3389/fnhum.2026.1811759) | FBCSP, EEGNet-style convolution, and attention fusion for MI | This is broad filter-bank/convolution/attention overlap rather than an equation-level collision with the quotient or skew-lag construction. It reinforces that ordinary multi-branch attention fusion is not available as the novelty claim. Collision risk: broad, low-medium. |
| [Novel Features for Brain-Computer Interfaces (Lal et al., JNE 2007)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2267903/) and [RSE MI feature study (2021)](https://biomed.bas.bg/bioautomation/2021/vol_25.1/files/25.1_02.pdf) | band-limited univariate cubic temporal-asymmetry features for MI | These are not multivariate second-order lag wedges, but they decisively prevent any broad claim that time-reversal-sensitive or temporal-asymmetry MI features are new. Collision risk: high at the concept level. |
| [The covariance perceptron (PLOS Computational Biology 2020)](https://doi.org/10.1371/journal.pcbi.1008127) | neural classification using spatial-temporal covariance patterns, including time lags | Establishes lagged covariance as a neural time-series representation beyond EEG. Our explicit skew component and function-preserving MI integration are narrower implementation choices. Collision risk: very high. |
| [Hu et al., The Statistics of EEG Unipolar References: Derivations and Properties](https://doi.org/10.1007/s10548-019-00706-y) | derives rank-deficiency-by-one and the orthogonal-projector centering identity `T_r^+ T_r = T_AR` for EEG reference operators | This is a direct equation-level collision with treating constant-mode removal or centering projection as a new principle. Our mask-specific use inside a continuation is an implementation composition only; the projector and EEG reference principle are established. Collision risk: direct and very high. |

### Dated search ledger

This ledger records targeted additions to the bounded scan; it is not a
systematic-review search record.

| Date checked | Primary source | Search implication |
|---|---|---|
| 2026-07-30 | [DST-GNN, Scientific Reports](https://doi.org/10.1038/s41598-026-58803-5), published 15 July 2026 | Added learned physical/functional graph adjacency and spatiotemporal interaction overlap; no expansion of the candidate claim. |
| 2026-07-30 | [ST-MA-SENet, Scientific Reports](https://doi.org/10.1038/s41598-026-58874-4), published 3 July 2026 | Added Euclidean/Riemannian covariance plus spatiotemporal/spectral attention overlap; retained amber-red status. |
| 2026-07-30 | Hechong Su, Jieren Xie, Zengyao Yang, Yuncheng Ge, Jingya Fu, Chengxi Xie, Kai Zhang, Xinyi Hu, Sicong Zhang, and Guanghua Xu, [CTCD based on Augmented Covariance Networks, Neurocomputing 634, 129911](https://doi.org/10.1016/j.neucom.2025.129911), published 14 June 2025 | Added as a direct, very-high-risk augmented-covariance neural/tensor collision alongside Carrara and Phase-SPDNet. |
| 2026-07-30 | [MCFANet, Frontiers in Human Neuroscience, volume 20](https://doi.org/10.3389/fnhum.2026.1811759) | Added as broad FBCSP/EEGNet/attention overlap, not an equation-level quotient or skew-lag collision. |

### Novelty verdict before scores

**Amber-red, compositional only.** The exact implemented bundle was not found
in this bounded scan, but several papers strongly overlap, augmented
covariance can contain the lag-wedge information implicitly, and Hindriks et
al. derive the same exterior-coordinate product and its lagged
cross-correlation asymmetry explicitly for EEG. A publishable method claim
must therefore be limited to the complete learned-source, band-window-lag,
mask-specific coordinate-quotient, zero-start neural composition. The
exterior coordinate, lagged covariance, antisymmetric lag decomposition,
time-asymmetry principle, and centering projection are not new. A publishable
method claim would require:

1. a systematic search across IEEE Xplore, PubMed, Scopus/Web of Science,
   Crossref, arXiv, and OpenReview;
2. full-method, equation-level comparison rather than title/abstract review;
3. author review of forward and backward citations;
4. a clear argument for why explicit skew-lag extraction and quotient
   parameterization improve bias, stability, calibration, or compute; and
5. decisive ablation and disjoint-subject evidence.

If that work finds an equivalent neural layer, the novelty claim must be
withdrawn even if accuracy is strong.

## Property tests required before any score

The candidate test suite must pass all of the following:

- native head prefix and bias are copied exactly; all 672 added columns are
  exactly zero;
- initialized logits match an independently copied native predictor in train
  and evaluation modes exactly, and all retained native state evolves exactly;
- every invalid candidate input condition is checked before any native
  stateful call; adversarial train-mode rejections leave parameters, all
  buffers and batch-normalization counters, native hooks, and RNG unchanged;
- stored float32 FIR coefficients meet the explicit absolute DC-sum and
  unit-norm tolerances; trainable RBF potential coefficients are not
  misdescribed as sum-zero;
- every projected filter sums to zero over its valid channels;
- quotient sources and features are invariant to an arbitrary time-varying
  common model-input offset;
- joint permutation of EEG channels, positions, and masks leaves features and
  logits unchanged within floating-point tolerance;
- changing masked finite channel values cannot change quotient features;
- a one-channel valid set yields the zero quotient and finite features; an
  empty set fails;
- exact CAR inputs expose the equivalence of unconstrained and projected
  linear responses;
- reversing time reverses window order, preserves log variance, and negates
  every lag wedge;
- global voltage sign inversion preserves quotient log variance and lag
  wedges, without assigning that property to the complete predictor;
- a same-phase/no-lead synthetic process has zero lag wedge, while a
  phase-shifted quadrature process has a nonzero wedge with the expected
  reversal sign;
- the lag-zero wedge is exactly zero;
- the latent source order is explicitly head-bound; no unqualified
  source-permutation invariance is claimed;
- one constructed model accepts both frozen 256- and 320-sample epoch lengths,
  and the stateful native floor filter and batch norm each execute once per
  forward;
- 2-class and 4-class output shapes, 256- and 320-sample inputs, and 3-, 15-,
  and 21-channel montages are finite;
- first-step gradients are finite, the new head columns receive a gradient,
  and the quotient field wakes after the zero head takes its first update;
- repeated seeded construction and evaluation are deterministic;
- trainable parameters remain below 30,000; and
- neither the model signature nor model state contains labels, outcomes,
  predictions, test identity, split identity, or subject identity; and
- nested screen rules are recursively immutable at module scope, returned as
  independent JSON objects, and bound to canonical JSON plus SHA-256 identity.

A property failure is an implementation stop, not a reason to relax the test.

## Scratch-only screens using frozen references

`ieee_mi/gauge_quotient_screen.py` is a read-only specification helper. It
creates no plan, writes no artifact, and launches no work. It maps candidate
jobs to existing reference records so CardinalFBC, FBCNet, and TCFormer are not
rerun or changed.

### Gate runner environment and cache contract

Gate execution must use the project-private UV virtual environment and cache.
The launch envelope sets `VIRTUAL_ENV`, `UV_CACHE_DIR`, and the required
`IEEE_MI_UV_EXECUTABLE` to canonical absolute paths, then invokes that exact UV
binary with `uv run --offline --active --no-project`. The runner never searches
ambient `PATH` for UV. It binds the executable path, bytes, SHA-256, version,
`pyvenv.cfg`, complete `uv pip freeze`, required package versions, CUDA/cuDNN,
and NVIDIA driver versions into the immutable environment identity. A worker
whose explicit binding or fingerprint differs cannot execute.

Each held cache snapshot is immutable and inode-bound. Every independent
`ZipFile` or `numpy.load` consumer reopens that exact inode with a separate
open-file description, explicitly starts at archive byte zero, and revalidates
the held snapshot afterward. No consumer may inherit or alter another
consumer's file offset. Archive EOF and format failures are Gate protocol
errors, not raw exceptions.

Evidence snapshots also retain every directory descriptor on the target's
exact relative path from its declared stable root: project source, run,
reference run, data cache, UV environment, or UV executable directory. Full
directory mutation fingerprints and component bindings are revalidated with
the leaf or sealed package. A same-parent leaf/package swap, or an ancestor
directory swap, remains a protocol failure even if the original inode and
bytes are restored before revalidation.

The concrete roots are: `project_root` for the 16-file source closure,
`reference_root` for the frozen comparator plan and 51 records, `cache_root`
for each subject NPZ, `run_root` for the immutable plan, every candidate
result, and the persisted analysis package, the active virtual-environment
prefix for `pyvenv.cfg`, and the explicitly bound UV executable's parent for
the UV binary. The analysis package is published at
`run_root/analysis/analysis_gate1`; `run_root/analysis` is created before the
plan snapshot exists. Under the run authority lock, initialization also
precreates the exact dataset/model/subject/fold parent for each of the 17
candidate records before publishing the immutable plan. Workers validate those
parents and never create shared layout. Each result publication therefore
mutates only its own precreated fold directory; sibling workers cannot change
an ancestor pinned by an already completed result. Analysis publication
mutates only the precreated `run_root/analysis` subtree, so neither result nor
analysis publication changes the `run_root` fingerprint held by the plan.

The quiescence scope is deliberately narrow. While an evidence snapshot is
held, its stable root and only the ancestor directories on that target's path
must not receive direct namespace mutations. Work inside an already-existing
unrelated sibling subtree does not change the pinned chain and is permitted.
Reference/cache/source roots are read-only for the Gate, and persisted
analysis runs only after candidate result publication is quiescent. Mutable
claim files are coordination state rather than scientific evidence; their
shared directory is protected by the run authority lock and exact-inode
checks instead of the quiescent evidence-snapshot contract.

### Gate 1: 17 already opened subjects

Use the exact 17-subject fold-0/seed-7 cohort from the corrected CHSD reference
screen:

| Dataset | Subjects |
|---|---|
| Local Exp4 | 1, 4, 7 |
| BNCI2014-001 | 1, 5, 9 |
| BNCI2014-004 | 1, 5, 9 |
| Cho2017 | 1, 18, 35, 52 |
| PhysioNet MI | 1, 18, 36, 54 |

Only 17 new candidate jobs are eligible. The three comparators are read from
the run bound to corrected reference plan SHA-256
`57298f7c18b8841e741d72194cb601faf3ac71cb81b1209b1c3964ddf43c0fc6`.

Proceed only if every condition holds:

1. candidate equal-dataset balanced accuracy strictly exceeds
   `cardinal_fbc_micro_extended`;
2. its dataset delta to that reference is nonnegative on at least 3/5
   datasets;
3. no dataset deficit exceeds 3.0 percentage points; and
4. strict subject wins are at least 9/17.

Otherwise the frozen action is `kill_candidate_before_disjoint_gate`. Do not
tune lags, bands, source count, shrinkage, windows, feature transform, optimizer
recipe, or head initialization in response.

### Gate 2: 115 disjoint opened-development subjects

Only after Gate 1 passes, run one candidate job for each of the 115 subjects
used by the completed CHSD robustness amendment. Reuse its 345 frozen
reference records from plan SHA-256
`1e4009ae67e7b11ae60c76c5d110ec49a27246422d2617c9c818199ca799b9ae`.
The candidate subjects are disjoint from Gate 1.

Proceed only if every condition holds against
`cardinal_fbc_micro_extended`:

1. equal-dataset balanced-accuracy margin is at least +0.25 percentage point;
2. dataset delta is nonnegative on at least 3/5 datasets;
3. no dataset deficit exceeds 2.0 percentage points;
4. strict subject win rate is at least 0.50; and
5. the lower endpoint of a 10,000-replicate, subject-paired 95% bootstrap
   confidence interval for the balanced-accuracy delta is strictly above zero
   (frozen seed 2026073001).

Any failure gives `stop_candidate_promotion`. Gate 2 remains opened-development
evidence, not confirmation.

Infrastructure failures may be repaired only without looking at aggregate
candidate scores and with the same frozen configuration. A scientific gate
failure cannot be retried under a renamed candidate.

## Fixed ablations if and only if the candidate passes both gates

Ablations are explanatory and cannot select a replacement configuration:

1. native `cardinal_fbc_micro_extended` (already frozen);
2. candidate with quotient projection replaced by the unprojected continuous
   potential, same parameter count;
3. candidate with lag-wedge head columns fixed to zero (log variance only);
4. candidate with log-variance head columns fixed to zero (lag wedges only);
5. candidate with lag direction symmetrized, using the same lags and count;
6. mask stress test with 10%, 25%, and 40% randomly missing channels, with one
   frozen mask seed schedule; and
7. standardized-input common-offset stress test, reported separately from any
   raw-reference interpretation.

The quotient-vs-unprojected ablation is expected to be indistinguishable on
exact unscaled CAR data. A claimed accuracy benefit there would trigger an
audit for scaling, mask, numerical, or implementation differences.

## Kill criteria and claim language

Kill or retain only as a documented negative prototype if:

- a property test fails;
- the parameter budget is exceeded;
- Gate 1 or Gate 2 fails;
- the literature audit finds the same bundle or an equation-level equivalent;
- performance depends on treating BNCI2014-004 derivations as point
  electrodes;
- the quotient effect disappears under a gauge-compatible scaling control
  while lag-wedge ablations show no independent contribution;
- gains are confined to one dataset or one subject subgroup;
- a required result needs a post-result architecture or recipe change; or
- the method materially worsens calibration, robustness, or compute without a
  compensating prespecified benefit.

Permitted language before confirmation:

> We evaluated one prespecified experimental continuation that explicitly
> extracts mask-aware quotient spatial sources and time-reversal-odd lag
> cross-moments. Its novelty is compositional and remains subject to a
> systematic prior-art review.

Forbidden language:

- "first reference-invariant EEG neural network";
- "globally top" or "state of the art";
- "clinically validated";
- "works for paralyzed patients";
- "reference invariant" for the complete predictor or raw preprocessing
  pipeline; and
- "globally sign invariant" or "common-mode invariant" for the complete
  predictor;
- any claim that the lag wedge, lagged covariance, or antisymmetric lag
  decomposition is a new statistic;
- any claim that constant-mode projection, centering, or quotienting is a new
  EEG reference principle;
- "causal," "causality," "directed connectivity," "directed information
  transfer," or "directed neural flow" as an interpretation of the lag wedge;
  and
- any interpretation of BNCI2014-004 nominal coordinates as point-electrode
  potentials.
