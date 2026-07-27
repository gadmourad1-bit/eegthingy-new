# Mathematical note: regularized cardinal-RBF scalp fields

This note states properties of the implementation in **models.py**, not of an
idealized variant. It accounts for the ridge, clamped spherical distance,
field normalization, pseudoinverse initialization in **CardinalFBCNet**, and
the joint spatiotemporal field in **CardinalFBCMicroDynamicsNet**.

## 1. Implemented operator and notation

Let \(A\in\{21,31\}\) be the number of fixed atlas anchors and let
\(q_1,\ldots,q_A\in\mathbb R^3\) be **CANONICAL_21_POSITIONS** or
**EXTENDED_31_POSITIONS**. Coordinates are normalized internally. For nonzero
\(p,q\in\mathbb R^3\), **spherical_gaussian_kernel** evaluates

\[
\begin{aligned}
 \widehat p &= p/\lVert p\rVert_2,\\
 d_\epsilon(p,q)
 &=\arccos\!\left(
   \operatorname{clip}(\widehat p^{\mathsf T}\widehat q,
                       -1+\epsilon,1-\epsilon)
   \right),\\
 \kappa_\ell(p,q)
 &=\exp\!\left[-\frac{d_\epsilon(p,q)^2}{2\ell^2}\right],
\end{aligned}
\]

with \(\epsilon=10^{-6}\) and, by default, \(\ell=0.35\). The vector-norm
denominator is clamped below at \(10^{-8}\); the results below assume valid
nonzero electrode coordinates.

Define

\[
 K_{ij}=\kappa_\ell(q_i,q_j),\qquad
 M_\lambda=K+\lambda I_A,\qquad
 \lambda=10^{-4},
\]

and, for a \(C\)-channel coordinate matrix
\(P=(p_1,\ldots,p_C)^{\mathsf T}\),

\[
 B(P)=K(P,Q)M_\lambda^{-1}\in\mathbb R^{C\times A}.
 \tag{1}
\]

Equation (1) is exactly **CardinalScalpField.basis**:
**cardinal_inverse** stores \(M_\lambda^{-1}\). For \(F\) temporal bands and
\(S\) spatial sources, the learned coefficient tensor is
\(\Theta\in\mathbb R^{F\times S\times A}\). Its raw sampled field is

\[
 H_{fsc}(P)=\sum_{a=1}^{A}\Theta_{fsa}B(P)_{ca}.
 \tag{2}
\]

Given channelwise filtered EEG
\(Z\in\mathbb R^{N\times C\times F\times T}\), the spatial contraction is

\[
 Y_{bsft}=\sum_{c=1}^{C}
   \mathcal N(H(P))_{fsc}Z_{bcft}.
 \tag{3}
\]

Here \(\mathcal N\) is the normalization selected by **field_norm**:

- **none** leaves (2) unchanged.
- **unit** divides each \((f,s)\) channel vector by its Euclidean norm,
  clamped below at \(10^{-7}\).
- **max** applies the rowwise Euclidean projection onto the ball of radius
  **max_norm** through **Tensor.renorm**. **CardinalFBCNet** uses radius 2.

With a batchwise channel mask, the code first multiplies each field by the
corresponding zero-one mask and then normalizes it. The normalization is
therefore recomputed on the surviving channels.

**CardinalFBCNet** uses \(F=9\), \(S=32\), and max normalization. Its FBC
filter bank, spatial bias, batch normalization, SiLU, four log-variance views,
and constrained classifier follow the pinned Braindecode FBCNet.
**CardinalFBCMicroDynamicsNet** retains that complete FBC reference path and
adds a second field. In the second field,
\(\Theta^{\mathrm{micro}}\in\mathbb R^{8\times8\times A}\) jointly mixes
eight learnable temporal FIR filters and the electrode axis into eight output
sources. **CardinalSpatiotemporalField** unit-normalizes each output source
over both its temporal-filter and channel axes.

## 2. Simultaneous channel-coordinate permutation invariance

**Proposition 1 (channel relabeling).** Let
\(\Pi\in\{0,1\}^{C\times C}\) be any permutation matrix. Simultaneously reorder
the EEG channel axis, coordinate rows, and, when present, channel-mask entries:

\[
 Z'_{b,:,f,t}=\Pi Z_{b,:,f,t},\qquad
 P'=\Pi P,\qquad
 m'_b=\Pi m_b.
\]

Then **CardinalScalpField** and **CardinalSpatiotemporalField** produce the same
source signals before and after this reordering. Consequently, the logits of
**CardinalFBCNet** and **CardinalFBCMicroDynamicsNet** are invariant to the
same simultaneous permutation.

**Proof.** Kernel evaluation is rowwise, so

\[
 B(P')=B(\Pi P)=\Pi B(P).
\]

After flattening the \((f,s)\) axes, (2) is
\(H(P)=\Theta_\flat B(P)^{\mathsf T}\). Therefore

\[
 H(P')=\Theta_\flat B(P)^{\mathsf T}\Pi^{\mathsf T}
       =H(P)\Pi^{\mathsf T}.
\]

Multiplication by a simultaneously permuted mask preserves this relation.
Euclidean norms, the radial max-norm projection, and clamped unit
normalization all commute with coordinate permutation. Hence

\[
 \mathcal N(H(P'))=\mathcal N(H(P))\Pi^{\mathsf T},
\]

and the contraction in (3) satisfies

\[
 \mathcal N(H(P'))\,\Pi Z
 =\mathcal N(H(P))\Pi^{\mathsf T}\Pi Z
 =\mathcal N(H(P))Z.
\]

The fixed FBC filters and learnable micro FIR filters apply the same temporal
operator independently to every channel and therefore commute with \(\Pi\).
Every later operation acts on source, band, time, or feature axes and receives
identical input. \(\square\)

This is a paired permutation property. Permuting EEG without its coordinates,
using a differently permuted mask, or applying unpermuted channel-specific
preprocessing need not preserve the output. Floating-point reduction order may
also cause negligible numerical differences.

## 3. Near-cardinality with ridge regularization

At the atlas itself, (1) becomes

\[
 B(Q)=K(K+\lambda I)^{-1}
     =I-\lambda(K+\lambda I)^{-1}.
 \tag{4}
\]

Thus a stored coefficient vector is close to, but is not exactly, the vector
of field values at the anchors. If \(K\) is symmetric positive definite with
smallest eigenvalue \(\mu_{\min}>0\), then

\[
 \left\lVert B(Q)-I\right\rVert_2
 =\frac{\lambda}{\mu_{\min}+\lambda}.
 \tag{5}
\]

More generally, whenever \(K+\lambda I\) is invertible,

\[
 \left\lVert B(Q)-I\right\rVert_2
 \leq \lambda\left\lVert(K+\lambda I)^{-1}\right\rVert_2.
 \tag{6}
\]

The two finite matrices produced by the stored atlases are numerically
positive definite. The following diagnostics use the current float32 kernel
and inverse; the bound is evaluated from eigenvalues in float64.

| anchors | \(\mu_{\min}(K)\) | \(\lVert B(Q)-I\rVert_2\) | bound in (5) | maximum entrywise deviation |
|---:|---:|---:|---:|---:|
| 21 | 0.00368253 | 0.0264117 | 0.0264381 | 0.00766277 |
| 31 | 0.00103137 | 0.0883299 | 0.0883958 | 0.0205160 |

Both atlas basis matrices are full rank in the current tests. The 21-anchor
basis is within about 0.77% entrywise of identity; the 31-anchor basis is
within about 2.06%. Neither basis is identity-valued at its anchors when
\(\lambda=10^{-4}\).

Positive definiteness is a checked property of these finite matrices, not a
universal statement about a geodesic Gaussian on every spherical point set. A
new atlas requires a separate rank and conditioning check.

## 4. Exact reproduction and pseudoinverse initialization

Ridge regularization changes coefficient coordinates but, when
\(M_\lambda\) is invertible, does not shrink the underlying
\(A\)-dimensional kernel span. Consider a scalar reference field in that span:

\[
 g(p)=K(p,Q)\alpha.
\]

Choosing

\[
 \theta=M_\lambda\alpha
 \tag{7}
\]

gives \(B(p)\theta=g(p)\) at every coordinate \(p\), not only at the anchors.
If \(K\) is invertible, any desired anchor-value vector
\(v\in\mathbb R^A\) has \(\alpha=K^{-1}v\), so it is reproduced exactly by

\[
 \theta=(K+\lambda I)K^{-1}v=B(Q)^{-1}v.
 \tag{8}
\]

Equations (7)--(8) reconcile exact representability with near-cardinality:
setting \(\theta=v\) is approximate, but an invertible change of coordinates
allows the raw field class to realize every anchor-value vector.

For paired FBC initialization, let \(P\) be the observed \(C\)-electrode
montage, \(B=B(P)\in\mathbb R^{C\times A}\), and
\(w\in\mathbb R^C\) one indexed FBC spatial-weight vector. The code computes
in float64

\[
 \theta=B^+w,
 \tag{9}
\]

then copies \(\theta\) to the model dtype. The reproduced samples are

\[
 \widehat w=B\theta=BB^+w.
 \tag{10}
\]

The exact rank statements are:

1. Equation (10) is exact if and only if
   \(w\in\operatorname{range}(B)\).
2. If \(\operatorname{rank}(B)=C\), so \(B\) has full row rank and necessarily
   \(C\leq A\), every \(w\in\mathbb R^C\) is reproduced exactly. Equation (9)
   is the unique exact solution of minimum Euclidean coefficient norm.
3. If \(C=A\) and \(B\) is nonsingular, then
   \(\theta=B^{-1}w\).
4. If \(B\) is rank deficient or \(C>A\), (10) is the orthogonal projection
   of \(w\) onto \(\operatorname{range}(B)\). The residual is
   \((I-BB^+)w\), so arbitrary indexed weights cannot all be reproduced.

These statements apply independently to all \(9\times32\) FBC spatial vectors.
The implementation also copies all \(9\times32\) indexed spatial biases. When
**channel_positions** is supplied and \(B(P)\) has full row rank, the raw
cardinal weights reproduce the freshly seeded indexed FBC weights on the
observed montage. Both implementations then apply the same rowwise radius-2
max-norm constraint, so their constrained spatial maps also agree. All
remaining FBC modules are transferred from that seeded instance. The resulting
initialization preserves the indexed FBC reference function up to pseudoinverse
and dtype roundoff. Regression tests cover the canonical 21-channel montage
and the C3/Cz/C4 montage.

Normalization is an important qualification. Equations (7)--(10) concern the
raw linear field. Max normalization preserves a reference inside the radius-2
ball and reproduces an indexed FBC field subjected to the same projection.
Unit normalization discards field-vector amplitude and retains only direction,
except in its clamped near-zero region; unrestricted linear-field reproduction
does not survive that normalization.

For **CardinalFBCMicroDynamicsNet**, the micro features need not be zero.
Instead, **ExpandedMaxNormLinear.extend** copies the FBC classifier columns and
bias and initializes all 80 new classifier columns to zero. The micro model's
initial logits therefore equal its FBC reference path up to numerical
precision. This statement requires paired initialization of that reference
path. If **channel_positions** is omitted, the cardinal reference-path
coefficients remain randomly initialized.

## 5. Continuity and a coordinate-perturbation bound

Let \(d\) be great-circle distance between normalized coordinates. The
clamped distance in the code is equivalent to clamping \(d\) to a closed
subinterval of \([0,\pi]\), a 1-Lipschitz scalar operation. The spherical
triangle inequality gives

\[
 |d_\epsilon(p,q_a)-d_\epsilon(p',q_a)|\leq d(p,p').
\]

For \(\phi(r)=\exp[-r^2/(2\ell^2)]\),

\[
 |\phi'(r)|=\frac{r}{\ell^2}e^{-r^2/(2\ell^2)}
 \leq \frac{e^{-1/2}}{\ell}
 \quad (\ell\leq\pi),
\]

with equality at \(r=\ell\). The default \(\ell=0.35\) meets the condition.
Each anchor kernel therefore obeys

\[
 |\kappa_\ell(p,q_a)-\kappa_\ell(p',q_a)|
 \leq L_\kappa d(p,p'),\qquad
 L_\kappa=\frac{e^{-1/2}}{\ell}.
 \tag{11}
\]

For paired coordinate sets \(P=(p_c)_{c=1}^C\) and
\(P'=(p'_c)_{c=1}^C\), define

\[
 D(P,P')=\left(\sum_{c=1}^{C}d(p_c,p'_c)^2\right)^{1/2}.
\]

Equations (1) and (11) imply

\[
 \left\lVert B(P)-B(P')\right\rVert_F
 \leq
 \sqrt A\,L_\kappa\,
 \left\lVert M_\lambda^{-1}\right\rVert_2 D(P,P').
 \tag{12}
\]

Flatten \(\Theta\) as
\(\Theta_\flat\in\mathbb R^{FS\times A}\). The raw-field bound is

\[
 \left\lVert H(P)-H(P')\right\rVert_F
 \leq
 \lVert\Theta_\flat\rVert_2
 \sqrt A\,L_\kappa
 \lVert M_\lambda^{-1}\rVert_2 D(P,P').
 \tag{13}
\]

The max-norm map in **CardinalFBCNet** is a Euclidean projection for every
\((f,s)\) row and is nonexpansive, so it does not increase the right side of
(13). For fixed filtered data \(Z_f\in\mathbb R^{C\times T}\), the source
perturbation in band \(f\) also satisfies

\[
 \left\lVert Y_f(P)-Y_f(P')\right\rVert_F
 \leq
 \left\lVert \mathcal N(H_f(P))-
                    \mathcal N(H_f(P'))\right\rVert_F
 \lVert Z_f\rVert_2.
 \tag{14}
\]

Unit normalization in the default **CardinalScalpField** and in
**CardinalSpatiotemporalField** needs an additional qualification. The coded
map is

\[
 u_\tau(h)=\frac{h}{\max(\lVert h\rVert_2,\tau)},
 \qquad\tau=10^{-7}.
\]

It is \(1/\tau\)-Lipschitz globally because it is a scaled Euclidean projection
onto the unit ball. This finite bound is deliberately loose. On a region where
both field norms are at least \(r>0\), the usual normalized-vector bound

\[
 \left\lVert \frac{h}{\lVert h\rVert_2}
       -\frac{h'}{\lVert h'\rVert_2}\right\rVert_2
 \leq \frac{2}{r}\lVert h-h'\rVert_2
\]

is more informative. The implemented spatial maps are continuous in valid
electrode coordinates, but a ridge or small coefficient-space norm alone does
not imply a tight practical robustness constant. For reference, the raw-basis
prefactor
\(\sqrt A L_\kappa\lVert M_\lambda^{-1}\rVert_2\) is approximately 2,100 for
the 21-anchor tensors and 8,529 for the 31-anchor tensors. These worst-case
bounds are conservative and are not empirical robustness results.

Bounds (11)--(14) hold with fixed sampled EEG values and perturbations only to
their associated coordinates. If moving an electrode also changes its signal,
that signal-perturbation term must be bounded separately.

## 6. Explicit failure of density invariance

Permutation invariance is not invariance to electrode density, duplication,
deletion, or discretization. Equation (3) is an unweighted channel sum; the
implementation has no surface quadrature or local-density correction.

A one-source counterexample suffices. Take one electrode at \(p\), a signal
\(z\neq0\), and a raw field value \(h(p)=a\) chosen so that
\(10^{-7}\leq|a|<\sqrt2\). With no normalization, or with the FBC max-norm
projection inactive, the output is \(az\). Duplicate the same coordinate and
signal. The two field values are both \(a\), and the output is \(2az\). With
unit normalization, the one-electrode output is
\(\operatorname{sign}(a)z\), whereas duplicated weights are each
\(\operatorname{sign}(a)/\sqrt2\), producing
\(\sqrt2\operatorname{sign}(a)z\). The output changes in both cases.

Identical coordinate rows also make \(B(P)\) row deficient. The model assigns
the same raw field value to those rows and cannot reproduce two different
indexed reference weights there. Channel masks permit missing-channel inputs,
but masking changes the sum and its normalization; it does not make the model
invariant to channel removal.

The implementation is therefore a coordinate-conditioned, variable-channel
spatial decoder, not a discretization-invariant neural operator. Any observed
cross-density robustness is empirical, not an architectural theorem.

## 7. Exact parameter accounting

Atlas coordinates and **cardinal_inverse** are buffers, not trainable
parameters. A standalone **CardinalScalpField** with \(F\) filters, \(S\)
sources, and \(A\) anchors has exactly \(FSA\) trainable coefficients.

For CardinalFBC, let \(F=9\), \(S=32\), \(V=4\) log-variance views, and
\(K_{\mathrm{cls}}\) output classes. The trainable count is

\[
 \underbrace{FSA}_{\text{cardinal coefficients}}
 +\underbrace{FS}_{\text{spatial bias}}
 +\underbrace{2FS}_{\text{batch-normalization affine}}
 +\underbrace{K_{\mathrm{cls}}(FSV+1)}_{\text{classifier weight and bias}}.
 \tag{15}
\]

The pinned filter bank additionally stores 1,982 frozen FIR numerator and
denominator coefficients. For four-class decoding, the exact current counts
are:

| model | anchors | cardinal coefficients | trainable parameters | all stored parameters, including frozen FIR |
|---|---:|---:|---:|---:|
| **cardinal_fbc** | 21 | 6,048 | 11,524 | 13,506 |
| **cardinal_fbc_extended** | 31 | 8,928 | 14,404 | 16,386 |

On a 21-channel input, the \(9\times32\times21\) cardinal tensor has exactly
the size of FBCNet's indexed grouped spatial-convolution weight tensor. With
the same spatial bias and transferred modules, **cardinal_fbc** has exactly the
same parameter count as indexed FBCNet on that montage. This equality is
specific to \(A=C=21\); it does not hold for a 3- or 15-channel indexed
baseline or for the 31-anchor variant.

The default micro branch adds

\[
\begin{aligned}
 &8\times25
 &&\text{learnable FIR coefficients}\\
 +{}&8\times8\times A
 &&\text{joint spatiotemporal-field coefficients}\\
 +{}&2\times8
 &&\text{micro batch-normalization affine terms}\\
 +{}&8\times15
 &&\text{depthwise dynamics convolution}\\
 +{}&8\times8
 &&\text{pointwise dynamics convolution}\\
 +{}&80K_{\mathrm{cls}}
 &&\text{new classifier columns}.
\end{aligned}
\]

The 80 added features are \(8\times8=64\) windowed log energies plus the mean
and standard deviation of eight dynamics channels, \(2\times8=16\). The micro
branch adds \(400+64A+80K_{\mathrm{cls}}\) trainable parameters to (15). For
four classes:

| model | anchors | added micro parameters | trainable parameters | all stored parameters, including frozen FIR |
|---|---:|---:|---:|---:|
| **cardinal_fbc_micro** | 21 | 2,064 | 13,588 | 15,570 |
| **cardinal_fbc_micro_extended** | 31 | 2,704 | 17,108 | 19,090 |

These counts use the implemented defaults. An input length of 320 adds no
learned weights. Different class counts or micro dimensions must be recomputed
from the formulas.

## 8. Terminology and claim constraints

The following formulations are supported:

- **regularized cardinal-RBF scalp field** or **near-cardinal geodesic-RBF
  field**;
- **coordinate-conditioned, montage-flexible spatial filtering**;
- **simultaneous channel-coordinate permutation invariant**;
- **variable-channel input with a fixed inducing atlas**;
- **full rank on the verified 21- and 31-anchor atlas matrices**, or **able to
  reproduce arbitrary observed indexed weights when \(B(P)\) has full row
  rank**;
- **function-preserving paired FBC initialization up to numerical precision**,
  when coordinates are supplied and the rank condition holds;
- **the same spatial coefficient budget and total parameter count as indexed
  FBCNet on the canonical 21-channel montage**;
- **a function-preserving FBC reference-path initialization, up to numerical
  precision, with a zero-initialized single-head micro-feature extension** for
  **CardinalFBCMicroDynamicsNet**.

The following stronger formulations are not supported without qualification:

- **an identity-valued cardinal or Lagrange basis**, or **interpolation of the
  coefficients as anchor values**: \(\lambda>0\) makes (4) nonidentity;
- **montage agnostic**: the model requires correctly paired 3-D coordinates in
  the atlas frame and compatible preprocessing;
- **rotation invariant**: rotating observed coordinates without the fixed
  atlas generally changes the field;
- **density invariant**, **sampling invariant**, **discretization invariant**,
  or **neural operator**: Section 6 gives a counterexample;
- **arbitrary continuous-field representation** or **universal approximation**:
  the raw field lies in a fixed finite \(A\)-dimensional kernel span;
- **universally full rank**: too many channels, repeated coordinates, or other
  degeneracy can violate the rank condition;
- **identical parameter count to FBCNet for every montage**: exact equality is
  the \(A=C=21\) case;
- **guaranteed cross-montage accuracy or robustness**: continuity and
  permutation invariance do not establish predictive generalization.

The architectural mathematics alone also does not establish literature
novelty or state-of-the-art performance. Those are separate claims requiring a
precise prior-art comparison and preregistered multi-dataset evidence.
