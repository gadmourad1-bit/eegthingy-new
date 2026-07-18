# Handoff prompt — scaffold a novel geometric EEG deep-net (copy this into ChatGPT)

You are a senior ML research engineer. Scaffold a **new, runnable PyTorch project** for a
**novel deep-learning architecture** for motor-imagery (MI) EEG classification, to be trained on
an RTX 5070 and evaluated both offline and in a closed-loop robot-navigation simulator. Produce
clean, importable, runnable code (not stubs) plus a README. Ask me before assuming anything you
cannot infer from below.

---

## 1. Research goal & the exact novelty (do not drift from this)
Build a **compact SPD / Riemannian tangent-space network** that **self-calibrates online and
unsupervised (label-free) at TWO levels**:
1. **Covariance-manifold recentering** of features to a session-invariant reference (like
   Euclidean Alignment / TSMNet's SPD domain batch-norm), and
2. **Decision-boundary recentering** — an adaptive, label-free tracking of the classifier's
   neutral operating point (defined in §5), applied online at inference.

It must be **small enough for real-time/embedded use** and is intended to be validated in a
**closed-loop robot-navigation task** (a maze simulator), not just offline accuracy.

**Prior art (position against, do NOT reinvent):** SPDNet (Huang & Van Gool, AAAI'17),
**TSMNet + SPDDSMBN** (Kobler et al., NeurIPS'22, arXiv:2206.01323 — closest work: online
unsupervised covariance alignment, but OFFLINE only and recenters covariance, not the decision
boundary), Tensor-CSPNet (TNNLS'22), Graph-CSPNet (TNNLS'23), MAtt (NeurIPS'22). The defensible
gap is: **online label-free DECISION-BOUNDARY self-calibration + embedded real-time + closed-loop
robot evaluation** — none of the above do this. Keep the architecture in that lane.

**Honest expectation:** at this data scale the net will likely *match, not beat*, classical
Riemannian methods offline — the contribution is the online-self-calibrating + closed-loop
capability, so build/evaluate for that, not for chasing offline SOTA.

## 2. Data (fixed — build to these exact shapes)
- Task: **binary** left-hand vs right-hand motor imagery. Labels `y ∈ {1, 2}` (1=left, 2=right).
- Source: MNE `.fif` files. An existing function `process_data(file_path, target_map)` returns
  `X` with shape **`(N_epochs, 4 bands, 15 channels, 251 samples)`** and `y` shape `(N,)`.
  - `N≈60` epochs per file (balanced), **~240 epochs per subject** (4 files/subject).
  - Sampling rate **125 Hz**; epoch window 0.5–2.5 s → **251 time samples**.
  - **4 filter-bank bands**: `[(8,12), (11,15), (14,20), (20,30)]` Hz (already applied — the band
    axis exists in X).
  - **15 EEG channels** (10–20 sensorimotor montage; one dead channel already dropped).
- **8 usable subjects**: IDs `1,3,4,5,6,7,8,10`. **Exclude subject 9.** For **subject 10 use only
  training files 5–8** (files 1–4 were a bad recording). Total ≈ 1,900 epochs.
- For the geometric net, convert each band's epoch to a **spatial covariance matrix**
  `C = (1/T) X_b X_bᵀ` → per epoch you get **4 SPD matrices of size 15×15**. Regularize toward
  SPD (e.g. shrinkage: `C ← (1-α)C + α·tr(C)/n·I`, α≈1e-3) so eigendecomposition is stable.

## 3. Hardware & environment (CRITICAL — Blackwell GPU)
- GPU: **RTX 5070, compute capability sm_120 (Blackwell)**, 12 GB. Ubuntu 24.04, driver 595+.
- **PyTorch MUST be the cu128 build** or you get `no kernel image is available`. Install torch AND
  torchaudio from `https://download.pytorch.org/whl/cu128` (versions `2.11.0+cu128`). A default
  PyPI torchaudio pulls a cu13 build → `libcudart.so.13: cannot open shared object file`; fix by
  reinstalling torchaudio from the cu128 index.
- Env manager is `uv`. Already installed: `braindecode 1.6.1`, `mne 1.12.1`, `pyriemann 0.12`,
  `scikit-learn 1.9.0`, `torch/torchaudio 2.11.0+cu128`.
- 12 GB VRAM is ample (these nets are tiny, <100 k params); batch size is not a constraint.

## 4. Architecture to implement (`model.py`)
Implement SPD-manifold layers in PyTorch (use `torch.linalg.eigh`, add small eigenvalue jitter for
stable backward):
- **BiMap**: `Y = W C Wᵀ`, `W ∈ R^{d_out × d_in}` (semi-orthogonal init; optionally use `geoopt`
  Stiefel manifold for a strict version — make it a flag).
- **ReEig**: rectify eigenvalues `λ ← max(λ, ε)`, reconstruct `V diag(λ) Vᵀ`.
- **LogEig**: `V diag(log λ) Vᵀ` → map to tangent space; then vectorize the upper triangle.
- **SPD momentum BatchNorm (the manifold-recentering level)**: recenter each batch's covariances
  by a running reference mean on the manifold. A robust, differentiable choice is a **log-Euclidean
  mean** recentering (mean in the matrix-log domain); keep a **running reference** updated by
  momentum so that at test time it adapts UNSUPERVISED to the target session (this is level-1
  self-calibration). Document the choice; note Riemannian (AIRM) mean as an upgrade.

**Net `GeoAdaptNet`**: input `(N, 4, 15, 15)` covariances → per band `[SPD-BN → BiMap(15→d) →
ReEig → LogEig → vec-upper]` → concat over the 4 bands → small Linear → **2 logits**. Keep it
<~50 k params. The signed logit difference is the log-odds `s` fed to §5.

## 5. Decision-boundary recentering head (`recenter.py`) — the novel level-2
Port this exact unsupervised, label-free online rule (from the project's classical pipeline) and
make it a first-class, optionally-learnable head that runs at inference:
- Log-odds toward class 2: `s`.
- Seed neutral from a calibration block: `c₀ = median_j s_j`.
- Recentred margin `z_t = s_t − c_t`; prediction = class 2 if `z_t>0` else class 1.
- Confidence `conf_t = σ(|z_t|) = 1/(1+e^{−|z_t|})`.
- Update `c` **only from rest-like windows** (`|z_t| < τ`, e.g. τ = logit(0.65) ≈ 0.62) via slow
  EMA `c_{t+1} = c_t + α(s_t − c_t)`, `α≈0.01`, **clamped** to `[c₀−Δ, c₀+Δ]`, `Δ=2.0`. This
  prevents sustained real MI from dragging the boundary and prevents the collapse that
  entropy-minimization TTA suffers on tiny streams.
- A window commits only if `conf_t ≥ 0.85` (this gate doubles as a rest dead-zone).
Provide an ablation switch to turn this head on/off, and a variant where `α` / gate are learned.

## 6. Baselines to compare against (`baselines.py`)
Wrap behind a common `fit/predict_proba` API:
- **Classical (the bar):** EA + Filter-Bank CSP(2) + shrinkage-LDA, and Riemannian tangent-space +
  logistic regression. (Reference implementations exist in the project's `classifier/run.py` as
  `EAFilterBankCSP` and `FilterBankTangentSpace` — import/adapt them.)
- **Deep baselines (braindecode):** EEGNet, ShallowFBCSPNet, Deep4Net, EEGConformer, ATCNet.
- **Key geometric baseline:** TSMNet (port the authors' released code) — the closest prior art.

## 7. Evaluation protocols & metrics (`protocols.py`, `evaluate.py`)
Implement three **leakage-free** protocols (no subject/session bleed across train/test):
- **Within-subject** k-fold CV (per subject, then averaged).
- **Cross-session** (train earlier sessions → test later sessions of the same subject).
- **Leave-one-subject-out (LOSO)** — true cross-subject transfer.
Metrics: accuracy, Cohen's κ, **param count**, **batch=1 inference latency (ms)** (the real-time
cost), and — for the online head — a simulated-online pass over concatenated windows. Save a
results table + bar charts. Fix seeds; report mean±std.

## 8. Project structure to scaffold
```
deepnet/
  __init__.py
  config.py        # hyperparams, subject list (exclude 9; S10 files 5-8), bands, paths
  data.py          # load .fif -> epochs -> covariances; protocol splitters; augmentation hooks
  spd.py           # BiMap, ReEig, LogEig, SPD momentum BatchNorm (log-Euclidean recentering)
  model.py         # GeoAdaptNet (the SPD net)
  recenter.py      # online unsupervised BoundaryRecenter head (§5)
  baselines.py     # classical + braindecode + TSMNet wrappers, common API
  engine.py        # training loop (Adam, early stop, standardization, augmentation)
  protocols.py     # within-subject / cross-session / LOSO splitters
  evaluate.py      # run protocols, compute metrics, save tables/figures
  run.py           # CLI entry: `python -m deepnet.run --protocol loso --model geonet`
  README.md        # how to install (cu128!), run, extend
```

## 9. Constraints & success criteria
- Small data (~1,900 epochs) → strong regularization (small width, dropout, weight decay, SPD-BN),
  and **data augmentation** (time shift, channel dropout, frequency masking, SPD/covariance mixup).
- **Cross-subject pretrain → per-subject unsupervised calibration** (no target labels), mirroring
  the online deployment; the SPD-BN running stats + the boundary head do the calibration.
- Keep the net tiny and report batch=1 latency (embedded target).
- Deliverable: code that runs `python -m deepnet.run --protocol within --model geonet` end-to-end
  on one subject and prints accuracy, and a baseline run for EEGNet + classical, on GPU.

## 10. Output format
Give me the files one by one with brief rationale per file, correct imports, and a final "how to
run" section. Flag any place you made an assumption. Do not fabricate results — only scaffold and
provide the commands to generate them.
