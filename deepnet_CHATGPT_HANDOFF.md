# Handoff — GeoAdaptNet EEG decoder (paste this into ChatGPT/Codex)

You are taking over a working, heavily-benchmarked EEG motor-imagery decoding project as a
senior ML research engineer **and scientific reviewer**. This is not a scaffold request: the
package is built, tested, and evaluated on two datasets. Your job is to continue from a
well-characterized state without repeating dead ends that have already been measured.

Read this document fully before changing anything, then read `deepnet/RESULTS.md`.

---

## 1. Operating rules (do not violate these)

1. **Never claim accuracy superiority.** The measured result is *competitive, not superior*.
   Several strong baselines beat this model on one dataset or the other.
2. **Never tune on outer test labels.** The deployment contract is: predict with state at `t`,
   then optionally update state for `t+1`. Selection happens on a held-out validation
   recording/participant only.
3. **Beware transductive numbers.** The classical `RiemannianTangentLogistic.calibrate()`
   recomputes alignment from the *unlabeled test set*. The honest, non-transductive classical
   bar is **89.94%** local, not the 90.06% that appears in some tables.
4. **Healthy volunteers only.** 8 healthy participants. No clinical claim, ever.
5. **Report failures.** A large fraction of the value here is the ledger of levers that did not
   work (§6). Do not silently drop them.
6. **Do not re-run and overwrite locked result files.** Use a new output stem per experiment.

---

## 2. Machine and repository state

- **Mac repo:** `/Users/admin/Documents/GitHub/eegthingy`, branch `main`.
- **GPU box:** `ssh gpu` → `user-desktop-ryzen9`, Ubuntu 24.04, Ryzen 9 9900X, **RTX 5070 12 GB
  (Blackwell sm_120)**. Repo at `~/Desktop/eegthingy`, venv `.venv`.
  **PyTorch must be cu128** (`torch/torchaudio == 2.11.0+cu128`); a default-PyPI torchaudio
  pulls a cu13 build and breaks with `libcudart.so.13`.
- **MOABB 1.5.0** is installed on the GPU box (for the external dataset); it did not disturb the
  cu128 torch stack.
- **Test suite: 71/71 passing** (`python -m pytest -q deepnet/tests`).

### Committed vs not

`HEAD = 1e89902 "geoadaptnet"` contains the bulk of the work (13 files, +1519 lines):
`augment.py`, `csp_init.py`, `dnn_baselines.py`, `dnn_benchmark.py`, `external_benchmark.py`,
`external_cho2017.py`, `config.py`, `data.py`, `engine.py`, `experiment.py`, `loso.py`,
`model.py`, `RESULTS.md`.

**Uncommitted at handoff time:**

| Status | File | What it is |
|---|---|---|
| modified | `deepnet/RESULTS.md` | full-52-subject external numbers + gap analysis |
| modified | `deepnet/dnn_benchmark.py` | adds the `geoadapt_anchor` arch |
| modified | `deepnet/external_benchmark.py` | adds `riemann`, `geoadapt_anchor`, `geoadapt_fb(sp)`, band presets, per-subject skip handling |
| modified | `deepnet/external_cho2017.py` | filter-bank presets (`default`/`rich9`/`rich7`) |
| new | `deepnet/tangent_anchor.py` | convex-head anchor (the local head-fix) |
| new | `deepnet/filterbank_net.py` | GeoAdaptNet-FB / FBSP (learnable Sinc filterbank) |
| new | `deepnet/fusion_benchmark.py` | decoder fusion / ensembling |
| new | `deepnet/GeoAdaptNet_Technical_Report.pdf` | 9-page technical report |

Result JSONs live in `deepnet/results/` (gitignored). Figure/report generation scripts are
currently only in a session scratchpad — **if the PDF must be regenerable, move them into the
repo.**

---

## 3. What the system is

Binary left/right-hand motor imagery. **Data contract (fixed, do not rediscover by glob):**
8 participants (IDs 1, 3, 4, 5, 6, 7, 8, 10 — subject 9 and S10 runs 1–4 are excluded) × 4
recordings = 32 sessions; 15 channels at 125 Hz; window **0.0–2.0 s** (251 samples); filter
bank 8–12 / 11–15 / 14–20 / 20–30 Hz; model input is **4 × (15×15) SPD covariances**;
~1,900 labelled windows total (~120 per training fold).

**GeoAdaptNet (9,678 params)** = a log-Euclidean **tangent-space anchor** (recenter → matrix log
→ 480-d vech → Linear) **plus a gated SPD residual** (per-band BiMap 15→8 → ReEig → LogEig →
attention), combined as `logits = anchor + sigmoid(gate) · residual`, gate initialized ≈ 0.02.
A separate scalar intent/rest head enables abstention.

**Deployment:** weights frozen; an unlabeled 20-window prefix seeds two bounded, constant-memory
causal states — a covariance recenter and an independent scalar decision-boundary recenter —
plus an intent gate and a 0.85 commit threshold. Batch-1 latency 4.54 ms on the RTX 5070.

---

## 4. Measured results

### 4.1 Deployment protocol (schema-v2, includes the online adapter)

Chronological: fit rec 1–2, select on rec 3, test rec 4 after an unlabeled prefix; seeds 7/17/27
aggregated within participant. LOSO: outer participant fully withheld, one inner participant for
selection, six fit the model; seed 7.

| Model | Chronological | LOSO | Coverage | Rest false-commit |
|---|---:|---:|---:|---:|
| EA-FBCSP (classical) | 87.53 | 85.67 | 69.4 | 50.9 |
| **GeoAdaptNet + swap aug + blend** | **86.62** | **85.02** | 42.3 | 19.8 |
| GeoAdaptNet (baseline) | 84.86 | 83.54 | 41.0 | 15.7 |
| Riemann TS+LR (classical) | 84.24 | 81.61 | 67.2 | 48.0 |

The distinctive property is the **operating point**: ~3× fewer commits on rest windows, bought
with lower coverage. This is a complete-decoder comparison — the baselines have the same
confidence threshold but no intent head.

### 4.2 Architecture comparison (uniform protocol, no online adapter, raw balanced accuracy)

**Local cohort** — 8 subjects, 3 seeds:

| Architecture | Params | no aug | + swap aug |
|---|---:|---:|---:|
| Riemann TS+LR (classical) | — | **90.06** | — |
| EA-FBCSP (classical) | — | 90.01 | — |
| Convex-head anchor (`tangent_anchor.py`) | 481 | 88.46 | — |
| GeoAdaptNet | 9,645 | 85.63 | **87.58** |
| ShallowConvNet | 26,002 | 85.87 | 87.04 |
| EEG-Conformer | 266,306 | 85.73 | 86.40 |
| EEGNet | 1,602 | 70.51 | 83.96 |
| ATCNet | 28,868 | 55.07 | 75.25 |
| DeepConvNet | 233,652 | 50.15 | 54.10 |

**External cohort — Cho2017 / GigaDB 100295, ALL 52 subjects, none skipped**, within-subject
5-fold CV, mapped onto the same 15-ch/125 Hz layout (10-20 ↔ 10-10: T5→P7, T6→P8, T3→T7, T4→T8):

| Architecture | no aug | + swap aug |
|---|---:|---:|
| **ShallowConvNet** | **62.91** | **63.16** |
| Riemann (classical, non-transductive) | 59.68 | — |
| Convex-head anchor (481 params) | 59.59 | — |
| EEGNet | 56.88 | 61.58 |
| EEG-Conformer | 56.94 | 58.91 |
| GeoAdaptNet-FB (learnable filterbank) | 57.67 | 58.75 |
| GeoAdaptNet | 57.69 | 58.44 |
| DeepConvNet / ATCNet | 50.9 / 50.3 | 51.2 / 51.3 |

GeoAdaptNet has the **lowest cross-subject variance of any model** (±8.5 vs ShallowConvNet ±11.5).

### 4.3 Fusion (tops both tables — with a caveat)

| Cohort | Best fusion | Best individual | Δ |
|---|---:|---:|---:|
| Local (8 subj) | 91.61 (tangent+fbcsp+riemann+shallow) | 89.94 | +1.67 |
| External (52 subj) | 64.77 (riemann+shallow+eegnet) | 62.91 | +1.86 |

**⚠ The winning combination was selected by looking at test scores** (~11 combos per dataset),
so these are optimistically biased. No fixed pre-registered combination wins both cohorts:
the all-members ensemble wins locally (91.61) but *loses* externally (62.25 < 62.91), and
riemann+shallow wins externally (63.64) but loses locally. **The correct next step is to select
the ensemble on validation data and report test once** (see §7).

---

## 5. The hard-won findings (read this before proposing anything)

1. **The gated SPD residual is redundant, not undertrained.** It contributes +0.17 pt and the gate
   never leaves its init. It was revived three ways — auxiliary deep supervision, raising/freeing
   the gate (it *does* open, to 0.21), and CSP-warm-starting each BiMap with supervised filters —
   and accuracy did not move. The rank-8 BiMap is a compressed view of the same covariance the
   full-rank anchor already reads. **Do not retry this.**

2. **The local gap to classical Riemann is the estimator, not the geometry.** Measured decomposition
   (identical features, isolation scripts):

   | Configuration | bal-acc | isolates |
   |---|---:|---|
   | Riemann **with** transductive `calibrate()` | 90.06 | the number in old tables |
   | Riemann **without** calibrate (fair) | 89.94 | transduction = +0.12 |
   | GeoAdaptNet geometry + convex StandardScaler+LogReg | 89.73 | AIRM vs log-Euclidean = +0.21 |
   | Same features + SGD BatchNorm+Linear head | 87.42 | **head estimator = −2.31** |
   | Full GeoAdaptNet | 85.63 | residual scaffolding = −1.8 |

   Replacing the SGD head with a convex L2 logistic head on frozen-reference tangent features
   (`tangent_anchor.py`, **481 parameters**) gives 88.46 local / 59.59 external. Note this fix
   converges *back toward classical tangent-space decoding*.

3. **Fixed filter banks lose at scale — decisive control.** Non-transductive classical Riemann with
   the same 4 fixed bands **also** loses to ShallowConvNet externally (59.68 vs 62.91). So no
   fixed-band geometric method reaches ShallowConvNet on that cohort, and the ~3.3 pt gap is not a
   GeoAdaptNet defect. A learnable Sinc filterbank recovered only +0.8 pt; adding learnable
   *spatial* filtering (BiMap, Tensor-CSPNet-style) **hurt** (58.12 < 58.98).

4. **Augmentation is the real accuracy lever.** Left/right electrode swap + label flip (a symmetric
   covariance permutation, stays on the SPD manifold) exploits motor imagery's lateral organization.
   It is architecture-agnostic and helps *every* net (EEGNet +13 pt), so it is a property of the
   problem, not the model.

5. **Everything strong sits near a data ceiling.** Cho2017 contains many genuinely near-chance
   participants. Chasing a single architecture that "blasts away" the field is not physically
   available; only fusion of complementary decoders beats the best individual.

---

## 6. Full lever ledger

| Lever | Result | Verdict |
|---|---|---|
| Left/right swap augmentation | +1.76 chronological / +1.48 LOSO | **adopted** |
| Convex head + frozen reference + no residual | +2.96 local, +1.9 external | **adopted** |
| Decoder fusion / ensembling | +1.67 local, +1.86 external | promising, needs validation-selection |
| Deep supervision / opening the residual gate | ≈ 0 | rejected |
| CSP warm-start of BiMap | ≈ 0 | rejected |
| Log-Euclidean covariance mixup | ≈ 0 | rejected |
| Seed ensembling (3 seeds) | −0.34 | rejected |
| Test-time mirror averaging | −0.79 | rejected (redundant with swap aug) |
| Adaptive OAS shrinkage | −1.05 (real-EEG OAS weight ≈ 0.01) | rejected |
| Longer optimizer budget | −1.38 | rejected |
| Learnable Sinc filterbank | +0.80 external | insufficient |
| + learnable spatial BiMap | −0.86 external | rejected |
| 9 filter bands instead of 4 | −2.7 external | rejected (p ≫ n) |

---

## 7. Recommended next work, in order

1. **Validation-selected fusion.** Select the ensemble composition on rec 3 (local) / an inner CV
   fold (external), freeze it, then report test **once**. This converts the cherry-picked 91.6/64.8
   into a number that is publishable. ~20 min of compute. *This is the highest-value open item.*
2. **Multi-seed LOSO.** Currently seed 7 only with one deterministic inner participant. A full
   inner-LOSO sweep with several seeds is required before any subject-independent claim.
3. **Move the report generators into the repo** (`make_figs.py`, `make_report.py` are in a scratchpad)
   so `GeoAdaptNet_Technical_Report.pdf` is regenerable.
4. **Commit the four uncommitted files** listed in §2 (the user has been committing selectively).
5. Matched modern comparators (TSMNet, OTTA-MI, T-TIME) if a methods claim is ever attempted.
6. Continuous asynchronous replay and a validated raw-board-to-covariance bridge before any
   closed-loop or clinical step.

---

## 8. Publication guidance (important — the user is targeting IEEE)

**This is not a novel classifier, and it should not be submitted as one.** Every component maps
to established prior art: tangent-space classification (Barachant 2012/2013), SPDNet layers
(Huang & Van Gool 2017), Euclidean/Riemannian alignment (He & Wu 2020; Zanini 2018), unsupervised
LDA-bias adaptation (Vidaurre 2011), Sinc filters (Ravanelli & Bengio 2018), TSMNet/Tensor-CSPNet
(Kobler 2022; Ju & Guan 2022). Empirically the deep parts are inert, the "fix" converges back to
classical tangent decoding, and it does not beat the classical bar or ShallowConvNet at scale.
An ensemble is standard practice (Lakshminarayanan 2017), not a contribution.

**What is defensible** is the *deployed system*: the decoupling of covariance recentering from
decision-boundary recentering as two bounded, label-free causal states; the separate intent gate
with explicit abstention; the safety/availability accounting (coverage and rest-false-commit) that
the geometric-DL literature does not report; and closed-loop robot evaluation. Target a
**systems/robotics venue (RA-L, IROS) or TNSRE**, framed as a self-calibrating, safety-gated
MI-BCI. Do **not** target TNNLS/TBME with a "novel classifier" claim — it would be identified and
rejected within a page. The user has decided this DNN work is **not** to be folded into the
existing RA-L navigation manuscript.

---

## 9. Reproduction

```bash
# adopted configuration (defaults reproduce the un-augmented baseline exactly)
python -m deepnet.experiment benchmark --subjects all \
  --models geoadapt,riemann,fbcsp --seeds 7,17,27 --window deployment \
  --calibration-windows 20 --max-epochs 180 --patience 25 --device cuda \
  --lr-swap-prob 0.5 --select-metric blend --output deepnet/results/<new_stem>.json

python -m deepnet.loso --subjects all --seeds 7 --calibration-task-events 10 \
  --max-epochs 180 --patience 25 --device cuda --lr-swap-prob 0.5 --select-metric blend

# architecture comparison (local, then external)
python -m deepnet.dnn_benchmark --archs geoadapt,geoadapt_anchor,eegnet,shallow,deep,conformer,atcnet,fbcsp,riemann \
  --subjects all --seeds 7,17,27 --device cuda [--augment]
python -m deepnet.external_benchmark --subjects 1-52 --folds 5 --seeds 7 --device cuda [--augment] \
  --archs geoadapt,geoadapt_anchor,geoadapt_fb,riemann,eegnet,shallow,deep,conformer,atcnet

# fusion
python -m deepnet.fusion_benchmark --subjects all --device cuda
python -m deepnet.fusion_benchmark --external --subjects 1-52 --device cuda
```

---

## 10. Environment gotchas (these cost real time — do not rediscover them)

- **Sinc filters:** `torch.where(cond, 1, sin(x)/x)` back-propagates a **NaN gradient** at the
  center tap `t=0` (0/0 in the untaken branch). Use a safe denominator. The symptom is a
  misleading `linalg.eigh: failed to converge / ill-conditioned` error on **both CUDA and CPU** —
  the matrix contains NaN, it is not a cuSOLVER quirk.
- **Learnable-filterbank covariances** need unit-trace normalization + ~0.05 shrinkage + a tiny
  diagonal ramp + per-channel input standardization to stay numerically stable.
- **GigaDB/Cho2017 downloads** are ~0.1–0.7 MB/s single-threaded (≈5 min/subject) but hit
  5–10 MB/s with 4–5 parallel workers. All 52 subjects download cleanly; none need skipping.
- **Run precache scripts from the repo directory**: `python ~/script.py` puts `~` on `sys.path`,
  not the repo, so `import deepnet` fails. Use `-m` or run from the repo root.
- **`uv` is not on the `nohup` PATH** — use `~/.local/bin/uv` or activate the venv first.
- **tqdm progress output merges into log lines**, so `grep '^OK'` undercounts; use `grep -o`.
- Heredoc + f-strings with escaped quotes break; write helper scripts to a file instead.
- SSH from an agent's shell may need network egress explicitly enabled.
