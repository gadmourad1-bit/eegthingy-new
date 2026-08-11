# UV reproducibility contract for the IEEE MI benchmark

## Status and scope

This document defines the dependency and environment contract for the clean
project that will be promoted to `/home/hanafy/neuralnetwork`. It also retains
the historical audit record from the broader source repository. The clean
release's current `pyproject.toml` and `uv.lock` are purpose-specific benchmark
manifests; they are not the broader repository manifests described in the
historical snapshot below.

The formal benchmark must use an isolated UV virtual environment. It must not
install into the system Python, alter the NVIDIA driver or CUDA toolkit, reuse
another user's environment, or synchronize the current repository lock into
the live scratch environment.

The most important historical-source warning was:

> The broader source repository's root `uv.lock` is not the benchmark lock.
> Do not run `uv sync`, `uv sync --frozen`, or an automatic `uv run` sync against the
> live scratch environment
> `/home/hanafy/scratchpad/eeg_novel_20260729/.venv-uv`.

UV performs an exact sync by default and removes undeclared packages. A dry-run
audit showed that the broader root project would remove the research packages
that were added to the scratch environment. The scratch environment is
evidence-bearing development infrastructure and must remain untouched. The
clean project gets a new `.venv` built from its purpose-specific release lock.

## Audit snapshot: 2026-07-29

### Repository root

The broader source repository root captured in this snapshot describes the Mac
acquisition, GUI, and robotics application, not the paper benchmark.

| File | Bytes | SHA-256 | Finding |
|---|---:|---|---|
| `pyproject.toml` | 436 | `fb717dd478f5ff3fe4bbf9c76ac39b454b9c7976b546a4eed2805fb00a684737` | Pins Python `==3.12.*` and 12 application/scientific dependencies |
| `uv.lock` | 63,834 | `af28e7ab7e08608f3fea666d22fd1ce237111bd8f152ed9a8901f3700fe8c286` | Contains 38 package records, but no Torch, Braindecode, MOABB, pandas, or PDF/test toolchain |
| `requirements.txt` | 193 | `c769eaabd56531d9ff0406b0a76f7386c77c44d7e4893a067689abd8b3864d3c` | Legacy application requirements, not a benchmark lock |
| `ieee_mi/requirements-cu128.txt` | 257 | `06171d32aaef187b2ba9047322a8e4a93e5f16d04b743cb75095beda9f858a16` | Legacy RTX 5070/cu128 environment; incompatible with the lab contract |

The root lock does contain MNE 1.12.1, NumPy 2.4.4, pyRiemann 0.11,
scikit-learn 1.8.0, SciPy 1.18.0, matplotlib 3.10.9, and threadpoolctl 3.6.0.
That partial overlap does not make it a valid benchmark lock.

The root-only dependencies `brainflow`, `panda3d`, `pyglet`, `pyserial`,
`trimesh`, and `websockets` belong to acquisition, simulation, or GUI code.
They should not be direct dependencies of the clean benchmark project.

### Validated lab scratch environment

The opened development screens were run successfully with this audited core
identity:

| Component | Observed value |
|---|---|
| UV | 0.12.0 |
| Python | CPython 3.12.13, UV-managed |
| Torch | 2.6.0+cu124 |
| TorchAudio | 2.6.0+cu124 |
| Torch CUDA build | 12.4 |
| cuDNN reported by Torch | 90100 |
| Braindecode | 1.6.1 |
| MNE | 1.12.1 |
| MOABB | 1.5.0 |
| NumPy | 2.4.4 |
| pandas | 3.0.3 |
| pyRiemann | 0.11 |
| scikit-learn | 1.8.0 |
| SciPy | 1.18.0 |
| `uv pip check` | Passed |

The lab host reported NVIDIA driver 550.127.05 and four RTX A5000 GPUs. The
driver is external to UV and must be recorded in every run plan; it must never
be changed by this project. The matching cu124 Torch and TorchAudio wheels use
the locked user-space CUDA runtime. No global CUDA or cuDNN installation is
required.

The scratch environment was assembled incrementally and is not represented by
its colocated legacy `pyproject.toml`/`uv.lock`. Its successful package
inventory is useful evidence for selecting versions, but it is not a
reproducible installation recipe. The final lock must be generated and tested
from a clean environment.

An explicit isolation probe on 2026-07-30 used the absolute UV executable with
`--active --offline --no-project`, the project-private `VIRTUAL_ENV` and
`UV_CACHE_DIR`, and `PYTHONNOUSERSITE=1`. It reported:

- active prefix
  `/home/hanafy/scratchpad/eeg_novel_20260729/.venv-uv`;
- UV-managed base interpreter CPython 3.12.13;
- `include-system-site-packages = false`;
- `site.ENABLE_USER_SITE = false`; and
- no `/usr` or `/usr/local` Python package directory in `sys.path`.

This proves isolation for that audited invocation. Every formal command must
still carry the same explicit environment envelope; the observation is not a
license to omit the clean final-project rebuild.

### Import and runner audit

The in-repository benchmark code in `ieee_mi` and `deepnet` directly imports
NumPy, SciPy, scikit-learn, Torch, MNE, pyRiemann, and Braindecode. Braindecode
declares TorchAudio as a runtime dependency, so the clean project pins the
matching binary build directly even though benchmark code does not call its
audio APIs. MOABB is imported lazily by the dataset loaders. The pinned
official TCFormer source imports `einops`. The CPU control runner imports
`threadpoolctl` at execution time and explicitly requires it to be a direct
project dependency.

Pandas and matplotlib are not imported directly by the core trainers, but are
runtime dependencies of MOABB and part of the validated environment. They are
kept as deliberate direct pins so that dataset metadata behavior and plotting
behavior cannot drift through a transitive upgrade.

The runner families have different provenance behavior:

| Runner family | Environment use | Provenance behavior |
|---|---|---|
| `development_screen`, `conditioned_screen`, `robustness_screen` | GPU, strict deterministic Torch | Freeze source hashes, selected scientific versions, CUDA/cuDNN, driver, Python, and GPU UUID bindings |
| `full_grid` and `full_grid_analysis` | GPU producer plus CPU analysis | Freeze all installed distributions, Python/UV/driver/Torch/TorchAudio identity, source closure, extra factory, executor, `pyproject.toml`, and `uv.lock` |
| `control_grid` | CPU-only with hard thread caps | Requires a UV venv and requires MNE, NumPy, pyRiemann, scikit-learn, SciPy, threadpoolctl, and Torch in both direct dependencies and the lock; hashes `pyproject.toml` and `uv.lock` |
| `local_model_tournament` and `benchmark` | GPU common-recipe development | Record installed distributions and source hashes, but still hash the legacy `requirements-cu128.txt` |
| native pretraining/transfer and FBMS transfer runners | GPU | Record package/CUDA identity, but several legacy source manifests still include `requirements-cu128.txt` |
| legacy `deepnet/*benchmark.py` procedures | CPU and GPU depending on model | Use the same scientific libraries, but predate the final UV lock contract |

Before promotion, references to `requirements-cu128.txt` in legacy runners must
either be replaced by the final `pyproject.toml`/`uv.lock` identity or be
explicitly retained as historical-only provenance. Installing that file on the
lab host is forbidden: it pins Torch 2.11.0+cu128, torchaudio, NumPy 2.5.1,
scikit-learn 1.9.0, pyRiemann 0.12, and a different matplotlib/pytest stack.
Those artifacts are not comparable environment identities for the new formal
grid.

## Recommended final project dependencies

The clean project is a Linux x86-64 benchmark application, not a distributable
library. Running modules from the project root is sufficient, so
`tool.uv.package = false` avoids an unnecessary editable build and build
backend. If a wheel or installed CLI is added later, that is a separate
reviewed change that requires a new lock and new preflight.

The proposed `pyproject.toml` dependency section is:

```toml
[project]
name = "ieee-mi-benchmark"
version = "0.1.0"
description = "Reproducible cross-dataset motor-imagery EEG benchmark"
requires-python = "==3.12.13"
dependencies = [
    "braindecode==1.6.1",
    "einops==0.8.2",
    "matplotlib==3.10.9",
    "mne==1.12.1",
    "moabb==1.5.0",
    "numpy==2.4.4",
    "pandas==3.0.3",
    "pyriemann==0.11",
    "scikit-learn==1.8.0",
    "scipy==1.18.0",
    "threadpoolctl==3.6.0",
    "torch==2.6.0+cu124",
    "torchaudio==2.6.0+cu124",
]

[dependency-groups]
test = [
    "pytest==9.0.2",
]
docs = [
    "pdfplumber==0.11.10",
    "pypdf==6.14.2",
    "reportlab==5.0.0",
]

[tool.uv]
package = false
default-groups = []
environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64' and implementation_name == 'cpython'",
]
required-environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
]

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchaudio = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

The exact Torch and TorchAudio direct requirements include the wheels' PEP 440
local build tag, `2.6.0+cu124`, and the explicit cu124 index must resolve and
install that same version string for both packages. This is stricter than
PyTorch's generic installation command because the formal validators compare
the declared, locked, and installed versions exactly. It follows the official
[PyTorch 2.6 CUDA 12.4 installation matrix](https://docs.pytorch.org/get-started/previous-versions/)
and UV's
[explicit PyTorch index pattern](https://docs.astral.sh/uv/guides/integration/pytorch/).
The `explicit = true` setting prevents unrelated dependencies from being
resolved from the PyTorch index.

TorchAudio is retained because Braindecode declares it as a runtime dependency.
Leaving that transitive requirement unconstrained allowed UV to select a newer
PyPI wheel built for a different Torch/CUDA generation. Making
`torchaudio==2.6.0+cu124` direct and assigning its own `tool.uv.sources` entry
keeps it aligned with `torch==2.6.0+cu124`. `torchvision` remains intentionally
absent because neither the benchmark nor this Braindecode use requires it.
`rotary-embedding-torch`, skorch, joblib, pooch, tqdm, h5py, MNE-BIDS, and the
NVIDIA user-space wheels may remain transitive dependencies, but their exact
versions and distribution hashes must be present in `uv.lock`.

The `docs` group is not part of model training. It exists for PDF generation
and render inspection. Poppler's `pdftoppm` is an external executable; use it
only if the lab already provides it. Do not run `apt`, `sudo`, Homebrew, or
another system package manager on the shared workstation. If Poppler is absent,
render and inspect PDFs on the Mac or another controlled documentation host.

Never add the docs group to an environment after a benchmark plan has frozen
its installed-distribution identity. Use a second project environment:

```bash
UV_PROJECT_ENVIRONMENT=/home/hanafy/neuralnetwork/.venv-docs \
uv sync --frozen --no-default-groups --group docs
```

Invoke report generators with the same `UV_PROJECT_ENVIRONMENT` value. This
keeps PDF tooling out of the formal training environment while resolving both
environments from the same reviewed lock.

The `test` group is also excluded from the formal training environment. Create
a separate test environment:

```bash
UV_PROJECT_ENVIRONMENT=/home/hanafy/neuralnetwork/.venv-test \
uv sync --frozen --no-default-groups --group test
```

`default-groups = []` is intentional. UV otherwise treats a dependency group
named `dev` specially and installs it by default during `uv sync` and
`uv run`. Every formal command still supplies `--no-default-groups`
explicitly so that command-line provenance does not depend on an implicit
project default.

The docs versions above are recommendations for the new generator, not evidence
that a PDF has already been validated with them. Lock them, generate the PDF,
render every page, and visually inspect the renders before freezing the final
documentation environment.

An isolated no-install lock probe on the lab host with UV 0.12.0 and CPython
3.12.13 resolved 109 package names. Its `pyproject.toml` SHA-256 was
`1fdb188881e3554f69c0009337a1848592042e10cd6104866dd91a71e589baa0`
and its generated `uv.lock` SHA-256 was
`fceb6d13f67d4ade2a8aab9da0b0dd4db65527c59eb65bd41e347e4233d2f576`.
Those exact hashes became part of the sealed Common Grid source identity, but
the lock selected TorchAudio 2.11.0 from PyPI while the validated installed
environment recorded TorchAudio 2.6.0+cu124. The resolution-only probe did not
prove that the lock recreated the installed binary environment.

A bounded post-assembly, pre-release repair added the exact TorchAudio pin and
explicit cu124 source without changing any sealed result artifact. The repaired
`pyproject.toml` SHA-256 is
`17e59040701e2ad0bdf7f0cc049ac551e9b6bf56c22847e715851f2697ca50fd`,
and the repaired `uv.lock` SHA-256 is
`739d5eae65a06a65ba69598c00c29fe8e2bcc49eed493869088b17af47a218f8`.
The lock still contains the same 109 package names; only the root project
metadata and TorchAudio package table changed. It now selects
`torchaudio-2.6.0+cu124-cp312-cp312-linux_x86_64.whl` from the cu124 index with
SHA-256
`3e5ffa69606171c74f3e2b969785ead50b782ca657e746aaee1ee7cc88dcfc08`.
Any run using the repaired manifests requires a new immutable plan and run
root. The original plan and preflight retain their original source hashes and
their separate installed-environment record.

## Compatibility findings

The validated scientific pins are mutually compatible in the lab environment:
`uv pip check` passed and the CPU/CUDA model tests ran. In particular:

- MOABB 1.5.0 requires NumPy at least 2.0, MNE at least 1.10,
  scikit-learn at least 1.6, pandas at least 1.5.2, and SciPy at least 1.9.3;
  the proposed pins satisfy those lower bounds.
- pyRiemann 0.11 requires NumPy at least 1.25 and scikit-learn at least 0.24;
  the proposed pins satisfy those bounds.
- Braindecode 1.6.1 requires Python at least 3.11 and works in the observed
  Python 3.12.13/Torch 2.6.0/TorchAudio 2.6.0 environment.
- TCFormer at pinned commit
  `74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5` uses Torch and einops. The
  clean release uses a compact vendored snapshot whose `SOURCE.json`, license,
  runtime file sizes, and runtime file SHA-256 values are verified by
  `ieee_mi.tcformer_source`; Git metadata is not a runtime requirement.

The main conflicts are provenance conflicts rather than resolver conflicts:

1. The root lock omits the benchmark stack and must not be promoted.
2. `requirements-cu128.txt` describes a different GPU, Torch, CUDA, and
   scientific stack and must not be installed on the lab host.
3. Historical result artifacts produced with Torch 2.11.0+cu128, cuDNN 91900,
   or NumPy 2.5.1 must retain their original provenance. Their timing and
   deterministic-byte identities cannot be relabeled as cu124 results.
4. The sealed Common Grid manifest hashes must remain historical facts. The
   repaired manifest pair is prospective and must never be substituted into
   the sealed plan or preflight report.
5. A driver update, Python patch update, UV minor update, package addition, or
   lock change after a plan is created is environment drift. Finish or abandon
   the old plan; never append mixed-environment records to it.

## Clean-room creation workflow

All commands in this section are intended for
`/home/hanafy/neuralnetwork`, after the final source tree and the proposed
project metadata have been placed there. They must not be run in the scratch
source tree.

### 1. Establish paths without touching system state

```bash
cd /home/hanafy/neuralnetwork

export UV_CACHE_DIR=/home/hanafy/scratchpad/eeg_novel_20260729/uv-cache
export UV_PROJECT_ENVIRONMENT=/home/hanafy/neuralnetwork/.venv
export PYTHONNOUSERSITE=1

uv --version
uv python find 3.12.13
uv python pin 3.12.13
```

The expected UV output is `uv 0.12.0`. The pin creates a project-local
`.python-version` containing `3.12.13`; it is not a global pin. The managed
Python is already present, so there is no reason to modify the system Python.

Never set `UV_PROJECT_ENVIRONMENT` to the scratch `.venv-uv`, `/usr`,
`/usr/local`, a shared environment, or any path outside the final project.
Never use `--system`, `sudo`, `uv pip install --system`, or bare system `pip`.

### 2. Generate the benchmark lock

Lock generation is a one-time, reviewed setup action and must occur before any
formal plan is created:

```bash
uv lock --python 3.12.13
uv lock --check
```

Review the resulting lock before installation:

```bash
uv tree --frozen
rg 'name = "(torch|torchaudio|braindecode|moabb|mne|numpy|pandas|pyriemann|scikit-learn|scipy|threadpoolctl)"' uv.lock
sha256sum pyproject.toml .python-version uv.lock
```

The lock must identify the `pytorch-cu124` registry for both Torch and
TorchAudio and must include their matching CPython 3.12 Linux x86-64 cu124
wheels. `uv lock --check` is important because `--frozen` uses the lock as the
source of truth without checking whether project metadata is newer. The
required pair for formal use is therefore:

```bash
uv lock --check
uv sync --frozen --no-default-groups
```

Do not use `uv lock --upgrade`, `uv add`, a formal `uv sync` without both
`--frozen` and `--no-default-groups`, or a formal `uv run` without both flags
after the lock is approved or after a run plan exists.

### 3. Create a fresh project venv and sync exactly

```bash
uv venv --python 3.12.13 /home/hanafy/neuralnetwork/.venv
uv sync --frozen --no-default-groups
uv pip check --python /home/hanafy/neuralnetwork/.venv/bin/python
```

Activation is optional for UV, but if an interactive shell is used:

```bash
source /home/hanafy/neuralnetwork/.venv/bin/activate
```

Verify that the interpreter is isolated:

```bash
uv run --frozen --no-default-groups python -c \
  'import pathlib,sys; print(sys.version); print(pathlib.Path(sys.prefix).resolve()); print(pathlib.Path(sys.base_prefix).resolve())'
```

`sys.prefix` must resolve to `/home/hanafy/neuralnetwork/.venv`, it must differ
from `sys.base_prefix`, and `.venv/pyvenv.cfg` must contain a UV marker and
`include-system-site-packages = false`. The formal runners enforce these
conditions.

### 4. Verify the accelerator build

Use one GPU that is idle according to the lab's cooperative allocation policy:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-<approved-idle-uuid> \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --frozen --no-default-groups python -c \
  'import torch,torchaudio; print(torch.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))'
```

Required values are Torch and TorchAudio `2.6.0+cu124`, CUDA build `12.4`,
cuDNN `90100`, and exactly one visible approved GPU. A different driver
version, build tag, cuDNN value, or device mapping is a failed preflight, not a
reason to install or upgrade global software.

## Cache and offline operation

Keep the UV cache on scratch storage to avoid filling the nearly full root
filesystem:

```bash
export UV_CACHE_DIR=/home/hanafy/scratchpad/eeg_novel_20260729/uv-cache
```

UV's cache is content-addressed and safe for concurrent readers/writers, but
the project `.venv` is not a shared package sandbox. Only one setup process may
synchronize it, and no package operation may run while benchmark workers are
active.

The first clean sync needs network access unless every locked wheel and index
response is already cached. After the online sync, prove that the cache is
sufficient by creating a separate disposable environment, not by deleting or
rewriting the validated one:

```bash
UV_PROJECT_ENVIRONMENT=/home/hanafy/scratchpad/eeg_novel_20260729/offline-venv-check \
UV_CACHE_DIR=/home/hanafy/scratchpad/eeg_novel_20260729/uv-cache \
UV_OFFLINE=1 \
uv sync --frozen --no-default-groups
```

The offline-check path must be newly and uniquely allocated for this
verification and must not contain a prior environment. Then run `uv pip check`
against that interpreter. Offline mode must fail if an artifact is missing; do
not weaken the lock, switch indexes, or fall back to an unhashed local install.
Preserve the approved cache until every formal run and report recreation is
complete. Do not run `uv cache clean` on a shared cache during the project.

For an archival handoff, export human- and tool-readable dependency views
without treating either as a replacement for `uv.lock`:

```bash
uv export --frozen --no-default-groups --format requirements.txt --output-file provenance/requirements.locked.txt
uv export --frozen --no-default-groups --format cyclonedx1.5 --output-file provenance/environment.cdx.json
```

The official UV documentation defines `--offline` as cache/local-files only
and describes lock checking, frozen syncs, and exact sync behavior in
[Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) and
[the CLI reference](https://docs.astral.sh/uv/reference/cli/).

## Runtime environment

Set deterministic and cooperative resource variables before Python starts.
The GPU supervisors also set these values for their child workers:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export BLIS_NUM_THREADS=4
export TBB_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR=/home/hanafy/scratchpad/eeg_novel_20260729/matplotlib-cache
export PYTHONNOUSERSITE=1
```

Use full physical GPU UUIDs, not unstable ordinal assumptions. `full_grid`
sets `CUDA_VISIBLE_DEVICES`, the CUDA device order, CUBLAS determinism, and
four-thread caps for each owned child. `control_grid` clears
`CUDA_VISIBLE_DEVICES`, uses `threadpoolctl`, and caps all CPU libraries.
Do not start DDP, use all host CPU threads, claim a busy GPU, or bypass the
50 GiB free-space low-water mark.

Do not change environment variables, packages, source files, the TCFormer
checkout, or the cache manifest after an immutable plan is created. Resume
after a power cut with the same venv, variables, source bytes, plan digest, and
GPU contract.

## Verification commands

Run dependency checks before tests:

```bash
cd /home/hanafy/neuralnetwork
uv lock --check
uv pip check --python .venv/bin/python
uv run --frozen --no-default-groups python -c \
  'import braindecode,mne,moabb,numpy,pandas,pyriemann,scipy,sklearn,torch,torchaudio; assert torch.__version__ == torchaudio.__version__ == "2.6.0+cu124"; print("imports-ok")'
```

Run the complete repository tests without writing configuration into another
user's home:

```bash
MNE_DONTWRITE_HOME=true \
MPLCONFIGDIR=/home/hanafy/scratchpad/eeg_novel_20260729/matplotlib-test-cache \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
UV_PROJECT_ENVIRONMENT=/home/hanafy/neuralnetwork/.venv-test \
uv run --frozen --no-default-groups --group test \
  python -m pytest -q --import-mode=importlib ieee_mi/tests deepnet/tests
```

At minimum, the benchmark infrastructure acceptance set is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
UV_PROJECT_ENVIRONMENT=/home/hanafy/neuralnetwork/.venv-test \
uv run --frozen --no-default-groups --group test python -m pytest -q --import-mode=importlib \
  ieee_mi/tests/test_development_screen.py \
  ieee_mi/tests/test_conditioned_screen.py \
  ieee_mi/tests/test_conditioned_analysis.py \
  ieee_mi/tests/test_robustness_screen.py \
  ieee_mi/tests/test_robustness_analysis.py \
  ieee_mi/tests/test_full_grid.py \
  ieee_mi/tests/test_full_grid_analysis.py \
  ieee_mi/tests/test_control_grid.py \
  ieee_mi/tests/test_chsd.py \
  ieee_mi/tests/test_chsd_conditioned.py
```

Run the CHSD CUDA tests with exactly one idle GPU visible:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-<approved-idle-uuid> \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
UV_PROJECT_ENVIRONMENT=/home/hanafy/neuralnetwork/.venv-test \
uv run --frozen --no-default-groups --group test python -m pytest -q --import-mode=importlib \
  ieee_mi/tests/test_chsd.py \
  ieee_mi/tests/test_chsd_conditioned.py
```

The formal producer should only be invoked with frozen project execution:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --frozen --no-default-groups \
  python -m ieee_mi.full_grid status --run-root <run-root>

uv run --frozen --no-default-groups python -m ieee_mi.control_grid audit \
  --run-root <control-run-root> \
  --output <control-run-root>/audit.json
```

Use the same `uv run --frozen --no-default-groups python -m ...` prefix for
plans, workers, analysis, native pretraining/transfer, and legacy procedure
runners. Never allow automatic lock or environment mutation as an incidental
effect of launching a benchmark.

## Byte- and version-level provenance manifest

`uv.lock` is the canonical dependency resolution, but it is only one part of a
paper-grade environment record. Before creating the first formal plan, write
an immutable `provenance/environment.json` and checksum sidecar. The manifest
must be generated by a checked-in script and must include:

1. schema version and UTC creation timestamp;
2. absolute project root and result root;
3. byte size and SHA-256 of `pyproject.toml`, `.python-version`, `uv.lock`, and
   the manifest-generating script;
4. `uv --version`, resolved UV executable path, UV executable byte size and
   SHA-256;
5. full Python version/build string, implementation, resolved executable,
   executable byte size/SHA-256, `sys.prefix`, `sys.base_prefix`, and
   `pyvenv.cfg` SHA-256;
6. every installed distribution as normalized name/version pairs, the sorted
   canonical list SHA-256, and a frozen text export SHA-256;
7. Torch and TorchAudio versions, Torch CUDA build, cuDNN integer, CUDA
   availability, visible device count, GPU name, full physical UUID, and
   compute capability;
8. `nvidia-smi` driver version, host/kernel/architecture, CPU model, and total
   RAM;
9. all determinism and thread environment variables;
10. source file path, byte size, and SHA-256 for the exact explicit dependency
    closures used by every selected runner;
11. Git commit and dirty-state digest for the final project;
12. TCFormer repository URL, exact pinned commit, vendored-source manifest
    hash, MIT license hash, and byte hashes for every dynamically imported
    file;
13. harmonized-cache manifest SHA-256 and every cache/split identity consumed;
14. run-plan path, byte size, SHA-256, model registry digest, model factory
    reference, training configuration, seeds, and exact command arguments.

The existing screen, full-grid, control-grid, and transfer plans already
capture much of this information and fail on drift. The bootstrap provenance
manifest complements those plan-specific records; it does not replace them.
Every published result must retain both its immutable plan and the environment
manifest that existed before the plan was made.

Useful read-only capture commands are:

```bash
uv --version
uv pip freeze --python .venv/bin/python
uv tree --frozen
sha256sum pyproject.toml .python-version uv.lock
git rev-parse HEAD
git status --porcelain=v1
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv,noheader
```

Do not hash the `.venv` directory as a substitute for package provenance and
do not copy `.venv` between machines. Recreate it from the reviewed lock,
verify it, then record the resolved interpreter, distributions, binary build
identity, and run plan.

## Shared-workstation non-negotiables

- No `sudo`, system package manager, global pip, `uv --system`, driver update,
  CUDA toolkit update, shell-profile edit, or daemon.
- No reuse, deletion, repair, or exact-sync of the scratch `.venv-uv`.
- No synchronization of a final-project lock into any environment owned by
  another user or project.
- No dependency change after planning. Use a new lock digest and new run root
  for an approved change.
- No automatic upgrade flags. A newer package is a new experiment, not a
  maintenance action.
- No removal of the shared UV cache while runs or offline recreation remain
  active.
- No claiming that old cu128 artifacts were produced by the cu124 environment.
- No formal run until a new clean `.venv` passes lock, dependency, CPU, CUDA,
  source, cache, and runner preflights.
