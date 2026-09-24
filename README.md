# ASPECT

**Agent-Specific Parameter-Efficient Core Tuning for Multi-Agent LLM Workflows**

Research implementation of ASPECT (AW-LoRA-SB) and six parameter-sharing
baselines. The repository contains the training and inference runtime, portable
launchers, task-aligned calibration, dataset reconstruction, checkpoint evaluation,
and regression tests. It does not require a proprietary scheduler, storage service,
or experiment platform.

**Validation:** all seven method entry points passed two-update GPU functional
checks, and ASPECT checkpoint reload passed real inference. The CPU suite passed
258 tests with one opt-in stress test skipped. See
[release validation](verification/RELEASE_VALIDATION.md) for the exact scope and
limitations; these checks do not constitute a new 200-step paper reproduction.

## Methods

| Entry point | Trainable parameters | Sharing unit |
| --- | --- | --- |
| `algorithms.WS` | LoRA A/B | Entire workflow |
| `algorithms.RS` | LoRA A/B | Role |
| `algorithms.AS` | LoRA A/B | Individual agent |
| `algorithms.ASPECT` | Private R; calibrated A/B are frozen | Individual agent |
| `algorithms.WS_Full_FT` | Original dense model | Entire workflow |
| `algorithms.RS_Full_FT` | Original dense model and optimizer state | Role |
| `algorithms.AS_Full_FT` | Original dense model and optimizer state | Individual agent |

Each method has a separate directory and configuration. The methods deliberately
share the GRPO, workflow, reward, diagnostics, and rollout implementation rather
than maintaining seven drifting copies. See [method details](docs/METHODS.md).

## Supported Experiments

- LoRA methods: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B; Math and Code;
  Eval-Opt, Voting, and Orch-Workers.
- Full-FT reference contract: Qwen3-1.7B Math, Eval-Opt and Orch-Workers.
- Formal cells: 200 updates, batch 64, eight rollouts, seed 42, warmup 15.
- LoRA methods retain atomic adapter/core checkpoints every 10 updates. Complete
  official-split validation is performed by the separate evaluator.
- Full-FT saves **no model, optimizer, or adapter checkpoints**. It validates
  the live updated policy on all 1,412 Math test problems every 10 updates.
- ASPECT uses rank/alpha 64/64, learning rate `1e-4` at 0.6B and `2e-4`
  at 1.7B/4B. WS/RS/AS use rank/alpha 64/32 and learning rate `2e-5`.
  Full-FT defaults to `1e-6`; `--learning-rate 2e-5` selects the separate ablation.

These are research reference configurations, not memory-adaptive defaults. A
smaller machine may need an explicitly documented non-canonical smoke setup.
The package does not promise bitwise-identical trajectories across GPU types,
asynchronous schedules, dependency changes, or older experimental frameworks.

## Environment

The reference runtime is **Linux x86-64**, Python **3.10.16**, NVIDIA GPUs with
BF16 support, a CUDA **12.6** toolkit/compiler, and a compatible NVIDIA driver.
Formal training uses four GPUs per cell and four independent TP1 rollout replicas
with one four-GPU FSDP actor. A GPU with approximately 140 GiB memory was used for
the main large-context runs. Do not assume a 24 GiB GPU can run these defaults.
Full-FT role/agent banks also require substantial host RAM; allow multiple dense
model and Adam-state copies in addition to rollout memory.

| Package | Reference version |
| --- | --- |
| PyTorch / torchvision / torchaudio | 2.7.1 / 0.22.1 / 2.7.1 |
| vLLM | 0.10.0 |
| Transformers | 4.57.0 |
| PEFT | 0.19.1 |
| Ray | 2.55.1 |
| FlashAttention | 2.8.3.post1 |
| FlashInfer / Ninja | 0.3.1 / 1.13.0 |
| Datasets / PyArrow | 5.0.0 / 24.0.0 |
| NumPy | 1.26.4 |

Install in a **fresh environment**, not into an unrelated running experiment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
python -m aspect.environment
```

`requirements/runtime.txt` pins the directly used research dependencies.
OpenCV 4.11 and CuPy 13.6 are compatibility pins for NumPy 1.26; the development
environment's newer optional image/CUDA packages had incompatible NumPy metadata
and are intentionally not reproduced. Their clean-install GPU behavior still
requires release acceptance testing.
The exact Transformers 4.57.0 and Polars 1.43.0 releases were marked yanked on
the public package index when inspected. Keep the reference pins for parity;
see the installation limitations in `verification/RELEASE_VALIDATION.md`.
FlashAttention needs a working CUDA compiler and build tools.
CUDA JIT extensions also need a C++ runtime compatible with the host compiler.
For example, GCC 13-built FlashInfer kernels can require `GLIBCXX_3.4.32`; an older
environment-provided `libstdc++` may shadow the system library. Resolve such a
native-library mismatch inside the experiment environment before training, rather
than disabling dependency or numerical checks.
Install the bundled
package as a regular wheel with `--no-deps` after the runtime dependencies; **do not install a separate
stock `verl` or `rllm` over the bundled modified modules**. `aspect.environment`
checks module provenance and the principal runtime versions.
The default is intentionally not an editable install: an existing framework
package can take precedence over an editable import hook. Run the provenance
check from outside the checkout as well when using a pre-populated environment.

For CPU configuration tests only, use Python 3.10-3.12 and
`python -m pip install '.[test]'`. Runtime numerical tests additionally need
PyTorch and the reference runtime dependencies. Training is not supported on
macOS or CPU-only hosts.

## Public Inputs

Model weights and datasets are downloaded separately and are not redistributed.
Use the respective upstream licenses and access conditions.

```bash
hf download Qwen/Qwen3-1.7B \
  --revision 0060bc56d46589041c1048efd1a397421b1142b5 \
  --local-dir models/Qwen3-1.7B

python -m aspect.data.download --task math --output datasets/math
python -m aspect.data.download --task code --output datasets/code
```

Math uses the pinned English DAPO source, split with `test_size=0.1, seed=42`:
12,704 train / 1,412 test. AIME is not substituted for this test set. Code uses
the pinned DeepCoder source plus a public runtime-filter bridge; every retained
row and test is checked against its source before the deterministic 13,995 train /
1,000 test split. Both loaders fail on SHA or split mismatches. See
[data provenance](docs/DATA.md) and the constants in `aspect/data/`.

Pass local model directories through `--model`; any filesystem is supported.
For other model scales download their official Qwen3 repositories, freeze the
reference revision in `docs/DATA.md`, and retain that revision with the run. The launcher does not
silently select or download an unpinned replacement model.

## Task-Aligned Subspace Calibration

Calibration runs once per compatible model/task/hyperparameter tuple. The public
builder uses the same 50 training examples, supervised targets, 512-token budget,
first-update approximation, and layer-wise randomized SVD as the research runtime.
It is **not** a full RL training step or a full-model fine-tuning run.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  --module aspect.calibrate \
  --task math --model-path models/Qwen3-1.7B --train-root datasets/math \
  --output-dir assets/math-1.7b-r64 --work-dir outputs/calibration-work \
  --rank 64 --lora-alpha 64 --actor-lr 2e-4 --warmup-steps 15 \
  --calibration-samples 50 --max-seq-length 512 --n-iter 10 --random-state 42 \
  --model-revision 0060bc56d46589041c1048efd1a397421b1142b5
```

For Code, use `--task code --train-root datasets/code/prepared`. For 0.6B use
`--actor-lr 1e-4`. Calibration supports a single GPU or distributed CUDA execution;
the research calibration used eight GPUs. The work directory must be fresh and
dedicated: temporary per-rank shards are removed after successful publication.
Do not supply a directory containing unrelated files. The artifact stores SHA,
configuration, selected-example hashes, spectrum diagnostics, and timing. Reuse
is allowed only for a matching immutable artifact, never a trained R checkpoint.

## Train One Cell

Select only GPUs allocated to you. The launcher refuses occupied GPUs and existing
output directories; it never kills unrelated processes. Logs and datasets live
under an isolated run directory.
Prefer a node-local SSD for the run and calibration directories. Archive complete,
SHA-checked artifacts to shared storage after completion; do not assume every
network filesystem has reliable POSIX rename and cache semantics.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m algorithms.ASPECT \
  --task math --workflow eval_opt --scale 1.7b \
  --model models/Qwen3-1.7B --data datasets/math \
  --basis assets/math-1.7b-r64 --output outputs/aspect-math-eval

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m algorithms.RS \
  --task code --workflow orch_workers --scale 1.7b \
  --model models/Qwen3-1.7B --data datasets/code/prepared \
  --output outputs/rs-code-orch
```

Replace the method module with `algorithms.WS`, `algorithms.AS`, or one of the
three Full-FT modules. Omit `--basis` outside ASPECT. Use `--dry-run` to inspect
the portable override contract without ML dependencies, and `--config-only` to
compose the actual Hydra configuration without GPU execution. `--cpus` controls
the Ray CPU budget and the bounded Code judge concurrency.

Reduced checks require `--smoke --steps 2 --gpus 1` and are explicitly marked
non-canonical. This does not make reference batch/context settings fit arbitrary
hardware. It does not stand in for a 200-step reproduction.
For a short functional check, additionally select `--quick-smoke`: eight training
problems per update, four rollouts per problem, and eight held-out problems for
online Full-FT validation. This flag is rejected without `--smoke`. Context
lengths, learning rate, warmup, role sharing and optimization code are unchanged.
The held-out subset is selected only in the run-local registry, with the original
and subset hashes recorded; formal runs always retain the complete split.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m algorithms.RS_Full_FT \
  --task math --workflow orch_workers --scale 1.7b \
  --model models/Qwen3-1.7B --data datasets/math \
  --output outputs/rs-full-quick --smoke --quick-smoke --steps 2 \
  --gpus 2 --gpu-memory 0.60
```

LoRA smoke runs retain and audit a checkpoint after each update; Full-FT smoke
runs remain checkpoint-free and use non-canonical online validation.

To check a saved LoRA/core smoke checkpoint through the same inference loader:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m aspect.evaluate \
  --run outputs/aspect-math-smoke --output outputs/reload-smoke/summary.json \
  --smoke --gpus 2 --steps 2 --max-samples 4 --gpu-memory 0.60
```

This writes trajectory-only diagnostic files and an explicitly non-canonical
summary, never a formal validation record. Use the actual smoke run directory.
Runs marked incomplete or invalid are rejected for canonical evaluation.

## Complete Validation

After a LoRA training cell finishes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m aspect.evaluate \
  --run outputs/aspect-math-eval \
  --output outputs/aspect-math-eval-validation/results.jsonl
```

Only completed checkpoints at steps 10,20,...,200 are evaluated. Every result
retains per-problem outcomes. Missing points remain missing; there is no
interpolation, forward filling, or substitution of reward for success.
Use a fresh output file for each disjoint evaluation request; `--steps` permits
explicit disjoint checkpoint subsets. Full-FT results are already written under
each run's `validation/` directory and cannot be reconstructed after exit because
no weights are saved.

## Two-Lane Full-FT

```bash
python -m aspect.full_ft_pair --method RS_Full_FT \
  --model models/Qwen3-1.7B --data datasets/math \
  --output outputs/rs-full-pair --gpu-groups 0,1,2,3 4,5,6,7 \
  --learning-rate 1e-6 --cpus-per-lane 80
```

Both workflows must finish 200 updates and 20 full validations. The early lane
holds its final live weights and performs real interruptible stochastic
repeatability checks under `AUDIT_ONLY_NOT_CANONICAL`; it neither continues
training nor contributes duplicate canonical results. Completion of the other
lane ends these auxiliary checks. Failure stops only this launcher's process
trees, retains partial outputs, and never resumes or silently publishes them.
See [reproduction and verification](docs/REPRODUCIBILITY.md).

An optional GitHub Actions definition is provided in `docs/ci_contracts.yml`.
It runs the lightweight CPU contracts and syntax checks. It is a template, not
an automatically enabled workflow; GPU acceptance uses the reference environment.

## Repository Layout

```text
algorithms/             Separate WS, RS, AS, ASPECT, and Full-FT entry points
aspect/                 Portable configuration, launch, preflight, audit, calibration
aspect/data/            Pinned public dataset download and deterministic preprocessing
rllm/                   Workflows, reward stack, rollout engine, GRPO workflow trainer
verl/                   Bundled modified FSDP, LoRA/core, full-parameter, vLLM backend
examples/math_reasoning Math workflow definitions and original trainer entry points
examples/deepcoder/     Code workflows, judge integration, diagnostics
dashboard/              Headless checkpoint evaluator and task registry (no dashboard server)
scripts/                Installation and filesystem/publication helpers
tests/                  Portable regression and method-contract tests
docs/                   Algorithm, data, reproducibility, and safety documentation
verification/           Release checks and source-equivalence manifest
licenses/               Required third-party license text
```

## Tests and Verification

```bash
CUDA_VISIBLE_DEVICES= python -m pytest tests -q
python -m compileall -q aspect algorithms rllm verl examples dashboard
```

The source-equivalence manifest records copied runtime file hashes and packaging
changes. See `verification/` for the actual release-validation scope. A passing
unit suite is not a claim of bug-free software or a fresh complete experimental
campaign. Please retain resolved commands, model/dataset revisions, environment
versions, hardware, raw metrics, and split hashes for each reproduction.

## Safety and Attribution

Generated Code is untrusted. Run Code experiments only inside a disposable,
network-restricted container or isolated compute account with no credentials or
sensitive mounts. The process-based judge is **not a security sandbox**.

This project builds on [marl-llm-workflows](https://github.com/XHMY/marl-llm-workflows),
[rLLM](https://github.com/rllm-org/rllm), [verl](https://github.com/volcengine/verl),
and [LoRA-SB](https://github.com/CERT-Lab/lora-sb). The bundled runtime retains
required upstream copyright and license notices. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No private experiment logs,
credentials, datasets, model weights, or original repository history are included.
