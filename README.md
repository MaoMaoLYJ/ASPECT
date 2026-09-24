# ASPECT

**Agent-Specific Parameter-Efficient Core Tuning for Multi-Agent LLM Workflows**

A minimal, standalone runtime for ASPECT and six parameter-sharing baselines.
Each command runs one independent experiment and exits after its requested
training and evaluation are complete. No proprietary scheduler, account, data
mount, multi-lane controller, or utilization-based lifecycle is required.

This distribution contains runtime code, configuration, prompts, installation
instructions and required licenses. Development tests, verification reports,
benchmarks and alternate training backends are not distributed.

## Algorithms

| Module | Trainable parameters | Sharing |
| --- | --- | --- |
| `algorithms.WS` | LoRA A/B | One adapter per workflow |
| `algorithms.RS` | LoRA A/B | One adapter per role |
| `algorithms.AS` | LoRA A/B | One adapter per individual agent |
| `algorithms.ASPECT` | R cores; calibrated A/B frozen | Private R per agent |
| `algorithms.WS_Full_FT` | All original model parameters | One model per workflow |
| `algorithms.RS_Full_FT` | All original model parameters | One model per role |
| `algorithms.AS_Full_FT` | All original model parameters | One model per agent |

The methods share the same GRPO, reward, workflow and rollout implementation.
Orch-Workers has three worker instances: RS shares their state as one worker
role, while AS and ASPECT give each worker its own state.

Included LoRA workflows are `eval_opt`, `voting` and `orch_workers` on Math and
Code. Full-FT currently provides Math `eval_opt` and `orch_workers`. The supplied
model presets are Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B. A new model architecture,
task format or workflow needs an appropriate adapter and validation.

## Installation

Use Linux x86-64, Python 3.10.16, a CUDA 12.6 toolkit/compiler, a compatible
NVIDIA driver, and GPUs supporting BF16. The runtime is not tied to a GPU model,
but model/context/batch sizes must fit GPU and host memory. The supplied stack
does not provide CPU, Apple GPU, AMD or Intel GPU training backends.

| Dependency | Reference version |
| --- | --- |
| PyTorch / torchvision / torchaudio | 2.7.1 / 0.22.1 / 2.7.1 |
| vLLM | 0.10.0 |
| Transformers / PEFT | 4.57.0 / 0.19.1 |
| Ray | 2.55.1 |
| FlashAttention / FlashInfer | 2.8.3.post1 / 0.3.1 |
| NumPy | 1.26.4 |

```bash
python3.10 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
python -m aspect.environment
```

Install the bundled package as a regular wheel. Do not install a separate
stock `rllm` or `verl` over these modified modules. `requirements/runtime.txt`
pins the reference dependencies. FlashAttention and FlashInfer need compatible
CUDA/C++ toolchains; an older environment `libstdc++` must not shadow the runtime
required by JIT-compiled kernels. Transformers 4.57.0 and Polars 1.43.0 were
marked yanked on the public index; exact versions are retained for reproducibility.

Full-FT role/agent state banks also require substantial host RAM. Changing GPU
type or parallelism can change asynchronous sampling and does not imply
bitwise-identical results. Not every GPU model or configuration has been tested.

## Model and Data

Weights and datasets are separate inputs, not included in this repository.

```bash
hf download Qwen/Qwen3-1.7B \
  --revision 0060bc56d46589041c1048efd1a397421b1142b5 \
  --local-dir models/Qwen3-1.7B

python -m aspect.data.download --task math --output datasets/math
python -m aspect.data.download --task code --output datasets/code
```

The Math loader reconstructs the pinned DAPO split: 12,704 training and 1,412
held-out problems, split seed 42. The Code loader reconstructs the pinned
DeepCoder split: 13,995 training and 1,000 held-out problems. Source revisions,
content hashes and preprocessing contracts are in `aspect/data/`; mismatches
fail explicitly. No alternative evaluation dataset is silently substituted.

Pass local model, dataset and output locations on the command line. For other
Qwen3 scales, download the corresponding official model and retain its revision.

## ASPECT Calibration

Create the immutable task-aligned basis before ASPECT training:

```bash
CUDA_VISIBLE_DEVICES=0 python -m aspect.calibrate \
  --model-path models/Qwen3-1.7B --train-root datasets/math --task math \
  --output-dir assets/math-1.7b-r64 --work-dir outputs/calibration-work \
  --rank 64 --lora-alpha 64 --actor-lr 2e-4 --warmup-steps 15 \
  --calibration-samples 50 --max-seq-length 512 --n-iter 10 --random-state 42
```

This estimates the first full-parameter update on training examples, computes
the low-rank SVD, and writes A/B, initial R and provenance hashes. Calibration
must match the model, task and training learning rate. Reuse a matching initial
basis, never a trained R checkpoint. For the Code task use `--task code` and
`--train-root datasets/code/prepared`.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m algorithms.ASPECT \
  --task math --workflow eval_opt --scale 1.7b \
  --model models/Qwen3-1.7B --data datasets/math \
  --basis assets/math-1.7b-r64 --output outputs/aspect-math-eval

CUDA_VISIBLE_DEVICES=0,1 python -m algorithms.RS \
  --task code --workflow orch_workers --scale 1.7b \
  --model models/Qwen3-1.7B --data datasets/code/prepared \
  --output outputs/rs-code-orch --gpus 2 --steps 200 \
  --batch-size 16 --rollouts 8 --max-prompt-length 8192 --max-response-length 2048

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m algorithms.RS_Full_FT \
  --task math --workflow orch_workers --scale 1.7b \
  --model models/Qwen3-1.7B --data datasets/math \
  --output outputs/rs-full-math --learning-rate 1e-6
```

Replace the method module to select another algorithm; omit `--basis` outside
ASPECT. The second command illustrates a custom resource/configuration choice,
not the paper's default comparison setting.

Defaults are four GPUs, 200 updates, batch 64, eight rollouts, seed 42 and
warmup 15. They are defaults, not fixed GPU-count or step-count requirements.
Use `--gpus`, `--steps`, `--batch-size`, `--rollouts`, `--cpus`, `--gpu-memory`,
`--max-prompt-length`, `--max-response-length`, `--checkpoint-interval` and
`--validation-interval` to configure an independent run. GRPO requires at least
two rollouts; distributed batch divisibility and hardware memory constraints
still apply. `--dry-run` displays overrides; `--config-only` resolves Hydra.

WS/RS/AS default to rank/alpha 64/32 and learning rate 2e-5. ASPECT uses 64/64
and learning rate 1e-4 at 0.6B or 2e-4 at 1.7B/4B. Full-FT defaults to 1e-6.
Specify `--learning-rate` for a separate ablation. Record any non-default choices
when reporting results; they are not automatically equivalent to paper settings.

LoRA/core checkpoints are published atomically at the requested interval
(default 10). Full-FT saves no model, optimizer or adapter checkpoints; it
validates live weights on the complete Math held-out split at the requested
interval (default 10). Each run keeps file/TensorBoard diagnostics. Existing
output directories are not overwritten. Ctrl-C cleans only the run's own
processes. There is no companion-run waiting or post-training keepalive work.

## Checkpoint Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m aspect.evaluate \
  --run outputs/aspect-math-eval \
  --output outputs/aspect-math-eval-results/results.jsonl --gpus 2
```

Evaluation uses independent TP1 servers with internal DP1 and loads each
adapter on every replica. By default it evaluates the saved checkpoint cadence
declared by the run; `--steps 10 20 30` selects existing points. It processes the
complete official split. Missing points are not interpolated. Full-FT results
are already stored under the run's `validation/`; there are no saved full-model
checkpoints to evaluate after the process exits.

## Layout

- `algorithms/`: seven public method entry points and defaults.
- `aspect/`: launchers, input preparation, calibration, evaluation and artifact checks.
- `examples/`: the six Math/Code workflow adapters and their prompt resources.
- `rllm/`: required workflow, reward and multi-agent training runtime.
- `verl/`: required GRPO, FSDP, adapter/full-parameter state and vLLM runtime.
- `dashboard/`: checkpoint discovery and independent-TP1 evaluation implementation.
- `requirements/`, `scripts/install.sh`: installation.
- `LICENSE`, `licenses/`, `THIRD_PARTY_NOTICES.md`: required licensing and attribution.

The Code judge executes generated programs. Run it only in an isolated Linux
environment without secrets or sensitive mounts; process limits are not a
complete security boundary. See `THIRD_PARTY_NOTICES.md` for upstream attribution.
