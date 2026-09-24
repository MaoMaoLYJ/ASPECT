# Release Validation

Updated: 2026-09-24. **All seven method entry points passed two-update GPU
functional acceptance. ASPECT checkpoint reload and real inference also passed.**

These checks verify execution and implementation integrity. They are not a new
200-step reproduction, a claim of identical accuracy, or a guarantee of no bugs.

## GPU Acceptance

The installed package was exercised on Qwen3-1.7B Math with the Orch-Workers
workflow. Each cell used two A100 80GB GPUs, FSDP2, two independent TP1 rollout
replicas, and an A100 memory fraction of 0.60. The explicit `--smoke --quick-smoke`
profile used eight training problems and four rollouts per problem for two
updates. Seed 42, warmup 15, learning rates, context lengths, sharing semantics,
reward, optimizer and update computation were retained.

| Method | Two Updates | Required Routes Updated | Checkpoint Contract | Online Validation |
| --- | --- | --- | --- | --- |
| WS | Passed | One shared adapter | Atomic adapter checkpoints | Separate evaluator |
| RS | Passed | Three role adapters | Atomic adapter checkpoints | Separate evaluator |
| AS | Passed | Five agent adapters | Atomic adapter checkpoints | Separate evaluator |
| ASPECT | Passed | Five private R cores | Atomic core checkpoints | Separate evaluator |
| WS_Full_FT | Passed | One shared dense model | No checkpoints | Two passes, eight problems each |
| RS_Full_FT | Passed | Three role models | No checkpoints | Two passes, eight problems each |
| AS_Full_FT | Passed | Five agent models | No checkpoints | Two passes, eight problems each |

Independent audits verified continuous steps, finite metrics, nonzero step-two
gradients and actual parameter changes on every required route, TensorBoard
agreement, checkpoint integrity or absence, and clean process exit. The first
warmup update has learning rate zero under the unchanged scheduler; step two
was therefore required to demonstrate actual parameter changes.

A fresh ASPECT calibration used 50 real training examples, rank/alpha 64/64,
196 target modules, sequence length 512 and the reference SVD settings. Saved
step-two cores were subsequently reloaded through the installed public evaluator
from outside the checkout. Two independent TP1 servers, each with internal DP1,
loaded the agent adapters and completed four real held-out trajectories. Servers
and test GPUs were released after completion. These small samples are functional
checks, not accuracy estimates. See `gpu_acceptance.json`.

## CPU and Source Checks

- Final Linux regression suite: **258 passed, 1 skipped**. The skip is an opt-in
  large Code worker-pool stress test, not a skipped algorithm acceptance case.
- Full-FT resolved Hydra parity: **12/12** cases passed, covering WS/RS/AS,
  Eval-Opt/Orch-Workers and learning rates 1e-6/2e-5. Only experiment labels and
  output/log locations are normalized in this comparison.
- **562** core runtime/workflow files are byte- or Python-AST-equivalent to the
  frozen research implementation; **556** are byte-identical. Calibration
  computation functions retain identical ASTs. See `source_equivalence.json`.
- Formal defaults remain four GPUs, batch 64, eight rollouts, 200 updates and
  complete official validation splits. Quick-smoke subsets cannot be canonical.
- Both completion orders of the Full-FT tail protocol passed CPU regression
  tests. The original final-policy, interruptible audit logic is retained.
- Python/JSON/shell syntax and private-identifier/credential scans passed.
  Required third-party license attribution is retained. Original repository
  history, private logs, training data and model weights are not included.

## Portability Fixes Verified

The release preserves the research computation and supplies portable wrappers.
Acceptance found and corrected checkpoint cadence for smoke tests, runtime
audit logging, worker import isolation, evaluator-owned process cleanup,
internal-DP versus independent-TP1 evaluation, and prompt-resource resolution
when launched outside the checkout. The checkpoint evaluator reuses the
research replica implementation; its replica count is configurable for smoke
checks, while the formal wrapper still requires four replicas.

All seven training checks used the same package-root working directory that the
final launcher now selects explicitly. The final evaluator was verified through
the corrected installed package. Training computation was not changed during
these evaluation/resource-path repairs.

## Installation Scope

Regular wheel building, installed backend provenance, Hydra composition and
`pip check` passed in a separate environment with existing reference CUDA
dependencies visible. Documented runtime pins matched the installed packages;
FlashInfer 0.3.1 and Ninja 1.13.0 are explicitly pinned after checking the actual
GPU runtime. This is **not** a fresh installation of the complete CUDA stack.

An older environment C++ runtime initially could not load a FlashInfer kernel
compiled with GCC 13. Selecting the compatible host C++ runtime for the test
processes resolved the native-library error; no algorithm fallback was used.
Use a compiler-compatible runtime as described in the README.

Transformers 4.57.0 and Polars 1.43.0 were marked yanked when the public package
index was inspected. Their exact research versions remain pinned. The former
has an upstream installation-related reason; the latter had no listed reason.

## Coverage Limits

The new GPU matrix covers all seven methods on Qwen3-1.7B Math Orch-Workers,
not every task, scale and workflow combination. Other workflow configurations,
route/state isolation, Code rewards and tail coordination have CPU regression
coverage and retain the research source. This acceptance did not rerun the
paper's full experimental grid or the complete 200-step, two-lane GPU schedule.
