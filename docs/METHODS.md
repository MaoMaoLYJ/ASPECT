# Parameter-Sharing Semantics

For each adapted dense layer, ordinary LoRA uses `W + (alpha/r) B A`.
ASPECT uses `W + (alpha/r) B R_agent A`: W, A, and B remain frozen;
only each agent's square rank-r core R is optimized.

## Task-aligned calibration

Fifty supervised training examples contribute gradients for the PEFT-selected
all-linear base weights; embeddings, output heads not selected by PEFT, and other
parameters do not enter this update estimate. Gradients are accumulated and
all-reduced. The first AdamW update is approximated by
`Delta = -(learning_rate / warmup_steps) * sign(sum_gradients)`.
This is an approximation, not a literal full-model optimizer step including
all AdamW epsilon and weight-decay terms.

For each target layer, `torch.svd_lowrank(Delta, q=r, niter=10)` produces U,S,V.
The builder stores B=U, A=V^T and the initial core R=diag(S), in BF16, using
seed 42. Each agent receives an independent R initialized from this same task
basis. No previously trained R is used. Calibration artifacts are compatible
only when model, task, rank, alpha, calibration rows, learning rate, warmup,
sequence limit, and SVD settings agree.

## Route identities

| Workflow | Role routes | Individual-agent routes |
| --- | --- | --- |
| Eval-Opt | generator, evaluator | generator, evaluator |
| Voting | generator, aggregator | generator0, generator1, generator2, aggregator |
| Orch-Workers | orchestrator, worker, synthesizer | orchestrator, worker0, worker1, worker2, synthesizer |

WS shares one adapter throughout a workflow. RS shares within a role. AS and
ASPECT use individual-agent routes. Role sharing is not interchangeable with
agent sharing even when the routes happen to coincide in Eval-Opt.

Full-FT does not use adapters. Each non-shared dense route has independent model,
AdamW, and scheduler state in a CPU bank, with a single FSDP execution slot.
RS_Full_FT merges all three worker trajectories into the shared worker update;
AS_Full_FT keeps five independent Orch models. Rollout routes are switched only
after the engine's pending requests are drained and its prefix cache is cleared.
Application identity remains sticky for replica selection while each backend
generation request receives a unique UUID suffix.

## Implementation map

- ASPECT factors/core: `verl/workers/actor/agent_lorasb.py`.
- Calibration computation: `aspect/calibrate.py`.
- Route selection: `rllm/trainer/verl/agent_workflow_trainer.py` and `rllm/engine/rollout/verl_engine.py`.
- FSDP and adapter collection: `verl/workers/fsdp_workers.py`, `verl/utils/fsdp_utils.py`.
- Full-FT banks: `verl/workers/actor/full_parameter_bank.py`.
- Request draining: `verl/experimental/agent_loop/full_parameter_batcher.py`.
- Canonical inline evaluation and tail protection: `rllm/trainer/verl/inline_full_validation.py`.

Historical internal identifiers `sp`, `ip`, `agent_lora`, and `agent_lorasb`
remain in checkpoint schemas for compatibility. The public method names are
WS, RS, AS, and ASPECT, respectively.
