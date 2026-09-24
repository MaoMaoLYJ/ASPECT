# Reproduction Contract

Keep the following fixed for a scientific comparison: official model/revision,
dataset bytes and split, seed, workflow prompts and iteration limits, effective
batch, rollout count, rewards, optimizer update order, learning rate, warmup,
training horizon, and validation protocol. Learning-rate ablations are separate
runs; never splice their trajectories.

Every public cell writes a resolved run manifest, per-step paper metrics and
TensorBoard events. Gradient diagnostics run every ten updates. LoRA/core
checkpoints are adapter-only or R-only atomic artifacts. Full-FT intentionally
does not save checkpoints and therefore cannot resume after interruption.
Interrupted runs retain their diagnostics but do not receive a success marker.

LoRA post-hoc Math evaluation uses the training prompt/response budgets;
Code uses the reference evaluator's enlarged response budgets. Full-FT online
evaluation uses its training workflow and length budget. Canonical validation
always uses one rollout, temperature 0.7, and the complete declared test split.
Do not report training batch success as validation accuracy.

The source manifest distinguishes unmodified runtime files from packaging
adaptations. The release wrapper changes path handling and public entry points,
not the underlying gradient/update implementation. Regression tests check route
identity, independent dense optimizer state, same-role worker sharing, unique
request IDs, calibration math, and data/progress integrity. Actual execution
evidence and any unverified hardware scope are recorded separately in
`verification/RELEASE_VALIDATION.md`.

Reproduction does not mean bitwise identical output on arbitrary hardware.
BF16 arithmetic, distributed reduction order, randomized SVD, vLLM batching,
and asynchronous scheduling can alter sampled trajectories. Report the exact
GPU topology and dependency versions. Historical result curves produced by
older runtime versions are not claimed to be bitwise reproducible with this
latest fixed framework.

The two-lane Full-FT launcher is for two disjoint four-GPU allocations. Its
canonical horizon is fixed at 200; only the separate non-canonical tail audit
continues on the earlier lane. Tail records must not enter paper curves.

No absolute-time or GPU-memory measurements are inferred from missing logs.
Calibration timing in the inherited artifact starts after model loading and
ends after gradient accumulation, reductions, SVD and artifact preparation;
it is not total process startup time and does not record peak GPU memory.
