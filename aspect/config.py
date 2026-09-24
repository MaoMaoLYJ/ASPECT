"""Portable method contracts layered over the original training defaults."""
from __future__ import annotations

import json
from pathlib import Path

METHODS = ("WS", "RS", "AS", "ASPECT", "WS_Full_FT", "RS_Full_FT", "AS_Full_FT")
WORKFLOWS = ("eval_opt", "voting", "orch_workers")
ROLE_ROUTES = {
    "eval_opt": ["generator", "evaluator"],
    "voting": ["generator", "aggregator"],
    "orch_workers": ["orchestrator", "worker", "synthesizer"],
}
AGENT_ROUTES = {
    "eval_opt": ["generator", "evaluator"],
    "voting": ["generator0", "generator1", "generator2", "aggregator"],
    "orch_workers": ["orchestrator", "worker0", "worker1", "worker2", "synthesizer"],
}
WORKFLOW_NAMES = {"eval_opt": "evaluator_optimizer", "voting": "voting", "orch_workers": "orchestrator_workers_propose"}


def method_config(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return json.loads((Path(__file__).parents[1] / "algorithms" / method / "config.json").read_text())


def build_overrides(*, method, workflow, task, scale, model, output, ray_temp,
                    cpus=16, gpus=4, basis=None, learning_rate=None, steps=200,
                    gpu_memory=0.85, batch_size=64, rollouts=8,
                    checkpoint_interval=10, validation_interval=10,
                    max_prompt_length=None, max_response_length=None):
    if workflow not in WORKFLOWS or task not in ("math", "code") or scale not in ("0.6b", "1.7b", "4b"):
        raise ValueError("Unsupported workflow, task, or scale")
    spec = method_config(method)
    full = spec["full_parameter"]
    if full and (task != "math" or workflow == "voting"):
        raise ValueError("Full-FT currently supports Math Eval-Opt and Orch-Workers")
    if gpus < 1 or steps < 1 or batch_size < 1 or rollouts < 2:
        raise ValueError("GPU count, steps and batch size must be positive; GRPO needs at least two rollouts")
    if checkpoint_interval < 1 or validation_interval < 1:
        raise ValueError("Checkpoint and validation intervals must be positive")
    if cpus < 2 or not 0 < gpu_memory < 1:
        raise ValueError("Invalid CPU count or GPU-memory fraction")
    lr = learning_rate if learning_rate is not None else (1e-4 if method == "ASPECT" and scale == "0.6b" else spec["learning_rate"])
    if lr <= 0:
        raise ValueError("Learning rate must be positive")
    if method == "ASPECT" and not basis:
        raise ValueError("ASPECT requires a completed task-aligned calibration artifact (--basis)")
    cfg = json.loads(Path(__file__).with_name("defaults.json").read_text())
    cfg.pop("+rllm.workflow.max_iterations")
    routes = AGENT_ROUTES[workflow] if spec["sharing"] == "agent" else ROLE_ROUTES[workflow]
    legacy = {"WS": "share_policy", "RS": "multi_lora", "AS": "agent_lora", "ASPECT": "agent_lorasb"}.get(method, method)
    name = f"{WORKFLOW_NAMES[workflow]}-qwen3_{scale}-{legacy}-{task}"
    cfg.update({
        "actor_rollout_ref.model.path": str(model),
        "actor_rollout_ref.model.lora_rank": 0 if full else 64,
        "actor_rollout_ref.model.lora_alpha": 64 if method == "ASPECT" else 32,
        "actor_rollout_ref.actor.optim.lr": lr,
        "trainer.agent_names": routes,
        "trainer.share_policy": spec["sharing"] == "workflow",
        "trainer.agent_wise_lora": method == "AS",
        "trainer.project_name": "ASPECT",
        "trainer.experiment_name": name,
        "trainer.n_gpus_per_node": gpus,
        "trainer.total_training_steps": steps,
        "trainer.default_local_dir": str(Path(output) / "checkpoints" / name),
        "ray_kwargs.ray_init.num_cpus": cpus,
        "+ray_kwargs.ray_init._temp_dir": str(ray_temp),
        "+ray_init.num_cpus": cpus,
        "+ray_init._temp_dir": str(ray_temp),
        "actor_rollout_ref.rollout.gpu_memory_utilization": gpu_memory,
        "+actor_rollout_ref.rollout.replica_sticky_max_load_gap": 0,
        "+data.dataset_name": "dapo_math" if task == "math" else "deepcoder_primeintellect",
    })
    if task == "math":
        cfg["data.max_prompt_length"] = 30720 if workflow == "eval_opt" else 20480
        cfg["data.max_response_length"] = 5120
    else:
        cfg["data.max_prompt_length"] = 20480 if scale == "4b" else 10240
        cfg["data.max_response_length"] = 5120 if scale == "4b" else 2048
        cfg.update({"+rllm.workflow.generation_admission_window": 0,
                    "rllm.workflow.code_executor_workers": max(1, min(88, cpus - 8)),
                    "rllm.workflow.max_concurrent_code_execs": 0,
                    "+rllm.workflow.code_batch_scheduler": True})
    option, value = {"eval_opt": ("max_iterations", 3 if task == "math" else 2),
                     "voting": ("n_votes", 3), "orch_workers": ("max_subtasks", 3)}[workflow]
    cfg[f"+rllm.workflow.{option}"] = value
    if method in ("AS", "ASPECT"):
        cfg.update({
            "+actor_rollout_ref.rollout.route_homogeneous_batching": True,
            "+actor_rollout_ref.rollout.route_homogeneous_batch_size": 512,
            "+actor_rollout_ref.rollout.route_homogeneous_max_requests_per_turn": 1536 if workflow == "orch_workers" else 512,
            "+actor_rollout_ref.rollout.route_homogeneous_coalesce_ms": 2,
            "+actor_rollout_ref.rollout.require_explicit_lora_route": True,
            "+actor_rollout_ref.rollout.route_semantic_canary": True,
            "+actor_rollout_ref.rollout.route_semantic_canary_max_tokens": 16,
            "+actor_rollout_ref.rollout.route_semantic_canary_logprob_atol": 0.0625,
            "+actor_rollout_ref.rollout.route_diagnostics_log_interval": 256,
        })
        if workflow != "eval_opt":
            flag = "agent_lorasb_instance_routing" if method == "ASPECT" else "agent_wise_lora_instance_routing"
            cfg[f"+rllm.workflow.workflow_args.{flag}"] = True
    if method == "ASPECT":
        cfg.pop("+trainer.peft_eval_checkpoint")
        cfg["+trainer.agent_lorasb"] = {"enable": True}
        cfg["+actor_rollout_ref.model.agent_lorasb"] = {"enable": True, "basis_path": str(basis), "routes": routes}
    if full:
        cfg.pop("+trainer.peft_eval_checkpoint")
        cfg.update({"actor_rollout_ref.model.lora_adapter_path": None,
                    "+actor_rollout_ref.model.require_full_parameter_training": True,
                    "trainer.save_freq": -1, "trainer.test_freq": 10,
                    "data.val_batch_size": 512, "+rllm.load_validation_dataset": True,
                    "+trainer.inline_full_validation": {
                        "enable": True, "expected_rows": 1412, "canonical": True,
                        "workflow": workflow, "output": str(Path(output) / "validation"),
                        "dataset_sha256": None}})
        if method != "WS_Full_FT":
            cfg.update({"+trainer.agent_wise_full_parameter": method == "AS_Full_FT",
                        "+actor_rollout_ref.model.full_parameter_routes": routes,
                        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
                        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True})
        if method == "RS_Full_FT":
            cfg["+trainer.role_wise_full_parameter"] = True
    cfg['data.train_batch_size'] = batch_size
    cfg['actor_rollout_ref.actor.ppo_mini_batch_size'] = batch_size
    cfg['actor_rollout_ref.rollout.n'] = rollouts
    cfg['trainer.save_freq'] = -1 if full else checkpoint_interval
    cfg['trainer.test_freq'] = validation_interval if full else -1
    cfg['trainer.paper_metrics.validation_checkpoint_interval'] = checkpoint_interval
    if not full and method != 'ASPECT':
        cfg['+trainer.peft_eval_checkpoint'] = {'enable': True, 'retain_interval': checkpoint_interval}
    for name, value in [('max_prompt_length', max_prompt_length), ('max_response_length', max_response_length)]:
        if value is not None:
            if value < 1:
                raise ValueError('Context lengths must be positive')
            cfg['data.' + name] = value
    return cfg


def render(cfg):
    def hydra(value):
        if isinstance(value, dict):
            return "{" + ",".join(str(k) + ":" + hydra(v) for k, v in value.items()) + "}"
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(hydra(v) for v in value) + "]"
        if value is None or isinstance(value, bool):
            return json.dumps(value)
        if isinstance(value, str):
            # The inherited config contains literal Hydra lists/maps.
            return value if value.startswith(("[", "{")) else json.dumps(value)
        return str(value)
    return [key + "=" + hydra(value) for key, value in cfg.items()]


def entrypoint(method, workflow, task):
    if method.endswith("_Full_FT"):
        return "examples.math_reasoning.train_shared_full_math"
    suffix = {"eval_opt": "evaluator_optimizer", "voting": "voting", "orch_workers": "orchestrator_workers"}[workflow]
    return (f"examples.math_reasoning.train_{suffix}_math" if task == "math"
            else f"examples.deepcoder.train_deepcoder_{suffix}")
