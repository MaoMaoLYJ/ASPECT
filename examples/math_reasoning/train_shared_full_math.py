"""Full-parameter Math entrypoint using the official DAPO held-out split."""

import hashlib
from pathlib import Path

import hydra
from omegaconf import open_dict

from examples.math_reasoning.evaluator_optimizer_math_workflow import EvaluatorOptimizerMathWorkflow
from examples.math_reasoning.orchestrator_workers_math_workflow import OrchestratorWorkersMathWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="multi_agent_ppo_trainer", version_base=None)
def main(config):
    train = DatasetRegistry.load_dataset("dapo_math", "train")
    test = DatasetRegistry.load_dataset("dapo_math", "test")
    online = config.trainer.inline_full_validation
    if train is None or test is None or len(test) != online.expected_rows:
        raise ValueError("Official DAPO train/test registry missing or incomplete")
    with open_dict(online):
        online.dataset_sha256 = hashlib.sha256(Path(test.get_data_path()).read_bytes()).hexdigest()
    args = {"reward_function": math_reward_fn, "use_final_outcome_reward": True,
            "initial_lora_weights": None}
    if online.workflow == "eval_opt":
        workflow = EvaluatorOptimizerMathWorkflow
        args["max_iterations"] = config.rllm.workflow.max_iterations
    elif online.workflow == "orch_workers":
        workflow = OrchestratorWorkersMathWorkflow
        args["max_subtasks"] = config.rllm.workflow.max_subtasks
        args["agent_full_parameter_instance_routing"] = config.trainer.get("agent_wise_full_parameter", False)
    else:
        raise ValueError(online.workflow)
    AgentTrainer(workflow_class=workflow, workflow_args=args, config=config,
                 train_dataset=train, val_dataset=test).train()


if __name__ == "__main__":
    main()
