"""Scalar policy diagnostics used by the paper-facing experiment runners."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import torch

from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics


def validate_paper_metrics_runtime_config(config: Any) -> None:
    """Fail fast when a formal run would silently omit required diagnostics."""

    trainer = config.trainer
    contract = trainer.get("paper_metrics", {})
    if not bool(contract.get("require_complete", False)):
        return

    gradient = trainer.get("gradient_diagnostics", {})
    if not bool(gradient.get("enable", False)):
        raise ValueError(
            "Formal paper run requires trainer.gradient_diagnostics.enable=true"
        )
    gradient_interval = int(gradient.get("interval", 0))
    if gradient_interval <= 0:
        raise ValueError(
            "Formal paper run requires a positive gradient diagnostics interval"
        )

    validation_interval = int(
        contract.get("validation_checkpoint_interval", 10)
    )
    checkpoint_interval = int(trainer.get("save_freq", 0))
    online = trainer.get("inline_full_validation", {})
    if online.get("enable", False):
        model = config.actor_rollout_ref.model
        rollout = config.actor_rollout_ref.rollout
        agent_private = bool(trainer.get("agent_wise_full_parameter", False))
        role_private = bool(trainer.get("role_wise_full_parameter", False))
        if agent_private and role_private:
            raise ValueError("Full-parameter agent-wise and role-wise modes are mutually exclusive")
        private = agent_private or role_private
        valid_policy = bool(trainer.get("share_policy")) != bool(private)
        if private:
            valid_policy = valid_policy and list(model.get("full_parameter_routes", [])) == list(trainer.get("agent_names", [])) and bool(trainer.get("agent_names"))
        if (not valid_policy or model.get("lora_rank", 0) != 0
                or model.get("lora_adapter_path") or not model.get("require_full_parameter_training")
                or checkpoint_interval != -1 or trainer.get("test_freq") != validation_interval
                or validation_interval <= 0 or rollout.val_kwargs.n != 1
                or (online.get("canonical", True) and validation_interval != 10)
                or (online.get("canonical", True) and online.get("expected_rows") != 1412)
                or trainer.get("peft_eval_checkpoint", {}).get("enable", False)):
            raise ValueError("Invalid checkpoint-free shared full-parameter validation contract")
        return
    if validation_interval <= 0:
        raise ValueError("Paper validation checkpoint interval must be positive")
    if checkpoint_interval <= 0 or validation_interval % checkpoint_interval != 0:
        raise ValueError(
            "Formal paper run must atomically retain every validation checkpoint: "
            f"save_freq={checkpoint_interval}, validation_interval={validation_interval}"
        )


def _metric_label(value: object) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")
    if not normalized:
        raise ValueError(f"Empty policy diagnostic label from {value!r}")
    return normalized


def compute_grouped_policy_diagnostics(
    *,
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor | None,
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    group_labels: Sequence[object],
    group_prefix: str,
    loss_agg_mode: str,
    include_global: bool = True,
) -> dict[str, Any]:
    """Compute the same chi-squared/PPL formula globally and per route/role.

    This function is diagnostic-only: tensors are detached and neither masks nor
    optimizer state are mutated. The chi-squared and perplexity formulas are the
    canonical VERL ``compute_offpolicy_metrics`` implementation used by the
    paper analysis.
    """

    batch_size = old_log_prob.shape[0]
    if not (
        entropys.shape[0]
        == response_mask.shape[0]
        == len(group_labels)
        == batch_size
    ):
        raise ValueError("Policy diagnostic tensors and group labels must align")
    if rollout_log_prob is not None and rollout_log_prob.shape != old_log_prob.shape:
        raise ValueError("Rollout and training log probabilities must align")

    metrics: dict[str, Any] = {}

    def add_scope(scope: str, indices: list[int]) -> None:
        mask = response_mask[indices]
        if not bool(mask.any().item()):
            return
        scoped = compute_offpolicy_metrics(
            old_log_prob=old_log_prob[indices].detach(),
            rollout_log_prob=(
                rollout_log_prob[indices].detach()
                if rollout_log_prob is not None
                else None
            ),
            response_mask=mask.detach(),
        )
        scoped["entropy"] = float(
            agg_loss(
                loss_mat=entropys[indices].detach(),
                loss_mask=mask.detach(),
                loss_agg_mode=loss_agg_mode,
            ).item()
        )
        metrics.update({f"{scope}/{name}": value for name, value in scoped.items()})

    if include_global:
        add_scope("paper_diag/policy/global", list(range(batch_size)))

    groups = list(dict.fromkeys(str(label) for label in group_labels))
    for group in groups:
        indices = [
            index
            for index, label in enumerate(group_labels)
            if str(label) == group
        ]
        add_scope(f"{group_prefix}/{_metric_label(group)}", indices)

    for metric_name in (
        "chi2_token",
        "chi2_seq",
        "training_ppl",
        "rollout_ppl",
        "ppl_ratio",
        "kl",
        "k3_kl",
        "entropy",
    ):
        values = [
            float(value)
            for key, value in metrics.items()
            if key.startswith(f"{group_prefix}/")
            and key.endswith(f"/{metric_name}")
        ]
        if values:
            metrics[f"{group_prefix}_max/{metric_name}"] = max(values)
            metrics[f"{group_prefix}_min/{metric_name}"] = min(values)
    return metrics
