"""Non-invasive gradient diagnostics for multi-agent LoRA training.

The functions in this module only summarize probe gradients. They never alter
advantages, losses, model parameters, optimizer state, or sampling behavior.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import torch


def parse_agent_labels(
    trajectory_ids: Sequence[object],
    known_roles: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return base-role and instantiated-slot labels from trajectory IDs.

    Workflow trajectory IDs end in ``_<slot>``. Repeated workflow instances use
    numeric slot suffixes (for example ``generator0``), while LoRA routing uses
    the base role (``generator``).
    """

    normalized_roles = sorted(
        {str(role) for role in (known_roles or ()) if str(role)},
        key=len,
        reverse=True,
    )
    roles: list[str] = []
    slots: list[str] = []
    for trajectory_id in trajectory_ids:
        value = str(trajectory_id)
        matched_role = next(
            (
                role
                for role in normalized_roles
                if re.search(rf"_{re.escape(role)}\d*$", value)
            ),
            None,
        )
        slot = (
            value[value.rfind(f"_{matched_role}") + 1 :]
            if matched_role is not None
            else value.rsplit("_", 1)[-1]
        )
        if not slot:
            raise ValueError(f"Cannot parse an agent slot from trajectory ID {trajectory_id!r}")
        role = matched_role or re.sub(r"\d+$", "", slot) or slot
        roles.append(role)
        slots.append(slot)
    return roles, slots


def should_collect_gradient_diagnostics(config: Mapping | None, global_step: int) -> bool:
    """Whether a diagnostic probe is enabled for ``global_step``."""

    if not config or not bool(config.get("enable", False)):
        return False
    interval = int(config.get("interval", 1))
    if interval <= 0:
        raise ValueError("trainer.gradient_diagnostics.interval must be positive")
    return int(global_step) % interval == 0


def group_population_statistics(
    *,
    groups: Sequence[str],
    labels: Sequence[object],
    trajectory_ids: Sequence[object],
    response_token_counts: Sequence[int | float],
) -> dict[str, list[float]]:
    """Return full-batch population statistics aligned with ``groups``.

    Probe gradients may use only a few sequences per group. Contribution
    weights must nevertheless reflect the complete PPO batch. The ratio of
    sequence count to unique trajectory ID also exposes temporal role reuse:
    Eval-Opt emits multiple generator/evaluator trajectories with the same
    task-role ID when refinement repeats.
    """

    if not (len(labels) == len(trajectory_ids) == len(response_token_counts)):
        raise ValueError("labels, trajectory_ids, and response_token_counts must align")
    if len(set(groups)) != len(groups):
        raise ValueError(f"Gradient group names must be unique: {groups!r}")

    normalized_labels = [str(label) for label in labels]
    normalized_trajectory_ids = [str(trajectory_id) for trajectory_id in trajectory_ids]
    unknown_labels = sorted(set(normalized_labels).difference(groups))
    if unknown_labels:
        raise ValueError(f"Labels are missing from gradient groups: {unknown_labels}")
    tokens = [float(count) for count in response_token_counts]
    if any(not math.isfinite(count) or count < 0 for count in tokens):
        raise ValueError("response_token_counts must be finite and non-negative")

    sequence_counts: list[float] = []
    trajectory_counts: list[float] = []
    token_totals: list[float] = []
    for group in groups:
        indices = [index for index, label in enumerate(normalized_labels) if label == group]
        sequence_counts.append(float(len(indices)))
        trajectory_counts.append(
            float(len({normalized_trajectory_ids[index] for index in indices}))
        )
        token_totals.append(float(sum(tokens[index] for index in indices)))

    return {
        "sequence_counts": sequence_counts,
        "trajectory_counts": trajectory_counts,
        "response_token_counts": token_totals,
    }


def _metric_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))


def _safe_cosine(gram: torch.Tensor, i: int, j: int, eps: float) -> tuple[float, bool]:
    norm_i = float(torch.clamp(gram[i, i], min=0).sqrt().item())
    norm_j = float(torch.clamp(gram[j, j], min=0).sqrt().item())
    valid = norm_i > eps and norm_j > eps
    if not valid:
        return 0.0, False
    cosine = float(gram[i, j].item()) / (norm_i * norm_j)
    return max(-1.0, min(1.0, cosine)), True


def _effective_count(shares: torch.Tensor, eps: float) -> float:
    positive = shares[shares > eps]
    if positive.numel() == 0:
        return 0.0
    entropy = -(positive * positive.log()).sum()
    return float(entropy.exp().item())


def align_lorasb_route_gradient_shards(
    shards: Mapping[str, torch.Tensor],
    *,
    route: str,
) -> dict[str, torch.Tensor]:
    """Replace a private route path with a common LoRA-SB R coordinate path."""

    route_token = f".cores.{route}."
    aligned: dict[str, torch.Tensor] = {}
    for name, shard in shards.items():
        if route_token not in str(name):
            raise ValueError(
                f"LoRA-SB route gradient {name!r} does not contain {route_token!r}"
            )
        canonical = str(name).replace(route_token, ".cores.<route>.", 1)
        if canonical in aligned:
            raise ValueError(f"Duplicate aligned LoRA-SB gradient key {canonical!r}")
        aligned[canonical] = shard
    if not aligned:
        raise ValueError(f"LoRA-SB route {route!r} produced no gradient shards")
    return aligned


def summarize_gradient_gram(
    *,
    gram: torch.Tensor,
    groups: Sequence[str],
    sequence_counts: Sequence[int | float],
    trajectory_counts: Sequence[int | float] | None = None,
    response_token_counts: Sequence[int | float] | None = None,
    probe_counts: Sequence[int | float] | None = None,
    mode: str,
    scope: str | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Summarize a distributed gradient Gram matrix as scalar diagnostics.

    ``gram[i, j]`` is the global inner product between the mean probe
    gradients for groups ``i`` and ``j``. In shared-policy (SP) mode, groups
    are logical roles and their contributions are weighted by sequence count,
    matching the official ``seq-mean-token-mean`` PPO aggregation. In
    independent-policy (IP) mode, groups are repeated slots of one role.
    """

    if mode not in {"sp", "ip"}:
        raise ValueError(f"mode must be 'sp' or 'ip', got {mode!r}")
    if mode == "ip" and not scope:
        raise ValueError("scope is required for IP diagnostics")
    if len(groups) == 0:
        raise ValueError("At least one gradient group is required")
    if len(set(groups)) != len(groups):
        raise ValueError(f"Gradient group names must be unique: {groups!r}")
    if len(sequence_counts) != len(groups):
        raise ValueError("sequence_counts must align with groups")
    if trajectory_counts is not None and len(trajectory_counts) != len(groups):
        raise ValueError("trajectory_counts must align with groups")
    if response_token_counts is not None and len(response_token_counts) != len(groups):
        raise ValueError("response_token_counts must align with groups")
    if probe_counts is not None and len(probe_counts) != len(groups):
        raise ValueError("probe_counts must align with groups")

    matrix = torch.as_tensor(gram, dtype=torch.float64).detach().cpu()
    expected_shape = (len(groups), len(groups))
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"Expected a {expected_shape} Gram matrix, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError("Gradient Gram matrix contains a non-finite value")
    matrix = 0.5 * (matrix + matrix.T)

    counts = torch.as_tensor(sequence_counts, dtype=torch.float64)
    if (counts < 0).any() or not torch.isfinite(counts).all():
        raise ValueError("sequence_counts must be finite and non-negative")
    trajectories = (
        torch.as_tensor(trajectory_counts, dtype=torch.float64)
        if trajectory_counts is not None
        else counts.clone()
    )
    if (trajectories < 0).any() or not torch.isfinite(trajectories).all():
        raise ValueError("trajectory_counts must be finite and non-negative")
    tokens = (
        torch.as_tensor(response_token_counts, dtype=torch.float64)
        if response_token_counts is not None
        else torch.zeros_like(counts)
    )
    if (tokens < 0).any() or not torch.isfinite(tokens).all():
        raise ValueError("response_token_counts must be finite and non-negative")
    probes = (
        torch.as_tensor(probe_counts, dtype=torch.float64)
        if probe_counts is not None
        else counts.clone()
    )
    if (probes < 0).any() or not torch.isfinite(probes).all():
        raise ValueError("probe_counts must be finite and non-negative")

    prefix = "grad_diag/sp" if mode == "sp" else f"grad_diag/ip/{_metric_label(scope or '')}"
    metrics: dict[str, float] = {
        f"{prefix}/probe": 1.0,
        f"{prefix}/probe_sequence_count": float(probes.sum().item()),
        f"{prefix}/sequence_count": float(counts.sum().item()),
        f"{prefix}/trajectory_count": float(trajectories.sum().item()),
    }
    total_trajectories = float(trajectories.sum().item())
    metrics[f"{prefix}/sequences_per_trajectory"] = (
        float(counts.sum().item()) / total_trajectories
        if total_trajectories > 0
        else 0.0
    )
    norms = torch.clamp(torch.diag(matrix), min=0).sqrt()
    total_count = float(counts.sum().item())
    sequence_shares = counts / total_count if total_count > 0 else torch.zeros_like(counts)

    group_kind = "role" if mode == "sp" else "slot"
    for i, group in enumerate(groups):
        group_prefix = f"{prefix}/{group_kind}/{_metric_label(group)}"
        metrics[f"{group_prefix}/grad_norm"] = float(norms[i].item())
        metrics[f"{group_prefix}/sequence_count"] = float(counts[i].item())
        metrics[f"{group_prefix}/sequence_share"] = float(sequence_shares[i].item())
        metrics[f"{group_prefix}/trajectory_count"] = float(trajectories[i].item())
        metrics[f"{group_prefix}/sequences_per_trajectory"] = (
            float(counts[i].item() / trajectories[i].item())
            if trajectories[i].item() > 0
            else 0.0
        )
        metrics[f"{group_prefix}/response_tokens"] = float(tokens[i].item())
        metrics[f"{group_prefix}/probe_count"] = float(probes[i].item())

    conflicts = 0
    valid_pairs = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            cosine, valid = _safe_cosine(matrix, i, j, eps)
            pair = f"{_metric_label(groups[i])}__{_metric_label(groups[j])}"
            metrics[f"{prefix}/pair/{pair}/cosine"] = cosine
            metrics[f"{prefix}/pair/{pair}/conflict"] = float(valid and cosine < 0.0)
            if valid:
                valid_pairs += 1
                conflicts += int(cosine < 0.0)
    metrics[f"{prefix}/valid_pair_count"] = float(valid_pairs)
    metrics[f"{prefix}/conflict_rate"] = conflicts / valid_pairs if valid_pairs else 0.0

    if mode == "ip":
        slot_count = len(groups)
        contribution_norms = sequence_shares * norms
        contribution_norm_sum = float(contribution_norms.sum().item())
        contribution_shares = (
            contribution_norms / contribution_norm_sum
            if contribution_norm_sum > eps
            else torch.zeros_like(contribution_norms)
        )
        weighted_resultant_sq = float(
            (sequence_shares.unsqueeze(0) @ matrix @ sequence_shares.unsqueeze(1))
            .squeeze()
            .clamp(min=0)
            .item()
        )
        weighted_resultant = math.sqrt(weighted_resultant_sq)
        coherence = (
            weighted_resultant / contribution_norm_sum
            if contribution_norm_sum > eps
            else 0.0
        )
        coherence = max(0.0, min(1.0, coherence))
        for i, group in enumerate(groups):
            group_prefix = f"{prefix}/slot/{_metric_label(group)}"
            metrics[f"{group_prefix}/contribution_norm"] = float(
                contribution_norms[i].item()
            )
            metrics[f"{group_prefix}/contribution_share"] = float(
                contribution_shares[i].item()
            )
        metrics[f"{prefix}/slot_count"] = float(slot_count)
        metrics[f"{prefix}/resultant_norm"] = weighted_resultant
        metrics[f"{prefix}/resultant_ratio"] = coherence
        metrics[f"{prefix}/coherence"] = coherence
        metrics[f"{prefix}/cancellation"] = 1.0 - coherence
        metrics[f"{prefix}/dominance"] = (
            float(contribution_shares.max().item())
            if contribution_shares.numel()
            else 0.0
        )
        metrics[f"{prefix}/effective_slots"] = _effective_count(
            contribution_shares, eps
        )
        metrics[f"{prefix}/amplification"] = float(slot_count) * coherence
        return metrics

    contribution_norms = sequence_shares * norms
    contribution_norm_sum = float(contribution_norms.sum().item())
    if contribution_norm_sum > eps:
        contribution_shares = contribution_norms / contribution_norm_sum
    else:
        contribution_shares = torch.zeros_like(contribution_norms)

    for i, group in enumerate(groups):
        group_prefix = f"{prefix}/role/{_metric_label(group)}"
        metrics[f"{group_prefix}/contribution_norm"] = float(contribution_norms[i].item())
        metrics[f"{group_prefix}/contribution_share"] = float(contribution_shares[i].item())

    weighted_sum_norm_sq = float(
        (sequence_shares.unsqueeze(0) @ matrix @ sequence_shares.unsqueeze(1))
        .squeeze()
        .clamp(min=0)
        .item()
    )
    weighted_sum_norm = math.sqrt(weighted_sum_norm_sq)
    coherence = (
        weighted_sum_norm / contribution_norm_sum
        if contribution_norm_sum > eps
        else 0.0
    )
    coherence = max(0.0, min(1.0, coherence))
    metrics[f"{prefix}/role_count"] = float(len(groups))
    metrics[f"{prefix}/resultant_norm"] = weighted_sum_norm
    metrics[f"{prefix}/resultant_ratio"] = coherence
    metrics[f"{prefix}/coherence"] = coherence
    metrics[f"{prefix}/cancellation"] = 1.0 - coherence
    metrics[f"{prefix}/dominance"] = (
        float(contribution_shares.max().item()) if contribution_shares.numel() else 0.0
    )
    metrics[f"{prefix}/effective_roles"] = _effective_count(contribution_shares, eps)
    return metrics


def summarize_aligned_route_gradient_gram(
    *,
    gram: torch.Tensor,
    routes: Sequence[str],
    sequence_counts: Sequence[int | float],
    eps: float = 1e-12,
) -> dict[str, float]:
    """Summarize LoRA-SB route gradients in their shared R coordinates.

    Every LoRA-SB route owns a distinct R tensor but all tensors use the same
    frozen A/B basis and identical module ordering.  Aligning by module and R
    coordinate therefore compares genuine task-update directions without
    pretending that the routes share optimizer parameters.
    """

    if not routes or len(set(routes)) != len(routes):
        raise ValueError("LoRA-SB gradient routes must be non-empty and unique")
    if len(sequence_counts) != len(routes):
        raise ValueError("sequence_counts must align with LoRA-SB routes")
    matrix = torch.as_tensor(gram, dtype=torch.float64).detach().cpu()
    expected_shape = (len(routes), len(routes))
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"Expected a {expected_shape} Gram matrix, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError("LoRA-SB route-gradient Gram matrix contains non-finite values")
    matrix = 0.5 * (matrix + matrix.T)
    counts = torch.as_tensor(sequence_counts, dtype=torch.float64)
    if (counts < 0).any() or not torch.isfinite(counts).all():
        raise ValueError("LoRA-SB route sequence counts must be finite and non-negative")

    prefix = "paper_diag/lorasb/gradient"
    norms = torch.clamp(torch.diag(matrix), min=0).sqrt()
    total_count = float(counts.sum().item())
    sequence_shares = counts / total_count if total_count > 0 else torch.zeros_like(counts)
    contribution_norms = sequence_shares * norms
    contribution_norm_sum = float(contribution_norms.sum().item())
    contribution_shares = (
        contribution_norms / contribution_norm_sum
        if contribution_norm_sum > eps
        else torch.zeros_like(contribution_norms)
    )
    metrics: dict[str, float] = {
        f"{prefix}/probe": 1.0,
        f"{prefix}/route_count": float(len(routes)),
        f"{prefix}/sequence_count": total_count,
    }
    for index, route in enumerate(routes):
        route_prefix = f"{prefix}/route/{_metric_label(route)}"
        metrics[f"{route_prefix}/grad_norm"] = float(norms[index].item())
        metrics[f"{route_prefix}/sequence_count"] = float(counts[index].item())
        metrics[f"{route_prefix}/sequence_share"] = float(sequence_shares[index].item())
        metrics[f"{route_prefix}/contribution_norm"] = float(
            contribution_norms[index].item()
        )
        metrics[f"{route_prefix}/contribution_share"] = float(
            contribution_shares[index].item()
        )

    conflicts = 0
    valid_pairs = 0
    for left in range(len(routes)):
        for right in range(left + 1, len(routes)):
            cosine, valid = _safe_cosine(matrix, left, right, eps)
            pair = f"{_metric_label(routes[left])}__{_metric_label(routes[right])}"
            metrics[f"{prefix}/pair/{pair}/cosine"] = cosine
            metrics[f"{prefix}/pair/{pair}/conflict"] = float(valid and cosine < 0.0)
            if valid:
                valid_pairs += 1
                conflicts += int(cosine < 0.0)
    metrics[f"{prefix}/valid_pair_count"] = float(valid_pairs)
    metrics[f"{prefix}/conflict_rate"] = conflicts / valid_pairs if valid_pairs else 0.0

    weighted_resultant_sq = float(
        (sequence_shares.unsqueeze(0) @ matrix @ sequence_shares.unsqueeze(1))
        .squeeze()
        .clamp(min=0)
        .item()
    )
    weighted_resultant = math.sqrt(weighted_resultant_sq)
    coherence = (
        weighted_resultant / contribution_norm_sum
        if contribution_norm_sum > eps
        else 0.0
    )
    coherence = max(0.0, min(1.0, coherence))
    metrics[f"{prefix}/resultant_norm"] = weighted_resultant
    metrics[f"{prefix}/resultant_ratio"] = coherence
    metrics[f"{prefix}/coherence"] = coherence
    metrics[f"{prefix}/cancellation"] = 1.0 - coherence
    metrics[f"{prefix}/dominance"] = (
        float(contribution_shares.max().item()) if contribution_shares.numel() else 0.0
    )
    metrics[f"{prefix}/effective_routes"] = _effective_count(contribution_shares, eps)
    return metrics


def summarize_update_accumulation(
    *,
    summed_gradient_norm: float,
    individual_gradient_norm_sum: float,
    update_count: int,
    mode: str,
    scope: str | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Summarize the real pre-clip gradients across optimizer updates.

    This separates update multiplicity from directional agreement. If ``U``
    updates are perfectly aligned, coherence is one and amplification is
    ``U``; opposing updates can have the same count but near-zero coherence.
    """

    if mode not in {"sp", "ip"}:
        raise ValueError(f"mode must be 'sp' or 'ip', got {mode!r}")
    if mode == "ip" and not scope:
        raise ValueError("scope is required for IP update diagnostics")
    if update_count < 0:
        raise ValueError("update_count must be non-negative")
    if summed_gradient_norm < 0 or individual_gradient_norm_sum < 0:
        raise ValueError("gradient norms must be non-negative")
    prefix = "grad_diag/sp" if mode == "sp" else f"grad_diag/ip/{_metric_label(scope or '')}"
    coherence = (
        float(summed_gradient_norm) / float(individual_gradient_norm_sum)
        if individual_gradient_norm_sum > eps
        else 0.0
    )
    coherence = max(0.0, min(1.0, coherence))
    return {
        f"{prefix}/optimizer_update_count": float(update_count),
        f"{prefix}/optimizer_update_coherence": coherence,
        f"{prefix}/optimizer_amplification": float(update_count) * coherence,
        f"{prefix}/optimizer_individual_norm_sum": float(individual_gradient_norm_sum),
        f"{prefix}/optimizer_summed_gradient_norm": float(summed_gradient_norm),
    }
