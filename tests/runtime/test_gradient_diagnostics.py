import math

import torch

from verl.trainer.ppo.gradient_diagnostics import (
    align_lorasb_route_gradient_shards,
    group_population_statistics,
    parse_agent_labels,
    should_collect_gradient_diagnostics,
    summarize_aligned_route_gradient_gram,
    summarize_gradient_gram,
    summarize_update_accumulation,
)


def test_parse_agent_labels_preserves_slots_and_strips_only_numeric_suffixes():
    roles, slots = parse_agent_labels(
        [
            "episode-a_generator0",
            "episode-b_generator2",
            "episode-c_evaluator",
            "episode-d_orchestrator_v2",
        ],
        known_roles=["generator", "evaluator", "orchestrator_v"],
    )

    assert roles == ["generator", "generator", "evaluator", "orchestrator_v"]
    assert slots == ["generator0", "generator2", "evaluator", "orchestrator_v2"]


def test_population_statistics_separate_temporal_reuse_from_parallel_slots():
    stats = group_population_statistics(
        groups=["generator", "evaluator"],
        labels=["generator", "evaluator", "generator", "evaluator", "generator"],
        trajectory_ids=["task0_generator", "task0_evaluator"] * 2 + ["task0_generator"],
        response_token_counts=[100, 10, 90, 8, 80],
    )

    assert stats == {
        "sequence_counts": [3.0, 2.0],
        "trajectory_counts": [1.0, 1.0],
        "response_token_counts": [270.0, 18.0],
    }


def test_ip_aligned_instances_reveal_multiplicity_amplification():
    gram = torch.ones((3, 3), dtype=torch.float64)

    metrics = summarize_gradient_gram(
        gram=gram,
        groups=["generator0", "generator1", "generator2"],
        sequence_counts=[4, 4, 4],
        trajectory_counts=[4, 4, 4],
        response_token_counts=[100, 100, 100],
        mode="ip",
        scope="generator",
    )

    assert metrics["grad_diag/ip/generator/coherence"] == 1.0
    assert metrics["grad_diag/ip/generator/resultant_ratio"] == 1.0
    assert metrics["grad_diag/ip/generator/cancellation"] == 0.0
    assert math.isclose(metrics["grad_diag/ip/generator/dominance"], 1.0 / 3.0)
    assert metrics["grad_diag/ip/generator/amplification"] == 3.0
    assert metrics["grad_diag/ip/generator/conflict_rate"] == 0.0


def test_ip_opposing_instances_cancel_instead_of_amplifying():
    gram = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)

    metrics = summarize_gradient_gram(
        gram=gram,
        groups=["worker0", "worker1"],
        sequence_counts=[2, 2],
        mode="ip",
        scope="worker",
    )

    assert metrics["grad_diag/ip/worker/coherence"] == 0.0
    assert metrics["grad_diag/ip/worker/resultant_ratio"] == 0.0
    assert metrics["grad_diag/ip/worker/cancellation"] == 1.0
    assert metrics["grad_diag/ip/worker/dominance"] == 0.5
    assert metrics["grad_diag/ip/worker/amplification"] == 0.0
    assert metrics["grad_diag/ip/worker/conflict_rate"] == 1.0


def test_eval_opt_ip_single_instance_cannot_have_multiplicity_amplification():
    metrics = summarize_gradient_gram(
        gram=torch.tensor([[9.0]], dtype=torch.float64),
        groups=["evaluator"],
        sequence_counts=[24],
        trajectory_counts=[8],
        probe_counts=[2],
        mode="ip",
        scope="evaluator",
    )

    assert metrics["grad_diag/ip/evaluator/slot_count"] == 1.0
    assert metrics["grad_diag/ip/evaluator/coherence"] == 1.0
    assert metrics["grad_diag/ip/evaluator/amplification"] == 1.0
    assert metrics["grad_diag/ip/evaluator/valid_pair_count"] == 0.0
    assert metrics["grad_diag/ip/evaluator/sequences_per_trajectory"] == 3.0
    assert metrics["grad_diag/ip/evaluator/probe_sequence_count"] == 2.0


def test_sp_opposing_role_gradients_report_conflict_and_cancellation():
    gram = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)

    metrics = summarize_gradient_gram(
        gram=gram,
        groups=["generator", "evaluator"],
        sequence_counts=[5, 5],
        mode="sp",
    )

    assert metrics["grad_diag/sp/pair/generator__evaluator/cosine"] == -1.0
    assert metrics["grad_diag/sp/conflict_rate"] == 1.0
    assert metrics["grad_diag/sp/resultant_ratio"] == 0.0
    assert metrics["grad_diag/sp/coherence"] == 0.0
    assert metrics["grad_diag/sp/cancellation"] == 1.0
    assert metrics["grad_diag/sp/dominance"] == 0.5
    assert metrics["grad_diag/sp/effective_roles"] == 2.0


def test_sp_role_dominance_uses_sequence_weighted_gradient_contributions():
    # Orthogonal role gradients with norms 10 and 1; equal sequence counts.
    gram = torch.tensor([[100.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    metrics = summarize_gradient_gram(
        gram=gram,
        groups=["generator", "evaluator"],
        sequence_counts=[4, 4],
        mode="sp",
    )

    assert math.isclose(metrics["grad_diag/sp/dominance"], 10.0 / 11.0)
    assert math.isclose(
        metrics["grad_diag/sp/role/generator/contribution_share"],
        10.0 / 11.0,
    )
    assert metrics["grad_diag/sp/conflict_rate"] == 0.0


def test_sp_population_weight_is_not_replaced_by_balanced_probe_count():
    metrics = summarize_gradient_gram(
        gram=torch.eye(2, dtype=torch.float64),
        groups=["worker", "synthesizer"],
        sequence_counts=[9, 1],
        trajectory_counts=[9, 1],
        probe_counts=[1, 1],
        mode="sp",
    )

    assert math.isclose(metrics["grad_diag/sp/dominance"], 0.9)
    assert metrics["grad_diag/sp/probe_sequence_count"] == 2.0
    assert metrics["grad_diag/sp/sequence_count"] == 10.0


def test_diagnostics_schedule_is_disabled_by_default_and_interval_gated():
    assert not should_collect_gradient_diagnostics({}, global_step=10)
    config = {"enable": True, "interval": 10}
    assert not should_collect_gradient_diagnostics(config, global_step=9)
    assert should_collect_gradient_diagnostics(config, global_step=10)


def test_actual_optimizer_update_accumulation_distinguishes_count_from_coherence():
    metrics = summarize_update_accumulation(
        summed_gradient_norm=3.0,
        individual_gradient_norm_sum=3.0,
        update_count=3,
        mode="ip",
        scope="generator",
    )

    assert metrics["grad_diag/ip/generator/optimizer_update_count"] == 3.0
    assert metrics["grad_diag/ip/generator/optimizer_update_coherence"] == 1.0
    assert metrics["grad_diag/ip/generator/optimizer_amplification"] == 3.0


def test_opposing_optimizer_updates_report_cancellation_without_fake_amplification():
    metrics = summarize_update_accumulation(
        summed_gradient_norm=0.0,
        individual_gradient_norm_sum=2.0,
        update_count=2,
        mode="sp",
    )

    assert metrics["grad_diag/sp/optimizer_update_coherence"] == 0.0
    assert metrics["grad_diag/sp/optimizer_amplification"] == 0.0


def test_lorasb_aligned_route_gradients_report_conflict_and_cancellation():
    metrics = summarize_aligned_route_gradient_gram(
        gram=torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float64),
        routes=["generator0", "generator1"],
        sequence_counts=[8, 8],
    )

    prefix = "paper_diag/lorasb/gradient"
    assert metrics[f"{prefix}/route/generator0/grad_norm"] == 2.0
    assert metrics[f"{prefix}/pair/generator0__generator1/cosine"] == -1.0
    assert metrics[f"{prefix}/pair/generator0__generator1/conflict"] == 1.0
    assert metrics[f"{prefix}/conflict_rate"] == 1.0
    assert metrics[f"{prefix}/resultant_ratio"] == 0.0
    assert metrics[f"{prefix}/coherence"] == 0.0
    assert metrics[f"{prefix}/cancellation"] == 1.0
    assert metrics[f"{prefix}/dominance"] == 0.5
    assert metrics[f"{prefix}/effective_routes"] == 2.0


def test_lorasb_route_gradient_names_align_private_cores_to_shared_coordinates():
    aligned = align_lorasb_route_gradient_shards(
        {
            "model.layer.core_router.cores.generator0._fsdp_wrapped_module.weight": (
                torch.tensor([1.0, 2.0])
            )
        },
        route="generator0",
    )

    assert set(aligned) == {
        "model.layer.core_router.cores.<route>._fsdp_wrapped_module.weight"
    }
    torch.testing.assert_close(
        aligned[
            "model.layer.core_router.cores.<route>._fsdp_wrapped_module.weight"
        ],
        torch.tensor([1.0, 2.0]),
    )
