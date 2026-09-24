import json

from pathlib import Path

import torch

from dashboard.evaluate_checkpoints import discover_checkpoints

from aspect.calibrate import (
    _supervised_example,
    first_adamw_update_approximation,
    summarize_calibration_statistics,
    summarize_singular_spectrum,
)

ROOT = Path(__file__).resolve().parents[2]

class _BoundaryTokenizer:
    def __init__(self, prompt_ids: list[int], target_ids: list[int]):
        self.prompt_ids = prompt_ids
        self.target_ids = target_ids

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        if len(messages) == 1:
            assert add_generation_prompt is True
            return self.prompt_ids
        assert add_generation_prompt is False
        return self.prompt_ids + self.target_ids

def test_lorasb_update_approximation_matches_official_first_adamw_formula():
    gradient_sum = torch.tensor([[-2.0, 0.0, 3.0]])
    update = first_adamw_update_approximation(
        gradient_sum,
        effective_lr=2e-4 / 15,
    )
    torch.testing.assert_close(
        update,
        torch.tensor([[2e-4 / 15, 0.0, -2e-4 / 15]]),
    )

def test_lorasb_calibration_statistics_keep_example_and_token_weighted_losses():
    stats = summarize_calibration_statistics(
        example_loss_sum=5.0,
        token_nll_sum=18.0,
        example_count=4,
        target_token_count=12,
    )

    assert stats == {
        "example_count": 4,
        "target_token_count": 12,
        "example_loss_mean": 1.25,
        "token_nll_mean": 1.5,
    }

def test_lorasb_singular_spectrum_is_plot_ready_and_energy_weighted():
    spectrum = summarize_singular_spectrum(
        {
            "layer0": {"top_r_singular_values": [3.0, 1.0]},
            "layer1": {"top_r_singular_values": [4.0, 2.0]},
        },
        rank=2,
    )

    assert spectrum == [
        {
            "rank": 1,
            "singular_value_mean": 3.5,
            "cumulative_energy_ratio": 25.0 / 30.0,
        },
        {
            "rank": 2,
            "singular_value_mean": 1.5,
            "cumulative_energy_ratio": 1.0,
        },
    ]

def test_lorasb_calibration_truncates_prompt_before_assistant_targets():
    tokenizer = _BoundaryTokenizer(
        prompt_ids=list(range(100)),
        target_ids=list(range(100, 110)),
    )

    example = _supervised_example(
        tokenizer,
        {"question": "long code problem", "solution": "supervised solution"},
        max_length=16,
    )

    assert example["input_ids"].tolist() == [
        [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    ]
    assert example["labels"].tolist() == [
        [-100, -100, -100, -100, -100, -100, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    ]

def test_lorasb_calibration_keeps_a_target_when_solution_exceeds_context():
    tokenizer = _BoundaryTokenizer(
        prompt_ids=list(range(20)),
        target_ids=list(range(20, 40)),
    )

    example = _supervised_example(
        tokenizer,
        {"question": "problem", "solution": "very long solution"},
        max_length=8,
    )

    assert example["input_ids"].tolist() == [list(range(20, 28))]
    assert example["labels"].tolist() == [list(range(20, 28))]

def test_full_evaluator_discovers_atomic_lorasb_r_only_checkpoint(tmp_path):
    checkpoint = (
        tmp_path
        / "voting-qwen3_1.7b-agentwise_lorasb-math"
        / "global_step_10"
        / "agent_lorasb"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "_SUCCESS").write_text("ok\n")
    (checkpoint / "cores.safetensors").write_bytes(b"test-discovery-only")
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "policy_mode": "agentwise_lorasb",
                "global_step": 10,
                "routes": ["generator0", "generator1", "generator2", "aggregator"],
                "basis_path": "/tmp/immutable-lorasb-basis",
            }
        )
        + "\n"
    )

    discovered = discover_checkpoints(str(tmp_path), task_type="math")
    assert len(discovered) == 1
    assert discovered[0].checkpoint_format == "agent_lorasb"
    assert discovered[0].checkpoint_step == 10
    assert discovered[0].agent_names == (
        "generator0",
        "generator1",
        "generator2",
        "aggregator",
    )
    assert discovered[0].basis_path == "/tmp/immutable-lorasb-basis"
