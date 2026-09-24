from examples.deepcoder.diagnostics import (
    aggregator_response_metrics,
    code_response_metrics,
    evaluator_response_metrics,
    mean_pairwise_ngram_jaccard,
    output_is_truncated,
    output_token_count,
)
from rllm.workflows.paper_trajectory_diagnostics import (
    aggregate_scalar_or_label_metric,
    contains_parseable_boxed,
    normalized_first_strategy_label,
)


class Output:
    def __init__(
        self,
        content: str,
        token_ids=None,
        completion_ids=None,
        finish_reason=None,
    ):
        self.content = content
        self.token_ids = token_ids
        self.completion_ids = completion_ids
        self.finish_reason = finish_reason


def test_output_token_count_prefers_runtime_token_ids():
    assert output_token_count(
        Output("one two", token_ids=[1, 2, 3], completion_ids=[1, 2, 3, 4])
    ) == 4
    assert output_token_count(Output("one two", token_ids=None)) == 2
    assert output_is_truncated(Output("x", finish_reason="length")) is True


def test_code_metrics_capture_fence_parseability_and_length():
    metrics = code_response_metrics(
        [
            Output("```python\nprint(1)\n```", token_ids=[1, 2]),
            Output("def", token_ids=[1, 2, 3, 4]),
        ],
        prefix="generator",
    )

    assert metrics["generator/code_fence_retention"] == 0.5
    assert metrics["generator/python_parseable"] == 0.5
    assert metrics["generator/response_tokens_p50"] == 3.0
    assert metrics["generator/response_tokens_p95"] == 3.9
    assert metrics["generator/response_tokens_mean"] == 3.0


def test_pairwise_jaccard_is_one_for_identical_responses():
    responses = ["a b c d", "a b c d", "a b c d"]
    assert mean_pairwise_ngram_jaccard(responses, n=3) == 1.0


def test_paper_trajectory_signatures_match_appendix_definitions():
    evaluator = Output(
        "```python\na = 1\nb = 2\nprint(a + b)\n```",
        completion_ids=list(range(12)),
    )
    metrics = evaluator_response_metrics([evaluator], prefix="evaluator")
    assert metrics["evaluator/form_python_code_fence_rate"] == 1.0
    assert metrics["evaluator/form_bare_stamp_rate"] == 0.0
    assert metrics["evaluator/form_other_rate"] == 0.0
    assert metrics["evaluator/verdict_unknown_rate"] == 1.0

    stamp = Output("\\boxed{Correct}", completion_ids=[1, 2, 3])
    metrics = evaluator_response_metrics([stamp], prefix="evaluator")
    assert metrics["evaluator/verdict_tag_retention"] == 1.0
    assert metrics["evaluator/form_bare_stamp_rate"] == 1.0

    aggregator = Output("\\boxed{2}", completion_ids=list(range(6)))
    metrics = aggregator_response_metrics([aggregator], prefix="aggregator")
    assert metrics["aggregator/terse_rate"] == 1.0
    assert contains_parseable_boxed("answer \\boxed{\\frac{1}{2}}") is True


def test_strategy_labels_are_aggregated_as_unique_counts_not_numeric_means():
    label = normalized_first_strategy_label(
        ["Dynamic programming over prefixes\nwith a short explanation"]
    )
    assert label == "dynamic programming over prefixes"
    assert aggregate_scalar_or_label_metric(
        "paper_diag/orchestrator/first_strategy_label",
        [label, label, "greedy invariant"],
    ) == {
        "paper_diag/orchestrator/first_strategy_label_unique_count": 2.0
    }
