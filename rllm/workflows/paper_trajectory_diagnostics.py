"""Lightweight trajectory-shape diagnostics for paper-facing MARL runs.

The helpers in this module consume only in-memory rollout outputs and return
small scalar summaries. They never retain prompts, responses, token IDs, or
model tensors, so enabling them does not change optimization or the persistent
storage contract.
"""

from __future__ import annotations

import ast
import itertools
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

_CODE_FENCE = re.compile(
    r"```(?:python|py)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_PYTHON_CODE_FENCE = re.compile(
    r"```(?:python|py)\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_VERDICT = re.compile(r"\\boxed\{\s*(Correct|Incorrect)\s*\}", re.IGNORECASE)
_HEDGING_PHRASES = (
    "wait",
    "alternatively",
    "actually",
    "hmm",
    "let me reconsider",
    "on second thought",
    "not correct",
    "this is wrong",
)
_HEDGING = re.compile(
    r"(?:^|\W)(?:"
    + "|".join(re.escape(phrase) for phrase in _HEDGING_PHRASES)
    + r")(?:$|\W)",
    re.IGNORECASE,
)


def output_text(output: Any) -> str:
    """Return the generated text without serializing the rollout object."""

    for attribute in ("text", "content"):
        value = getattr(output, attribute, None)
        if isinstance(value, str) and value:
            return value
    return "" if output is None else str(output)


def output_token_count(output: Any) -> int:
    """Return the exact completion length when the rollout engine exposes it."""

    for attribute in ("completion_ids", "token_ids", "response_ids"):
        token_ids = getattr(output, attribute, None)
        if token_ids is not None:
            return len(token_ids)
    completion_length = getattr(output, "completion_length", None)
    if completion_length is not None and int(completion_length) > 0:
        return int(completion_length)
    return len(output_text(output).split())


def output_is_truncated(output: Any) -> bool:
    """Match vLLM's length-termination signal and legacy rollout flags."""

    finish_reason = str(getattr(output, "finish_reason", "") or "").lower()
    return finish_reason == "length" or bool(getattr(output, "truncated", False))


def _quantile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def contains_parseable_boxed(text: str) -> bool:
    """Return whether ``text`` contains a non-empty balanced ``\\boxed{...}``."""

    start = 0
    marker = "\\boxed{"
    while True:
        marker_index = text.find(marker, start)
        if marker_index < 0:
            return False
        content_start = marker_index + len(marker)
        depth = 1
        for index in range(content_start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    if text[content_start:index].strip():
                        return True
                    break
        start = content_start


def _extract_python(text: str) -> tuple[str, bool]:
    match = _CODE_FENCE.search(text)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def _is_parseable_python(text: str) -> bool:
    source, _ = _extract_python(text)
    if not source:
        return False
    try:
        ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


def text_response_metrics(outputs: Sequence[Any], *, prefix: str) -> dict[str, float]:
    """Summarize role-level response shape using only scalar outputs."""

    if not outputs:
        return {}
    lengths = [output_token_count(output) for output in outputs]
    texts = [output_text(output) for output in outputs]
    char_lengths = [len(text) for text in texts]
    return {
        f"{prefix}/response_count": float(len(outputs)),
        f"{prefix}/response_tokens_mean": float(np.mean(lengths)),
        f"{prefix}/response_tokens_min": float(min(lengths)),
        f"{prefix}/response_tokens_max": float(max(lengths)),
        f"{prefix}/response_tokens_p50": _quantile(lengths, 0.50),
        f"{prefix}/response_tokens_p95": _quantile(lengths, 0.95),
        f"{prefix}/response_chars_mean": float(np.mean(char_lengths)),
        f"{prefix}/response_chars_p50": _quantile(char_lengths, 0.50),
        f"{prefix}/response_chars_p95": _quantile(char_lengths, 0.95),
        f"{prefix}/truncation_rate": float(
            np.mean([output_is_truncated(output) for output in outputs])
        ),
        f"{prefix}/boxed_retention": float(
            np.mean([contains_parseable_boxed(text) for text in texts])
        ),
        f"{prefix}/hedging_rate": float(
            np.mean([bool(_HEDGING.search(text)) for text in texts])
        ),
    }


def code_response_metrics(outputs: Sequence[Any], *, prefix: str) -> dict[str, float]:
    """Add code-format and Python syntax health to response-shape metrics."""

    metrics = text_response_metrics(outputs, prefix=prefix)
    if not outputs:
        return metrics
    texts = [output_text(output) for output in outputs]
    metrics.update(
        {
            f"{prefix}/code_fence_retention": float(
                np.mean([bool(_CODE_FENCE.search(text)) for text in texts])
            ),
            f"{prefix}/python_parseable": float(
                np.mean([_is_parseable_python(text) for text in texts])
            ),
        }
    )
    return metrics


def evaluator_response_metrics(
    outputs: Sequence[Any],
    *,
    prefix: str,
) -> dict[str, float]:
    """Summarize evaluator verdict retention and role-capture surface forms."""

    metrics = text_response_metrics(outputs, prefix=prefix)
    if not outputs:
        return metrics
    texts = [output_text(output) for output in outputs]
    token_lengths = [output_token_count(output) for output in outputs]
    verdicts = []
    for text in texts:
        match = _VERDICT.search(text)
        verdicts.append(match.group(1).lower() if match else "unknown")

    python_fence = []
    bare_stamp = []
    for text, token_length, verdict in zip(texts, token_lengths, verdicts, strict=True):
        match = _PYTHON_CODE_FENCE.search(text)
        body_lines = (
            [line for line in match.group(1).splitlines() if line.strip()]
            if match
            else []
        )
        is_python_fence = len(body_lines) >= 3
        python_fence.append(is_python_fence)
        bare_stamp.append(
            verdict != "unknown"
            and not bool(_CODE_FENCE.search(text))
            and token_length <= 200
        )

    metrics.update(
        {
            f"{prefix}/verdict_tag_retention": float(
                np.mean([verdict != "unknown" for verdict in verdicts])
            ),
            f"{prefix}/verdict_correct_rate": float(
                np.mean([verdict == "correct" for verdict in verdicts])
            ),
            f"{prefix}/verdict_incorrect_rate": float(
                np.mean([verdict == "incorrect" for verdict in verdicts])
            ),
            f"{prefix}/verdict_unknown_rate": float(
                np.mean([verdict == "unknown" for verdict in verdicts])
            ),
            f"{prefix}/form_python_code_fence_rate": float(np.mean(python_fence)),
            f"{prefix}/form_bare_stamp_rate": float(np.mean(bare_stamp)),
            f"{prefix}/form_other_rate": float(
                np.mean(
                    [
                        not is_python_fence and not is_bare_stamp
                        for is_python_fence, is_bare_stamp in zip(
                            python_fence, bare_stamp, strict=True
                        )
                    ]
                )
            ),
        }
    )
    return metrics


def aggregator_response_metrics(
    outputs: Sequence[Any],
    *,
    prefix: str,
) -> dict[str, float]:
    """Summarize whether a Voting aggregator keeps its terse selection role."""

    metrics = text_response_metrics(outputs, prefix=prefix)
    if not outputs:
        return metrics
    metrics[f"{prefix}/terse_rate"] = float(
        np.mean(
            [
                output_token_count(output) <= 30
                and contains_parseable_boxed(output_text(output))
                for output in outputs
            ]
        )
    )
    return metrics


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = text.split()
    return {
        tuple(tokens[index : index + n])
        for index in range(max(0, len(tokens) - n + 1))
    }


def mean_pairwise_ngram_jaccard(
    responses: Iterable[str],
    *,
    n: int = 3,
) -> float:
    """Average pairwise n-gram overlap within one workflow episode."""

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    grams = [_ngrams(str(response), n) for response in responses]
    pairs = list(itertools.combinations(grams, 2))
    if not pairs:
        return 1.0
    similarities = []
    for left, right in pairs:
        union = left | right
        similarities.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(similarities))


def normalized_first_strategy_label(strategies: Sequence[str]) -> str:
    """Return a bounded, deterministic label for batch-level diversity counts."""

    if not strategies:
        return ""
    lines = str(strategies[0]).strip().splitlines()
    if not lines:
        return ""
    first_line = lines[0]
    normalized = re.sub(r"\s+", " ", first_line).strip().lower()
    return " ".join(normalized.split()[:16])


def aggregate_scalar_or_label_metric(
    key: str,
    values: Sequence[Any],
) -> dict[str, float]:
    """Aggregate workflow metrics while keeping label strings out of logs."""

    if key.endswith("/first_strategy_label"):
        labels = {str(value) for value in values if str(value)}
        return {f"{key}_unique_count": float(len(labels))}
    numeric = [float(value) for value in values]
    return {key: float(np.mean(numeric))} if numeric else {}
