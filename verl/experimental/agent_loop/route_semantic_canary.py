"""Deterministic audits for adapter-routed rollout scheduling."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

BFLOAT16_ROUTE_CANARY_LOGPROB_ATOL = 8 * 2**-7


class RouteSemanticCanaryError(RuntimeError):
    """Fatal malformed output from a routed generation canary."""

    fatal_workflow_error = True


def token_fingerprint(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def deterministic_canary_sampling_params(
    sampling_params: dict[str, Any], *, max_tokens: int
) -> dict[str, Any]:
    if max_tokens <= 0:
        raise ValueError("route semantic canary max_tokens must be positive")
    params = dict(sampling_params)
    params.update(
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
            "min_tokens": 0,
            "max_tokens": max_tokens,
            "logprobs": True,
        }
    )
    return params


def compare_route_canary_outputs(
    direct: Any, batched: Any, *, logprob_atol: float
) -> dict[str, Any]:
    if logprob_atol < 0:
        raise ValueError("route semantic canary logprob_atol cannot be negative")
    direct_ids = list(direct.token_ids)
    batched_ids = list(batched.token_ids)
    direct_logprobs = direct.log_probs
    batched_logprobs = batched.log_probs
    if direct_logprobs is None or batched_logprobs is None:
        raise RouteSemanticCanaryError(
            "Route semantic canary requires logprobs from both paths"
        )
    direct_logprobs = list(direct_logprobs)
    batched_logprobs = list(batched_logprobs)
    if len(direct_logprobs) != len(direct_ids):
        raise RouteSemanticCanaryError(
            "Route semantic canary direct output/logprob length mismatch: "
            f"tokens={len(direct_ids)} logprobs={len(direct_logprobs)}"
        )
    if len(batched_logprobs) != len(batched_ids):
        raise RouteSemanticCanaryError(
            "Route semantic canary batched output/logprob length mismatch: "
            f"tokens={len(batched_ids)} logprobs={len(batched_logprobs)}"
        )
    for path, values in (
        ("direct", direct_logprobs),
        ("batched", batched_logprobs),
    ):
        if any(not math.isfinite(float(value)) for value in values):
            raise RouteSemanticCanaryError(
                f"Route semantic canary encountered non-finite {path} logprobs"
            )

    common_prefix_tokens = 0
    for direct_id, batched_id in zip(direct_ids, batched_ids):
        if direct_id != batched_id:
            break
        common_prefix_tokens += 1
    token_match = direct_ids == batched_ids
    comparable_tokens = len(direct_ids) if token_match else common_prefix_tokens
    deltas = []
    for direct_value, batched_value in zip(
        direct_logprobs[:comparable_tokens],
        batched_logprobs[:comparable_tokens],
        strict=True,
    ):
        direct_value = float(direct_value)
        batched_value = float(batched_value)
        if direct_value == batched_value:
            deltas.append(0.0)
            continue
        deltas.append(abs(direct_value - batched_value))
    max_delta = max(deltas, default=0.0)
    mean_delta = math.fsum(deltas) / len(deltas) if deltas else 0.0
    ordered_deltas = sorted(deltas)
    p95_index = max(0, math.ceil(0.95 * len(ordered_deltas)) - 1)
    p95_delta = ordered_deltas[p95_index] if ordered_deltas else 0.0
    logprob_within_atol = max_delta <= logprob_atol
    numerical_warning = not token_match or not logprob_within_atol
    warning_reasons = []
    if not token_match:
        warning_reasons.append("finite_token_drift")
    if not logprob_within_atol:
        warning_reasons.append("finite_logprob_drift")
    return {
        "token_count": len(direct_ids),
        "token_sha256": token_fingerprint(direct_ids),
        "direct_token_count": len(direct_ids),
        "batched_token_count": len(batched_ids),
        "direct_token_sha256": token_fingerprint(direct_ids),
        "batched_token_sha256": token_fingerprint(batched_ids),
        "token_match": token_match,
        "common_prefix_tokens": common_prefix_tokens,
        "comparable_logprob_tokens": comparable_tokens,
        "max_logprob_delta": max_delta,
        "mean_logprob_delta": mean_delta,
        "p95_logprob_delta": p95_delta,
        "logprob_atol": logprob_atol,
        "logprob_within_atol": logprob_within_atol,
        "numerical_warning": numerical_warning,
        "warning_reasons": warning_reasons,
    }
