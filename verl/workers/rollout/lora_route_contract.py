"""Fail-closed contracts for explicit per-agent LoRA routing."""

from __future__ import annotations

from collections.abc import Iterable


def require_explicit_lora_route_loaded(
    lora_int_id: int | None,
    loaded_lora_ids: Iterable[int],
    *,
    model_lora_rank: int,
) -> None:
    if lora_int_id is None:
        return
    if model_lora_rank <= 0:
        raise RuntimeError(
            f"Explicit LoRA route {lora_int_id} was requested with LoRA disabled"
        )
    loaded = set(loaded_lora_ids)
    if lora_int_id not in loaded:
        raise RuntimeError(
            "Explicit LoRA route is not loaded; refusing base-model fallback: "
            f"requested={lora_int_id} loaded={sorted(loaded)}"
        )
