"""Small, dependency-free helpers for selecting PEFT adapters."""

from collections.abc import Collection


def validate_peft_adapter_route(
    *,
    requested: str,
    peft_adapters: Collection[str],
    configured_roles: Collection[str] = (),
) -> None:
    """Validate against adapters actually installed in PEFT.

    ``trainer.agent_names`` describes workflow roles. In shared-policy mode those
    roles intentionally map to PEFT's single ``default`` adapter, so role names
    are not a valid source of truth for checkpoint routing.
    """

    available = sorted(str(adapter) for adapter in peft_adapters)
    if requested in available:
        return
    roles = sorted(str(role) for role in configured_roles)
    raise ValueError(
        f"Adapter {requested!r} not found. Available PEFT adapters: {available}; "
        f"configured workflow roles: {roles}"
    )
