"""Launch vLLM with a narrow Starlette routing compatibility patch."""

from __future__ import annotations

import runpy
import os
from collections.abc import Callable
from typing import Any


def patch_instrumentator_route_name() -> bool:
    """Handle Starlette's pathless ``_IncludedRouter`` in HTTP metrics."""
    try:
        from prometheus_fastapi_instrumentator import routing
    except ImportError:
        return False

    original: Callable[..., Any] = routing._get_route_name
    if getattr(original, "_vllm_included_router_compat", False):
        return True

    def compatible_get_route_name(scope, routes, route_name=None):
        try:
            return original(scope, routes, route_name)
        except AttributeError as exc:
            message = str(exc)
            if "'_IncludedRouter'" not in message or "has no attribute 'path'" not in message:
                raise
            route = scope.get("route")
            return getattr(route, "path", None) or scope.get("path") or route_name

    setattr(compatible_get_route_name, "_vllm_included_router_compat", True)
    routing._get_route_name = compatible_get_route_name
    return True


def main() -> None:
    port_base = os.environ.get('UNITYMAS_VLLM_DP_PORT_BASE')
    if port_base is not None:
        from dashboard.vllm_dp_ports import install_port_contract, WIDTH
        install_port_contract(port_base)
        print(f'[vllm-dp-ports] leased={port_base}..{int(port_base)+WIDTH-1}', flush=True)
    if patch_instrumentator_route_name():
        print("[vllm-compat] enabled pathless Starlette router handling", flush=True)
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
