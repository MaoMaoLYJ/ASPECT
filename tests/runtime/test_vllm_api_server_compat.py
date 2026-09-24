from prometheus_fastapi_instrumentator import routing
from starlette.routing import Match

from dashboard.vllm_api_server_compat import patch_instrumentator_route_name


class _IncludedRouter:
    def matches(self, scope):
        return Match.FULL, {}


def test_pathless_included_router_falls_back_to_request_path():
    original = routing._get_route_name
    try:
        assert patch_instrumentator_route_name()
        assert routing._get_route_name({"path": "/health"}, [_IncludedRouter()]) == "/health"
    finally:
        routing._get_route_name = original
