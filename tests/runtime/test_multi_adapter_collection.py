"""CPU regressions for the production collector, with real PEFT filtering.

Only the FSDP gather boundary is mocked. Load the source functions without the
CUDA/Ray worker imports so these tests can also run on a CPU-only workstation.
"""

import ast
import asyncio
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from peft import LoraConfig, get_peft_model


ROOT = Path(__file__).resolve().parents[2]
ROUTES = ["orchestrator", "worker0", "worker1", "worker2", "synthesizer"]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in names]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.fixture
def collector():
    counts = SimpleNamespace(gathers=0, in_gather=False, snapshots=0)

    @contextmanager
    def summon(module, writeback):
        assert writeback is False and not counts.in_gather
        counts.gathers += 1
        counts.in_gather = True
        try:
            yield
        finally:
            counts.in_gather = False

    namespace = dict(OrderedDict=OrderedDict, FSDP=SimpleNamespace(summon_full_params=summon),
                     fsdp_version=lambda module: 1, get_torch_device=lambda: SimpleNamespace(empty_cache=lambda: None),
                     layered_summon_lora_params=Mock())
    functions = load_functions(ROOT / "verl/utils/fsdp_utils.py",
                              ["collect_lora_params", "collect_multi_adapter_lora_params"], namespace)
    return functions, namespace, counts


def make_model(dtype=torch.float32, modules_to_save=False):
    torch.manual_seed(42)
    base = torch.nn.Sequential(OrderedDict([
        ("projection", torch.nn.Linear(8, 8, bias=False)),
        ("head", torch.nn.Linear(8, 8, bias=False)),
    ]))

    def config():
        return LoraConfig(r=4, lora_alpha=2, target_modules=["projection"],
                          modules_to_save=["head"] if modules_to_save else None)

    model = get_peft_model(base, config(), adapter_name=ROUTES[0])
    for route in ROUTES[1:]:
        model.add_adapter(route, config())
    model.to(dtype=dtype)
    # Different, nonzero B weights catch stale, swapped and cross-route exports.
    with torch.no_grad():
        for index, route in enumerate(ROUTES):
            for name, parameter in model.named_parameters():
                if f".{route}." in name:
                    parameter.copy_(torch.randn_like(parameter) * (index + 1))
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("modules_to_save", [False, True])
def test_one_gather_exactly_matches_five_old_exports(collector, dtype, modules_to_save):
    functions, namespace, counts = collector
    model = make_model(dtype, modules_to_save)
    original_state_dict = model.state_dict

    def snapshot(*args, **kwargs):
        assert counts.in_gather
        counts.snapshots += 1
        return original_state_dict(*args, **kwargs)

    model.state_dict = snapshot
    wrapped = SimpleNamespace(_fsdp_wrapped_module=model)
    before = {name: value.clone() for name, value in original_state_dict().items()}
    expected = {route: functions.collect_lora_params(wrapped, False, True, route) for route in ROUTES}
    assert counts.gathers == counts.snapshots == 5
    rng = torch.get_rng_state().clone()
    actual = functions.collect_multi_adapter_lora_params(wrapped, ROUTES, False)
    assert counts.gathers == counts.snapshots == 6
    assert torch.equal(rng, torch.get_rng_state())
    assert list(actual) == ROUTES
    for route in ROUTES:
        assert list(actual[route]) == list(expected[route])
        assert actual[route]
        for name, tensor in actual[route].items():
            assert torch.equal(tensor, expected[route][name])
            assert tensor.dtype == dtype and tensor.device.type == "cpu"
            assert not tensor.requires_grad
    for name, tensor in original_state_dict().items():
        assert torch.equal(tensor, before[name])
    key = next(key for key in actual["worker0"] if "lora_B" in key)
    assert not torch.equal(actual["worker0"][key], actual["worker1"][key])


@pytest.mark.parametrize("version,layered", [(0, False), (2, False), (1, True), (2, True)])
def test_other_collection_modes_keep_existing_path(collector, version, layered):
    functions, namespace, counts = collector
    namespace["fsdp_version"] = lambda module: version
    model = make_model()
    legacy = Mock(side_effect=lambda **kw: {"route": kw["adapter_name"]})
    namespace["collect_lora_params"] = legacy
    wrapped = SimpleNamespace(_fsdp_wrapped_module=model)
    actual = functions.collect_multi_adapter_lora_params(wrapped, ROUTES, layered)
    assert list(actual) == ROUTES
    assert counts.gathers == 0
    assert [call.kwargs["adapter_name"] for call in legacy.call_args_list] == ROUTES
    assert all(call.kwargs["base_sync_done"] is True for call in legacy.call_args_list)
    assert all(call.kwargs["layered_summon"] is layered for call in legacy.call_args_list)


@pytest.mark.parametrize("routes", [[], ["worker0", "worker0"], ["missing"]])
def test_invalid_route_list_fails_before_collective(collector, routes):
    functions, namespace, counts = collector
    with pytest.raises(ValueError):
        functions.collect_multi_adapter_lora_params(SimpleNamespace(_fsdp_wrapped_module=make_model()), routes, False)
    assert counts.gathers == 0


@pytest.mark.parametrize("base_sync_done,sleep_level", [(False, 1), (True, 1), (True, 2)])
def test_worker_keeps_base_sync_route_ids_hashes_and_upload_order(collector, base_sync_done, sleep_level):
    functions, namespace, counts = collector
    model = make_model()
    wrapped = SimpleNamespace(_fsdp_wrapped_module=model)
    events = []

    async def resume(tags):
        events.append(("resume", tags))

    async def update_weights(weights, **kwargs):
        events.append(("upload", kwargs, dict(weights)))

    def collect_base(**kwargs):
        assert kwargs["base_sync_done"] is False
        events.append(("collect_base",))
        return {"base.weight": torch.ones(2)}

    worker = SimpleNamespace(rank=0, actor_module_fsdp=wrapped, base_sync_done=base_sync_done,
                             _is_offload_param=True,
                             config=SimpleNamespace(rollout=SimpleNamespace(
                                 get=lambda name, default: False, free_cache_engine=True)),
                             rollout=SimpleNamespace(sleep_level=sleep_level, resume=resume, update_weights=update_weights),
                             _compute_weight_hash=Mock(return_value=dict(md5="test", num_params=2, num_elements=64)))
    worker_source = ROOT / "verl/workers/fsdp_workers.py"
    tree = ast.parse(worker_source.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
                  and node.name == "_sync_multi_agent_lora_weights")
    env = dict(logger=Mock(), collect_multi_adapter_lora_params=functions.collect_multi_adapter_lora_params,
               collect_lora_params=collect_base, convert_weight_keys=lambda params, model: params,
               replace_lora_wrapper=lambda key, config: key, get_device_id=lambda: "cpu",
               log_gpu_memory_usage=lambda *args, **kwargs: None, set_expandable_segments=lambda *args: None,
               offload_fsdp_model_to_cpu=lambda module: events.append(("offload",)),
               aggressive_empty_cache=lambda **kwargs: None, VLLM_LORA_INT_ID=123,
               DTensor=type("FakeDTensor", (), {}))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(worker_source), "exec"), env)
    asyncio.run(env[method.name](worker, model, ROUTES))
    assert counts.gathers == 1
    assert [call.args[1] for call in worker._compute_weight_hash.call_args_list] == ROUTES
    uploads = [event for event in events if event[0] == "upload"]
    needs_base = not base_sync_done or sleep_level == 2
    assert len(uploads) == 5 + int(needs_base)
    if needs_base:
        assert uploads.pop(0)[1] == {"base_sync_done": False}
    assert events[-1] == ("resume", ["kv_cache"])
    assert events.index(("offload",)) < events.index(("resume", ["weights"]))
    expected = functions.collect_multi_adapter_lora_params(wrapped, ROUTES, False)
    for index, (event, kwargs, weights) in enumerate(uploads):
        route = ROUTES[index]
        assert kwargs == dict(base_sync_done=True, lora_int_id=123 + index, peft_config=model.peft_config[route])
        for name, tensor in weights.items():
            assert torch.equal(tensor, expected[route][name])
