"""Exercise the real rollout bridge with a capturing, non-model backend."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.workflows import TerminationEvent
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServerBase


class CaptureServer:
    def __init__(self, response_length=3):
        self.calls = []
        self.response_length = response_length

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(token_ids=[10] * self.response_length,
                               log_probs=[-0.1] * self.response_length)


def make_engine(*, shared=False, response_length=3, prompt_length=300):
    engine = VerlEngine.__new__(VerlEngine)
    engine.validate = False
    engine.config = SimpleNamespace(trainer={"share_policy": shared,
                                            "agent_names": ["worker0", "worker1"]})
    engine._audited_agent_routes = set()
    engine.accumulate_reasoning = False
    engine.max_prompt_length = 20480
    engine.max_response_length = 5120
    engine.train_sampling_params = {"temperature": 0.7, "top_p": 1.0,
                                    "top_k": -1, "logprobs": 1}
    engine.val_sampling_params = {**engine.train_sampling_params, "temperature": 0.0}
    engine.processor = None
    engine.tokenizer = SimpleNamespace(
        encode=lambda *a, **k: [1] * prompt_length,
        decode=lambda *a, **k: "contract test response")
    engine.chat_parser = SimpleNamespace(
        parse=lambda *a, **k: "contract test prompt",
        parse_completion=lambda *a, **k: {"content": "contract test response",
                                         "reasoning": None, "tool_calls": []})
    engine.server_manager = CaptureServer(response_length)
    return engine


def request(engine, **kwargs):
    return asyncio.run(engine.get_model_response(
        [{"role": "user", "content": "contract test"}],
        agent_name="worker1", application_id="NOT_CANONICAL", **kwargs))


@pytest.mark.parametrize("options,expected", [
    ({}, 5120), ({"max_tokens": 64}, 64), ({"max_new_tokens": 32}, 32),
    ({"max_tokens": 64, "max_new_tokens": 32}, 64),
])
@pytest.mark.parametrize("validation_mode", ["training", "request", "engine"])
def test_budget_reaches_backend_without_changing_sampling_or_route(options, expected, validation_mode):
    engine = make_engine()
    train_before = engine.train_sampling_params.copy()
    val_before = engine.val_sampling_params.copy()
    engine.validate = validation_mode == "engine"
    request(engine, validate=validation_mode == "request", **options)
    call, = engine.server_manager.calls
    expected_sampling = train_before if validation_mode == "training" else val_before
    assert call["sampling_params"] == {**expected_sampling, "max_tokens": expected}
    assert call["prompt_ids"] == [1] * 300
    assert call["lora_int_id"] == VLLM_LORA_INT_ID + 1
    assert call["request_id"] == "NOT_CANONICAL"
    assert engine.train_sampling_params == train_before
    assert engine.val_sampling_params == val_before


def test_shared_policy_route_and_per_request_budget_are_preserved():
    engine = make_engine(shared=True)
    request(engine, max_tokens=16)
    request(engine)
    first, second = engine.server_manager.calls
    assert first["lora_int_id"] is None and second["lora_int_id"] is None
    assert first["sampling_params"]["max_tokens"] == 16
    assert second["sampling_params"]["max_tokens"] == 5120


@pytest.mark.parametrize("length,expected,reason", [(3, 3, "stop"), (64, 64, "length"), (80, 64, "length")])
def test_existing_completion_and_logprob_contract_is_unchanged(length, expected, reason):
    engine = make_engine(response_length=length)
    result = request(engine, max_tokens=64)
    assert result.completion_ids == [10] * expected
    assert result.logprobs == [-0.1] * expected
    assert result.completion_length == expected
    assert result.finish_reason == reason


def test_overlong_prompt_is_rejected_before_dispatch():
    engine = make_engine(prompt_length=20481)
    with pytest.raises(TerminationEvent):
        request(engine)
    assert engine.server_manager.calls == []


class CaptureVLLM:
    async def list_loras(self):
        return [VLLM_LORA_INT_ID, VLLM_LORA_INT_ID + 1]

    async def generate(self, **kwargs):
        self.received = kwargs
        output = SimpleNamespace(token_ids=[10],
                                 logprobs=[{10: SimpleNamespace(logprob=-0.1)}])
        yield SimpleNamespace(outputs=[output])


@pytest.mark.parametrize("options,prompt_length,expected", [
    ({}, 300, 5120), ({"max_tokens": 64}, 300, 64),
    ({"max_new_tokens": 32}, 300, 32), ({}, 25480, 120),
])
def test_real_async_server_builds_bounded_vllm_sampling_params(options, prompt_length, expected):
    engine = make_engine(prompt_length=prompt_length)
    server = vLLMHttpServerBase.__new__(vLLMHttpServerBase)
    server.config = OmegaConf.create({"max_model_len": 25600, "repetition_penalty": 1.0})
    server.model_config = SimpleNamespace(processor=None, lora_rank=64)
    server._audited_explicit_lora_routes = set()
    server.engine = CaptureVLLM()
    engine.server_manager = server
    result = request(engine, enforce_max_prompt_length=False, **options)
    received = server.engine.received
    assert received["sampling_params"].max_tokens == expected
    assert received["sampling_params"].temperature == 0.7
    assert received["sampling_params"].logprobs == 0
    assert received["lora_request"].lora_int_id == VLLM_LORA_INT_ID + 1
    assert result.completion_ids == [10] and result.logprobs == [-0.1]
