"""Non-model regressions for request admission and final-result delivery."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from vllm.sampling_params import RequestOutputKind

from verl.experimental.agent_loop.route_homogeneous_batcher import RouteHomogeneousBatcher
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServerBase


@pytest.mark.parametrize("late_route", ["worker0", "worker1"])
def test_late_arrival_uses_free_slot_before_slow_request_finishes(late_route):
    async def scenario():
        batcher = RouteHomogeneousBatcher(max_batch_size=2, max_requests_per_turn=4, coalesce_ms=0)
        slow_started, release_slow, late_started = (asyncio.Event() for _ in range(3))

        async def slow(route):
            assert route == "worker0"
            slow_started.set()
            await release_slow.wait()
            return (route, "slow")

        async def late(route):
            assert route == late_route
            late_started.set()
            return (route, "late")

        first = asyncio.create_task(batcher.submit("worker0", slow))
        await slow_started.wait()
        second = asyncio.create_task(batcher.submit(late_route, late))
        try:
            await asyncio.wait_for(late_started.wait(), 0.5)
            assert not first.done()
        finally:
            release_slow.set()
            results = await asyncio.gather(first, second)
        assert results == [("worker0", "slow"), (late_route, "late")]

    asyncio.run(scenario())


@pytest.mark.parametrize("finish", ["length", "stop"])
@pytest.mark.parametrize("logprobs", [True, False])
def test_real_vllm_output_processor_preserves_tokens_probabilities_and_termination(finish, logprobs):
    from scripts.repro.benchmark_rollout_delivery import replay_output, toy_tokenizer

    tokenizer = toy_tokenizer()
    tokens = [1, 2, 1, 2, 4]
    before = replay_output(tokenizer, tokens, final_only=False, detokenize=True, finish=finish, logprobs=logprobs)
    after = replay_output(tokenizer, tokens, final_only=True, detokenize=False, finish=finish, logprobs=logprobs)
    assert before["contract"] == after["contract"]
    assert before["deliveries"] == len(tokens)
    assert after["deliveries"] == 1


def test_real_vllm_string_stop_still_terminates_before_remaining_tokens():
    from scripts.repro.benchmark_rollout_delivery import replay_output, toy_tokenizer

    tokenizer = toy_tokenizer()
    before = replay_output(tokenizer, [1, 3, 2, 4], final_only=False, detokenize=True, stop=["STOP"])
    after = replay_output(tokenizer, [1, 3, 2, 4], final_only=True, detokenize=True, stop=["STOP"])
    assert before["contract"] == after["contract"]
    assert after["contract"]["token_ids"] == [1, 3]
    assert after["contract"]["stop_reason"] == "STOP"
    assert after["deliveries"] == 1


def test_late_route_respects_fifo_route_and_global_caps():
    async def scenario():
        batcher = RouteHomogeneousBatcher(max_batch_size=1, max_requests_per_turn=2, coalesce_ms=0)
        starts, active, peak = [], {}, 0
        release = {name: asyncio.Event() for name in ("a0", "a1", "b0", "b1")}
        started = {name: asyncio.Event() for name in release}

        async def invoke(route, name):
            nonlocal peak
            starts.append(name)
            active[route] = active.get(route, 0) + 1
            peak = max(peak, sum(active.values()))
            assert active[route] == 1 and sum(active.values()) <= 2
            started[name].set()
            await release[name].wait()
            active[route] -= 1
            return name

        tasks = [asyncio.create_task(batcher.submit("a", lambda route: invoke(route, "a0")))]
        await started["a0"].wait()
        for route, name in (("a", "a1"), ("b", "b0"), ("b", "b1")):
            tasks.append(asyncio.create_task(batcher.submit(route, lambda r, n=name: invoke(r, n))))
        try:
            await asyncio.wait_for(started["b0"].wait(), 0.5)
            assert starts == ["a0", "b0"]
            release["b0"].set()
            await asyncio.wait_for(started["b1"].wait(), 0.5)
            assert starts == ["a0", "b0", "b1"]
        finally:
            for event in release.values():
                event.set()
            assert await asyncio.gather(*tasks) == ["a0", "a1", "b0", "b1"]
        assert peak == 2
        assert [name for name in starts if name.startswith("a")] == ["a0", "a1"]

    asyncio.run(scenario())


def test_request_errors_cancellation_and_idle_restart_remain_isolated():
    async def scenario():
        batcher = RouteHomogeneousBatcher(max_batch_size=1, max_requests_per_turn=2, coalesce_ms=0)
        started, release = asyncio.Event(), asyncio.Event()
        called = []

        async def slow(route):
            started.set()
            await release.wait()
            return route

        async def fail(route):
            raise ValueError("backend failure")

        async def good(route):
            called.append(route)
            return route

        first = asyncio.create_task(batcher.submit("a", slow))
        await started.wait()
        cancelled = asyncio.create_task(batcher.submit("a", good))
        await asyncio.sleep(0)
        cancelled.cancel()
        try:
            with pytest.raises(ValueError, match="backend failure"):
                await asyncio.wait_for(batcher.submit("b", fail), 0.5)
            assert await asyncio.wait_for(batcher.submit("c", good), 0.5) == "c"
        finally:
            release.set()
            await first
            await asyncio.gather(cancelled, return_exceptions=True)
        assert await batcher.submit("d", good) == "d"
        assert called == ["c", "d"]
        for _ in range(5):
            await asyncio.sleep(0)
        assert batcher._dispatcher is None

    asyncio.run(scenario())


class CaptureEngine:
    async def list_loras(self):
        return [123, 124]

    async def generate(self, **kwargs):
        self.received = kwargs
        params = kwargs["sampling_params"]
        assert params.output_kind == RequestOutputKind.FINAL_ONLY
        output = SimpleNamespace(token_ids=[7, 8], logprobs=[
            {7: SimpleNamespace(logprob=-0.25)}, {8: SimpleNamespace(logprob=-0.5)}])
        yield SimpleNamespace(outputs=[output])


@pytest.mark.parametrize("route", [None, 123, 124])
@pytest.mark.parametrize("stop", [None, ["STOP"]])
def test_token_backend_requests_final_only_without_changing_generation(route, stop):
    async def scenario():
        server = vLLMHttpServerBase.__new__(vLLMHttpServerBase)
        server.config = OmegaConf.create({"max_model_len": 25600, "repetition_penalty": 1.0})
        server.model_config = SimpleNamespace(processor=None, lora_rank=64)
        server._audited_explicit_lora_routes = set()
        server.engine = CaptureEngine()
        params = {"max_tokens": 5120, "temperature": 0.7, "top_p": 0.9,
                  "top_k": -1, "logprobs": 1, "seed": 42, "stop": stop}
        before = params.copy()
        output = await server.generate([1, 2], params, "NOT_CANONICAL", lora_int_id=route)
        actual = server.engine.received["sampling_params"]
        for name in ("max_tokens", "temperature", "top_p", "top_k", "seed"):
            assert getattr(actual, name) == before[name]
        assert actual.stop == (stop or [])
        assert actual.detokenize is bool(stop)
        assert actual.logprobs == 0
        assert server.engine.received["lora_request"].lora_int_id == (route or 123)
        assert params == before
        assert output.token_ids == [7, 8] and output.log_probs == [-0.25, -0.5]

    asyncio.run(scenario())


def test_explicit_detokenization_request_is_preserved():
    async def scenario():
        server = vLLMHttpServerBase.__new__(vLLMHttpServerBase)
        server.config = OmegaConf.create({"max_model_len": 25600})
        server.model_config = SimpleNamespace(processor=None, lora_rank=0)
        server.engine = CaptureEngine()
        await server.generate([1], {"max_tokens": 10, "detokenize": True}, "explicit")
        assert server.engine.received["sampling_params"].detokenize is True

    asyncio.run(scenario())


@pytest.mark.parametrize("validate", [False, True])
def test_workflow_phase_timings_do_not_change_results_or_execution_order(monkeypatch, validate):
    import numpy as np
    import rllm.engine.agent_workflow_engine as module

    async def scenario():
        engine = module.AgentWorkflowEngine.__new__(module.AgentWorkflowEngine)
        order = []
        output = SimpleNamespace(meta_info={"repeat_counts": [1]}, batch=object())

        async def wake():
            order.append("wake")

        async def sleep():
            order.append("sleep")

        async def execute(tasks, ids, **kwargs):
            order.append("execute")
            assert tasks == [{"q": 1}] and ids == ["task"]
            assert engine.rollout_engine.validate is validate
            return ["exact episode"]

        def transform(results, ids):
            order.append("transform")
            assert results == ["exact episode"] and ids == ["task"]
            return output

        clock = iter([0.0, 2.0, 12.0, 14.0, 17.0])
        monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
        engine.rollout_engine = SimpleNamespace(wake_up=wake, sleep=sleep, validate=False)
        engine.execute_tasks = execute
        engine.transform_results_for_verl = transform
        batch = SimpleNamespace(meta_info={"validate": validate}, non_tensor_batch={
            "extra_info": np.array([{"q": 1}], dtype=object), "task_ids": np.array(["task"])})
        assert await engine.execute_tasks_verl(batch) is output
        assert output.meta_info == {"repeat_counts": [1], "workflow_timing": {
            "workflow_wake_sync": 2.0, "workflow_execute": 10.0,
            "workflow_sleep": 2.0, "workflow_transform": 3.0}}
        assert order == ["wake", "execute", "sleep", "transform"]
        assert engine.rollout_engine.validate is False
        assert engine.current_mode == "train"

    asyncio.run(scenario())


@pytest.mark.parametrize("phase_metrics", [{}, {"workflow_execute": 2.5}])
def test_trainer_consumes_phase_metadata_once_and_keeps_existing_timing(monkeypatch, phase_metrics):
    import rllm.trainer.verl.agent_workflow_trainer as module

    trainer = module.AgentWorkflowPPOTrainer.__new__(module.AgentWorkflowPPOTrainer)
    trainer._loop = object()
    output = SimpleNamespace(meta_info={"repeat_counts": [1], "workflow_timing": phase_metrics.copy()})

    async def execute(batch, **kwargs):
        assert batch == "input"
        return output

    trainer.agent_execution_engine = SimpleNamespace(execute_tasks_verl=execute)
    monkeypatch.setattr(module.asyncio, "run_coroutine_threadsafe", lambda coro, loop: SimpleNamespace(result=lambda: asyncio.run(coro)))
    timing = {"workflow_execute": 1.0}
    assert trainer.generate_trajectories("input", timing_raw=timing) is output
    assert output.meta_info == {"repeat_counts": [1]}
    assert timing["workflow_execute"] == 1.0 + phase_metrics.get("workflow_execute", 0.0)
    assert timing["generate_trajectories"] >= 0.0
