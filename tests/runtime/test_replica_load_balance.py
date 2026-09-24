import asyncio

import heapq

from collections import Counter

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

from verl.experimental.agent_loop.route_homogeneous_batcher import RouteHomogeneousBatcher

def manager(gap):
    m = AsyncLLMServerManager.__new__(AsyncLLMServerManager)
    m.weighted_serveres = [[0, (i, object())] for i in range(4)]
    heapq.heapify(m.weighted_serveres)
    m.request_id_to_server = {}
    m.replica_sticky_max_load_gap = gap
    return m

def test_parallel_same_workflow_can_use_all_four_replicas():
    m = manager(0)
    handles = [m._choose_server("one-workflow") for _ in range(12)]
    assert sorted(Counter(handles).values()) == [3, 3, 3, 3]
    for handle in reversed(handles):
        m._release_server(handle)
    assert all(entry[0] == 0 for entry in m.weighted_serveres)

def test_idle_replica_takes_work_while_prefix_affinity_is_preserved_on_ties():
    m = manager(0)
    first = m._choose_server("first")
    m._release_server(first)
    assert m._choose_server("first") is first
    assert m._choose_server("first") is not first

def test_legacy_sticky_behavior_is_unchanged():
    m = manager(-1)
    assert len({m._choose_server("sticky") for _ in range(12)}) == 1

def test_dispatch_preserves_payload_and_releases_counts_on_cancel_and_error():
    from types import SimpleNamespace

    async def run():
        m = manager(0)
        seen = []
        gate = asyncio.Event()

        async def remote(**kwargs):
            seen.append(kwargs)
            if kwargs['lora_int_id'] == 3:
                raise RuntimeError('test backend failure')
            await gate.wait()

        for entry in m.weighted_serveres:
            entry[1] = (entry[1][0], SimpleNamespace(generate=SimpleNamespace(remote=remote)))
        prompt, sampling = [10, 20], {'temperature': 0.7, 'max_tokens': 2048}
        tasks = [asyncio.create_task(m._generate_direct('same', prompt_ids=prompt,
            sampling_params=sampling, image_data=None, lora_int_id=route)) for route in (2, 3)]
        await asyncio.sleep(0)
        tasks[0].cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        assert isinstance(outcomes[0], asyncio.CancelledError)
        assert isinstance(outcomes[1], RuntimeError)
        assert all(entry[0] == 0 for entry in m.weighted_serveres)
        assert [x['lora_int_id'] for x in seen] == [2, 3]
        assert all(x['prompt_ids'] is prompt and x['sampling_params'] is sampling for x in seen)
        assert len({x['request_id'] for x in seen}) == 2
    asyncio.run(run())
