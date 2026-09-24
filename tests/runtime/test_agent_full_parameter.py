import asyncio

import pytest

import torch

from verl.workers.actor.full_parameter_bank import FullParameterBank, TensorBank

from verl.experimental.agent_loop.full_parameter_batcher import FullParameterBatcher

def test_weights_optimizer_scheduler_are_private():
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    bank = FullParameterBank(model, optimizer, scheduler, ['worker1', 'worker2', 'worker3'])
    initial = [p.clone() for p in model.parameters()]
    optimizer.zero_grad()
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    trained = [p.clone() for p in model.parameters()]
    bank.activate('worker2')
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), initial))
    assert not optimizer.state
    assert scheduler.get_last_lr() == [0.01]
    bank.activate('worker3')
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), initial))
    bank.activate('worker1')
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), trained))
    assert optimizer.state and scheduler.get_last_lr() == [pytest.approx(0.009)]
    with pytest.raises(ValueError):
        bank.activate('worker')

def test_rollout_tensor_bank_switches_in_place_without_aliases():
    model = torch.nn.Linear(3, 2)
    bank = TensorBank(model)
    bank.capture('a')
    original = model.weight.clone()
    pointer = model.weight.data_ptr()
    with torch.no_grad():
        model.weight.add_(1)
    bank.capture('b')
    bank.activate('a')
    assert torch.equal(model.weight, original)
    assert model.weight.data_ptr() == pointer
    bank.activate('b')
    assert torch.equal(model.weight, original + 1)
    with pytest.raises(ValueError):
        bank.activate('missing')

def test_interleaved_full_updates_equal_standalone_models():
    import copy
    torch.manual_seed(42)
    shared = torch.nn.Linear(3, 2)
    originals = {r: copy.deepcopy(shared) for r in ('a', 'b')}
    opt = torch.optim.AdamW(shared.parameters(), lr=2e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
    bank = FullParameterBank(shared, opt, sched, list(originals))
    separate = {r: torch.optim.AdamW(m.parameters(), lr=2e-5) for r, m in originals.items()}
    schedules = {r: torch.optim.lr_scheduler.StepLR(o, 1, gamma=0.9) for r, o in separate.items()}
    for i, route in enumerate(('a', 'b', 'b', 'a', 'b', 'a')):
        inputs = torch.randn(4, 3)
        bank.activate(route)
        for model, optimizer, scheduler in ((shared, opt, sched), (originals[route], separate[route], schedules[route])):
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().sum().backward()
            optimizer.step()
            scheduler.step()
        assert all(torch.equal(x, y) for x, y in zip(shared.parameters(), originals[route].parameters()))

def test_full_sync_wakes_weights_and_kv_once():
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / 'verl/workers/fsdp_workers.py').read_text()
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'rollout_mode')
    private = next(n for n in ast.walk(method) if isinstance(n, ast.If) and 'full_parameter_bank' in ast.unparse(n.test))
    loop = next(n for n in private.body if isinstance(n, ast.For))
    assert 'resume' not in ast.unparse(loop)
    assert 'manage_rollout_memory=False' in ast.unparse(loop)
    assert ast.unparse(private).count('self.rollout.resume') == 2

def test_full_route_batches_never_overlap_or_switch_active_requests():
    async def run():
        active = set()
        selected = None
        switches = []

        async def activate(route):
            nonlocal selected
            assert not active
            selected = route
            switches.append(route)

        batcher = FullParameterBatcher(activate, max_batch_size=3)

        async def invoke(route, i):
            assert selected == route
            active.add(i)
            await asyncio.sleep(0.001)
            assert selected == route
            active.remove(i)
            return route, i

        calls = [batcher.submit(str(i % 3), lambda route, i=i: invoke(route, i)) for i in range(12)]
        assert await asyncio.gather(*calls) == [(str(i % 3), i) for i in range(12)]
        assert set(switches) == {'0', '1', '2'}
    asyncio.run(run())
