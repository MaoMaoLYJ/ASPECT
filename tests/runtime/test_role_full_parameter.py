"""RS full models: repeated worker slots share one real model and optimizer."""

import ast

import asyncio

from types import SimpleNamespace

import numpy as np

import pytest

import torch

from hydra import compose, initialize_config_module

from omegaconf import OmegaConf

def test_role_training_combines_three_workers_once():
    from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
    from verl import DataProto
    trainer = AgentWorkflowPPOTrainer.__new__(AgentWorkflowPPOTrainer)
    trainer.config = OmegaConf.create({'trainer': {'share_policy': False,
        'agent_wise_full_parameter': False, 'role_wise_full_parameter': True},
        'actor_rollout_ref': {'model': {}}})
    trainer.actor_rollout_wg = SimpleNamespace(world_size=4)
    names = ['x_orchestrator', 'x_worker0', 'x_worker1', 'x_worker2', 'x_synthesizer']
    batch = DataProto.from_dict(tensors={'input_ids': torch.ones(5, 2, dtype=torch.long)},
                               non_tensors={'trajectory_ids': np.asarray(names, dtype=object)})
    groups = trainer._split_batch_by_agent(batch)
    assert list(groups) == ['orchestrator', 'worker', 'synthesizer']
    assert groups['worker'][1] == [1, 2, 3]
    assert len(groups['worker'][0]) == 4  # Existing transparent FSDP padding.
    updates = trainer._agent_update_batches(batch)
    assert [item[0] for item in updates] == ['orchestrator', 'worker', 'synthesizer']
    assert updates[1][2] == [1, 2, 3]

def test_role_rollout_uses_full_weights_and_rejects_instance_routes():
    from rllm.engine.rollout.verl_engine import VerlEngine
    engine = VerlEngine.__new__(VerlEngine)
    engine.config = OmegaConf.create({'trainer': {'share_policy': False,
        'agent_wise_full_parameter': False, 'role_wise_full_parameter': True},
        'actor_rollout_ref': {'model': {'full_parameter_routes': ['orchestrator', 'worker', 'synthesizer']}}})
    engine.validate = False
    engine.accumulate_reasoning = False
    engine.max_prompt_length, engine.max_response_length = 32, 16
    engine.processor = None
    engine.train_sampling_params = {'temperature': .7}
    engine.val_sampling_params = {'temperature': .7}
    engine.chat_parser = SimpleNamespace(parse=lambda *a, **k: 'prompt',
        parse_completion=lambda *a, **k: {'content': 'answer', 'reasoning': None, 'tool_calls': []})
    engine.tokenizer = SimpleNamespace(encode=lambda *a, **k: [1], decode=lambda *a, **k: 'answer')
    calls, released = [], []
    async def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(token_ids=[2], log_probs=[-.1])
    server = SimpleNamespace(generate=SimpleNamespace(remote=generate))
    engine.server_manager = SimpleNamespace(_choose_server=lambda _: server,
                                           _release_server=lambda s: released.append(s))
    for validate in (False, True):
        asyncio.run(engine.get_model_response([{'role': 'user', 'content': 'test'}],
                    agent_name='worker', validate=validate))
    assert all(r['full_parameter_route'] == 'worker' and 'lora_int_id' not in r for r in calls)
    assert len(released) == 2
    with pytest.raises(ValueError):
        asyncio.run(engine.get_model_response([], agent_name='worker1'))

def test_repeated_worker_updates_share_state_but_other_roles_remain_private():
    from verl.workers.actor.full_parameter_bank import FullParameterBank
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.9)
    bank = FullParameterBank(model, optimizer, scheduler, ['orchestrator', 'worker', 'synthesizer'])
    initial = [p.detach().clone() for p in model.parameters()]
    bank.activate('worker')
    for _ in range(3):
        bank.activate('worker')
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2,3)).square().sum().backward()
        optimizer.step()
        scheduler.step()
    trained = [p.detach().clone() for p in model.parameters()]
    assert all(s['step'].item() == 3 for s in optimizer.state.values())
    bank.activate('synthesizer')
    assert not optimizer.state
    assert all(torch.equal(p,q) for p,q in zip(model.parameters(),initial))
    bank.activate('worker')
    assert all(torch.equal(p,q) for p,q in zip(model.parameters(),trained))
    assert all(s['step'].item() == 3 for s in optimizer.state.values())
