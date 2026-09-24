"""Concurrent same-role workers must not reuse a vLLM generation request ID."""
import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rllm.engine.rollout.verl_engine import VerlEngine
from verl.experimental.agent_loop.full_parameter_batcher import FullParameterBatcher


@pytest.mark.parametrize('validate', [False, True])
@pytest.mark.parametrize('scope', ['role', 'agent'])
def test_parallel_worker_ids_unique_without_changing_sticky_routing(scope, validate):
    async def run():
        calls, sticky, released, active = [], [], [], set()
        route = 'worker' if scope == 'role' else 'worker0'
        async def activate(selected):
            assert not active and selected == route
        batcher = FullParameterBatcher(activate)
        async def backend(**kwargs):
            request_id = kwargs['request_id']
            if request_id in active:
                raise ValueError(f'Request id {request_id} already running.')
            active.add(request_id)
            calls.append(kwargs)
            try:
                await asyncio.sleep(.001)
                return SimpleNamespace(token_ids=[2], log_probs=[-.1])
            finally:
                active.remove(request_id)
        async def generate(**kwargs):
            return await batcher.submit(kwargs['full_parameter_route'], lambda _: backend(**kwargs))
        server = SimpleNamespace(generate=SimpleNamespace(remote=generate))
        engine = VerlEngine.__new__(VerlEngine)
        engine.config = OmegaConf.create({'trainer': {'share_policy': False,
            'role_wise_full_parameter': scope == 'role', 'agent_wise_full_parameter': scope == 'agent'},
            'actor_rollout_ref': {'model': {'full_parameter_routes': [route]}}})
        engine.validate = validate
        engine.accumulate_reasoning = False
        engine.max_prompt_length, engine.max_response_length = 32, 16
        engine.processor = None
        engine.train_sampling_params = {'temperature': .7, 'top_p': 1.0}
        engine.val_sampling_params = {'temperature': .7, 'top_p': 1.0}
        engine.chat_parser = SimpleNamespace(parse=lambda *a, **k: 'prompt',
            parse_completion=lambda *a, **k: {'content': 'answer', 'reasoning': None, 'tool_calls': []})
        engine.tokenizer = SimpleNamespace(encode=lambda *a, **k: [1], decode=lambda *a, **k: 'answer')
        def choose(key):
            sticky.append(key)
            return server
        engine.server_manager = SimpleNamespace(_choose_server=choose,
                                                _release_server=lambda s: released.append(s))
        outputs = await asyncio.gather(*[engine.get_model_response(
            [{'role':'user','content':'same episode parallel worker'}],
            application_id='episode:4', agent_name=route, max_tokens=8) for _ in range(12)])
        assert len(outputs) == len(calls) == len(released) == 12
        assert sticky == ['episode:4'] * 12
        assert len({c['request_id'] for c in calls}) == 12
        assert all(c['request_id'] != 'episode:4' for c in calls)
        assert all(c['full_parameter_route'] == route and 'lora_int_id' not in c for c in calls)
        assert all(c['sampling_params'] == {'temperature':.7,'top_p':1.0,'max_tokens':8} for c in calls)
        assert not active
    asyncio.run(run())


@pytest.mark.parametrize('first,other', [('eval_opt','orch_workers'),('orch_workers','eval_opt')])
def test_tail_keeps_first_lane_working_until_peer_final_validation(tmp_path, first, other):
    from rllm.trainer.verl.inline_full_validation import tail_verification, write_record, TailAuditComplete
    output=tmp_path/'validation'
    for step in range(10,201,10):
        write_record(output/f'step_{step:03d}.json',{'canonical':True,'step':step,'num_total':1412})
    original=(output/'step_200.json').read_bytes()
    cfg=OmegaConf.create({'trainer':{'role_wise_full_parameter':True,'inline_full_validation':{
        'enable':True,'canonical':True,'workflow':first,'output':str(output),
        'tail_control_dir':str(tmp_path/'control')}}})
    calls=[]
    def validate():
        assert not (tmp_path/'control'/f'{other}.json').exists()
        calls.append(cfg.trainer.inline_full_validation.output)
        assert not cfg.trainer.inline_full_validation.canonical
        write_record(tmp_path/'control'/f'{other}.json',{'completed_step':200,'online_validations':20})
        raise TailAuditComplete()
    trainer=SimpleNamespace(config=cfg,global_steps=201,_validate_agent=validate)
    tail_verification(trainer)
    assert len(calls)==1 and 'AUDIT_ONLY_NOT_CANONICAL' in calls[0]
    from pathlib import Path
    assert (Path(calls[0])/'AUDIT_ABORTED_AFTER_PRIMARY_COMPLETION.json').is_file()
    assert trainer.global_steps==201 and cfg.trainer.inline_full_validation.canonical
    assert (output/'step_200.json').read_bytes()==original
