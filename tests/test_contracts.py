import itertools
import json
from pathlib import Path

import pytest

from aspect.config import AGENT_ROUTES, METHODS, ROLE_ROUTES, WORKFLOWS, build_overrides, render


def cfg(method='ASPECT', workflow='eval_opt', task='math', scale='1.7b', **kwargs):
    return build_overrides(method=method, workflow=workflow, task=task, scale=scale,
                           model='/model', output='/run', ray_temp='/tmp/as_test',
                           basis='/basis' if method == 'ASPECT' else None, **kwargs)


@pytest.mark.parametrize('method,workflow,task,scale', tuple(itertools.product(METHODS[:4], WORKFLOWS, ('math', 'code'), ('0.6b', '1.7b', '4b'))))
def test_lora_matrix(method, workflow, task, scale):
    c = cfg(method, workflow, task, scale)
    assert c['data.train_batch_size'] == 64
    assert c['actor_rollout_ref.rollout.n'] == 8
    assert c['data.seed'] == 42
    assert c['actor_rollout_ref.actor.optim.lr_warmup_steps'] == 15
    assert c['actor_rollout_ref.rollout.tensor_model_parallel_size'] == 1
    assert c['actor_rollout_ref.rollout.data_parallel_size'] == 1
    assert c['actor_rollout_ref.rollout.max_num_batched_tokens'] == 65536
    assert c['trainer.total_training_steps'] == 200
    assert c['trainer.save_freq'] == 10 and c['trainer.test_freq'] == -1
    assert c['trainer.paper_metrics.require_complete']
    assert c['trainer.agent_names'] == (AGENT_ROUTES if method in ('AS', 'ASPECT') else ROLE_ROUTES)[workflow]
    assert c['trainer.share_policy'] == (method == 'WS')
    assert c['trainer.agent_wise_lora'] == (method == 'AS')
    if method == 'ASPECT':
        assert c['actor_rollout_ref.model.lora_alpha'] == 64
        assert c['+actor_rollout_ref.model.agent_lorasb']['routes'] == AGENT_ROUTES[workflow]
        assert c['actor_rollout_ref.actor.optim.lr'] == (1e-4 if scale == '0.6b' else 2e-4)
        if workflow != 'eval_opt':
            assert c['+rllm.workflow.workflow_args.agent_lorasb_instance_routing']
            assert '+rllm.workflow.workflow_args.agent_wise_lora_instance_routing' not in c
    else:
        assert c['actor_rollout_ref.model.lora_alpha'] == 32
        assert c['actor_rollout_ref.actor.optim.lr'] == 2e-5


@pytest.mark.parametrize('method,workflow,lr', tuple(itertools.product(METHODS[4:], ('eval_opt', 'orch_workers'), (1e-6, 2e-5))))
def test_full_parameter_contract(method, workflow, lr):
    c = cfg(method, workflow, learning_rate=lr)
    assert c['actor_rollout_ref.model.lora_rank'] == 0
    assert c['trainer.save_freq'] == -1
    assert c['trainer.resume_mode'] == 'disable'
    assert c['trainer.test_freq'] == 10
    assert c['+trainer.inline_full_validation']['expected_rows'] == 1412
    assert c['+trainer.inline_full_validation']['canonical']
    if method != 'WS_Full_FT':
        assert c['+actor_rollout_ref.model.full_parameter_routes'] == (AGENT_ROUTES if method == 'AS_Full_FT' else ROLE_ROUTES)[workflow]


@pytest.mark.parametrize('method,workflow', tuple(itertools.product(METHODS, ('eval_opt', 'orch_workers'))))
def test_actual_hydra_composition(method, workflow):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    from aspect.entrypoint import register_resolvers
    register_resolvers()
    c = cfg(method, workflow)
    with initialize_config_module(config_module='rllm.trainer.config', version_base=None):
        resolved = compose(config_name='multi_agent_ppo_trainer', overrides=render(c))
    resolved = OmegaConf.to_container(resolved, resolve=True)
    assert resolved['actor_rollout_ref']['actor']['optim']['lr'] > 0
    assert resolved['trainer']['total_training_steps'] == 200


def test_guards_reject_unvalidated_formal_variants():
    with pytest.raises(ValueError, match='Formal'):
        cfg(steps=2)
    with pytest.raises(ValueError, match='validated Full-FT'):
        cfg('RS_Full_FT', 'voting')
    with pytest.raises(ValueError, match='Learning rate'):
        cfg(learning_rate=0)


def test_smoke_is_explicitly_noncanonical():
    c = cfg('RS_Full_FT', smoke=True, steps=2, gpus=1)
    assert c['trainer.test_freq'] == 1
    assert c['trainer.save_freq'] == -1
    assert not c['+trainer.inline_full_validation']['canonical']
    assert 'SMOKE_NOT_CANONICAL' in c['trainer.experiment_name']


def test_lora_smoke_keeps_the_complete_diagnostics_gate():
    c = cfg('ASPECT', smoke=True, steps=2, gpus=1)
    assert c['trainer.save_freq'] == 1
    assert c['trainer.paper_metrics.validation_checkpoint_interval'] == 1
    assert c['trainer.paper_metrics.require_complete']


def test_renderer_quotes_paths_and_keeps_hydra_mappings():
    value = {'+x': {'path': '/a path/b', 'route': ['worker0', 'worker1'], 'enable': True}}
    rendered = render(value)
    assert rendered == ['+x={path:"/a path/b",route:["worker0","worker1"],enable:true}']
