from pathlib import Path
import json
import sys

import pytest

from aspect.evaluate import build_command


def run_manifest(canonical=False, method='ASPECT'):
    return {'canonical': canonical, 'method': method, 'task': 'math', 'overrides': {
        'trainer.total_training_steps': 200 if canonical else 2,
        'trainer.experiment_name': 'display-name-SMOKE_NOT_CANONICAL',
        'trainer.default_local_dir': '/run/checkpoints/orchestrator_workers_propose-qwen3_1.7b-agent_lorasb-math',
        'actor_rollout_ref.model.path': '/model',
        '+actor_rollout_ref.model.agent_lorasb': {'basis_path': '/basis'},
    }}


def command(run=None, **kwargs):
    return build_command(run or run_manifest(), Path('/output/result.json'), Path('/tmp/test_eval'), **kwargs)


def test_smoke_requires_explicit_noncanonical_selection():
    with pytest.raises(ValueError, match='smoke'):
        command(steps=[2])


def test_smoke_uses_real_checkpoint_directory_and_disables_canonical_results():
    result = command(steps=[2], smoke=True, gpus=2, max_samples=4)
    assert result[2] == 'dashboard.tp1_replica_eval'
    assert result[result.index('--data-parallel') + 1] == '2'
    assert result[result.index('--eval-interval') + 1] == '1'
    assert result[result.index('--max-samples') + 1] == '4'
    assert '--trajectory-output-dir' in result
    assert result[result.index('--experiment-filter') + 1] == '^orchestrator_workers_propose\\-qwen3_1\\.7b\\-agent_lorasb\\-math$'
    assert '--use-training-lengths' in result
    assert result[result.index('--agent-lorasb-basis-path') + 1] == '/basis'


def test_formal_contract_stays_four_gpu_full_split():
    result = command(run_manifest(canonical=True))
    assert result[2] == 'dashboard.tp1_replica_eval'
    assert '--max-samples' not in result
    assert result[result.index('--data-parallel') + 1] == '4'
    assert result[result.index('--eval-interval') + 1] == '10'
    assert result[result.index('--dataset') + 1] == 'dapo_math'
    assert result[result.index('--n-rollouts') + 1] == '1'


@pytest.mark.parametrize('options', [{'gpus': 2}, {'max_samples': 4}, {'steps': [1]}, {'steps': [10, 10]}])
def test_formal_contract_rejects_reduced_or_duplicate_points(options):
    with pytest.raises(ValueError):
        command(run_manifest(canonical=True), **options)


def test_full_parameter_weights_cannot_be_reconstructed_from_checkpoints():
    with pytest.raises(ValueError, match='live'):
        command(run_manifest(method='RS_Full_FT'), smoke=True, gpus=2, max_samples=4, steps=[2])


def test_smoke_cannot_evaluate_an_untrained_step():
    with pytest.raises(ValueError, match='checkpoint'):
        command(smoke=True, gpus=2, max_samples=4, steps=[3])


@pytest.mark.parametrize('marker', ['INCOMPLETE.json', 'INVALID_DO_NOT_USE.json'])
def test_failed_run_cannot_publish_canonical_validation(tmp_path, monkeypatch, marker):
    import aspect.evaluate as evaluate
    import aspect.preflight as preflight
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'run.json').write_text(json.dumps(run_manifest(canonical=True)))
    (root / marker).write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--run', str(root), '--output', str(tmp_path/'result.jsonl')])
    monkeypatch.setattr(preflight, 'check_gpus', lambda *_: None)
    monkeypatch.setattr(preflight, 'reward_canary', lambda *_: None)
    monkeypatch.setattr(evaluate, 'run_evaluator', lambda *_a, **_k: None)
    with pytest.raises(ValueError, match='canonical'):
        evaluate.main()
