from aspect.train import runtime_env
from pathlib import Path
import os
import aspect.train


def test_reference_audit_logging_levels_are_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv('VERL_LOGGING_LEVEL', raising=False)
    monkeypatch.delenv('VLLM_LOGGING_LEVEL', raising=False)
    env = runtime_env(tmp_path)
    assert env['VERL_LOGGING_LEVEL'] == 'INFO'
    assert env['VLLM_LOGGING_LEVEL'] == 'INFO'


def test_launcher_does_not_attach_to_an_unrelated_ray_cluster(tmp_path, monkeypatch):
    monkeypatch.setenv('RAY_ADDRESS', 'unrelated-cluster:9999')
    env = runtime_env(tmp_path)
    assert 'RAY_ADDRESS' not in env
    assert env['VERL_FILE_LOGGER_PATH'] == str(tmp_path / 'paper_metrics.jsonl')
    assert env['TENSORBOARD_DIR'] == str(tmp_path / 'tensorboard')


def test_worker_imports_use_this_release_before_unrelated_checkouts(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/unrelated/checkouts')
    env = runtime_env(tmp_path)
    expected = str(Path(aspect.train.__file__).resolve().parents[1])
    assert env['PYTHONPATH'].split(os.pathsep)[0] == expected
