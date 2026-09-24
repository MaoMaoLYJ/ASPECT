from pathlib import Path


def test_installed_training_uses_its_bundled_resource_directory(tmp_path, monkeypatch):
    import aspect.audit as audit
    import aspect.preflight as preflight
    import aspect.train as train
    import rllm.utils.process_tree as lifecycle

    model, data = tmp_path / 'model', tmp_path / 'data'
    model.mkdir()
    data.mkdir()
    (model / 'config.json').write_text('{}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preflight, 'check_gpus', lambda *_: None)
    monkeypatch.setattr(preflight, 'reward_canary', lambda *_: None)
    monkeypatch.setattr(train.subprocess, 'run', lambda *_a, **_k: None)
    monkeypatch.setattr(train.resource, 'setrlimit', lambda *_: None)
    monkeypatch.setattr(train.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(audit, 'audit_training', lambda *_: None)
    seen = []

    class Process:
        returncode = 0
        def poll(self):
            return 0

    def popen(command, **kwargs):
        assert kwargs['cwd'] == Path(train.__file__).resolve().parents[1]
        assert kwargs['start_new_session'] is True
        seen.append(command)
        return Process()

    class Tree:
        def __init__(self, process):
            pass
        def stop(self):
            pass

    monkeypatch.setattr(train.subprocess, 'Popen', popen)
    monkeypatch.setattr(lifecycle, 'ProcessTree', Tree)
    train.main(['--method', 'WS', '--workflow', 'orch_workers', '--task', 'math',
                '--model', str(model), '--data', str(data), '--output', str(tmp_path/'run'),
                '--smoke', '--quick-smoke', '--steps', '2', '--gpus', '2'])
    assert len(seen) == 1
