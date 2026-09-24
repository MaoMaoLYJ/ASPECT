import subprocess
from pathlib import Path

import pytest

from aspect.evaluate import run_evaluator


@pytest.mark.parametrize('outcome', [0, 7, KeyboardInterrupt()])
def test_evaluator_always_cleans_only_its_owned_process_tree(monkeypatch, outcome):
    import aspect.evaluate as evaluate
    import rllm.utils.process_tree as lifecycle
    seen = []

    class Process:
        def wait(self):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    process = Process()

    def popen(command, **kwargs):
        assert kwargs['start_new_session'] is True
        assert kwargs['cwd'] == Path(evaluate.__file__).resolve().parents[1]
        seen.append(('start', command))
        return process

    class Tree:
        def __init__(self, owned):
            assert owned is process
        def stop(self):
            seen.append(('stop', process))

    monkeypatch.setattr(evaluate.subprocess, 'Popen', popen)
    monkeypatch.setattr(lifecycle, 'ProcessTree', Tree)
    if isinstance(outcome, BaseException):
        with pytest.raises(KeyboardInterrupt):
            run_evaluator(['owned-evaluator'], {})
    elif outcome:
        with pytest.raises(subprocess.CalledProcessError):
            run_evaluator(['owned-evaluator'], {})
    else:
        run_evaluator(['owned-evaluator'], {})
    assert seen == [('start', ['owned-evaluator']), ('stop', process)]
