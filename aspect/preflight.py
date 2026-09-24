"""Fail closed on occupied GPUs, invalid basis provenance, and broken rewards."""
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import warnings


def check_gpus(count):
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(selected) != count or any(not item.strip() for item in selected):
        raise ValueError('Set CUDA_VISIBLE_DEVICES explicitly to exactly the GPUs allocated to this run')
    for gpu in selected:
        result = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used',
                                          '--format=csv,noheader,nounits'], text=True)
        if int(result.strip()) > 64:
            warnings.warn(f'GPU {gpu} already has allocated memory; ensure sufficient capacity and permission to use it', RuntimeWarning)
    import torch
    if torch.cuda.device_count() != count:
        raise RuntimeError('Visible CUDA device count differs from the allocation')


def validate_basis(basis, model, task, learning_rate):
    from verl.workers.actor.agent_lorasb import load_basis_artifact
    _, manifest = load_basis_artifact(basis)
    expected = {'rank': 64, 'lora_alpha': 64, 'calibration_task': task,
                'calibration_samples': 50, 'max_seq_length': 512, 'n_iter': 10,
                'random_state': 42, 'actor_learning_rate': learning_rate, 'warmup_steps': 15,
                'model_config_sha256': hashlib.sha256((Path(model) / 'config.json').read_bytes()).hexdigest()}
    differences = {k: (manifest.get(k), value) for k, value in expected.items() if manifest.get(k) != value}
    if differences:
        raise ValueError(f'Calibration artifact does not match this experiment: {differences}')


def _reward_probe(task):
    if task == 'math':
        from rllm.rewards.math_reward import rllm_reward_fn_math
        good = rllm_reward_fn_math('dapo_math', r'The answer is \boxed{42}.', '42')
        bad = rllm_reward_fn_math('dapo_math', r'The answer is \boxed{41}.', '42')
    else:
        from rllm.rewards.reward_fn import code_reward_fn
        problem = {'problem': 'Echo the input integer.', 'data_source': 'livecodebench',
                   'ground_truth': [{'input': '17\n', 'output': '17\n', 'testtype': 'stdin'}]}
        good = code_reward_fn(problem, '```python\nimport sys\nprint(sys.stdin.read().strip())\n```')
        bad = code_reward_fn(problem, '```python\nprint(0)\n```')
    if not good.is_correct or bad.is_correct or good.reward <= bad.reward:
        raise RuntimeError(f'{task} reward canary failed')
    return {'correct': good.reward, 'incorrect': bad.reward}


def reward_canary(task):
    main = _reward_probe(task)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        spawned = pool.submit(_reward_probe, task).result(timeout=120)
    return {'passed': True, 'main': main, 'spawned': spawned}
