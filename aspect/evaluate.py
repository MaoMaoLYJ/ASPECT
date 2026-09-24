"""Evaluate real adapter checkpoints, with a separate non-canonical smoke mode."""
import argparse
import json
import os
import re
import signal
from pathlib import Path
import subprocess
import sys
import tempfile


def run_evaluator(command, env):
    from rllm.utils.process_tree import ProcessTree
    previous = {item: signal.getsignal(item) for item in (signal.SIGTERM, signal.SIGINT)}
    process, tree = None, None
    def interrupt(*_):
        raise KeyboardInterrupt('Evaluation interrupted')
    for item in previous:
        signal.signal(item, interrupt)
    try:
        process = subprocess.Popen(command, env=env, start_new_session=True,
                                   cwd=Path(__file__).resolve().parents[1])
        tree = ProcessTree(process)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        try:
            if tree is not None:
                tree.stop()
            elif process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        finally:
            for item, handler in previous.items():
                signal.signal(item, handler)


def build_command(run, output, temp, *, steps=None, smoke=False, gpus=4,
                  max_samples=None, gpu_memory=0.85, port=8000):
    if run['method'].endswith('_Full_FT'):
        raise ValueError('Full-FT evaluates live weights online, without saved checkpoints')
    if not run['canonical'] and not smoke:
        raise ValueError('A smoke training run requires explicit --smoke evaluation')
    if not 1 <= gpus <= 8 or not 0 < gpu_memory < 1 or not 1024 <= port <= 65536 - gpus:
        raise ValueError('Invalid evaluation GPU, memory, or port budget')
    cfg = run['overrides']
    if smoke:
        expected_rows = 1412 if run['task'] == 'math' else 1000
        if max_samples is None or not 1 <= max_samples <= expected_rows:
            raise ValueError('Smoke evaluation requires --max-samples within the task split')
        last_step = min(200, cfg['trainer.total_training_steps'])
        steps = steps if steps is not None else [last_step]
        allowed = range(1, last_step + 1)
    else:
        if gpus != 4 or max_samples is not None:
            raise ValueError('Formal evaluation uses four GPUs and the complete split')
        steps = steps if steps is not None else list(range(10, 201, 10))
        allowed = range(10, 201, 10)
    if not steps or len(steps) != len(set(steps)) or any(s not in allowed for s in steps):
        raise ValueError('Choose unique, actually trained checkpoint steps')
    checkpoint_root = Path(cfg['trainer.default_local_dir'])
    command = [sys.executable, '-m', 'dashboard.tp1_replica_eval',
               '--task-type', 'math' if run['task'] == 'math' else 'deepcoder',
               '--checkpoints-dir', str(checkpoint_root.parent),
               '--base-model', cfg['actor_rollout_ref.model.path'],
               '--experiment-filter', '^' + re.escape(checkpoint_root.name) + '$',
               '--dataset', 'dapo_math' if run['task'] == 'math' else 'deepcoder_primeintellect',
               '--step-filter', *map(str, steps), '--include-problem-results',
               '--eval-interval', '1' if smoke else '10',
               '--output-json', str(output.resolve()), '--tensor-parallel', '1', '--data-parallel', str(gpus),
               '--n-parallel', '512', '--n-rollouts', '1', '--temperature', '0.7',
               '--gpu-memory-utilization', str(gpu_memory), '--port', str(port),
               '--discovery-manifest', str(temp / 'discovery.json')]
    if run['task'] == 'math':
        command += ['--use-training-lengths']
    if run['method'] == 'ASPECT':
        basis = cfg['+actor_rollout_ref.model.agent_lorasb']['basis_path']
        command += ['--agent-lorasb-basis-path', basis, '--agent-lorasb-temp-root', str(temp / 'adapters')]
    if smoke:
        command += ['--max-samples', str(max_samples), '--trajectory-output-dir',
                    str(output.with_suffix('.trajectories'))]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', nargs='+', type=int)
    p.add_argument('--gpu-memory', type=float, default=0.85)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--smoke', action='store_true', help='Trajectory-only, never a canonical result')
    p.add_argument('--gpus', type=int, default=4)
    p.add_argument('--max-samples', type=int)
    args = p.parse_args()
    root = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    run = json.loads((root / 'run.json').read_text())
    if not args.smoke and ((root / 'INCOMPLETE.json').exists() or any(root.glob('INVALID_DO_NOT_USE*'))):
        raise ValueError('An interrupted or invalid run cannot publish canonical validation')
    if output.exists() or (args.smoke and output.with_suffix('.trajectories').exists()):
        raise FileExistsError('Choose fresh validation output paths to avoid duplicate records')
    temp = Path(tempfile.mkdtemp(prefix='as_eval_'))
    command = build_command(run, output, temp, steps=args.steps, smoke=args.smoke,
                            gpus=args.gpus, max_samples=args.max_samples,
                            gpu_memory=args.gpu_memory, port=args.port)
    from aspect.train import runtime_env
    from aspect.preflight import check_gpus, reward_canary
    check_gpus(args.gpus)
    reward_canary(run['task'])
    env = runtime_env(root)
    env['UNITYMAS_EVAL_OWNED_TEMP_ROOT'] = str(temp)
    env['UNITYMAS_EVAL_STARTUP_TIMEOUT'] = '1800'
    env['UNITYMAS_EVAL_ENFORCE_EAGER'] = '1'
    output.parent.mkdir(parents=True, exist_ok=True)
    run_evaluator(command, env)
    if args.smoke:
        # The evaluator deliberately disables its canonical result writer in this mode.
        if output.exists():
            raise ValueError('Smoke evaluation unexpectedly wrote canonical results')
        steps = args.steps or [min(200, run['overrides']['trainer.total_training_steps'])]
        dataset = 'dapo_math' if run['task'] == 'math' else 'deepcoder_primeintellect'
        name = Path(run['overrides']['trainer.default_local_dir']).name
        for step in steps:
            directory = output.with_suffix('.trajectories') / name / dataset / f'step_{step}'
            files = list(directory.glob('eval_*.json'))
            if len(files) != args.max_samples:
                raise ValueError(f'Incomplete smoke trajectories at checkpoint {step}')
            for path in files:
                json.loads(path.read_text())
        from aspect.audit import write_json
        write_json(output, {'passed': True, 'canonical': False, 'purpose': 'checkpoint_reload_smoke',
                            'steps': steps, 'problems_per_step': args.max_samples})
    print(f'Completed result records: {output}')


if __name__ == '__main__':
    main()
