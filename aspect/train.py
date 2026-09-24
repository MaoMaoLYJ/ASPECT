"""Launch an isolated training cell on user-selected GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from aspect.config import METHODS, WORKFLOWS, build_overrides, entrypoint, render


def runtime_env(output):
    env = os.environ.copy()
    env.pop("RAY_ADDRESS", None)
    source_root = str(Path(__file__).resolve().parents[1])
    paths = [source_root, *env.get('PYTHONPATH', '').split(os.pathsep)]
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    env.update(VLLM_ATTENTION_BACKEND="FLASH_ATTN", VLLM_USE_V1="1",
               VLLM_ALLOW_LONG_MAX_MODEL_LEN="1", VLLM_ALLOW_RUNTIME_LORA_UPDATING="True",
               VLLM_ENGINE_ITERATION_TIMEOUT_S="100000000000",
               VLLM_LOGGING_LEVEL="INFO", VERL_LOGGING_LEVEL="INFO",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               RLLM_REGISTRY_DIR=str(output / "registry"), RLLM_DATASET_DIR=str(output / "datasets"),
               VERL_FILE_LOGGER_PATH=str(output / "paper_metrics.jsonl"),
               TENSORBOARD_DIR=str(output / "tensorboard"))
    return env


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--workflow", choices=WORKFLOWS, required=True)
    p.add_argument("--task", choices=("math", "code"), required=True)
    p.add_argument("--scale", choices=("0.6b", "1.7b", "4b"), default="1.7b")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--basis", type=Path)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--cpus", type=int, default=min(88, len(os.sched_getaffinity(0))) if hasattr(os, 'sched_getaffinity') else 8)
    p.add_argument("--gpu-memory", type=float, default=0.85)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dry-run", action="store_true", help="Print the resolved override contract without loading ML libraries")
    p.add_argument("--config-only", action="store_true", help="Compose the actual Hydra configuration without training")
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--rollouts', type=int, default=8)
    p.add_argument('--checkpoint-interval', type=int, default=10)
    p.add_argument('--validation-interval', type=int, default=10)
    p.add_argument('--max-prompt-length', type=int)
    p.add_argument('--max-response-length', type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    model, data = args.model.expanduser().resolve(), args.data.expanduser().resolve()
    basis = args.basis.expanduser().resolve() if args.basis else None
    # Ray appends a long session/socket suffix; keep the parent short.
    ray_temp = '/tmp/as_' + str(os.getpid())
    cfg = build_overrides(method=args.method, workflow=args.workflow, task=args.task,
                          scale=args.scale, model=model, output=output, ray_temp=ray_temp,
                          cpus=args.cpus, gpus=args.gpus, basis=basis,
                          learning_rate=args.learning_rate, steps=args.steps,
                          gpu_memory=args.gpu_memory, batch_size=args.batch_size, rollouts=args.rollouts,
                          checkpoint_interval=args.checkpoint_interval, validation_interval=args.validation_interval,
                          max_prompt_length=args.max_prompt_length, max_response_length=args.max_response_length)
    command = [sys.executable, '-m', 'aspect.entrypoint', entrypoint(args.method, args.workflow, args.task), *render(cfg)]
    if args.dry_run:
        print(json.dumps({'canonical': True, 'overrides': cfg, 'command': command}, indent=2))
        return
    env = runtime_env(output)
    if args.config_only:
        subprocess.run([*command, '--cfg', 'job', '--resolve'], env=env, check=True)
        return
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {output}')
    if not (model / 'config.json').is_file() or not data.exists():
        raise ValueError('Download the model and prepare the task dataset before training')
    from aspect.preflight import check_gpus, reward_canary, validate_basis
    check_gpus(args.gpus)
    if basis:
        validate_basis(basis, model, args.task, cfg['actor_rollout_ref.actor.optim.lr'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    (output / 'run.json').write_text(json.dumps({'method': args.method, 'task': args.task,
        'workflow': args.workflow, 'scale': args.scale, 'canonical': True,
        'overrides': cfg, 'command': command,
        'started_unix': time.time()}, indent=2) + '\n')
    prepare = (['-m', 'aspect.data.math', '--train-root', str(data)] if args.task == 'math' else
               ['-m', 'aspect.data.code', 'register', '--package-root', str(data)])
    subprocess.run([sys.executable, *prepare, '--summary', str(output / 'data_summary.json')], env=env, check=True)
    reward_canary(args.task)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from rllm.utils.process_tree import ProcessTree
    process, tree = None, None
    def stop(*_):
        raise KeyboardInterrupt('Training interrupted')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with (output / 'logs/train.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, cwd=Path(__file__).resolve().parents[1])
            tree = ProcessTree(process)
            code = process.wait()
            if code:
                raise RuntimeError(f'Training failed (exit {code}); inspect {output / "logs/train.log"}')
        from aspect.audit import audit_training
        audit_training(output)
    except BaseException as exc:
        (output / 'INCOMPLETE.json').write_text(json.dumps({'error': str(exc)}) + '\n')
        raise
    finally:
        if tree is not None:
            tree.stop()
        elif process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == '__main__':
    main()
