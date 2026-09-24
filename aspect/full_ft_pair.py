"""Two four-GPU Full-FT lanes with real, interruptible tail validation."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('WS_Full_FT', 'RS_Full_FT', 'AS_Full_FT'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-groups', nargs=2, required=True, metavar=('LANE0', 'LANE1'))
    p.add_argument('--learning-rate', type=float, default=1e-6)
    p.add_argument('--cpus-per-lane', type=int, default=16)
    p.add_argument('--gpu-memory', type=float, default=0.85)
    args = p.parse_args()
    groups = [g.split(',') for g in args.gpu_groups]
    if any(len(g) != 4 for g in groups) or len(set(groups[0] + groups[1])) != 8:
        raise ValueError('Provide two disjoint groups of four allocated GPUs')
    available = sorted(os.sched_getaffinity(0))
    lanes = [available[::2], available[1::2]]
    if args.cpus_per_lane < 2 or any(len(lane) < args.cpus_per_lane for lane in lanes):
        raise ValueError('CPU budget exceeds the disjoint per-lane allocation')
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    from rllm.utils.process_tree import ProcessTree
    from aspect.audit import write_json
    processes, trees = [], []
    def interrupt(*_):
        raise KeyboardInterrupt('Full-FT pair interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for workflow, group, cpus in zip(('eval_opt', 'orch_workers'), args.gpu_groups, lanes):
            command = ['taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-m', 'aspect.train', '--method', args.method,
                '--workflow', workflow, '--task', 'math', '--scale', '1.7b', '--model', str(args.model.resolve()),
                '--data', str(args.data.resolve()), '--output', str(root / workflow),
                '--learning-rate', str(args.learning_rate), '--cpus', str(args.cpus_per_lane),
                '--gpu-memory', str(args.gpu_memory), '--tail-control', str(root / 'primary_completion')]
            proc = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': group}, start_new_session=True)
            processes.append(proc)
            trees.append(ProcessTree(proc))
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in (None, 0) for proc in processes):
                raise RuntimeError('A lane failed; preserving partial outputs and stopping only this pair')
            time.sleep(2)
        records = []
        for workflow in ('eval_opt', 'orch_workers'):
            success = json.loads((root / workflow / '_SUCCESS.json').read_text())
            if not success['passed'] or success['steps'] != 200 or success['full_validations'] != 20:
                raise ValueError('A primary lane is not complete')
            records.extend(json.loads((root / workflow / 'canonical_results_merged.json').read_text())['records'])
        write_json(root / 'canonical_results_merged.json', {'passed': True, 'records': records})
        write_json(root / '_SUCCESS.json', {'passed': True, 'training_cells': 2, 'full_validations': 40, 'checkpoint_saved': False})
    except BaseException as exc:
        write_json(root / 'INCOMPLETE.json', {'error': str(exc), 'canonical_complete': False})
        raise
    finally:
        for tree in trees:
            tree.stop()


if __name__ == '__main__':
    main()
