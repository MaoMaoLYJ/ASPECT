"""Artifact completion checks; missing results are never interpolated."""
import hashlib
import json
import math
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def validation_record(row, step, canonical=True, expected_rows=1412):
    if expected_rows < 1 or (canonical and expected_rows != 1412):
        raise ValueError('Formal validation must cover all 1412 problems')
    if row['step'] != step or row['num_total'] != expected_rows or row['canonical'] != canonical or row['checkpoint_saved']:
        raise ValueError('Invalid full-split online validation contract')
    counts = row['per_problem_n_correct']
    if len(counts) != expected_rows or any(n not in (0, 1) for n in counts) or sum(counts) != row['num_correct']:
        raise ValueError('Per-problem validation counts do not match')
    if (row['dataset'] != 'dapo_math' or row['split'] != 'test' or row['n_rollouts'] != 1
            or not math.isclose(row['accuracy'], sum(counts) / expected_rows, rel_tol=1e-12)):
        raise ValueError('Official validation split or accuracy mismatch')
    if len(row['dataset_sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['dataset_sha256']):
        raise ValueError('Missing validation split hash')


def audit_training(output):
    output = Path(output)
    run = json.loads((output / 'run.json').read_text())
    cfg, canonical = run['overrides'], run['canonical']
    expected = list(range(1, cfg['trainer.total_training_steps'] + 1))
    rows = [json.loads(line) for line in (output / 'paper_metrics.jsonl').read_text().splitlines() if line.strip()]
    seen = []
    for row in rows:
        metrics = row.get('data', {})
        step = int(metrics.get('training/global_step', row.get('step', -1)))
        if step == 0:
            continue
        seen.append(step)
        if 'batch/success' not in metrics or not any(k.startswith('paper_diag/policy/') for k in metrics):
            raise ValueError(f'Missing success or policy diagnostics at step {step}')
        for key, value in metrics.items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f'Non-finite metric {key} at step {step}')
        if step % cfg['trainer.gradient_diagnostics.interval'] == 0 and not any(k.endswith('/probe') and v == 1 for k, v in metrics.items()):
            raise ValueError(f'Missing gradient probe at step {step}')
    if seen != expected:
        raise ValueError('Training steps are incomplete or duplicated')
    if not any(p.stat().st_size for p in (output / 'tensorboard').rglob('events.out.tfevents.*')):
        raise ValueError('Missing TensorBoard diagnostics')
    full = run['method'].endswith('_Full_FT')
    validations = []
    if full:
        forbidden = [p for p in output.rglob('*') if p.suffix in ('.safetensors', '.bin', '.pt') or p.name.startswith('global_step_')]
        if forbidden:
            raise ValueError('Full-FT must not save model/optimizer/adapter checkpoints')
        for step in expected:
            if step % cfg['trainer.test_freq']:
                continue
            record = json.loads((output / 'validation' / f'step_{step:03d}.json').read_text())
            validation_record(record, step, canonical,
                              expected_rows=cfg['+trainer.inline_full_validation']['expected_rows'])
            validations.append(record)
    elif cfg['trainer.save_freq'] > 0:
        root = Path(cfg['trainer.default_local_dir'])
        interval = cfg['trainer.save_freq']
        for step in range(interval, cfg['trainer.total_training_steps'] + 1, interval):
            path = root / f'global_step_{step}'
            if run['method'] == 'ASPECT':
                artifact = path / 'agent_lorasb'
                manifest = json.loads((artifact / 'manifest.json').read_text())
                digest = hashlib.sha256((artifact / 'cores.safetensors').read_bytes()).hexdigest()
                if not (artifact / '_SUCCESS').is_file() or manifest.get('global_step') != step or manifest.get('cores_sha256') != digest:
                    raise ValueError(f'Invalid ASPECT checkpoint at step {step}')
            else:
                manifest = json.loads((path / 'actor/peft_checkpoint_manifest.json').read_text())
                if manifest.get('global_step') != step or manifest.get('checkpoint_format') != 'peft_eval_v1':
                    raise ValueError(f'Invalid adapter checkpoint at step {step}')
                for route in manifest['adapter_routes']:
                    directory = path / 'actor' / route['directory']
                    if not (directory / 'adapter_config.json').is_file() or not (directory / 'adapter_model.safetensors').stat().st_size:
                        raise ValueError(f'Missing adapter route at step {step}')
    report = {'passed': True, 'canonical': canonical, 'steps': len(seen),
              'full_validations': sum(r['num_total'] == 1412 for r in validations),
              'online_validations': len(validations),
              'checkpoint_saved': cfg['trainer.save_freq'] > 0}
    write_json(output / 'training_audit.json', report)
    if full:
        write_json(output / 'canonical_results_merged.json', {'passed': True, 'canonical': canonical, 'records': validations})
        write_json(output / ('_SUCCESS.json' if canonical else 'SMOKE_SUCCESS.json'), report)
    return report
