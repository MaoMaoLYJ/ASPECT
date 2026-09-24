"""Bounded recovery of ESTALE for idempotent artifact I/O, never model updates."""
import errno
from functools import wraps
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def retry_estale(operation):
    @wraps(operation)
    def reopen(*args, **kwargs):
        for attempt in range(4):
            try:
                return operation(*args, **kwargs)
            except OSError as exc:
                if exc.errno != errno.ESTALE or attempt == 3:
                    raise
                print(json.dumps({'event': 'shared_volume_estale_retry',
                    'operation': operation.__qualname__, 'retry': attempt + 1,
                    'error': str(exc)}), flush=True)
                time.sleep(2 ** attempt)
    return reopen


@retry_estale
def stage_checkpoint(source, target, identity):
    from scripts.repro.run_qwen4_math_recovery import checkpoint_identity
    source, target = Path(source), Path(target)
    if not identity or checkpoint_identity(source) != identity:
        raise ValueError('Source checkpoint identity changed before staging')
    if target.exists():
        if checkpoint_identity(target) != identity:
            raise ValueError('Staged checkpoint identity mismatch')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        for relative in identity:
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe checkpoint identity path')
            destination = temporary/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/path, destination)
        if checkpoint_identity(temporary) != identity or checkpoint_identity(source) != identity:
            raise ValueError('Checkpoint identity changed during staging')
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target
